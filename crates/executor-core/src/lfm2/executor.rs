//! Backend-neutral LFM2 token execution over the finite portable inference operations.
//!
//! Prefixes are immutable snapshots. Append and fork stage independent backend-resident state
//! and publish only after a fence and final-logit readback complete.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use minifield_engine_api::{
    AllocationClass, BackendLease, CandidateScore, CompletionPoll, ExecutorError, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, PackedHeadSpec, Result,
    RotarySpec, Shape, TokenChunk, TokenExecutor, TokenId, TokenIds,
};

use super::{
    LayerKind, Lfm2Config, Lfm2LayerWeightRole, Lfm2ResolvedWeight, Lfm2TypedWeights,
    Lfm2WeightRole, NumericalMode,
};

/// Caller-selected logical cache capacity for one loaded model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lfm2ExecutionLimits {
    pub max_logical_tokens: u64,
}

impl Lfm2ExecutionLimits {
    fn validate(self, config: &Lfm2Config) -> Result<()> {
        if self.max_logical_tokens == 0 {
            return Err(ExecutorError::InvalidArgument(
                "LFM2 logical token capacity must be nonzero",
            ));
        }
        if let Some(maximum) = config.max_position_embeddings
            && self.max_logical_tokens > maximum
        {
            return Err(ExecutorError::OutOfBounds(
                "requested LFM2 cache capacity exceeds configured positions",
            ));
        }
        Ok(())
    }
}

struct ModelContext<B: InferenceOps> {
    backend: Rc<RefCell<B>>,
    retirement: Rc<B::FenceRetirement>,
    weights: Rc<Lfm2TypedWeights<B::Buffer>>,
    lease: BackendLease,
    owner: Rc<()>,
    limits: Lfm2ExecutionLimits,
    // A fence submission error after enqueued backend work is terminal for this executor
    // instance. Retain the affected source snapshots and allocations locally rather than
    // publishing or dropping them with an unknown device completion.
    quarantined: Cell<bool>,
    unfenced_buffers: RefCell<Vec<B::Buffer>>,
    unfenced_prefixes: RefCell<Vec<Lfm2Prefix<B>>>,
    // Source cache snapshots retained after an abandoned fenced task. They are released only
    // after the backend retirement queue reports no unresolved fence.
    abandoned_prefixes: RefCell<Vec<Lfm2Prefix<B>>>,
}

impl<B: InferenceOps> ModelContext<B> {
    fn borrow_backend(&self) -> Result<std::cell::RefMut<'_, B>> {
        self.backend.try_borrow_mut().map_err(|_| {
            ExecutorError::BackendFailure("backend is busy with another portable task")
        })
    }

    fn validate_backend(&self) -> Result<()> {
        if self.quarantined.get() {
            return Err(ExecutorError::BackendFailure(
                "LFM2 executor is quarantined after an unconfirmed fence submission failure",
            ));
        }
        let backend = self.backend.try_borrow().map_err(|_| {
            ExecutorError::BackendFailure("backend is busy with another portable task")
        })?;
        let observed = backend.lease();
        if !self.lease.same_actual_instance(&observed) {
            return Err(ExecutorError::WrongBackend);
        }
        if self.lease.identity() != observed.identity() {
            return Err(ExecutorError::StaleBuffer);
        }
        if !self.weights.inner().matches_backend_lease(&observed) {
            return Err(ExecutorError::WrongBackend);
        }
        Ok(())
    }

    fn validate_prefix(&self, prefix: &Lfm2Prefix<B>) -> Result<()> {
        if !Rc::ptr_eq(&self.owner, &prefix.owner)
            || !self.lease.same_actual_instance(&prefix.lease)
        {
            return Err(ExecutorError::WrongBackend);
        }
        if self.lease.identity() != prefix.lease.identity() {
            return Err(ExecutorError::StaleBuffer);
        }
        if prefix.config_sha256 != self.weights.inner().config_sha256()
            || prefix.asset_sha256 != self.weights.inner().asset_sha256()
            || prefix.numerical_mode != self.weights.config().numerical_mode
        {
            return Err(ExecutorError::InvalidArgument(
                "prefix model identity differs from executor model",
            ));
        }
        self.validate_backend()
    }

    fn config(&self) -> &Lfm2Config {
        self.weights.config()
    }

    fn quarantine_unfenced(&self, buffers: Vec<B::Buffer>, prefixes: Vec<Lfm2Prefix<B>>) {
        self.quarantined.set(true);
        self.unfenced_buffers.borrow_mut().extend(buffers);
        self.unfenced_prefixes.borrow_mut().extend(prefixes);
    }

    fn retain_abandoned_prefixes(&self, prefixes: Vec<Lfm2Prefix<B>>) {
        self.abandoned_prefixes.borrow_mut().extend(prefixes);
    }

    fn release_retired_prefixes_if_safe(&self) {
        if !self.retirement.has_unresolved() {
            self.abandoned_prefixes.borrow_mut().clear();
        }
    }
}

struct Tensor<B: InferenceOps> {
    buffer: B::Buffer,
    shape: Shape,
}

enum LayerCache<B: InferenceOps> {
    Conv {
        history: Tensor<B>,
    },
    Attention {
        key: Tensor<B>,
        value: Tensor<B>,
        length: u64,
    },
}

struct PrefixStorage<B: InferenceOps> {
    length: u64,
    history: Vec<TokenId>,
    layers: Vec<LayerCache<B>>,
    next_logits: Option<Tensor<B>>,
    /// Device-resident greedy argmax of `next_logits`, shape `[1]`. Feeding it
    /// to an embedding gather keeps token selection off the host. `sampled_id`
    /// is the same value resolved on the host during the publish readback.
    sampled: Option<Tensor<B>>,
    sampled_id: Option<TokenId>,
}

impl<B: InferenceOps> PrefixStorage<B> {
    fn into_buffers(self) -> Vec<B::Buffer> {
        let mut values = Vec::new();
        for layer in self.layers {
            match layer {
                LayerCache::Conv { history } => values.push(history.buffer),
                LayerCache::Attention { key, value, .. } => {
                    values.push(key.buffer);
                    values.push(value.buffer);
                }
            }
        }
        if let Some(logits) = self.next_logits {
            values.push(logits.buffer);
        }
        if let Some(sampled) = self.sampled {
            values.push(sampled.buffer);
        }
        values
    }
}

/// Opaque immutable causal prefix, including all cache and identity state.
pub struct Lfm2Prefix<B: InferenceOps> {
    owner: Rc<()>,
    lease: BackendLease,
    config_sha256: [u8; 32],
    asset_sha256: [u8; 32],
    numerical_mode: NumericalMode,
    storage: Rc<PrefixStorage<B>>,
}

impl<B: InferenceOps> Clone for Lfm2Prefix<B> {
    fn clone(&self) -> Self {
        Self {
            owner: Rc::clone(&self.owner),
            lease: self.lease.clone(),
            config_sha256: self.config_sha256,
            asset_sha256: self.asset_sha256,
            numerical_mode: self.numerical_mode,
            storage: Rc::clone(&self.storage),
        }
    }
}

impl<B: InferenceOps> Lfm2Prefix<B> {
    #[must_use]
    pub fn logical_length(&self) -> u64 {
        self.storage.length
    }
    #[must_use]
    pub fn token_history(&self) -> &[TokenId] {
        &self.storage.history
    }
}

/// Portable LFM2 executor that owns its backend and immutable typed weights.
pub struct Lfm2Executor<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
}

impl<B: InferenceOps> Lfm2Executor<B> {
    pub fn new(
        backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
    ) -> Result<Self> {
        weights.config().validate()?;
        limits.validate(weights.config())?;
        if !weights.config().tie_embedding {
            return Err(ExecutorError::InvalidTie);
        }
        let backend = Rc::new(RefCell::new(backend));
        let (lease, retirement) = {
            let observed = backend.borrow().lease();
            if !weights.inner().matches_backend_lease(&observed) {
                return Err(ExecutorError::WrongBackend);
            }
            let capabilities = backend.borrow().capabilities();
            for operation in [
                minifield_engine_api::OperationKind::Copy,
                minifield_engine_api::OperationKind::RectCopy2d,
                minifield_engine_api::OperationKind::GatherRows,
                minifield_engine_api::OperationKind::Add,
                minifield_engine_api::OperationKind::Multiply,
                minifield_engine_api::OperationKind::Linear,
                minifield_engine_api::OperationKind::RowRmsNorm,
                minifield_engine_api::OperationKind::Rotary,
                minifield_engine_api::OperationKind::GroupedQueryAttention,
                minifield_engine_api::OperationKind::GatedShortConvolution,
                minifield_engine_api::OperationKind::SwiGlu,
            ] {
                capabilities.validate(minifield_engine_api::DType::F32, operation, 2, 0, 0)?;
            }
            (observed, backend.borrow().fence_retirement())
        };
        validate_roles::<B>(&weights)?;
        Ok(Self {
            context: Rc::new(ModelContext {
                backend,
                retirement,
                weights: Rc::new(weights),
                lease,
                owner: Rc::new(()),
                limits,
                quarantined: Cell::new(false),
                unfenced_buffers: RefCell::new(Vec::new()),
                unfenced_prefixes: RefCell::new(Vec::new()),
                abandoned_prefixes: RefCell::new(Vec::new()),
            }),
        })
    }

    #[must_use]
    pub fn limits(&self) -> Lfm2ExecutionLimits {
        self.context.limits
    }
    pub fn poll_retired(&self) -> Result<()> {
        self.context.backend.borrow().poll_retired_fences()?;
        self.context.release_retired_prefixes_if_safe();
        Ok(())
    }

    /// Return current backend resource classifications without forcing synchronization.
    pub fn resource_report(&self) -> Result<minifield_engine_api::ResourceReport> {
        self.context.validate_backend()?;
        Ok(self.context.backend.borrow().resource_report())
    }

    /// Invalidate this loaded executor after its backend has been reloaded or quarantined.
    ///
    /// Loaded weights and every prefix remain bound to the old generation, so callers must
    /// construct a new executor from a freshly loaded model before submitting more work.
    pub fn advance_backend_generation(&mut self) -> Result<()> {
        self.context.borrow_backend()?.advance_generation()
    }

    fn accepted_tokens(&self, input: TokenChunk<'_>, base: u64) -> Result<Vec<TokenId>> {
        input.validate()?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(input.ids.len())
            .map_err(|_| ExecutorError::ResourceLimit("token chunk host allocation failed"))?;
        for (index, token) in input.ids.iter().copied().enumerate() {
            if input.is_valid(index)? {
                if token >= self.context.config().vocab_size {
                    return Err(ExecutorError::OutOfBounds(
                        "valid token ID exceeds loaded model vocabulary",
                    ));
                }
                tokens.push(token);
            }
        }
        let length = base
            .checked_add(
                u64::try_from(tokens.len()).map_err(|_| {
                    ExecutorError::Overflow("token chunk logical length exceeds u64")
                })?,
            )
            .ok_or(ExecutorError::Overflow(
                "prefix logical length overflows u64",
            ))?;
        if length > self.context.limits.max_logical_tokens {
            return Err(ExecutorError::OutOfBounds(
                "append exceeds configured logical prefix capacity",
            ));
        }
        Ok(tokens)
    }
}

fn validate_roles<B: InferenceOps>(weights: &Lfm2TypedWeights<B::Buffer>) -> Result<()> {
    let _ = weights.resolve(Lfm2WeightRole::TokenEmbedding)?;
    let _ = weights.resolve(Lfm2WeightRole::TiedLmHead)?;
    let _ = weights.resolve(Lfm2WeightRole::EmbeddingNorm)?;
    for (index, kind) in weights.config().layers.iter().copied().enumerate() {
        match kind {
            LayerKind::Conv => {
                for role in [
                    Lfm2LayerWeightRole::ConvKernel,
                    Lfm2LayerWeightRole::ConvInProjection,
                    Lfm2LayerWeightRole::ConvOutProjection,
                ] {
                    let _ = weights.resolve(layer_role(index, role))?;
                }
            }
            LayerKind::FullAttention => {
                for role in [
                    Lfm2LayerWeightRole::QueryNorm,
                    Lfm2LayerWeightRole::KeyNorm,
                    Lfm2LayerWeightRole::QueryProjection,
                    Lfm2LayerWeightRole::KeyProjection,
                    Lfm2LayerWeightRole::ValueProjection,
                    Lfm2LayerWeightRole::OutputProjection,
                ] {
                    let _ = weights.resolve(layer_role(index, role))?;
                }
            }
        }
        for role in [
            Lfm2LayerWeightRole::FfnW1,
            Lfm2LayerWeightRole::FfnW2,
            Lfm2LayerWeightRole::FfnW3,
            Lfm2LayerWeightRole::FfnNorm,
            Lfm2LayerWeightRole::OperatorNorm,
        ] {
            let _ = weights.resolve(layer_role(index, role))?;
        }
    }
    Ok(())
}

/// Linear projection through whichever operand set the role is stored as:
/// dense f32 weights or `minifield.ternary.v1` packed code/scale streams.
fn weight_linear<B: InferenceOps>(
    backend: &B,
    output: &mut B::Buffer,
    input: &B::Buffer,
    weights: &Lfm2TypedWeights<B::Buffer>,
    role: Lfm2WeightRole,
) -> Result<()> {
    match weights.resolve(role)? {
        Lfm2ResolvedWeight::Dense(weight) => backend.linear(output, input, weight),
        Lfm2ResolvedWeight::Packed { codes, scales } => {
            backend.packed_linear(output, input, codes, scales)
        }
    }
}

fn layer_role(index: usize, role: Lfm2LayerWeightRole) -> Lfm2WeightRole {
    Lfm2WeightRole::Layer { index, role }
}
fn shape(rows: u64, columns: u64) -> Result<Shape> {
    Shape::new(&[rows, columns])
}
fn allocate<B: InferenceOps>(
    backend: &mut B,
    shape: Shape,
    class: AllocationClass,
) -> Result<Tensor<B>> {
    Ok(Tensor {
        buffer: backend.allocate_f32_classified(shape, class)?,
        shape,
    })
}
fn clone_tensor<B: InferenceOps>(backend: &mut B, source: &Tensor<B>) -> Result<Tensor<B>> {
    let mut output = allocate(backend, source.shape, AllocationClass::Cache)?;
    backend.copy(&mut output.buffer, &source.buffer)?;
    Ok(output)
}

fn publish<B: InferenceOps>(context: &ModelContext<B>, storage: PrefixStorage<B>) -> Lfm2Prefix<B> {
    Lfm2Prefix {
        owner: Rc::clone(&context.owner),
        lease: context.lease.clone(),
        config_sha256: context.weights.inner().config_sha256(),
        asset_sha256: context.weights.inner().asset_sha256(),
        numerical_mode: context.config().numerical_mode,
        storage: Rc::new(storage),
    }
}
fn allocate_empty<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
) -> Result<PrefixStorage<B>> {
    let config = context.config();
    let mut history = Vec::new();
    history
        .try_reserve_exact(
            usize::try_from(context.limits.max_logical_tokens).map_err(|_| {
                ExecutorError::Overflow("configured logical capacity exceeds usize")
            })?,
        )
        .map_err(|_| ExecutorError::ResourceLimit("prefix token-history allocation failed"))?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers.len())
        .map_err(|_| ExecutorError::ResourceLimit("prefix layer-state allocation failed"))?;
    let kv_width = u64::from(config.key_value_heads)
        .checked_mul(u64::from(config.head_dim))
        .ok_or(ExecutorError::Overflow("LFM2 cache width overflows u64"))?;
    for kind in &config.layers {
        match kind {
            LayerKind::Conv => layers.push(LayerCache::Conv {
                history: allocate(
                    backend,
                    shape(
                        u64::from(config.conv_width - 1),
                        u64::from(config.hidden_size),
                    )?,
                    AllocationClass::Cache,
                )?,
            }),
            LayerKind::FullAttention => layers.push(LayerCache::Attention {
                key: allocate(
                    backend,
                    shape(context.limits.max_logical_tokens, kv_width)?,
                    AllocationClass::Cache,
                )?,
                value: allocate(
                    backend,
                    shape(context.limits.max_logical_tokens, kv_width)?,
                    AllocationClass::Cache,
                )?,
                length: 0,
            }),
        }
    }
    Ok(PrefixStorage {
        length: 0,
        history,
        layers,
        next_logits: None,
        sampled: None,
        sampled_id: None,
    })
}

fn clone_storage<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    source: &PrefixStorage<B>,
) -> Result<PrefixStorage<B>> {
    if source.length > context.limits.max_logical_tokens
        || source.history.len() as u64 != source.length
    {
        return Err(ExecutorError::InvalidArgument(
            "prefix cache metadata differs from logical token history",
        ));
    }
    let mut history = Vec::new();
    history
        .try_reserve_exact(source.history.len())
        .map_err(|_| ExecutorError::ResourceLimit("fork token-history allocation failed"))?;
    history.extend_from_slice(&source.history);
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(source.layers.len())
        .map_err(|_| ExecutorError::ResourceLimit("fork layer-state allocation failed"))?;
    for layer in &source.layers {
        match layer {
            LayerCache::Conv { history } => layers.push(LayerCache::Conv {
                history: clone_tensor(backend, history)?,
            }),
            LayerCache::Attention { key, value, length } => layers.push(LayerCache::Attention {
                key: clone_tensor(backend, key)?,
                value: clone_tensor(backend, value)?,
                length: *length,
            }),
        }
    }
    let next_logits = match &source.next_logits {
        Some(tensor) => Some(clone_tensor(backend, tensor)?),
        None => None,
    };
    let sampled = match &source.sampled {
        Some(tensor) => Some(clone_tensor(backend, tensor)?),
        None => None,
    };
    Ok(PrefixStorage {
        length: source.length,
        history,
        layers,
        next_logits,
        sampled,
        sampled_id: source.sampled_id,
    })
}

fn push_scratch<B: InferenceOps>(
    scratch: &mut Vec<B::Buffer>,
    values: impl IntoIterator<Item = B::Buffer>,
) {
    scratch.extend(values);
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn append_token<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    state: &mut PrefixStorage<B>,
    token: TokenId,
    ids: TokenIds<'_, B>,
    scratch: &mut Vec<B::Buffer>,
) -> Result<()> {
    let config = context.config();
    if token >= config.vocab_size {
        return Err(ExecutorError::OutOfBounds(
            "valid token ID exceeds loaded model vocabulary",
        ));
    }
    if state.length >= context.limits.max_logical_tokens {
        return Err(ExecutorError::OutOfBounds(
            "append exceeds configured logical prefix capacity",
        ));
    }
    let hidden = u64::from(config.hidden_size);
    let intermediate = u64::from(config.effective_intermediate_size);
    let kv_width = u64::from(config.key_value_heads)
        .checked_mul(u64::from(config.head_dim))
        .ok_or(ExecutorError::Overflow(
            "LFM2 key/value width overflows u64",
        ))?;
    let mut x = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
    match context.weights.resolve(Lfm2WeightRole::TokenEmbedding)? {
        Lfm2ResolvedWeight::Dense(embedding) => {
            backend.gather_rows(&mut x.buffer, embedding, ids)?;
        }
        Lfm2ResolvedWeight::Packed { codes, scales } => {
            backend.packed_gather_rows(&mut x.buffer, codes, scales, ids)?;
        }
    }

    // Decode-fusion layout: every residual add is fused with the RMS norm that
    // consumes its sum, so the loop carries `x` (the residual base) alongside
    // `u` (the already-normed input for the layer about to run). The final
    // iteration's fused norm uses the embedding norm weight, producing the
    // lm_head input without a separate pass.
    let mut u = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
    if !config.layers.is_empty() {
        backend.row_rms_norm(
            &mut u.buffer,
            &x.buffer,
            context
                .weights
                .buffer_for(layer_role(0, Lfm2LayerWeightRole::OperatorNorm))?,
            config.block_norm_epsilon,
        )?;
    }
    for (index, kind) in config.layers.iter().copied().enumerate() {
        let mut operator = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        match kind {
            LayerKind::Conv => {
                let mut projection = allocate(
                    backend,
                    shape(
                        1,
                        hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
                            "LFM2 convolution projection width overflows u64",
                        ))?,
                    )?,
                    AllocationClass::Scratch,
                )?;
                weight_linear(
                    backend,
                    &mut projection.buffer,
                    &u.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::ConvInProjection),
                )?;
                let mut convolved = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
                let Some(LayerCache::Conv { history }) = state.layers.get_mut(index) else {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix convolution cache does not match model layer",
                    ));
                };
                backend.gated_short_convolution(
                    &mut convolved.buffer,
                    &projection.buffer,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::ConvKernel))?,
                    &mut history.buffer,
                    GatedShortConvSpec::new(config.hidden_size, config.conv_width)?,
                )?;
                weight_linear(
                    backend,
                    &mut operator.buffer,
                    &convolved.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::ConvOutProjection),
                )?;
                push_scratch::<B>(scratch, [projection.buffer, convolved.buffer]);
            }
            LayerKind::FullAttention => {
                let mut q = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
                let mut k = allocate(backend, shape(1, kv_width)?, AllocationClass::Scratch)?;
                let mut v = allocate(backend, shape(1, kv_width)?, AllocationClass::Scratch)?;
                weight_linear(
                    backend,
                    &mut q.buffer,
                    &u.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::QueryProjection),
                )?;
                if let (
                    Lfm2ResolvedWeight::Packed {
                        codes: k_codes,
                        scales: k_scales,
                    },
                    Lfm2ResolvedWeight::Packed {
                        codes: v_codes,
                        scales: v_scales,
                    },
                ) = (
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::KeyProjection))?,
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::ValueProjection))?,
                ) {
                    backend.packed_linear_pair(
                        &mut k.buffer,
                        &mut v.buffer,
                        &u.buffer,
                        k_codes,
                        k_scales,
                        v_codes,
                        v_scales,
                    )?;
                } else {
                    weight_linear(
                        backend,
                        &mut k.buffer,
                        &u.buffer,
                        &context.weights,
                        layer_role(index, Lfm2LayerWeightRole::KeyProjection),
                    )?;
                    weight_linear(
                        backend,
                        &mut v.buffer,
                        &u.buffer,
                        &context.weights,
                        layer_role(index, Lfm2LayerWeightRole::ValueProjection),
                    )?;
                }
                let query_heads = PackedHeadSpec::new(config.attention_heads, config.head_dim)?;
                let key_value_heads = PackedHeadSpec::new(config.key_value_heads, config.head_dim)?;
                let mut q_rope = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
                let mut k_rope = allocate(backend, shape(1, kv_width)?, AllocationClass::Scratch)?;
                backend.qk_norm_rope(
                    &mut q_rope.buffer,
                    &mut k_rope.buffer,
                    &q.buffer,
                    &k.buffer,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::QueryNorm))?,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::KeyNorm))?,
                    &[state.length],
                    RotarySpec::new(query_heads, config.rope_theta)?,
                    key_value_heads,
                    config.block_norm_epsilon,
                )?;
                let mut attention = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
                let Some(LayerCache::Attention { key, value, length }) =
                    state.layers.get_mut(index)
                else {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix attention cache does not match model layer",
                    ));
                };
                backend.causal_gqa(
                    &mut attention.buffer,
                    &q_rope.buffer,
                    &k_rope.buffer,
                    &v.buffer,
                    &mut key.buffer,
                    &mut value.buffer,
                    length,
                    GqaSpec::new(
                        config.attention_heads,
                        config.key_value_heads,
                        config.head_dim,
                    )?,
                )?;
                weight_linear(
                    backend,
                    &mut operator.buffer,
                    &attention.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::OutputProjection),
                )?;
                push_scratch::<B>(
                    scratch,
                    [
                        q.buffer,
                        k.buffer,
                        v.buffer,
                        q_rope.buffer,
                        k_rope.buffer,
                        attention.buffer,
                    ],
                );
            }
        }
        let mut residual = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        let mut ffn_input = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        backend.add_row_rms_norm(
            &mut residual.buffer,
            &mut ffn_input.buffer,
            &x.buffer,
            &operator.buffer,
            context
                .weights
                .buffer_for(layer_role(index, Lfm2LayerWeightRole::FfnNorm))?,
            config.block_norm_epsilon,
        )?;
        let mut gate = allocate(backend, shape(1, intermediate)?, AllocationClass::Scratch)?;
        let mut up = allocate(backend, shape(1, intermediate)?, AllocationClass::Scratch)?;
        if let (
            Lfm2ResolvedWeight::Packed {
                codes: gate_codes,
                scales: gate_scales,
            },
            Lfm2ResolvedWeight::Packed {
                codes: up_codes,
                scales: up_scales,
            },
        ) = (
            context
                .weights
                .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW1))?,
            context
                .weights
                .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW3))?,
        ) {
            backend.packed_linear_pair(
                &mut gate.buffer,
                &mut up.buffer,
                &ffn_input.buffer,
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
            )?;
        } else {
            weight_linear(
                backend,
                &mut gate.buffer,
                &ffn_input.buffer,
                &context.weights,
                layer_role(index, Lfm2LayerWeightRole::FfnW1),
            )?;
            weight_linear(
                backend,
                &mut up.buffer,
                &ffn_input.buffer,
                &context.weights,
                layer_role(index, Lfm2LayerWeightRole::FfnW3),
            )?;
        }
        let mut down = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        match context
            .weights
            .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW2))?
        {
            Lfm2ResolvedWeight::Packed { codes, scales } => {
                backend.packed_swiglu_linear(
                    &mut down.buffer,
                    &gate.buffer,
                    &up.buffer,
                    codes,
                    scales,
                )?;
            }
            Lfm2ResolvedWeight::Dense(_) => {
                let mut activated =
                    allocate(backend, shape(1, intermediate)?, AllocationClass::Scratch)?;
                backend.swiglu(&mut activated.buffer, &gate.buffer, &up.buffer)?;
                weight_linear(
                    backend,
                    &mut down.buffer,
                    &activated.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::FfnW2),
                )?;
                push_scratch::<B>(scratch, [activated.buffer]);
            }
        }
        // The next layer's operator norm (or the final embedding norm) rides
        // on the same fused add+norm pass that produces the residual base.
        let mut next_x = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        let mut next_u = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
        let (next_norm, next_epsilon) = if index + 1 < config.layers.len() {
            (
                layer_role(index + 1, Lfm2LayerWeightRole::OperatorNorm),
                config.block_norm_epsilon,
            )
        } else {
            (Lfm2WeightRole::EmbeddingNorm, config.norm_epsilon)
        };
        backend.add_row_rms_norm(
            &mut next_x.buffer,
            &mut next_u.buffer,
            &residual.buffer,
            &down.buffer,
            context.weights.buffer_for(next_norm)?,
            next_epsilon,
        )?;
        push_scratch::<B>(
            scratch,
            [
                x.buffer,
                u.buffer,
                operator.buffer,
                residual.buffer,
                ffn_input.buffer,
                gate.buffer,
                up.buffer,
                down.buffer,
            ],
        );
        x = next_x;
        u = next_u;
    }
    if config.layers.is_empty() {
        backend.row_rms_norm(
            &mut u.buffer,
            &x.buffer,
            context.weights.buffer_for(Lfm2WeightRole::EmbeddingNorm)?,
            config.norm_epsilon,
        )?;
    }
    let mut logits = allocate(
        backend,
        shape(1, u64::from(config.vocab_size))?,
        AllocationClass::Cache,
    )?;
    weight_linear(
        backend,
        &mut logits.buffer,
        &u.buffer,
        &context.weights,
        Lfm2WeightRole::TiedLmHead,
    )?;
    push_scratch::<B>(scratch, [x.buffer, u.buffer]);
    // Reduce the fresh logits to a device-resident greedy token. The publish
    // readback then only needs this one f32: NaN means the logits row held a
    // non-finite value, and a finite value doubles as the next append's
    // embedding-gather selector.
    if state.sampled.is_none() {
        state.sampled = Some(allocate(
            backend,
            Shape::new(&[1])?,
            AllocationClass::Cache,
        )?);
    }
    if let Some(sampled) = state.sampled.as_mut() {
        backend.argmax(&mut sampled.buffer, &logits.buffer)?;
    }
    state.sampled_id = None;
    state.next_logits = Some(logits);
    state.history.push(token);
    state.length = state.length.checked_add(1).ok_or(ExecutorError::Overflow(
        "prefix logical length overflows u64",
    ))?;
    Ok(())
}

enum PrefixAction<B: InferenceOps> {
    Prefill(Vec<TokenId>),
    Append {
        source: Lfm2Prefix<B>,
        tokens: Vec<TokenId>,
    },
    /// Single-token greedy append: the token is the source prefix's resolved
    /// argmax sample, embedded directly from its device-resident id buffer.
    AppendArgmax {
        source: Lfm2Prefix<B>,
    },
    Fork {
        source: Lfm2Prefix<B>,
    },
}

enum PrefixPhase<B: InferenceOps> {
    New,
    /// A cooperative token step is ready. Each poll submits at most one model token.
    Building,
    Fence(B::Fence),
    Readback(B::Readback),
    Terminal,
}

/// Nonblocking staged prefix prefill, append, or deep cache fork.
pub struct PrefixTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    action: Option<PrefixAction<B>>,
    phase: PrefixPhase<B>,
    staged: Option<PrefixStorage<B>>,
    scratch: Vec<B::Buffer>,
    /// Source snapshots retained while queued copies or kernels read their state.
    retained_prefixes: Vec<Lfm2Prefix<B>>,
    next_token: usize,
    check_logits: bool,
}

impl<B: InferenceOps> PrefixTask<B> {
    fn prefill(context: Rc<ModelContext<B>>, tokens: Vec<TokenId>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Prefill(tokens)),
            phase: PrefixPhase::New,
            staged: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    fn append(context: Rc<ModelContext<B>>, source: Lfm2Prefix<B>, tokens: Vec<TokenId>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Append { source, tokens }),
            phase: PrefixPhase::New,
            staged: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    fn append_argmax(context: Rc<ModelContext<B>>, source: Lfm2Prefix<B>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::AppendArgmax { source }),
            phase: PrefixPhase::New,
            staged: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    fn fork(context: Rc<ModelContext<B>>, source: Lfm2Prefix<B>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Fork { source }),
            phase: PrefixPhase::New,
            staged: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: false,
        }
    }

    fn terminal(&mut self, result: Result<Lfm2Prefix<B>>) -> CompletionPoll<Lfm2Prefix<B>> {
        self.action = None;
        self.staged = None;
        self.scratch.clear();
        self.retained_prefixes.clear();
        self.phase = PrefixPhase::Terminal;
        CompletionPoll::Ready(result)
    }

    fn quarantine_unfenced(&mut self) {
        let mut buffers = core::mem::take(&mut self.scratch);
        if let Some(staged) = self.staged.take() {
            buffers.extend(staged.into_buffers());
        }
        let prefixes = core::mem::take(&mut self.retained_prefixes);
        self.context.quarantine_unfenced(buffers, prefixes);
        self.action = None;
        self.phase = PrefixPhase::Terminal;
    }

    fn submit_fence(&mut self) -> Result<()> {
        let result = {
            let backend = self.context.borrow_backend()?;
            backend.fence()
        };
        match result {
            Ok(fence) => {
                self.phase = PrefixPhase::Fence(fence);
                Ok(())
            }
            Err(error) => {
                // Kernel/copy calls preceding a failed fence may still be visible to a device
                // queue. Preserve every owned source and destination until this executor is
                // discarded or reloaded; no later submission is admitted from this instance.
                self.quarantine_unfenced();
                Err(error)
            }
        }
    }

    fn start(&mut self) -> Result<Option<Lfm2Prefix<B>>> {
        self.context.validate_backend()?;
        let Some(action) = self.action.as_ref() else {
            return Err(ExecutorError::CompletionConsumed);
        };
        match action {
            PrefixAction::Prefill(tokens) => {
                let mut backend = self.context.borrow_backend()?;
                let staged = allocate_empty(&self.context, &mut *backend)?;
                drop(backend);
                if tokens.is_empty() {
                    self.action = None;
                    return Ok(Some(publish(&self.context, staged)));
                }
                self.staged = Some(staged);
                self.phase = PrefixPhase::Building;
                Ok(None)
            }
            PrefixAction::Append { source, tokens } => {
                self.context.validate_prefix(source)?;
                if tokens.is_empty() {
                    let Some(PrefixAction::Append { source, .. }) = self.action.take() else {
                        return Err(ExecutorError::CompletionConsumed);
                    };
                    return Ok(Some(source));
                }
                // The action owns one snapshot and this additional retained clone survives after
                // the action is consumed at submission. Device copies may read its cache.
                self.retained_prefixes.push(source.clone());
                let mut backend = self.context.borrow_backend()?;
                let staged = clone_storage(&self.context, &mut *backend, &source.storage)?;
                drop(backend);
                self.staged = Some(staged);
                self.phase = PrefixPhase::Building;
                Ok(None)
            }
            PrefixAction::AppendArgmax { source } => {
                self.context.validate_prefix(source)?;
                if source.storage.sampled.is_none() || source.storage.sampled_id.is_none() {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix has no resolved greedy sample to append",
                    ));
                }
                self.retained_prefixes.push(source.clone());
                let mut backend = self.context.borrow_backend()?;
                let staged = clone_storage(&self.context, &mut *backend, &source.storage)?;
                drop(backend);
                self.staged = Some(staged);
                self.phase = PrefixPhase::Building;
                Ok(None)
            }
            PrefixAction::Fork { source } => {
                self.context.validate_prefix(source)?;
                self.retained_prefixes.push(source.clone());
                let mut backend = self.context.borrow_backend()?;
                self.staged = Some(clone_storage(
                    &self.context,
                    &mut *backend,
                    &source.storage,
                )?);
                drop(backend);
                self.action = None;
                self.submit_fence()?;
                Ok(None)
            }
        }
    }

    fn build_one_token(&mut self) -> Result<()> {
        enum Step<'a, B: InferenceOps> {
            Host(TokenId),
            Device(TokenId, &'a B::Buffer),
        }
        self.context.validate_backend()?;
        let step: Option<Step<'_, B>> = match self.action.as_ref() {
            Some(PrefixAction::Prefill(tokens) | PrefixAction::Append { tokens, .. }) => {
                tokens.get(self.next_token).copied().map(Step::Host)
            }
            Some(PrefixAction::AppendArgmax { source }) => {
                if self.next_token == 0 {
                    match (source.storage.sampled_id, source.storage.sampled.as_ref()) {
                        (Some(id), Some(sampled)) => Some(Step::Device(id, &sampled.buffer)),
                        _ => {
                            return Err(ExecutorError::BackendFailure(
                                "append source lost its resolved greedy sample",
                            ));
                        }
                    }
                } else {
                    None
                }
            }
            Some(PrefixAction::Fork { .. }) | None => {
                return Err(ExecutorError::CompletionConsumed);
            }
        };
        let Some(step) = step else {
            self.action = None;
            return self.submit_fence();
        };
        let mut backend = self.context.borrow_backend()?;
        let Some(staged) = self.staged.as_mut() else {
            return Err(ExecutorError::BackendFailure(
                "staged prefix state is unavailable while building",
            ));
        };
        match step {
            Step::Host(token) => append_token(
                &self.context,
                &mut *backend,
                staged,
                token,
                TokenIds::Host(&[token]),
                &mut self.scratch,
            )?,
            Step::Device(token, ids) => append_token(
                &self.context,
                &mut *backend,
                staged,
                token,
                TokenIds::Device(ids),
                &mut self.scratch,
            )?,
        }
        drop(backend);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(ExecutorError::Overflow(
                "prefix token progress overflows usize",
            ))?;
        if matches!(
            self.action.as_ref(),
            Some(PrefixAction::Prefill(tokens)) if self.next_token == tokens.len()
        ) || matches!(
            self.action.as_ref(),
            Some(PrefixAction::Append { tokens, .. }) if self.next_token == tokens.len()
        ) || matches!(
            self.action.as_ref(),
            Some(PrefixAction::AppendArgmax { .. }) if self.next_token == 1
        ) {
            self.action = None;
            self.submit_fence()?;
        }
        Ok(())
    }

    fn retired_buffers(&mut self) -> Vec<B::Buffer> {
        let mut retained = core::mem::take(&mut self.scratch);
        if let Some(staged) = self.staged.take() {
            retained.extend(staged.into_buffers());
        }
        retained
    }

    fn abandon_source_prefixes(&mut self) {
        let prefixes = core::mem::take(&mut self.retained_prefixes);
        self.context.retain_abandoned_prefixes(prefixes);
    }
}

impl<B: InferenceOps> Drop for PrefixTask<B> {
    fn drop(&mut self) {
        let phase = core::mem::replace(&mut self.phase, PrefixPhase::Terminal);
        let fence = match phase {
            PrefixPhase::Fence(fence) => Some(fence),
            PrefixPhase::Building => {
                if let Ok(fence) = self
                    .context
                    .borrow_backend()
                    .and_then(|backend| backend.fence())
                {
                    Some(fence)
                } else {
                    // No completion boundary exists to prove queued copies are done. Keep all
                    // model-owned resources quarantined through the executor lifetime.
                    self.quarantine_unfenced();
                    None
                }
            }
            PrefixPhase::New | PrefixPhase::Readback(_) | PrefixPhase::Terminal => None,
        };
        if let Some(fence) = fence {
            let retained = self.retired_buffers();
            self.abandon_source_prefixes();
            if let Err(rejected) = self.context.retirement.retire(fence, retained) {
                self.context.retirement.quarantine_rejected(rejected);
            }
        }
    }
}

#[allow(clippy::if_not_else)]
impl<B: InferenceOps> InferenceCompletion for PrefixTask<B> {
    type Output = Lfm2Prefix<B>;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match &mut self.phase {
            PrefixPhase::New => match self.start() {
                Ok(Some(prefix)) => self.terminal(Ok(prefix)),
                Ok(None) => CompletionPoll::Pending,
                Err(error)
                    if self.staged.is_some()
                        || !self.scratch.is_empty()
                        || !self.retained_prefixes.is_empty() =>
                {
                    self.quarantine_unfenced();
                    CompletionPoll::Ready(Err(error))
                }
                Err(error) => self.terminal(Err(error)),
            },
            PrefixPhase::Building => match self.build_one_token() {
                Ok(()) => CompletionPoll::Pending,
                Err(error) => {
                    // A backend may have accepted an earlier primitive in this token before a
                    // later primitive reports failure. Do not drop its inputs without a fence.
                    self.quarantine_unfenced();
                    CompletionPoll::Ready(Err(error))
                }
            },
            PrefixPhase::Fence(fence) => match fence.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Err(error)) => self.terminal(Err(error)),
                CompletionPoll::Ready(Ok(())) => {
                    if !self.check_logits {
                        if let Err(error) = self.context.validate_backend() {
                            return self.terminal(Err(error));
                        }
                        let Some(staged) = self.staged.take() else {
                            return self.terminal(Err(ExecutorError::BackendFailure(
                                "staged prefix state is unavailable",
                            )));
                        };
                        self.scratch.clear();
                        self.retained_prefixes.clear();
                        self.phase = PrefixPhase::Terminal;
                        self.action = None;
                        CompletionPoll::Ready(Ok(publish(&self.context, staged)))
                    } else {
                        let Some(sampled) = self
                            .staged
                            .as_ref()
                            .and_then(|state| state.sampled.as_ref())
                        else {
                            return self.terminal(Err(ExecutorError::BackendFailure(
                                "appended prefix has no logits boundary",
                            )));
                        };
                        // The argmax output is one f32: NaN reports non-finite
                        // logits and a finite value is the next greedy token.
                        // This replaces the full-vocab logits readback.
                        let readback = match self
                            .context
                            .borrow_backend()
                            .and_then(|backend| backend.read_f32_async(&sampled.buffer))
                        {
                            Ok(readback) => readback,
                            Err(error) => return self.terminal(Err(error)),
                        };
                        self.phase = PrefixPhase::Readback(readback);
                        CompletionPoll::Pending
                    }
                }
            },
            PrefixPhase::Readback(readback) => match readback.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Err(error)) => self.terminal(Err(error)),
                CompletionPoll::Ready(Ok(values)) => {
                    if values.len() != 1 {
                        return self.terminal(Err(ExecutorError::BackendFailure(
                            "greedy-sample readback returned the wrong value count",
                        )));
                    }
                    let sampled = values[0];
                    if !sampled.is_finite() {
                        return self.terminal(Err(ExecutorError::BackendFailure(
                            "final LFM2 logits contain a non-finite value",
                        )));
                    }
                    if let Err(error) = self.context.validate_backend() {
                        return self.terminal(Err(error));
                    }
                    let Some(mut staged) = self.staged.take() else {
                        return self.terminal(Err(ExecutorError::BackendFailure(
                            "staged prefix state is unavailable",
                        )));
                    };
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        staged.sampled_id = Some(sampled as u32);
                    }
                    self.scratch.clear();
                    self.retained_prefixes.clear();
                    self.phase = PrefixPhase::Terminal;
                    self.action = None;
                    CompletionPoll::Ready(Ok(publish(&self.context, staged)))
                }
            },
            PrefixPhase::Terminal => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        let phase = core::mem::replace(&mut self.phase, PrefixPhase::Terminal);
        match phase {
            PrefixPhase::New => {
                self.action = None;
                Ok(())
            }
            PrefixPhase::Building => {
                // Building has not yet produced a completion boundary. Submit one solely to
                // establish whether queued source/destination buffers can be released.
                match self.submit_fence() {
                    Ok(()) => self.cancel(),
                    Err(error) => Err(error),
                }
            }
            PrefixPhase::Fence(mut fence) => match fence.cancel() {
                Ok(()) => {
                    self.action = None;
                    self.staged = None;
                    self.scratch.clear();
                    self.retained_prefixes.clear();
                    Ok(())
                }
                Err(error) => {
                    let retained = self.retired_buffers();
                    self.abandon_source_prefixes();
                    if let Err(rejected) = self.context.retirement.retire(fence, retained) {
                        self.context.retirement.quarantine_rejected(rejected);
                    }
                    self.action = None;
                    Err(error)
                }
            },
            PrefixPhase::Readback(mut readback) => match readback.cancel() {
                Ok(()) => {
                    self.action = None;
                    self.staged = None;
                    self.scratch.clear();
                    self.retained_prefixes.clear();
                    Ok(())
                }
                Err(error) => {
                    // Cancellation is not confirmation that the backend stopped reading the
                    // staged logits. Keep the completion and every source allocation owned
                    // by this task so callers may poll or retry cancellation safely.
                    self.phase = PrefixPhase::Readback(readback);
                    Err(error)
                }
            },
            PrefixPhase::Terminal => Err(ExecutorError::CompletionConsumed),
        }
    }
}

/// Nonblocking host-visible final logits. The prefix remains retained until readback is terminal.
pub struct LogitsTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    prefix: Option<Lfm2Prefix<B>>,
    readback: Option<B::Readback>,
    terminal: bool,
}

impl<B: InferenceOps> LogitsTask<B> {
    fn new(context: Rc<ModelContext<B>>, prefix: Lfm2Prefix<B>) -> Self {
        Self {
            context,
            prefix: Some(prefix),
            readback: None,
            terminal: false,
        }
    }
    fn finish(&mut self, result: Result<Vec<f32>>) -> CompletionPoll<Vec<f32>> {
        self.prefix = None;
        self.readback = None;
        self.terminal = true;
        CompletionPoll::Ready(result)
    }
}

impl<B: InferenceOps> InferenceCompletion for LogitsTask<B> {
    type Output = Vec<f32>;
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.terminal {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        if self.readback.is_none() {
            let Some(prefix) = self.prefix.as_ref() else {
                return self.finish(Err(ExecutorError::CompletionConsumed));
            };
            if let Err(error) = self.context.validate_prefix(prefix) {
                return self.finish(Err(error));
            }
            let Some(logits) = prefix.storage.next_logits.as_ref() else {
                return self.finish(Err(ExecutorError::InvalidArgument(
                    "empty prefix has no next-token logits",
                )));
            };
            let readback = match self
                .context
                .borrow_backend()
                .and_then(|backend| backend.read_f32_async(&logits.buffer))
            {
                Ok(readback) => readback,
                Err(error) => return self.finish(Err(error)),
            };
            self.readback = Some(readback);
            return CompletionPoll::Pending;
        }
        let Some(readback) = self.readback.as_mut() else {
            return self.finish(Err(ExecutorError::CompletionConsumed));
        };
        match readback.poll_step() {
            CompletionPoll::Pending => CompletionPoll::Pending,
            CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            CompletionPoll::Ready(Ok(values)) if values.iter().all(|value| value.is_finite()) => {
                let Some(prefix) = self.prefix.as_ref() else {
                    return self.finish(Err(ExecutorError::CompletionConsumed));
                };
                if let Err(error) = self.context.validate_prefix(prefix) {
                    return self.finish(Err(error));
                }
                self.finish(Ok(values))
            }
            CompletionPoll::Ready(Ok(_)) => self.finish(Err(ExecutorError::BackendFailure(
                "final LFM2 logits contain a non-finite value",
            ))),
        }
    }
    fn cancel(&mut self) -> Result<()> {
        if self.terminal {
            return Err(ExecutorError::CompletionConsumed);
        }
        if let Some(readback) = self.readback.as_mut() {
            readback.cancel()?;
        }
        self.prefix = None;
        self.readback = None;
        self.terminal = true;
        Ok(())
    }
}

fn candidate_score_transport(accumulated: f64) -> Result<f32> {
    if !accumulated.is_finite() {
        return Err(ExecutorError::BackendFailure(
            "candidate probability sum is non-finite before F32 transport",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let transported = accumulated as f32;
    if !transported.is_finite() {
        return Err(ExecutorError::BackendFailure(
            "candidate probability sum exceeds finite F32 transport",
        ));
    }
    Ok(transported)
}

fn token_log_probability(logits: &[f32], token: TokenId) -> Result<f64> {
    let token = usize::try_from(token)
        .map_err(|_| ExecutorError::Overflow("candidate token index exceeds usize"))?;
    let token_value = *logits.get(token).ok_or(ExecutorError::OutOfBounds(
        "candidate token index exceeds logits vocabulary",
    ))?;
    if !token_value.is_finite() || logits.iter().any(|value| !value.is_finite()) {
        return Err(ExecutorError::BackendFailure(
            "candidate normalization received a non-finite logit",
        ));
    }
    let maximum = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for logit in logits {
        sum += (f64::from(*logit) - maximum).exp();
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(ExecutorError::BackendFailure(
            "candidate normalization denominator is invalid",
        ));
    }
    // Subtract the common maximum before adding the denominator term. Combining maximum and
    // ln(sum) first loses ln(sum) for equal very-large finite F32 logits, violating the
    // translation invariance of log-softmax even with F64 accumulation.
    Ok((f64::from(token_value) - maximum) - sum.ln())
}

/// Serial complete-candidate scorer. Every candidate begins from a deep fork of the same parent.
pub struct ScoreTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    parent: Lfm2Prefix<B>,
    candidates: Vec<Vec<TokenId>>,
    scores: Vec<CandidateScore>,
    candidate_index: usize,
    token_index: usize,
    accumulated: f64,
    branch: Option<Lfm2Prefix<B>>,
    fork: Option<PrefixTask<B>>,
    logits: Option<LogitsTask<B>>,
    append: Option<PrefixTask<B>>,
    terminal: bool,
}

impl<B: InferenceOps> ScoreTask<B> {
    fn new(
        context: Rc<ModelContext<B>>,
        parent: Lfm2Prefix<B>,
        candidates: Vec<Vec<TokenId>>,
    ) -> Self {
        Self {
            context,
            parent,
            candidates,
            scores: Vec::new(),
            candidate_index: 0,
            token_index: 0,
            accumulated: 0.0,
            branch: None,
            fork: None,
            logits: None,
            append: None,
            terminal: false,
        }
    }
    fn finish(
        &mut self,
        result: Result<Vec<CandidateScore>>,
    ) -> CompletionPoll<Vec<CandidateScore>> {
        self.fork = None;
        self.logits = None;
        self.append = None;
        self.branch = None;
        self.terminal = true;
        CompletionPoll::Ready(result)
    }
    fn complete_candidate(&mut self) -> Result<()> {
        let candidate = self
            .candidates
            .get(self.candidate_index)
            .ok_or(ExecutorError::OutOfBounds("candidate index is unavailable"))?;
        self.scores.push(CandidateScore {
            candidate_index: self.candidate_index,
            token_count: candidate.len(),
            // CandidateScore is a fixed F32 transport contract; all normalization and summation
            // above are F64, and the one narrowing conversion is checked for finiteness.
            log_probability: candidate_score_transport(self.accumulated)?,
        });
        self.candidate_index = self
            .candidate_index
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("candidate index overflows usize"))?;
        self.token_index = 0;
        self.accumulated = 0.0;
        self.branch = None;
        Ok(())
    }
}

impl<B: InferenceOps> InferenceCompletion for ScoreTask<B> {
    type Output = Vec<CandidateScore>;
    #[allow(clippy::too_many_lines)]
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.terminal {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        // A scorer can spend its final polling boundary between the last child completion and
        // publication, and an empty candidate has no child completion at all. Revalidate the
        // original immutable parent on every poll so neither path can publish after reload,
        // quarantine, or actual-backend replacement.
        if let Err(error) = self.context.validate_prefix(&self.parent) {
            return self.finish(Err(error));
        }
        if self.candidate_index == self.candidates.len() {
            let scores = core::mem::take(&mut self.scores);
            return self.finish(Ok(scores));
        }
        let candidate_len = self.candidates[self.candidate_index].len();
        if candidate_len == 0 {
            return match self.complete_candidate() {
                Ok(()) => CompletionPoll::Pending,
                Err(error) => self.finish(Err(error)),
            };
        }
        if self.branch.is_none() {
            if self.fork.is_none() {
                self.fork = Some(PrefixTask::fork(
                    Rc::clone(&self.context),
                    self.parent.clone(),
                ));
            }
            let Some(fork) = self.fork.as_mut() else {
                return self.finish(Err(ExecutorError::BackendFailure(
                    "fork task is unavailable",
                )));
            };
            return match fork.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Ok(branch)) => {
                    self.branch = Some(branch);
                    self.fork = None;
                    CompletionPoll::Pending
                }
                CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            };
        }
        if let Some(append) = self.append.as_mut() {
            return match append.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Ok(branch)) => {
                    self.branch = Some(branch);
                    self.append = None;
                    self.token_index = match self.token_index.checked_add(1) {
                        Some(index) => index,
                        None => {
                            return self.finish(Err(ExecutorError::Overflow(
                                "candidate token index overflows usize",
                            )));
                        }
                    };
                    if self.token_index == candidate_len {
                        match self.complete_candidate() {
                            Ok(()) => CompletionPoll::Pending,
                            Err(error) => self.finish(Err(error)),
                        }
                    } else {
                        CompletionPoll::Pending
                    }
                }
                CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            };
        }
        if self.logits.is_none() {
            let Some(branch) = self.branch.as_ref() else {
                return self.finish(Err(ExecutorError::BackendFailure(
                    "candidate branch is unavailable",
                )));
            };
            self.logits = Some(LogitsTask::new(Rc::clone(&self.context), branch.clone()));
        }
        let Some(logits) = self.logits.as_mut() else {
            return self.finish(Err(ExecutorError::BackendFailure(
                "candidate logits task is unavailable",
            )));
        };
        match logits.poll_step() {
            CompletionPoll::Pending => CompletionPoll::Pending,
            CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            CompletionPoll::Ready(Ok(values)) => {
                let token = self.candidates[self.candidate_index][self.token_index];
                match token_log_probability(&values, token) {
                    Ok(value) => self.accumulated += value,
                    Err(error) => return self.finish(Err(error)),
                }
                let Some(branch) = self.branch.as_ref() else {
                    return self.finish(Err(ExecutorError::BackendFailure(
                        "candidate branch is unavailable",
                    )));
                };
                self.append = Some(PrefixTask::append(
                    Rc::clone(&self.context),
                    branch.clone(),
                    vec![token],
                ));
                self.logits = None;
                CompletionPoll::Pending
            }
        }
    }
    fn cancel(&mut self) -> Result<()> {
        if self.terminal {
            return Err(ExecutorError::CompletionConsumed);
        }
        if let Some(task) = self.fork.as_mut() {
            task.cancel()?;
        }
        if let Some(task) = self.logits.as_mut() {
            task.cancel()?;
        }
        if let Some(task) = self.append.as_mut() {
            task.cancel()?;
        }
        self.fork = None;
        self.logits = None;
        self.append = None;
        self.branch = None;
        self.terminal = true;
        Ok(())
    }
}

impl<B: InferenceOps> TokenExecutor for Lfm2Executor<B> {
    type Prefix = Lfm2Prefix<B>;
    type Prefill = PrefixTask<B>;
    type Append = PrefixTask<B>;
    type Fork = PrefixTask<B>;
    type Scores = ScoreTask<B>;
    type Logits = LogitsTask<B>;
    fn prefill(&mut self, input: TokenChunk<'_>) -> Result<Self::Prefill> {
        Ok(PrefixTask::prefill(
            Rc::clone(&self.context),
            self.accepted_tokens(input, 0)?,
        ))
    }
    fn append_known(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
    ) -> Result<Self::Append> {
        self.context.validate_prefix(prefix)?;
        Ok(PrefixTask::append(
            Rc::clone(&self.context),
            prefix.clone(),
            self.accepted_tokens(input, prefix.storage.length)?,
        ))
    }
    fn sampled_token(&mut self, prefix: &Self::Prefix) -> Result<Option<TokenId>> {
        self.context.validate_prefix(prefix)?;
        Ok(prefix.storage.sampled_id)
    }
    fn append_argmax(&mut self, prefix: &Self::Prefix) -> Result<Self::Append> {
        self.context.validate_prefix(prefix)?;
        if prefix.storage.sampled.is_none() || prefix.storage.sampled_id.is_none() {
            return Err(ExecutorError::InvalidArgument(
                "prefix has no resolved greedy sample to append",
            ));
        }
        Ok(PrefixTask::append_argmax(
            Rc::clone(&self.context),
            prefix.clone(),
        ))
    }
    fn fork(&mut self, prefix: &Self::Prefix) -> Result<Self::Fork> {
        self.context.validate_prefix(prefix)?;
        Ok(PrefixTask::fork(Rc::clone(&self.context), prefix.clone()))
    }
    fn next_logits(&mut self, prefix: &Self::Prefix) -> Result<Self::Logits> {
        self.context.validate_prefix(prefix)?;
        if prefix.storage.next_logits.is_none() {
            return Err(ExecutorError::InvalidArgument(
                "empty prefix has no next-token logits",
            ));
        }
        Ok(LogitsTask::new(Rc::clone(&self.context), prefix.clone()))
    }
    fn score_candidates(
        &mut self,
        prefix: &Self::Prefix,
        candidates: &[&[TokenId]],
    ) -> Result<Self::Scores> {
        self.context.validate_prefix(prefix)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(candidates.len())
            .map_err(|_| ExecutorError::ResourceLimit("candidate list allocation failed"))?;
        for candidate in candidates {
            if !candidate.is_empty() && prefix.storage.next_logits.is_none() {
                return Err(ExecutorError::InvalidArgument(
                    "nonempty candidate requires an anchored prefix logits row",
                ));
            }
            let length = u64::try_from(candidate.len())
                .map_err(|_| ExecutorError::Overflow("candidate length exceeds u64"))?;
            if prefix
                .storage
                .length
                .checked_add(length)
                .ok_or(ExecutorError::Overflow(
                    "candidate logical length overflows u64",
                ))?
                > self.context.limits.max_logical_tokens
            {
                return Err(ExecutorError::OutOfBounds(
                    "candidate continuation exceeds configured logical prefix capacity",
                ));
            }
            let mut copy = Vec::new();
            copy.try_reserve_exact(candidate.len())
                .map_err(|_| ExecutorError::ResourceLimit("candidate token allocation failed"))?;
            for token in *candidate {
                if *token >= self.context.config().vocab_size {
                    return Err(ExecutorError::OutOfBounds(
                        "candidate token ID exceeds loaded model vocabulary",
                    ));
                }
                copy.push(*token);
            }
            owned.push(copy);
        }
        Ok(ScoreTask::new(
            Rc::clone(&self.context),
            prefix.clone(),
            owned,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::candidate_score_transport;

    #[test]
    fn token_log_probability_keeps_normalization_when_equal_finite_logits_are_large() {
        let large = f32::from_bits(0x62b5_02e8);
        let logits = [large, large, large, large];
        let expected = -(4.0_f64).ln();
        let actual = match super::token_log_probability(&logits, 0) {
            Ok(value) => value,
            Err(error) => panic!("finite logits unexpectedly failed: {error:?}"),
        };
        assert!((actual - expected).abs() <= f64::EPSILON);
        let normal = match super::token_log_probability(&[0.0, 1.0], 1) {
            Ok(value) => value,
            Err(error) => panic!("normal logits unexpectedly failed: {error:?}"),
        };
        assert!((normal - (1.0_f64 - (1.0_f64.exp() + 1.0).ln())).abs() <= f64::EPSILON);
    }

    #[test]
    fn candidate_score_transport_rejects_nonfinite_or_unrepresentable_f64() {
        match candidate_score_transport(-1.25) {
            Ok(value) => assert!((value + 1.25).abs() <= f32::EPSILON),
            Err(error) => panic!("finite transport unexpectedly failed: {error:?}"),
        }
        assert!(candidate_score_transport(f64::INFINITY).is_err());
        assert!(candidate_score_transport(f64::NEG_INFINITY).is_err());
        assert!(candidate_score_transport(f64::MAX).is_err());
        assert!(candidate_score_transport(-f64::MAX).is_err());
    }
}
