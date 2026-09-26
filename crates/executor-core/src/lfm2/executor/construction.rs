//! Validated construction and optional packed execution policy.

use super::{
    AllocationClass, AppendChoiceTask, Cell, ExecutorError, FenceRetirement, HashMap, InferenceOps,
    LayerKind, Lfm2ExecutionLimits, Lfm2ExecutionOptions, Lfm2Executor, Lfm2LayerWeightRole,
    Lfm2Lut2Mode, Lfm2Prefix, Lfm2ResolvedWeight, Lfm2TypedWeights, Lfm2WeightFormat,
    Lfm2WeightRole, ModelContext, PrefillChoiceTask, PrefixTask, Rc, RefCell, Result,
    TokenChoiceExecutor, TokenChunk, TokenId, TokenIds, allocate, allocate_empty, append_tokens,
    layer_role, shape,
};

impl<B: InferenceOps> Lfm2Executor<B> {
    pub fn new(
        backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
    ) -> Result<Self> {
        Self::new_with_options(backend, weights, limits, Lfm2ExecutionOptions::default())
    }

    pub fn new_with_options(
        backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
        options: Lfm2ExecutionOptions,
    ) -> Result<Self> {
        if weights.classes().is_some() {
            return Err(ExecutorError::InvalidArgument(
                "use Lfm2Classifier for classification weights",
            ));
        }
        Self::new_inner(backend, weights, limits, options)
    }

    fn new_inner(
        mut backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
        options: Lfm2ExecutionOptions,
    ) -> Result<Self> {
        weights.config().validate()?;
        limits.validate(weights.config())?;
        if !weights.config().tie_embedding {
            return Err(ExecutorError::InvalidTie);
        }
        let lease = backend.lease();
        if !weights.inner().matches_backend_lease(&lease) {
            return Err(ExecutorError::WrongBackend);
        }
        let capabilities = backend.capabilities();
        for operation in [
            minifield_engine_api::OperationKind::Copy,
            minifield_engine_api::OperationKind::RectCopy2d,
            minifield_engine_api::OperationKind::GatherRows,
            minifield_engine_api::OperationKind::GatherColumns,
            minifield_engine_api::OperationKind::Argmax,
            minifield_engine_api::OperationKind::Add,
            minifield_engine_api::OperationKind::Multiply,
            minifield_engine_api::OperationKind::Linear,
            minifield_engine_api::OperationKind::RowRmsNorm,
            minifield_engine_api::OperationKind::Rotary,
            minifield_engine_api::OperationKind::GroupedQueryAttention,
            minifield_engine_api::OperationKind::GatedShortConvolution,
            minifield_engine_api::OperationKind::SwiGlu,
            minifield_engine_api::OperationKind::AddRowRmsNorm,
            minifield_engine_api::OperationKind::QkNormRope,
        ] {
            capabilities.validate(minifield_engine_api::DType::F32, operation, 2, 0, 0)?;
        }
        if weights.has_packed() {
            for operation in [
                minifield_engine_api::OperationKind::PackedGatherRows,
                minifield_engine_api::OperationKind::PackedLinear,
                minifield_engine_api::OperationKind::PackedLinearPair,
                minifield_engine_api::OperationKind::PackedSwigluLinear,
                minifield_engine_api::OperationKind::PackedSwigluPair,
            ] {
                capabilities.validate(minifield_engine_api::DType::F32, operation, 2, 0, 0)?;
            }
        }
        validate_roles::<B>(&weights)?;
        let retirement = backend.fence_retirement();
        let lut2 = repack_lut2(&mut backend, &weights, &*retirement, options)?;
        Ok(Self {
            context: Rc::new(ModelContext {
                backend: Rc::new(RefCell::new(backend)),
                retirement,
                weights: Rc::new(weights),
                lease,
                owner: Rc::new(()),
                limits,
                quarantined: Cell::new(false),
                unfenced_buffers: RefCell::new(Vec::new()),
                unfenced_prefixes: RefCell::new(Vec::new()),
                abandoned_prefixes: RefCell::new(Vec::new()),
                lut2_codes: lut2.codes,
                lut2_mode: Cell::new(options.lut2_mode),
                skipped_lut2_roles: lut2.skipped_roles,
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

    /// Inspect the current backend through a checked shared borrow.
    pub fn inspect_backend<T>(&self, inspect: impl FnOnce(&B) -> T) -> Result<T> {
        self.context.validate_backend()?;
        let backend = self.context.backend.try_borrow().map_err(|_| {
            ExecutorError::BackendFailure("backend is busy with another portable task")
        })?;
        Ok(inspect(&*backend))
    }

    /// Invalidate this loaded executor after its backend has been reloaded or quarantined.
    ///
    /// Loaded weights and every prefix remain bound to the old generation, so callers must
    /// construct a new executor from a freshly loaded model before submitting more work.
    pub fn advance_backend_generation(&mut self) -> Result<()> {
        self.context.borrow_backend()?.advance_generation()
    }

    /// Select which FFN ops consume LUT2-repacked codes (see [`Lfm2Lut2Mode`]).
    /// This selects among streams admitted by the construction policy; it never repacks.
    /// Unavailable streams keep using the raw kernels.
    pub fn set_lut2_mode(&self, mode: Lfm2Lut2Mode) {
        self.context.lut2_mode.set(mode);
    }

    /// The active LUT2 dispatch mode.
    #[must_use]
    pub fn lut2_mode(&self) -> Lfm2Lut2Mode {
        self.context.lut2_mode.get()
    }

    /// Whether construction admitted a LUT2 stream for this validated weight role.
    #[must_use]
    pub fn has_lut2_codes(&self, role: Lfm2WeightRole) -> bool {
        self.context.lut2_codes.contains_key(&role)
    }

    /// Eligible ternary FFN roles kept on raw codes by policy, capability, or resource limits.
    #[must_use]
    pub fn skipped_lut2_roles(&self) -> &[Lfm2WeightRole] {
        &self.context.skipped_lut2_roles
    }

    pub(super) fn prefill_selected(
        &mut self,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<PrefillChoiceTask<B>> {
        let tokens = self.accepted_tokens(input, 0)?;
        if tokens.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "choice prefill requires a nonempty prompt",
            ));
        }
        if token_ids.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "choice selector list is empty",
            ));
        }
        for token in token_ids {
            if *token >= self.context.weights.output_width() {
                return Err(ExecutorError::OutOfBounds(
                    "choice token ID exceeds loaded model vocabulary",
                ));
            }
        }
        self.context.validate_backend()?;
        let count = u64::try_from(token_ids.len())
            .map_err(|_| ExecutorError::Overflow("choice token count exceeds u64"))?;
        let mut backend = self.context.borrow_backend()?;
        let mut staged = allocate_empty(&self.context, &mut *backend)?;
        let mut scratch = Vec::new();
        let mut gathered = allocate(&mut *backend, shape(1, count)?, AllocationClass::Scratch)?;
        let recorded = (|| -> Result<B::Readback> {
            append_tokens(
                &self.context,
                &mut *backend,
                &mut staged,
                &tokens,
                TokenIds::Host(&tokens),
                true,
                &mut scratch,
            )?;
            let Some(logits) = staged.next_logits.as_ref() else {
                return Err(ExecutorError::BackendFailure(
                    "choice prefill produced no logits boundary",
                ));
            };
            backend.gather_columns(&mut gathered.buffer, &logits.buffer, token_ids)?;
            backend.read_f32_async(&gathered.buffer)
        })();
        let readback = match recorded {
            Ok(readback) => readback,
            Err(error) => {
                // Recorded work may be live in a pending device queue without a
                // completion boundary; retain every owned buffer rather than
                // dropping it into an unknown device state.
                scratch.push(gathered.buffer);
                scratch.extend(staged.into_buffers());
                drop(backend);
                self.context.quarantine_unfenced(scratch, Vec::new());
                return Err(error);
            }
        };
        drop(backend);
        Ok(PrefillChoiceTask::new(
            Rc::clone(&self.context),
            staged,
            gathered,
            scratch,
            readback,
            token_ids.len(),
        ))
    }

    pub(super) fn accepted_tokens(&self, input: TokenChunk<'_>, base: u64) -> Result<Vec<TokenId>> {
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
        self.context.checked_length(
            base,
            u64::try_from(tokens.len())
                .map_err(|_| ExecutorError::Overflow("token chunk logical length exceeds u64"))?,
        )?;
        Ok(tokens)
    }
}

/// Last-valid-token classifier sharing the LFM2 backbone and finite backend ops.
/// Every call starts with empty attention and convolution state.
pub struct Lfm2Classifier<B: InferenceOps> {
    executor: Lfm2Executor<B>,
    selectors: Vec<TokenId>,
}

impl<B: InferenceOps> Lfm2Classifier<B> {
    pub fn new(
        backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
    ) -> Result<Self> {
        Self::new_with_options(backend, weights, limits, Lfm2ExecutionOptions::default())
    }

    pub fn new_with_options(
        backend: B,
        weights: Lfm2TypedWeights<B::Buffer>,
        limits: Lfm2ExecutionLimits,
        options: Lfm2ExecutionOptions,
    ) -> Result<Self> {
        let classes = weights.classes().ok_or(ExecutorError::InvalidArgument(
            "classification head required",
        ))?;
        Ok(Self {
            executor: Lfm2Executor::new_inner(backend, weights, limits, options)?,
            selectors: (0..classes).collect(),
        })
    }

    /// Return raw class logits. The caller owns any action mask and sampling policy.
    pub fn classify(&mut self, input: TokenChunk<'_>) -> Result<PrefillChoiceTask<B>> {
        self.executor.prefill_selected(input, &self.selectors)
    }

    /// Select which FFN ops consume LUT2-repacked codes; forwards to the
    /// inner executor (see [`Lfm2Executor::set_lut2_mode`]).
    pub fn set_lut2_mode(&self, mode: Lfm2Lut2Mode) {
        self.executor.set_lut2_mode(mode);
    }

    #[must_use]
    pub fn lut2_mode(&self) -> Lfm2Lut2Mode {
        self.executor.lut2_mode()
    }

    #[must_use]
    pub fn has_lut2_codes(&self, role: Lfm2WeightRole) -> bool {
        self.executor.has_lut2_codes(role)
    }

    #[must_use]
    pub fn skipped_lut2_roles(&self) -> &[Lfm2WeightRole] {
        self.executor.skipped_lut2_roles()
    }

    pub fn resource_report(&self) -> Result<minifield_engine_api::ResourceReport> {
        self.executor.resource_report()
    }

    /// Inspect the current backend through a checked shared borrow.
    pub fn inspect_backend<T>(&self, inspect: impl FnOnce(&B) -> T) -> Result<T> {
        self.executor.inspect_backend(inspect)
    }

    /// Prefill the shared prompt head once; the returned prefix is a reusable
    /// source for [`Self::classify_tail`] calls that append each decision's
    /// varying suffix instead of re-prefilling the whole prompt.
    pub fn prefill_base(&mut self, input: TokenChunk<'_>) -> Result<PrefixTask<B>> {
        self.executor.prefill_choice_base(input)
    }

    /// Classify a prompt that continues from `prefix`, returning the class
    /// logits gathered from the appended tail's final position.
    pub fn classify_tail(
        &mut self,
        prefix: &Lfm2Prefix<B>,
        tail: TokenChunk<'_>,
    ) -> Result<AppendChoiceTask<B>> {
        self.executor
            .append_choice_logits(prefix, tail, &self.selectors)
    }
}

fn validate_roles<B: InferenceOps>(weights: &Lfm2TypedWeights<B::Buffer>) -> Result<()> {
    let _ = weights.resolve(Lfm2WeightRole::TokenEmbedding)?;
    let _ = weights.resolve(weights.output_role())?;
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

struct Lut2Storage<Buffer> {
    codes: HashMap<Lfm2WeightRole, Buffer>,
    skipped_roles: Vec<Lfm2WeightRole>,
}

fn repack_lut2<B: InferenceOps>(
    backend: &mut B,
    weights: &Lfm2TypedWeights<B::Buffer>,
    retirement: &B::FenceRetirement,
    options: Lfm2ExecutionOptions,
) -> Result<Lut2Storage<B::Buffer>> {
    // Every validated FFN matrix has hidden * intermediate weights, four ternary codes per byte.
    let code_bytes = u64::from(weights.config().hidden_size)
        .checked_mul(u64::from(weights.config().effective_intermediate_size))
        .ok_or(ExecutorError::Overflow("LUT2 code extent overflows u64"))?
        / 4;
    let pair_bytes = code_bytes
        .checked_mul(2)
        .ok_or(ExecutorError::Overflow("LUT2 pair extent overflows u64"))?;
    let supported = backend.supports_ternary_lut2();
    let mut remaining = options.max_lut2_bytes;
    let mut codes = HashMap::new();
    let mut skipped_roles = Vec::new();
    let mut attempted = false;
    for index in 0..weights.config().layers.len() {
        // A limited budget can fund a useful down projection before the two-stream producer.
        let down_role = layer_role(index, Lfm2LayerWeightRole::FfnW2);
        if weights.role_quant(down_role) == Lfm2WeightFormat::TernaryV1 {
            if supported && options.lut2_mode != Lfm2Lut2Mode::Off && code_bytes <= remaining {
                let Lfm2ResolvedWeight::Packed { codes: raw, .. } = weights.resolve(down_role)?
                else {
                    return Err(ExecutorError::InvalidDType(
                        "ternary LUT2 role requires packed weight streams",
                    ));
                };
                attempted = true;
                match backend.repack_ternary_lut2(raw) {
                    Ok(repacked) => {
                        remaining -= code_bytes;
                        codes.insert(down_role, repacked);
                    }
                    Err(ExecutorError::ResourceLimit(_)) => skipped_roles.push(down_role),
                    Err(error) => return Err(error),
                }
            } else {
                skipped_roles.push(down_role);
            }
        }
        let gate_role = layer_role(index, Lfm2LayerWeightRole::FfnW1);
        let up_role = layer_role(index, Lfm2LayerWeightRole::FfnW3);
        let gate_ternary = weights.role_quant(gate_role) == Lfm2WeightFormat::TernaryV1;
        let up_ternary = weights.role_quant(up_role) == Lfm2WeightFormat::TernaryV1;
        if !supported
            || options.lut2_mode != Lfm2Lut2Mode::Auto
            || !gate_ternary
            || !up_ternary
            || pair_bytes > remaining
        {
            if gate_ternary {
                skipped_roles.push(gate_role);
            }
            if up_ternary {
                skipped_roles.push(up_role);
            }
            continue;
        }
        let (
            Lfm2ResolvedWeight::Packed { codes: gate, .. },
            Lfm2ResolvedWeight::Packed { codes: up, .. },
        ) = (weights.resolve(gate_role)?, weights.resolve(up_role)?)
        else {
            return Err(ExecutorError::InvalidDType(
                "ternary LUT2 pair requires packed weight streams",
            ));
        };
        attempted = true;
        let gate = match backend.repack_ternary_lut2(gate) {
            Ok(repacked) => repacked,
            Err(ExecutorError::ResourceLimit(_)) => {
                skipped_roles.extend([gate_role, up_role]);
                continue;
            }
            Err(error) => return Err(error),
        };
        match backend.repack_ternary_lut2(up) {
            Ok(up) => {
                remaining -= pair_bytes;
                codes.insert(gate_role, gate);
                codes.insert(up_role, up);
            }
            Err(ExecutorError::ResourceLimit(_)) => {
                // A single producer stream has no consumer. Backend RAII retains any queued use.
                drop(gate);
                skipped_roles.extend([gate_role, up_role]);
            }
            Err(error) => return Err(error),
        }
    }
    if attempted {
        // Fence every attempted repack, including an optional allocation that was rejected.
        // The backend owns queued inputs; retirement advances completion without blocking the host.
        let fence = backend.fence()?;
        if let Err(rejected) = retirement.retire(fence, Vec::new()) {
            let cause = rejected.cause().clone();
            retirement.quarantine_rejected(rejected);
            return Err(cause);
        }
    }
    Ok(Lut2Storage {
        codes,
        skipped_roles,
    })
}
