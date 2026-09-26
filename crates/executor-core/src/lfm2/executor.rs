//! Backend-neutral LFM2 token execution over the finite portable inference operations.
//!
//! Prefixes are immutable snapshots. Append and fork stage independent backend-resident state
//! and publish only after a fence and final-logit readback complete.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

use minifield_engine_api::{
    AllocationClass, BackendLease, CandidateScore, CompletionPoll, ExecutorError, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, PackedHeadSpec, RectCopy2d,
    Result, RotarySpec, Shape, TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenId, TokenIds,
};

use super::{
    LayerKind, Lfm2Config, Lfm2LayerWeightRole, Lfm2ResolvedWeight, Lfm2TypedWeights,
    Lfm2WeightFormat, Lfm2WeightRole, NumericalMode,
};

/// Which packed FFN ops consume LUT2-repacked ternary code streams when the
/// backend produced them at load. The raw streams always stay resident, so
/// every mode is semantically identical; this only picks the kernel path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Lfm2Lut2Mode {
    /// Pair+SwiGLU and the down projection both take LUT2 streams.
    #[default]
    Auto,
    /// Only the down projection takes a LUT2 stream; the pair+SwiGLU
    /// producer keeps the raw fused kernel.
    DownOnly,
    /// Raw ternary kernels everywhere even when LUT2 buffers were loaded.
    Off,
}

/// Construction policy for optional ternary LUT2 storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lfm2ExecutionOptions {
    pub lut2_mode: Lfm2Lut2Mode,
    /// Maximum logical bytes reserved for duplicate LUT2 code streams.
    /// Backend admission still enforces its physical allocation limits.
    pub max_lut2_bytes: u64,
}

impl Default for Lfm2ExecutionOptions {
    fn default() -> Self {
        Self {
            lut2_mode: Lfm2Lut2Mode::Auto,
            max_lut2_bytes: 64 * 1024 * 1024,
        }
    }
}

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
    // LUT2-repacked ternary code buffers for the FFN roles, present only on
    // backends that opted in at load. Raw codes remain the source of truth.
    lut2_codes: HashMap<Lfm2WeightRole, B::Buffer>,
    lut2_mode: Cell<Lfm2Lut2Mode>,
    skipped_lut2_roles: Vec<Lfm2WeightRole>,
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

    fn checked_length(&self, base: u64, additional: u64) -> Result<u64> {
        let length = base.checked_add(additional).ok_or(ExecutorError::Overflow(
            "prefix logical length overflows u64",
        ))?;
        if length > self.limits.max_logical_tokens {
            return Err(ExecutorError::OutOfBounds(
                "append exceeds configured logical prefix capacity",
            ));
        }
        Ok(length)
    }

    fn validate_sample_mask(&self, mask: &[u64]) -> Result<()> {
        let words = u64::from(self.weights.output_width()).div_ceil(64);
        if u64::try_from(mask.len())
            .map_err(|_| ExecutorError::Overflow("sample mask word count exceeds u64"))?
            != words
        {
            return Err(ExecutorError::InvalidArgument(
                "masked argmax mask length must be ceil(width / 64)",
            ));
        }
        Ok(())
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

mod construction;
mod dispatch;
mod execution;
mod prefix_task;
mod readback;
mod scoring;
mod storage;

pub use construction::Lfm2Classifier;
pub use prefix_task::PrefixTask;
pub use readback::{AppendChoiceTask, ChoiceLogitsTask, LogitsTask, PrefillChoiceTask};
pub use scoring::ScoreTask;

use execution::{append_token, append_tokens, sample_epilogue};
use storage::{
    allocate, allocate_branch_storage, allocate_empty, clone_storage, copy_branch_storage, publish,
};

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

fn layer_role(index: usize, role: Lfm2LayerWeightRole) -> Lfm2WeightRole {
    Lfm2WeightRole::Layer { index, role }
}
fn shape(rows: u64, columns: u64) -> Result<Shape> {
    Shape::new(&[rows, columns])
}
