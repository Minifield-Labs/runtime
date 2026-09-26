//! Prefix completion lifecycle, fences, and quarantine.

use super::{
    CompletionPoll, ExecutorError, FenceRetirement, InferenceCompletion, InferenceOps, Lfm2Prefix,
    ModelContext, PrefixStorage, Rc, Result, Tensor, TokenId, TokenIds, allocate_empty,
    append_token, append_tokens, clone_storage, publish, sample_epilogue,
};

enum PrefixAction<B: InferenceOps> {
    Prefill {
        tokens: Vec<TokenId>,
        /// Allowed-token bitset applied to the final token's epilogue argmax.
        mask: Option<Rc<[u64]>>,
    },
    /// Unscored base prefill: populates history and caches without producing
    /// a final logits row or greedy sample, so the published prefix is only a
    /// branch source.
    BasePrefill {
        tokens: Vec<TokenId>,
    },
    Append {
        tokens: Vec<TokenId>,
    },
    /// Single-token greedy append: the token is the source prefix's resolved
    /// argmax sample, embedded directly from its device-resident id buffer.
    AppendArgmax {
        /// Allowed-token bitset applied to this step's epilogue argmax.
        mask: Option<Rc<[u64]>>,
    },
    Fork {
        source: Lfm2Prefix<B>,
    },
}

enum PrefixPhase<B: InferenceOps> {
    New,
    /// A cooperative build step is ready. Prefill submits the whole prompt in
    /// one pass; appends still submit at most one model token per poll.
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
    /// The prefix being extended, consumed by `start` to stage storage.
    source: Option<Lfm2Prefix<B>>,
    scratch: Vec<B::Buffer>,
    /// Source snapshots retained while queued copies or kernels read their state.
    retained_prefixes: Vec<Lfm2Prefix<B>>,
    next_token: usize,
    check_logits: bool,
}

impl<B: InferenceOps> PrefixTask<B> {
    pub(super) fn prefill(context: Rc<ModelContext<B>>, tokens: Vec<TokenId>) -> Self {
        Self::prefill_impl(context, tokens, None)
    }

    pub(super) fn prefill_masked(
        context: Rc<ModelContext<B>>,
        tokens: Vec<TokenId>,
        mask: Rc<[u64]>,
    ) -> Self {
        Self::prefill_impl(context, tokens, Some(mask))
    }

    fn prefill_impl(
        context: Rc<ModelContext<B>>,
        tokens: Vec<TokenId>,
        mask: Option<Rc<[u64]>>,
    ) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Prefill { tokens, mask }),
            phase: PrefixPhase::New,
            staged: None,
            source: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    pub(super) fn append(
        context: Rc<ModelContext<B>>,
        source: Lfm2Prefix<B>,
        tokens: Vec<TokenId>,
    ) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Append { tokens }),
            phase: PrefixPhase::New,
            staged: None,
            source: Some(source),
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    pub(super) fn append_argmax(context: Rc<ModelContext<B>>, source: Lfm2Prefix<B>) -> Self {
        Self::append_argmax_impl(context, source, None)
    }

    pub(super) fn append_argmax_masked(
        context: Rc<ModelContext<B>>,
        source: Lfm2Prefix<B>,
        mask: Rc<[u64]>,
    ) -> Self {
        Self::append_argmax_impl(context, source, Some(mask))
    }

    fn append_argmax_impl(
        context: Rc<ModelContext<B>>,
        source: Lfm2Prefix<B>,
        mask: Option<Rc<[u64]>>,
    ) -> Self {
        Self {
            context,
            action: Some(PrefixAction::AppendArgmax { mask }),
            phase: PrefixPhase::New,
            staged: None,
            source: Some(source),
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: true,
        }
    }

    pub(super) fn fork(context: Rc<ModelContext<B>>, source: Lfm2Prefix<B>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::Fork { source }),
            phase: PrefixPhase::New,
            staged: None,
            source: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: false,
        }
    }

    pub(super) fn base_prefill(context: Rc<ModelContext<B>>, tokens: Vec<TokenId>) -> Self {
        Self {
            context,
            action: Some(PrefixAction::BasePrefill { tokens }),
            phase: PrefixPhase::New,
            staged: None,
            source: None,
            scratch: Vec::new(),
            retained_prefixes: Vec::new(),
            next_token: 0,
            check_logits: false,
        }
    }

    fn terminal(&mut self, result: Result<Lfm2Prefix<B>>) -> CompletionPoll<Lfm2Prefix<B>> {
        self.action = None;
        self.staged = None;
        self.source = None;
        self.scratch.clear();
        self.retained_prefixes.clear();
        self.phase = PrefixPhase::Terminal;
        CompletionPoll::Ready(result)
    }

    fn quarantine_unfenced(&mut self) {
        // `self.source` is consumed before any copies that could read it are
        // recorded, so a still-present source owns no unfenced work.
        self.source = None;
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
            PrefixAction::Prefill { tokens, .. } | PrefixAction::BasePrefill { tokens } => {
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
            PrefixAction::Append { tokens } => {
                let Some(source) = self.source.take() else {
                    return Err(ExecutorError::BackendFailure(
                        "append source was already consumed",
                    ));
                };
                self.context.validate_prefix(&source)?;
                if tokens.is_empty() {
                    self.action = None;
                    return Ok(Some(source));
                }
                self.stage_storage(source)?;
                self.phase = PrefixPhase::Building;
                Ok(None)
            }
            PrefixAction::AppendArgmax { .. } => {
                let Some(source) = self.source.take() else {
                    return Err(ExecutorError::BackendFailure(
                        "append source was already consumed",
                    ));
                };
                self.context.validate_prefix(&source)?;
                if source.storage.sampled.is_none() || source.storage.sampled_id.is_none() {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix has no resolved greedy sample to append",
                    ));
                }
                self.stage_storage(source)?;
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
                    true,
                )?);
                drop(backend);
                self.action = None;
                self.submit_fence()?;
                Ok(None)
            }
        }
    }

    /// Stage an append's working storage. When this task holds the only
    /// reference to the source snapshot (the decode loop's case), its cache
    /// buffers are reused in place and no copies are recorded. Otherwise the
    /// storage is deep-copied and the source retained until the copies are
    /// fenced.
    fn stage_storage(&mut self, source: Lfm2Prefix<B>) -> Result<()> {
        let Lfm2Prefix {
            owner,
            lease,
            config_sha256,
            asset_sha256,
            numerical_mode,
            storage,
        } = source;
        let staged = match Rc::try_unwrap(storage) {
            Ok(storage) => storage,
            Err(storage) => {
                let mut backend = self.context.borrow_backend()?;
                let staged = clone_storage(&self.context, &mut *backend, &storage, false)?;
                drop(backend);
                self.retained_prefixes.push(Lfm2Prefix {
                    owner,
                    lease,
                    config_sha256,
                    asset_sha256,
                    numerical_mode,
                    storage,
                });
                staged
            }
        };
        self.staged = Some(staged);
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn build_step(&mut self) -> Result<()> {
        enum Step<B: InferenceOps> {
            Host(TokenId),
            Device(TokenId, Tensor<B>),
        }
        self.context.validate_backend()?;
        let bulk = match self.action.as_ref() {
            Some(PrefixAction::Prefill { tokens, mask }) => Some((tokens, mask.as_deref(), true)),
            Some(PrefixAction::BasePrefill { tokens }) => Some((tokens, None, false)),
            _ => None,
        };
        if let Some((tokens, mask, scored)) = bulk {
            let mut backend = self.context.borrow_backend()?;
            let Some(staged) = self.staged.as_mut() else {
                return Err(ExecutorError::BackendFailure(
                    "staged prefix state is unavailable while building",
                ));
            };
            append_tokens(
                &self.context,
                &mut *backend,
                staged,
                tokens,
                TokenIds::Host(tokens),
                scored,
                &mut self.scratch,
            )?;
            if scored {
                sample_epilogue(&mut *backend, staged, mask)?;
            }
            drop(backend);
            self.action = None;
            return self.submit_fence();
        }
        // The mask constrains only the argmax that publishes the next greedy
        // sample: for append_argmax that's the single appended token's.
        let (step, mask): (Option<Step<B>>, Option<Rc<[u64]>>) = match self.action.as_ref() {
            Some(PrefixAction::Append { tokens }) => {
                (tokens.get(self.next_token).copied().map(Step::Host), None)
            }
            Some(PrefixAction::AppendArgmax { mask }) => {
                if self.next_token == 0 {
                    let staged = self.staged.as_mut().ok_or(ExecutorError::BackendFailure(
                        "staged prefix state is unavailable while building",
                    ))?;
                    match (staged.sampled_id, staged.sampled.take()) {
                        (Some(id), Some(sampled)) => {
                            (Some(Step::Device(id, sampled)), mask.clone())
                        }
                        _ => {
                            return Err(ExecutorError::BackendFailure(
                                "append source lost its resolved greedy sample",
                            ));
                        }
                    }
                } else {
                    (None, None)
                }
            }
            Some(
                PrefixAction::Prefill { .. }
                | PrefixAction::BasePrefill { .. }
                | PrefixAction::Fork { .. },
            )
            | None => {
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
        let mask = mask.as_deref();
        match step {
            Step::Host(token) => append_token(
                &self.context,
                &mut *backend,
                staged,
                token,
                TokenIds::Host(&[token]),
                mask,
                &mut self.scratch,
            )?,
            Step::Device(token, sampled) => {
                append_token(
                    &self.context,
                    &mut *backend,
                    staged,
                    token,
                    TokenIds::Device(&sampled.buffer),
                    mask,
                    &mut self.scratch,
                )?;
                // The embedding gather just read this id buffer; keep it alive
                // until the submission completes.
                self.scratch.push(sampled.buffer);
            }
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
            Some(PrefixAction::Append { tokens }) if self.next_token == tokens.len()
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
            PrefixPhase::Building => match self.build_step() {
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
