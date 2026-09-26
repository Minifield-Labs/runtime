//! Bounded logit and choice readback tasks.

use super::{
    CompletionPoll, ExecutorError, InferenceCompletion, InferenceOps, Lfm2Prefix, ModelContext,
    PrefixStorage, Rc, Result, Tensor,
};

/// Nonblocking host-visible final logits. The prefix remains retained until readback is terminal.
pub struct LogitsTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    prefix: Option<Lfm2Prefix<B>>,
    readback: Option<B::Readback>,
    terminal: bool,
}

impl<B: InferenceOps> LogitsTask<B> {
    pub(super) fn new(context: Rc<ModelContext<B>>, prefix: Lfm2Prefix<B>) -> Self {
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

/// Nonblocking host readback of caller-selected logits columns. The prefix
/// and the `[1, K]` gathered buffer stay retained until the readback is
/// terminal, so only K f32 values ever cross to the host.
pub struct ChoiceLogitsTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    prefix: Option<Lfm2Prefix<B>>,
    gathered: Option<Tensor<B>>,
    readback: Option<B::Readback>,
    terminal: bool,
}

impl<B: InferenceOps> ChoiceLogitsTask<B> {
    pub(super) fn new(
        context: Rc<ModelContext<B>>,
        prefix: Lfm2Prefix<B>,
        gathered: Tensor<B>,
        readback: B::Readback,
    ) -> Self {
        Self {
            context,
            prefix: Some(prefix),
            gathered: Some(gathered),
            readback: Some(readback),
            terminal: false,
        }
    }
    fn finish(&mut self, result: Result<Vec<f32>>) -> CompletionPoll<Vec<f32>> {
        self.prefix = None;
        drop(self.gathered.take());
        self.readback = None;
        self.terminal = true;
        CompletionPoll::Ready(result)
    }
}

impl<B: InferenceOps> InferenceCompletion for ChoiceLogitsTask<B> {
    type Output = Vec<f32>;
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.terminal {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
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
                "selected LFM2 logits contain a non-finite value",
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
        drop(self.gathered.take());
        self.readback = None;
        self.terminal = true;
        Ok(())
    }
}

/// Nonblocking one-pass prefill plus caller-selected logits readback.
///
/// The whole prompt runs as one `[T, ...]` model pass over freshly staged
/// prefix storage; only the selected columns of its final `[1, V]` logits row
/// are gathered into `[1, K]` and read back. The staged storage, gathered
/// buffer, and scratch stay retained until the readback is terminal, so only
/// K f32 values ever cross to the host. No prefix is published and no greedy
/// sample is computed.
pub struct PrefillChoiceTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    staged: Option<PrefixStorage<B>>,
    gathered: Option<Tensor<B>>,
    scratch: Vec<B::Buffer>,
    readback: Option<B::Readback>,
    expected: usize,
    terminal: bool,
}

impl<B: InferenceOps> PrefillChoiceTask<B> {
    pub(super) fn new(
        context: Rc<ModelContext<B>>,
        staged: PrefixStorage<B>,
        gathered: Tensor<B>,
        scratch: Vec<B::Buffer>,
        readback: B::Readback,
        expected: usize,
    ) -> Self {
        Self {
            context,
            staged: Some(staged),
            gathered: Some(gathered),
            scratch,
            readback: Some(readback),
            expected,
            terminal: false,
        }
    }

    fn finish(&mut self, result: Result<Vec<f32>>) -> CompletionPoll<Vec<f32>> {
        self.staged = None;
        self.gathered = None;
        self.scratch.clear();
        self.readback = None;
        self.terminal = true;
        CompletionPoll::Ready(result)
    }
}

impl<B: InferenceOps> InferenceCompletion for PrefillChoiceTask<B> {
    type Output = Vec<f32>;
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.terminal {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        let Some(readback) = self.readback.as_mut() else {
            return self.finish(Err(ExecutorError::CompletionConsumed));
        };
        match readback.poll_step() {
            CompletionPoll::Pending => CompletionPoll::Pending,
            CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            CompletionPoll::Ready(Ok(values)) if values.len() != self.expected => self.finish(Err(
                ExecutorError::BackendFailure("choice readback returned the wrong value count"),
            )),
            CompletionPoll::Ready(Ok(values)) if values.iter().all(|value| value.is_finite()) => {
                if let Err(error) = self.context.validate_backend() {
                    return self.finish(Err(error));
                }
                self.finish(Ok(values))
            }
            CompletionPoll::Ready(Ok(_)) => self.finish(Err(ExecutorError::BackendFailure(
                "selected LFM2 logits contain a non-finite value",
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
        self.staged = None;
        self.gathered = None;
        self.scratch.clear();
        self.readback = None;
        self.terminal = true;
        Ok(())
    }
}

/// Nonblocking shared-base branch: clones an unscored base prefix's cache
/// state, appends one criterion tail in a single `[T, ...]` pass, gathers the
/// selected columns of its final `[1, V]` logits into `[1, K]`, and reads
/// them back. The base snapshot stays retained until the readback is
/// terminal, so the same base can serve every criterion serially. No branch
/// prefix is published and no greedy sample is computed.
pub struct AppendChoiceTask<B: InferenceOps> {
    context: Rc<ModelContext<B>>,
    base: Option<Lfm2Prefix<B>>,
    staged: Option<PrefixStorage<B>>,
    gathered: Option<Tensor<B>>,
    scratch: Vec<B::Buffer>,
    readback: Option<B::Readback>,
    expected: usize,
    terminal: bool,
}

impl<B: InferenceOps> AppendChoiceTask<B> {
    pub(super) fn new(
        context: Rc<ModelContext<B>>,
        base: Lfm2Prefix<B>,
        staged: PrefixStorage<B>,
        gathered: Tensor<B>,
        scratch: Vec<B::Buffer>,
        readback: B::Readback,
        expected: usize,
    ) -> Self {
        Self {
            context,
            base: Some(base),
            staged: Some(staged),
            gathered: Some(gathered),
            scratch,
            readback: Some(readback),
            expected,
            terminal: false,
        }
    }

    fn finish(&mut self, result: Result<Vec<f32>>) -> CompletionPoll<Vec<f32>> {
        self.base = None;
        self.staged = None;
        self.gathered = None;
        self.scratch.clear();
        self.readback = None;
        self.terminal = true;
        CompletionPoll::Ready(result)
    }
}

impl<B: InferenceOps> InferenceCompletion for AppendChoiceTask<B> {
    type Output = Vec<f32>;
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.terminal {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        let Some(readback) = self.readback.as_mut() else {
            return self.finish(Err(ExecutorError::CompletionConsumed));
        };
        match readback.poll_step() {
            CompletionPoll::Pending => CompletionPoll::Pending,
            CompletionPoll::Ready(Err(error)) => self.finish(Err(error)),
            CompletionPoll::Ready(Ok(values)) if values.len() != self.expected => self.finish(Err(
                ExecutorError::BackendFailure("choice readback returned the wrong value count"),
            )),
            CompletionPoll::Ready(Ok(values)) if values.iter().all(|value| value.is_finite()) => {
                if let Err(error) = self.context.validate_backend() {
                    return self.finish(Err(error));
                }
                self.finish(Ok(values))
            }
            CompletionPoll::Ready(Ok(_)) => self.finish(Err(ExecutorError::BackendFailure(
                "selected LFM2 logits contain a non-finite value",
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
        self.base = None;
        self.staged = None;
        self.gathered = None;
        self.scratch.clear();
        self.readback = None;
        self.terminal = true;
        Ok(())
    }
}
