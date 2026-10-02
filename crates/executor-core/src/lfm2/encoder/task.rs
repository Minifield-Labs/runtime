//! Pollable encoder execution, identity validation, and submission retirement.

use super::{EncoderInput, EncoderLimits, EncoderTypedWeights, PointerOutput, decode_pointer};
use minifield_engine_api::{
    BackendLease, CompletionPoll, EncoderOps, ExecutorError, FenceRetirement, InferenceCompletion,
    Result,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

struct Context<B: EncoderOps> {
    backend: Rc<RefCell<B>>,
    weights: Rc<EncoderTypedWeights<B::Buffer>>,
    lease: BackendLease,
    retirement: Rc<B::FenceRetirement>,
    limits: EncoderLimits,
    quarantined: Cell<bool>,
    unfenced: RefCell<Vec<B::Buffer>>,
}

impl<B: EncoderOps> Context<B> {
    fn backend(&self) -> Result<std::cell::RefMut<'_, B>> {
        self.backend
            .try_borrow_mut()
            .map_err(|_| ExecutorError::BackendFailure("encoder backend is busy with another task"))
    }
    fn validate(&self) -> Result<()> {
        if self.quarantined.get() {
            return Err(ExecutorError::BackendFailure(
                "encoder is quarantined after an unconfirmed submission failure",
            ));
        }
        let backend = self.backend()?;
        let lease = backend.lease();
        if !self.lease.same_actual_instance(&lease) {
            return Err(ExecutorError::WrongBackend);
        }
        if self.lease.identity() != lease.identity() {
            return Err(ExecutorError::StaleBuffer);
        }
        if !self.weights.inner().matches_backend_lease(&lease) {
            return Err(ExecutorError::WrongBackend);
        }
        backend.poll_retired_fences()
    }
    fn quarantine(&self, buffers: Vec<B::Buffer>) {
        self.quarantined.set(true);
        self.unfenced.borrow_mut().extend(buffers);
    }
}

/// A complete-sequence model. Causal prefix/cache execution remains independent.
pub struct Lfm2PointerEncoder<B: EncoderOps> {
    context: Rc<Context<B>>,
}

impl<B: EncoderOps> Lfm2PointerEncoder<B> {
    pub fn new(
        backend: Rc<RefCell<B>>,
        weights: Rc<EncoderTypedWeights<B::Buffer>>,
        limits: EncoderLimits,
    ) -> Result<Self> {
        if limits.max_tokens == 0
            || limits.max_tokens > 8192
            || limits.max_questions == 0
            || limits.max_questions > 4096
        {
            return Err(ExecutorError::InvalidArgument(
                "encoder limits require 1..=8192 tokens and 1..=4096 questions",
            ));
        }
        if weights
            .config()
            .backbone
            .max_position_embeddings
            .is_some_and(|maximum| limits.max_tokens > maximum)
        {
            return Err(ExecutorError::OutOfBounds(
                "encoder token limit exceeds model positions",
            ));
        }
        let observed = backend
            .try_borrow()
            .map_err(|_| ExecutorError::BackendFailure("encoder backend is busy"))?;
        let lease = observed.lease();
        if !weights.inner().matches_backend_lease(&lease) {
            return Err(ExecutorError::WrongBackend);
        }
        let retirement = observed.fence_retirement();
        drop(observed);
        Ok(Self {
            context: Rc::new(Context {
                backend,
                weights,
                lease,
                retirement,
                limits,
                quarantined: Cell::new(false),
                unfenced: RefCell::new(Vec::new()),
            }),
        })
    }

    pub fn begin_predict(&self, input: EncoderInput) -> Result<PointerTask<B>> {
        self.context.validate()?;
        input.validate(
            self.context.weights.config().backbone.vocab_size,
            self.context.limits,
        )?;
        Ok(PointerTask {
            context: Rc::clone(&self.context),
            input: Some(input),
            phase: Phase::New,
            scratch: super::execution::Scratch::default(),
            output: None,
            cursor: super::execution::EncodeCursor::default(),
        })
    }

    #[must_use]
    pub fn weights(&self) -> &EncoderTypedWeights<B::Buffer> {
        &self.context.weights
    }
}

enum Phase<B: EncoderOps> {
    New,
    Building,
    Fence(B::Fence),
    Readback(B::Readback),
    Terminal,
}

/// Submitted buffers stay owned until a terminal fence/readback, cancellation,
/// or transfer to the backend's existing abandoned-fence retirement queue.
pub struct PointerTask<B: EncoderOps> {
    context: Rc<Context<B>>,
    input: Option<EncoderInput>,
    phase: Phase<B>,
    scratch: super::execution::Scratch<B::Buffer>,
    output: Option<usize>,
    cursor: super::execution::EncodeCursor,
}

impl<B: EncoderOps> PointerTask<B> {
    fn terminal(&mut self, result: Result<PointerOutput>) -> CompletionPoll<PointerOutput> {
        self.scratch.clear();
        self.input = None;
        self.output = None;
        self.phase = Phase::Terminal;
        CompletionPoll::Ready(result)
    }

    fn retire(&mut self, fence: B::Fence) {
        let retained = core::mem::take(&mut self.scratch).into_buffers();
        if let Err(rejected) = self.context.retirement.retire(fence, retained) {
            self.context.retirement.quarantine_rejected(rejected);
        }
        self.output = None;
    }

    fn record(&mut self) -> Result<()> {
        self.context.validate()?;
        self.phase = Phase::Building;
        let result = self.context.backend().and_then(|mut backend| {
            let input = self
                .input
                .as_ref()
                .ok_or(ExecutorError::CompletionConsumed)?;
            super::execution::encode(
                &mut *backend,
                &self.context.weights,
                input,
                &mut self.scratch,
                &mut self.cursor,
            )
        });
        // Even a partially recorded pass needs a completion boundary before release.
        let fence = self.context.backend().and_then(|backend| backend.fence());
        match fence {
            Ok(fence) => match result {
                Ok(output) => {
                    self.output = output;
                    self.phase = Phase::Fence(fence);
                    Ok(())
                }
                Err(error) => {
                    self.retire(fence);
                    self.phase = Phase::Terminal;
                    Err(error)
                }
            },
            Err(error) => {
                self.context
                    .quarantine(core::mem::take(&mut self.scratch).into_buffers());
                self.phase = Phase::Terminal;
                Err(error)
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn finish(&mut self, mut values: Vec<f32>) -> Result<PointerOutput> {
        self.context.validate()?;
        let input = self.input.take().ok_or(ExecutorError::CompletionConsumed)?;
        let count = input
            .token_ids
            .len()
            .checked_mul(input.questions.len())
            .ok_or(ExecutorError::Overflow(
                "pointer output size overflows usize",
            ))?;
        if values.len()
            != count.checked_mul(2).ok_or(ExecutorError::Overflow(
                "pointer output size overflows usize",
            ))?
        {
            return Err(ExecutorError::BackendFailure(
                "pointer readback returned the wrong value count",
            ));
        }
        let scale = 1.0 / (self.context.weights.config().pointer_width as f32).sqrt();
        for value in &mut values {
            *value *= scale;
        }
        let end = values.split_off(count);
        decode_pointer(&input, values, end)
    }
}

impl<B: EncoderOps> InferenceCompletion for PointerTask<B> {
    type Output = PointerOutput;
    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match &mut self.phase {
            Phase::New => match self.record() {
                Ok(()) => CompletionPoll::Pending,
                Err(error) => self.terminal(Err(error)),
            },
            Phase::Building => self.terminal(Err(ExecutorError::BackendFailure(
                "encoder recording phase escaped its bounded step",
            ))),
            Phase::Fence(fence) => match fence.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Err(error)) => self.terminal(Err(error)),
                CompletionPoll::Ready(Ok(())) => {
                    if self.output.is_none() {
                        // The previous slice is complete before another fence is submitted.
                        return match self.record() {
                            Ok(()) => CompletionPoll::Pending,
                            Err(error) => self.terminal(Err(error)),
                        };
                    }
                    let result = self.context.validate().and_then(|()| {
                        let output = self
                            .output
                            .and_then(|index| self.scratch.get(index))
                            .ok_or(ExecutorError::BackendFailure(
                                "pointer result buffer is unavailable",
                            ))?;
                        self.context.backend()?.read_f32_async(output)
                    });
                    match result {
                        Ok(readback) => {
                            self.phase = Phase::Readback(readback);
                            CompletionPoll::Pending
                        }
                        Err(error) => self.terminal(Err(error)),
                    }
                }
            },
            Phase::Readback(readback) => match readback.poll_step() {
                CompletionPoll::Pending => CompletionPoll::Pending,
                CompletionPoll::Ready(Err(error)) => self.terminal(Err(error)),
                CompletionPoll::Ready(Ok(values)) => {
                    let result = self.finish(values);
                    self.terminal(result)
                }
            },
            Phase::Terminal => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        match core::mem::replace(&mut self.phase, Phase::Terminal) {
            Phase::New => {
                self.input = None;
                Ok(())
            }
            Phase::Fence(mut fence) => match fence.cancel() {
                Ok(()) => {
                    self.scratch.clear();
                    self.input = None;
                    Ok(())
                }
                Err(error) => {
                    self.retire(fence);
                    self.input = None;
                    Err(error)
                }
            },
            Phase::Readback(mut readback) => match readback.cancel() {
                Ok(()) => {
                    self.scratch.clear();
                    self.input = None;
                    Ok(())
                }
                Err(error) => {
                    self.phase = Phase::Readback(readback);
                    Err(error)
                }
            },
            Phase::Building => {
                self.context
                    .quarantine(core::mem::take(&mut self.scratch).into_buffers());
                Err(ExecutorError::BackendFailure(
                    "cannot cancel an unfenced encoder pass",
                ))
            }
            Phase::Terminal => Err(ExecutorError::CompletionConsumed),
        }
    }
}

impl<B: EncoderOps> Drop for PointerTask<B> {
    fn drop(&mut self) {
        match core::mem::replace(&mut self.phase, Phase::Terminal) {
            Phase::Fence(fence) => self.retire(fence),
            Phase::Building => {
                let result = self.context.backend().and_then(|backend| backend.fence());
                match result {
                    Ok(fence) => self.retire(fence),
                    Err(_) => self
                        .context
                        .quarantine(core::mem::take(&mut self.scratch).into_buffers()),
                }
            }
            // The readback owns its staging/source storage; all earlier consumers
            // were confirmed complete before the readback was submitted.
            Phase::New | Phase::Readback(_) | Phase::Terminal => {}
        }
    }
}
