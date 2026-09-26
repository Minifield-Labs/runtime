//! Complete candidate scoring and probability transport.

use super::{
    CandidateScore, CompletionPoll, ExecutorError, InferenceCompletion, InferenceOps, Lfm2Prefix,
    LogitsTask, ModelContext, PrefixTask, Rc, Result, TokenId,
};

pub(super) fn candidate_score_transport(accumulated: f64) -> Result<f32> {
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

pub(super) fn token_log_probability(logits: &[f32], token: TokenId) -> Result<f64> {
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
    pub(super) fn new(
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
