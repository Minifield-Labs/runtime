//! Shared-prefix admission and nonblocking cached classification.

use super::{
    AppendChoiceTask, CompletionPoll, ExecutorError, InferenceCompletion, InferenceOps,
    Lfm2Classifier, Lfm2Prefix, PrefillChoiceTask, PrefixTask, Result, TokenChunk, TokenId,
};

pub(super) struct ClassifierCache<B: InferenceOps> {
    anchor: Vec<TokenId>,
    shared_head: usize,
    base: Option<Lfm2Prefix<B>>,
}

impl<B: InferenceOps> Default for ClassifierCache<B> {
    fn default() -> Self {
        Self {
            anchor: Vec::new(),
            shared_head: 0,
            base: None,
        }
    }
}

/// Cache work admitted during one classification, including failed calls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassifierCacheStats {
    pub reused_tokens: u64,
    pub rebuilds: u64,
    pub fallback_used: bool,
}

enum Stage<B: InferenceOps> {
    Base(PrefixTask<B>),
    Full(PrefillChoiceTask<B>),
    Tail(AppendChoiceTask<B>),
    Finished,
}

/// A cached classification that publishes a rebuilt base only after it completes.
pub struct CachedClassifyTask<'a, B: InferenceOps> {
    classifier: &'a mut Lfm2Classifier<B>,
    ids: Vec<TokenId>,
    stage: Stage<B>,
    stats: ClassifierCacheStats,
}

impl<B: InferenceOps> Lfm2Classifier<B> {
    /// Reuse a common head of at least 16 tokens, always retaining a nonempty tail.
    /// Short or changed prompts fall back to full classification. Failed or cancelled
    /// rebuilds leave the previously published base intact.
    pub fn classify_cached(&mut self, input: TokenChunk<'_>) -> Result<CachedClassifyTask<'_, B>> {
        let ids = self.executor.accepted_tokens(input, 0)?;
        if ids.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "classification requires a nonempty prompt",
            ));
        }
        let head = ids.len() - 1;
        if self.cache.anchor.is_empty() {
            self.cache.anchor.clone_from(&ids);
            self.cache.shared_head = head;
        } else {
            self.cache.shared_head = self.cache.shared_head.min(head);
            self.cache.shared_head = ids
                .iter()
                .zip(&self.cache.anchor)
                .take(self.cache.shared_head)
                .take_while(|(a, b)| a == b)
                .count();
        }
        let mut stats = ClassifierCacheStats::default();
        let base = self
            .cache
            .base
            .as_ref()
            .filter(|base| {
                ids.len() > base.token_history().len() && ids.starts_with(base.token_history())
            })
            .cloned();
        let stage = if let Some(base) = base {
            stats.reused_tokens = base.logical_length();
            Stage::Tail(
                self.classify_tail(&base, TokenChunk::all(&ids[base.token_history().len()..]))?,
            )
        } else if self.cache.shared_head >= 16 {
            Stage::Base(self.prefill_base(TokenChunk::all(&ids[..self.cache.shared_head]))?)
        } else {
            stats.fallback_used = self.cache.base.is_some();
            Stage::Full(self.classify(TokenChunk::all(&ids))?)
        };
        Ok(CachedClassifyTask {
            classifier: self,
            ids,
            stage,
            stats,
        })
    }
}

impl<B: InferenceOps> CachedClassifyTask<'_, B> {
    #[must_use]
    pub fn cache_stats(&self) -> ClassifierCacheStats {
        self.stats
    }
}

impl<B: InferenceOps> InferenceCompletion for CachedClassifyTask<'_, B> {
    type Output = Vec<f32>;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        let result = match &mut self.stage {
            Stage::Base(task) => match task.poll_step() {
                CompletionPoll::Pending => return CompletionPoll::Pending,
                CompletionPoll::Ready(Err(error)) => Err(error),
                CompletionPoll::Ready(Ok(base)) => {
                    self.stats.rebuilds += 1;
                    self.stats.reused_tokens = base.logical_length();
                    self.classifier.cache.base = Some(base.clone());
                    match self.classifier.classify_tail(
                        &base,
                        TokenChunk::all(&self.ids[base.token_history().len()..]),
                    ) {
                        Ok(task) => {
                            self.stage = Stage::Tail(task);
                            return CompletionPoll::Pending;
                        }
                        Err(error) => Err(error),
                    }
                }
            },
            Stage::Full(task) => match task.poll_step() {
                CompletionPoll::Pending => return CompletionPoll::Pending,
                CompletionPoll::Ready(result) => result,
            },
            Stage::Tail(task) => match task.poll_step() {
                CompletionPoll::Pending => return CompletionPoll::Pending,
                CompletionPoll::Ready(result) => result,
            },
            Stage::Finished => Err(ExecutorError::CompletionConsumed),
        };
        self.stage = Stage::Finished;
        CompletionPoll::Ready(result)
    }

    fn cancel(&mut self) -> Result<()> {
        match &mut self.stage {
            Stage::Base(task) => task.cancel(),
            Stage::Full(task) => task.cancel(),
            Stage::Tail(task) => task.cancel(),
            Stage::Finished => Ok(()),
        }
    }
}
