//! Token executor admission and public contract forwarding.

use super::{
    AllocationClass, AppendChoiceTask, ChoiceLogitsTask, ExecutorError, InferenceOps, Lfm2Executor,
    Lfm2Prefix, LogitsTask, PrefillChoiceTask, PrefixTask, Rc, Result, ScoreTask,
    TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenId, TokenIds, allocate,
    allocate_branch_storage, append_tokens, copy_branch_storage, shape,
};

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
    fn prefill_masked(&mut self, input: TokenChunk<'_>, mask: Rc<[u64]>) -> Result<Self::Prefill> {
        self.context.validate_sample_mask(&mask)?;
        Ok(PrefixTask::prefill_masked(
            Rc::clone(&self.context),
            self.accepted_tokens(input, 0)?,
            mask,
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
    fn append_argmax(&mut self, prefix: Self::Prefix) -> Result<Self::Append> {
        self.context.validate_prefix(&prefix)?;
        self.context.checked_length(prefix.storage.length, 1)?;
        if prefix.storage.sampled.is_none() || prefix.storage.sampled_id.is_none() {
            return Err(ExecutorError::InvalidArgument(
                "prefix has no resolved greedy sample to append",
            ));
        }
        Ok(PrefixTask::append_argmax(Rc::clone(&self.context), prefix))
    }
    fn append_argmax_masked(
        &mut self,
        prefix: Self::Prefix,
        mask: Rc<[u64]>,
    ) -> Result<Self::Append> {
        self.context.validate_prefix(&prefix)?;
        self.context.checked_length(prefix.storage.length, 1)?;
        self.context.validate_sample_mask(&mask)?;
        if prefix.storage.sampled.is_none() || prefix.storage.sampled_id.is_none() {
            return Err(ExecutorError::InvalidArgument(
                "prefix has no resolved greedy sample to append",
            ));
        }
        Ok(PrefixTask::append_argmax_masked(
            Rc::clone(&self.context),
            prefix,
            mask,
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

impl<B: InferenceOps> TokenChoiceExecutor for Lfm2Executor<B> {
    type ChoiceLogits = ChoiceLogitsTask<B>;
    type ChoicePrefill = PrefillChoiceTask<B>;
    type ChoiceAppend = AppendChoiceTask<B>;

    fn choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoiceLogits> {
        self.context.validate_prefix(prefix)?;
        let Some(logits) = prefix.storage.next_logits.as_ref() else {
            return Err(ExecutorError::InvalidArgument(
                "empty prefix has no next-token logits",
            ));
        };
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
        let count = u64::try_from(token_ids.len())
            .map_err(|_| ExecutorError::Overflow("choice token count exceeds u64"))?;
        let mut backend = self.context.borrow_backend()?;
        let mut gathered = allocate(&mut *backend, shape(1, count)?, AllocationClass::Scratch)?;
        backend.gather_columns(&mut gathered.buffer, &logits.buffer, token_ids)?;
        let readback = backend.read_f32_async(&gathered.buffer)?;
        drop(backend);
        Ok(ChoiceLogitsTask::new(
            Rc::clone(&self.context),
            prefix.clone(),
            gathered,
            readback,
        ))
    }

    fn prefill_choice_logits(
        &mut self,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoicePrefill> {
        self.prefill_selected(input, token_ids)
    }

    fn prefill_choice_base(&mut self, input: TokenChunk<'_>) -> Result<Self::Prefill> {
        let tokens = self.accepted_tokens(input, 0)?;
        if tokens.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "choice base prefill requires a nonempty prompt",
            ));
        }
        Ok(PrefixTask::base_prefill(Rc::clone(&self.context), tokens))
    }

    fn append_choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoiceAppend> {
        self.context.validate_prefix(prefix)?;
        let tokens = self.accepted_tokens(input, prefix.storage.length)?;
        if tokens.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "choice branch tail is empty",
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
        let mut staged = allocate_branch_storage(&self.context, &mut *backend, &prefix.storage)?;
        let mut scratch = Vec::new();
        let mut gathered = allocate(&mut *backend, shape(1, count)?, AllocationClass::Scratch)?;
        let recorded = (|| -> Result<B::Readback> {
            copy_branch_storage(&mut *backend, &prefix.storage, &mut staged)?;
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
                    "choice branch produced no logits boundary",
                ));
            };
            backend.gather_columns(&mut gathered.buffer, &logits.buffer, token_ids)?;
            backend.read_f32_async(&gathered.buffer)
        })();
        let readback = match recorded {
            Ok(readback) => readback,
            Err(error) => {
                // Recorded work may be live in a pending device queue without a
                // completion boundary; retain every owned buffer and the base
                // snapshot rather than dropping them into an unknown device
                // state.
                scratch.push(gathered.buffer);
                scratch.extend(staged.into_buffers());
                drop(backend);
                self.context
                    .quarantine_unfenced(scratch, vec![prefix.clone()]);
                return Err(error);
            }
        };
        drop(backend);
        Ok(AppendChoiceTask::new(
            Rc::clone(&self.context),
            prefix.clone(),
            staged,
            gathered,
            scratch,
            readback,
            token_ids.len(),
        ))
    }
}
