//! Token inference, candidate scoring, and decode constraints.

use std::rc::Rc;

use crate::{ExecutorError, InferenceCompletion, Result};

/// A vocabulary index accepted by a loaded decoder.
pub type TokenId = u32;

/// A physical token chunk plus an optional logical-validity mask.
#[derive(Clone, Copy, Debug)]
pub struct TokenChunk<'a> {
    pub ids: &'a [TokenId],
    pub valid: Option<&'a [bool]>,
}

impl<'a> TokenChunk<'a> {
    #[must_use]
    pub const fn all(ids: &'a [TokenId]) -> Self {
        Self { ids, valid: None }
    }

    #[must_use]
    pub const fn masked(ids: &'a [TokenId], valid: &'a [bool]) -> Self {
        Self {
            ids,
            valid: Some(valid),
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.valid.is_some_and(|mask| mask.len() != self.ids.len()) {
            return Err(ExecutorError::InvalidArgument(
                "token validity mask length differs from token IDs",
            ));
        }
        Ok(())
    }

    pub fn is_valid(self, index: usize) -> Result<bool> {
        if index >= self.ids.len() {
            return Err(ExecutorError::OutOfBounds(
                "token index exceeds physical chunk",
            ));
        }
        match self.valid {
            None => Ok(true),
            Some(mask) => mask
                .get(index)
                .copied()
                .ok_or(ExecutorError::InvalidArgument(
                    "token validity mask length differs from token IDs",
                )),
        }
    }
}

/// A complete conditional candidate score.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScore {
    pub candidate_index: usize,
    pub token_count: usize,
    /// Finite f32 logits normalize and accumulate in f64 before this final f32 transport cast.
    pub log_probability: f32,
}

/// Async-compatible token inference contract. No universal blocking, Send, or Sync bound applies.
pub trait TokenExecutor {
    type Prefix;
    type Prefill: InferenceCompletion<Output = Self::Prefix>;
    /// A successful append publishes a new immutable prefix. The input snapshot is never
    /// mutated: callers replace it only after this completion is ready.
    type Append: InferenceCompletion<Output = Self::Prefix>;
    /// A fork copies backend-resident state asynchronously before publishing an independent
    /// prefix snapshot.
    type Fork: InferenceCompletion<Output = Self::Prefix>;
    type Scores: InferenceCompletion<Output = Vec<CandidateScore>>;
    type Logits: InferenceCompletion<Output = Vec<f32>>;

    fn prefill(&mut self, input: TokenChunk<'_>) -> Result<Self::Prefill>;
    fn append_known(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
    ) -> Result<Self::Append>;
    /// The greedy next-token id resolved when `prefix` was published, if it
    /// carries a logits boundary. Implementations resolve the argmax during
    /// the publish completion itself, so this accessor costs no readback and
    /// lets callers inspect the sampled token before deciding to append it.
    fn sampled_token(&mut self, prefix: &Self::Prefix) -> Result<Option<TokenId>>;
    /// Append the token currently reported by `sampled_token`. The sampled id
    /// stays backend-resident through the embedding gather, so greedy decode
    /// never reads a full logits row to the host. Takes the prefix by value:
    /// when the caller hands over the last reference, implementations may
    /// reuse its cache storage in place instead of deep-copying it. Returns
    /// `InvalidArgument` when the prefix has no resolved greedy sample.
    fn append_argmax(&mut self, prefix: Self::Prefix) -> Result<Self::Append>;
    /// `prefill` whose greedy sample is constrained to `mask`: bit `i` of the
    /// bitset (LSB-first u64 words, `ceil(vocab / 64)` long) marks an allowed
    /// token id. Implementations that cannot constrain report `Unsupported`
    /// instead of silently dropping the mask.
    fn prefill_masked(
        &mut self,
        _input: TokenChunk<'_>,
        _mask: Rc<[u64]>,
    ) -> Result<Self::Prefill> {
        Err(ExecutorError::Unsupported(
            "masked prefill is not implemented for this executor",
        ))
    }
    /// `append_argmax` whose next greedy sample is constrained to `mask`
    /// (same bitset layout as `prefill_masked`). The appended token itself
    /// was already sampled under the previous step's mask.
    fn append_argmax_masked(
        &mut self,
        _prefix: Self::Prefix,
        _mask: Rc<[u64]>,
    ) -> Result<Self::Append> {
        Err(ExecutorError::Unsupported(
            "masked append_argmax is not implemented for this executor",
        ))
    }
    fn fork(&mut self, prefix: &Self::Prefix) -> Result<Self::Fork>;
    fn next_logits(&mut self, prefix: &Self::Prefix) -> Result<Self::Logits>;
    fn score_candidates(
        &mut self,
        prefix: &Self::Prefix,
        candidates: &[&[TokenId]],
    ) -> Result<Self::Scores>;
}

/// Single-pass structured choice extension: reads back only the logits of
/// caller-selected one-token ids instead of the full vocabulary row. A
/// successful `choice_logits` completion returns the selected logits in
/// caller order, preserving duplicates, with exactly `token_ids.len()`
/// values. Implementations reject an empty selector list, ids outside the
/// vocabulary, and prefixes that carry no logits boundary.
pub trait TokenChoiceExecutor: TokenExecutor {
    type ChoiceLogits: InferenceCompletion<Output = Vec<f32>>;
    type ChoicePrefill: InferenceCompletion<Output = Vec<f32>>;
    type ChoiceAppend: InferenceCompletion<Output = Vec<f32>>;

    fn choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoiceLogits>;

    /// Prefill `input` as one single-sequence pass and resolve to the
    /// `token_ids` logits at the final position in caller order, preserving
    /// duplicates, with exactly `token_ids.len()` finite values. No prefix is
    /// published and no greedy sample is computed. Implementations reject an
    /// empty accepted prompt, an empty selector list, and ids outside the
    /// vocabulary.
    fn prefill_choice_logits(
        &mut self,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoicePrefill>;

    /// Prefill `input` as one single-sequence pass that publishes an
    /// immutable branch base: token history and layer caches are populated,
    /// but no final logits row or greedy sample is produced. The published
    /// prefix is only a valid source for [`Self::append_choice_logits`] and
    /// [`TokenExecutor::fork`]. Implementations reject an empty accepted prompt.
    fn prefill_choice_base(&mut self, input: TokenChunk<'_>) -> Result<Self::Prefill>;

    /// Branch from `prefix`, append `input` as one single-sequence pass, and
    /// resolve to the `token_ids` logits at the branch's final position in
    /// caller order, preserving duplicates, with exactly `token_ids.len()`
    /// finite values. `prefix` stays immutable and reusable for further
    /// branches; no branch prefix is published and no greedy sample is
    /// computed. Implementations reject an empty accepted tail, an empty
    /// selector list, ids outside the vocabulary, and base+tail overflow of
    /// the configured prefix capacity.
    fn append_choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoiceAppend>;
}

/// Per-step decode constraint driven by a generation loop: supplies the
/// allowed-token bitset before each sampled step and observes each emitted
/// id so the constraint can advance its own state. Bitsets are one bit per
/// token id, LSB-first u64 words (`ceil(vocab / 64)` long), matching the
/// `argmax_masked`/`prefill_masked`/`append_argmax_masked` layout.
pub trait DecodeConstraint {
    /// Allowed ids for the upcoming sample.
    fn allowed(&mut self) -> Rc<[u64]>;
    /// Records an emitted token. Called when the id is committed, after the
    /// decode checks pass and before the append that produced it publishes.
    fn advance(&mut self, token: TokenId);
}
