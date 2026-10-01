#![forbid(unsafe_code)]
//! Portable greedy plaintext generation over caller-provided executors and tokenizer assets.
//!
//! This crate has no filesystem, network, prompt-template, tool, routing, or model-loading
//! policy. The caller supplies a `TokenExecutor`, a loaded `Tokenizer`, prompt text, and limits.

use core::fmt;

use minifield_engine_api::{
    CompletionPoll, DecodeConstraint, ExecutorError, InferenceCompletion, TokenChoiceExecutor,
    TokenChunk, TokenExecutor, TokenId,
};
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerError};

#[cfg(test)]
use minifield_text_tokenizer::MODEL_VOCAB_SIZE;

/// The fixed generation policy supported by the first integration slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationPolicy {
    /// Choose the lowest token ID among exactly tied finite maximum logits.
    Greedy,
}

/// Why a completed generation returned normally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// The caller's requested output-token count was reached.
    MaxOutputTokens,
    /// The next selected ID was an explicit caller stop token and was not emitted.
    StopToken(TokenId),
}

/// Caller-provided bounded plaintext-generation request.
#[derive(Clone, Debug)]
pub struct GenerationRequest<'a> {
    pub prompt: &'a str,
    pub add_bos: bool,
    pub max_output_tokens: usize,
    pub max_context_tokens: usize,
    pub stop_token_ids: &'a [TokenId],
    pub skip_special_tokens: bool,
    pub policy: GenerationPolicy,
}

impl<'a> GenerationRequest<'a> {
    /// Constructs an explicit greedy request with EOS 7 as its only stop token.
    #[must_use]
    pub const fn with_eos(
        prompt: &'a str,
        max_output_tokens: usize,
        max_context_tokens: usize,
    ) -> Self {
        Self {
            prompt,
            add_bos: false,
            max_output_tokens,
            max_context_tokens,
            stop_token_ids: &[7],
            skip_special_tokens: true,
            policy: GenerationPolicy::Greedy,
        }
    }
}

/// Completed plaintext and the exact raw generated IDs that formed it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationResult {
    pub input_ids: Vec<TokenId>,
    pub generated_ids: Vec<TokenId>,
    pub text: String,
    pub stop_reason: StopReason,
}

/// Cooperative caller cancellation probe.
pub trait Cancellation {
    /// Returns true when the current generation should request cancellation.
    fn is_cancelled(&mut self) -> bool;
}

impl<F> Cancellation for F
where
    F: FnMut() -> bool,
{
    fn is_cancelled(&mut self) -> bool {
        self()
    }
}

/// A cancellation probe that never requests cancellation.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&mut self) -> bool {
        false
    }
}

/// Checked generation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationError {
    Tokenizer(TokenizerError),
    Executor(ExecutorError),
    Cancelled,
    ContextLimit { required: usize, limit: usize },
    LogitLength { actual: usize, expected: usize },
    NonFiniteLogit { token_id: TokenId },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Executor(error) => write!(formatter, "executor error: {error}"),
            Self::Cancelled => formatter.write_str("generation cancelled"),
            Self::ContextLimit { required, limit } => {
                write!(
                    formatter,
                    "generation requires {required} context tokens, above limit {limit}"
                )
            }
            Self::LogitLength { actual, expected } => {
                write!(
                    formatter,
                    "executor returned {actual} logits, expected full head of {expected}"
                )
            }
            Self::NonFiniteLogit { token_id } => write!(
                formatter,
                "executor returned non-finite logit for token {token_id}"
            ),
        }
    }
}

impl std::error::Error for GenerationError {}

impl From<TokenizerError> for GenerationError {
    fn from(error: TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<ExecutorError> for GenerationError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(error)
    }
}

/// Runs bounded greedy generation over the complete 65,536-token model head.
///
/// The executor resolves the greedy next token inside its append/prefill
/// completion (`sampled_token`), so no full logits row crosses to the host.
/// The executor prefix and output decoder are only replaced after an append
/// completion succeeds. A selected unmapped or malformed UTF-8 token is
/// rejected before that append begins.
///
/// # Errors
///
/// Returns an error for tokenizer admission/decoding, executor completion, cancellation, invalid
/// logits, or caller context limits. A failed operation never publishes its candidate prefix or
/// candidate decoded text.
pub fn generate<E, C>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &GenerationRequest<'_>,
    cancellation: &mut C,
) -> Result<GenerationResult, GenerationError>
where
    E: TokenExecutor,
    C: Cancellation,
{
    generate_impl(
        executor,
        tokenizer,
        request,
        None,
        cancellation,
        &mut |_, _| {},
    )
}

/// `generate` under a per-step token constraint: before each sampled step the
/// constraint's bitset gates the executor's argmax, so only an allowed id can
/// be emitted. Stop-token handling is unchanged: a constraint that allows the
/// stop id when complete lets generation end naturally.
///
/// # Errors
///
/// Same as [`generate`], plus `Unsupported` when the executor has no masked
/// prefill/append implementation.
pub fn generate_constrained<E, C>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &GenerationRequest<'_>,
    constraint: &mut dyn DecodeConstraint,
    cancellation: &mut C,
) -> Result<GenerationResult, GenerationError>
where
    E: TokenExecutor,
    C: Cancellation,
{
    generate_impl(
        executor,
        tokenizer,
        request,
        Some(constraint),
        cancellation,
        &mut |_, _| {},
    )
}

/// Content-free progress boundaries for host measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationProgress {
    Tokenized(usize),
    Prefilled,
    TokenEmitted,
}

/// Generate with count/timing hooks. The observer receives no prompt or output content.
///
/// # Errors
/// Returns the same errors as [`generate`].
pub fn generate_observed<E: TokenExecutor, C: Cancellation>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &GenerationRequest<'_>,
    cancellation: &mut C,
    observer: &mut impl FnMut(GenerationProgress, &E),
) -> Result<GenerationResult, GenerationError> {
    generate_impl(executor, tokenizer, request, None, cancellation, observer)
}

fn generate_impl<E, C>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &GenerationRequest<'_>,
    mut constraint: Option<&mut dyn DecodeConstraint>,
    cancellation: &mut C,
    observer: &mut impl FnMut(GenerationProgress, &E),
) -> Result<GenerationResult, GenerationError>
where
    E: TokenExecutor,
    C: Cancellation,
{
    if request.policy != GenerationPolicy::Greedy {
        return Err(GenerationError::Executor(ExecutorError::Unsupported(
            "only greedy generation is implemented",
        )));
    }
    let input_ids = tokenizer.encode(
        request.prompt,
        EncodeOptions {
            add_special_tokens: request.add_bos,
        },
    )?;
    observer(GenerationProgress::Tokenized(input_ids.len()), executor);
    ensure_context(input_ids.len(), request.max_context_tokens)?;
    if request.max_output_tokens == 0 {
        return Ok(GenerationResult {
            input_ids,
            generated_ids: Vec::new(),
            text: String::new(),
            stop_reason: StopReason::MaxOutputTokens,
        });
    }

    let mut prefill = match constraint.as_deref_mut() {
        Some(constraint) => {
            executor.prefill_masked(TokenChunk::all(&input_ids), constraint.allowed())?
        }
        None => executor.prefill(TokenChunk::all(&input_ids))?,
    };
    let mut prefix = complete(&mut prefill, cancellation)?;
    observer(GenerationProgress::Prefilled, executor);
    let mut generated_ids = Vec::new();
    let mut decoded = tokenizer.streaming_decoder(request.skip_special_tokens);
    let mut text = String::new();

    loop {
        let required = input_ids
            .len()
            .checked_add(generated_ids.len())
            .and_then(|length| length.checked_add(1))
            .ok_or(GenerationError::ContextLimit {
                required: usize::MAX,
                limit: request.max_context_tokens,
            })?;
        ensure_context(required, request.max_context_tokens)?;

        let next_id = executor
            .sampled_token(&prefix)?
            .ok_or(GenerationError::Executor(ExecutorError::BackendFailure(
                "executor published no greedy next-token sample",
            )))?;
        if request.stop_token_ids.contains(&next_id) {
            let tail = decoded.finish()?;
            text.push_str(&tail);
            return Ok(GenerationResult {
                input_ids,
                generated_ids,
                text,
                stop_reason: StopReason::StopToken(next_id),
            });
        }

        let mut candidate_decoder = decoded.clone();
        let decoded_fragment = candidate_decoder.push(&[next_id])?;
        if cancellation.is_cancelled() {
            return Err(GenerationError::Cancelled);
        }
        let mut append_task = match constraint.as_deref_mut() {
            Some(constraint) => {
                constraint.advance(next_id);
                executor.append_argmax_masked(prefix, constraint.allowed())?
            }
            None => executor.append_argmax(prefix)?,
        };
        let candidate_prefix = complete(&mut append_task, cancellation)?;
        prefix = candidate_prefix;
        decoded = candidate_decoder;
        generated_ids.push(next_id);
        observer(GenerationProgress::TokenEmitted, executor);
        text.push_str(&decoded_fragment);

        if generated_ids.len() == request.max_output_tokens {
            let tail = decoded.finish()?;
            text.push_str(&tail);
            return Ok(GenerationResult {
                input_ids,
                generated_ids,
                text,
                stop_reason: StopReason::MaxOutputTokens,
            });
        }
    }
}

fn ensure_context(required: usize, limit: usize) -> Result<(), GenerationError> {
    if required > limit {
        return Err(GenerationError::ContextLimit { required, limit });
    }
    Ok(())
}

/// One criterion in a structured choice: a caller-facing name plus the tail
/// appended to the shared base prompt for this criterion's true/false
/// continuation.
#[derive(Clone, Debug)]
pub struct ChoiceCriterion<'a> {
    pub name: &'a str,
    pub tail: &'a str,
}

/// Caller-provided structured-choice request. The shared base prompt is
/// prefilled once, then each criterion tail is appended serially and scored
/// against the shared one-token true and false selectors; no output tokens
/// are generated.
#[derive(Clone, Debug)]
pub struct ChoiceRequest<'a> {
    pub base_prompt: &'a str,
    pub criteria: &'a [ChoiceCriterion<'a>],
    pub true_selector: &'a str,
    pub false_selector: &'a str,
    pub add_bos: bool,
    pub max_context_tokens: usize,
}

/// One criterion's normalized probability within a completed choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceProbability {
    pub name: String,
    pub probability: f32,
}

/// Completed choice: the winning criterion name, a normalized confidence in
/// `[0, 1]`, and per-criterion probabilities in caller order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceResult {
    pub choice: String,
    pub confidence: f32,
    pub probabilities: Vec<ChoiceProbability>,
}

/// Checked structured-choice failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChoiceError {
    Tokenizer(TokenizerError),
    Executor(ExecutorError),
    Cancelled,
    ContextLimit { required: usize, limit: usize },
    InvalidChoices(&'static str),
    NonFiniteLogit { choice_index: usize },
}

impl fmt::Display for ChoiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Executor(error) => write!(formatter, "executor error: {error}"),
            Self::Cancelled => formatter.write_str("choice cancelled"),
            Self::ContextLimit { required, limit } => {
                write!(
                    formatter,
                    "choice requires {required} context tokens, above limit {limit}"
                )
            }
            Self::InvalidChoices(reason) => {
                write!(formatter, "invalid choice request: {reason}")
            }
            Self::NonFiniteLogit { choice_index } => write!(
                formatter,
                "executor returned non-finite logit for choice {choice_index}"
            ),
        }
    }
}

impl std::error::Error for ChoiceError {}

impl From<TokenizerError> for ChoiceError {
    fn from(error: TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<ExecutorError> for ChoiceError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(error)
    }
}

/// One prepared criterion: its caller-provided name and tokenized tail
/// appended to the shared base prefix for this criterion's scoring pass.
pub struct PreparedCriterion {
    name: String,
    tail_ids: Vec<TokenId>,
}

impl PreparedCriterion {
    /// The criterion's caller-provided name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tokenized tail appended to the shared base for this criterion.
    #[must_use]
    pub fn tail_ids(&self) -> &[TokenId] {
        &self.tail_ids
    }
}

/// Validated, tokenized choice state. Produced by [`prepare_choice`],
/// consumed by [`finish_choice`]; async hosts prefill `base_input_ids()` once
/// through `TokenChoiceExecutor::prefill_choice_base`, then serially pass
/// each criterion's `tail_ids()` and `token_ids()` to
/// `TokenChoiceExecutor::append_choice_logits`, so only two logits per
/// criterion cross to the host.
pub struct PreparedChoice {
    base_input_ids: Vec<TokenId>,
    criteria: Vec<PreparedCriterion>,
    token_ids: [TokenId; 2],
}

impl PreparedChoice {
    /// The tokenized shared base prompt, prefilled once for all criteria.
    #[must_use]
    pub fn base_input_ids(&self) -> &[TokenId] {
        &self.base_input_ids
    }

    /// The prepared criteria in caller order.
    #[must_use]
    pub fn criteria(&self) -> &[PreparedCriterion] {
        &self.criteria
    }

    /// The shared `[true, false]` selector token IDs.
    #[must_use]
    pub fn token_ids(&self) -> &[TokenId; 2] {
        &self.token_ids
    }
}

/// Tokenize and validate a choice request without touching an executor.
///
/// The base prompt must be nonempty and a request must list between 2 and
/// 255 criteria. Every criterion name and tail must be nonempty and every
/// name unique. The base is encoded once with the request's BOS policy;
/// every tail is encoded without special tokens and must compose strictly:
/// encoding `base_prompt + tail` with BOS must equal the base IDs followed
/// by the tail IDs, and `base_ids.len() + tail_ids.len()` is bounded by
/// `max_context_tokens`. The true and false selectors must be nonempty,
/// distinct, encode to exactly one distinct token each without special
/// tokens, and compose with every `base_prompt + tail`: encoding the full
/// text plus a selector must equal the criterion IDs followed by exactly the
/// selector token.
///
/// # Errors
///
/// Returns `InvalidChoices` when any base, criterion, or selector rule
/// fails, `ContextLimit` when a criterion's total exceeds the context bound,
/// and `Tokenizer` for encoding failures.
#[allow(clippy::too_many_lines)]
pub fn prepare_choice(
    tokenizer: &Tokenizer,
    request: &ChoiceRequest<'_>,
) -> Result<PreparedChoice, ChoiceError> {
    if request.base_prompt.is_empty() {
        return Err(ChoiceError::InvalidChoices("choice base prompt is empty"));
    }
    if request.criteria.len() < 2 {
        return Err(ChoiceError::InvalidChoices(
            "choice requires at least two criteria",
        ));
    }
    if request.criteria.len() > 255 {
        return Err(ChoiceError::InvalidChoices(
            "choice supports at most 255 criteria",
        ));
    }
    if request.true_selector.is_empty() || request.false_selector.is_empty() {
        return Err(ChoiceError::InvalidChoices("choice selector is empty"));
    }
    if request.true_selector == request.false_selector {
        return Err(ChoiceError::InvalidChoices(
            "choice selectors must be distinct",
        ));
    }
    let mut token_ids = [0; 2];
    for (slot, selector) in [request.true_selector, request.false_selector]
        .iter()
        .enumerate()
    {
        let ids = tokenizer.encode(
            selector,
            EncodeOptions {
                add_special_tokens: false,
            },
        )?;
        if ids.len() != 1 {
            return Err(ChoiceError::InvalidChoices(
                "choice selector must encode to exactly one token",
            ));
        }
        token_ids[slot] = ids[0];
    }
    if token_ids[0] == token_ids[1] {
        return Err(ChoiceError::InvalidChoices(
            "choice selector token IDs must be distinct",
        ));
    }
    let base_input_ids = tokenizer.encode(
        request.base_prompt,
        EncodeOptions {
            add_special_tokens: request.add_bos,
        },
    )?;
    let mut criteria = Vec::with_capacity(request.criteria.len());
    let mut continued = String::new();
    for criterion in request.criteria {
        if criterion.name.is_empty() {
            return Err(ChoiceError::InvalidChoices("criterion name is empty"));
        }
        if criteria
            .iter()
            .any(|prepared: &PreparedCriterion| prepared.name == criterion.name)
        {
            return Err(ChoiceError::InvalidChoices(
                "criterion names must be unique",
            ));
        }
        if criterion.tail.is_empty() {
            return Err(ChoiceError::InvalidChoices("criterion tail is empty"));
        }
        let tail_ids = tokenizer.encode(
            criterion.tail,
            EncodeOptions {
                add_special_tokens: false,
            },
        )?;
        let total = base_input_ids.len() + tail_ids.len();
        if total > request.max_context_tokens {
            return Err(ChoiceError::ContextLimit {
                required: total,
                limit: request.max_context_tokens,
            });
        }
        continued.clear();
        continued.push_str(request.base_prompt);
        continued.push_str(criterion.tail);
        let composed = tokenizer.encode(
            &continued,
            EncodeOptions {
                add_special_tokens: request.add_bos,
            },
        )?;
        if composed.len() != total
            || composed[..base_input_ids.len()] != base_input_ids[..]
            || composed[base_input_ids.len()..] != tail_ids[..]
        {
            return Err(ChoiceError::InvalidChoices(
                "criterion tail is not a compositional continuation of the base prompt",
            ));
        }
        for (selector, token_id) in [request.true_selector, request.false_selector]
            .iter()
            .zip(token_ids.iter())
        {
            continued.push_str(selector);
            let continued_ids = tokenizer.encode(
                &continued,
                EncodeOptions {
                    add_special_tokens: request.add_bos,
                },
            )?;
            continued.truncate(continued.len() - selector.len());
            if continued_ids.len() != total + 1
                || continued_ids[..total] != composed[..]
                || continued_ids.last() != Some(token_id)
            {
                return Err(ChoiceError::InvalidChoices(
                    "choice selector is not a compositional continuation of the prompt",
                ));
            }
        }
        criteria.push(PreparedCriterion {
            name: criterion.name.to_owned(),
            tail_ids,
        });
    }
    Ok(PreparedChoice {
        base_input_ids,
        criteria,
        token_ids,
    })
}

/// Normalize the per-criterion `[true, false]` logit pairs of a completed
/// choice.
///
/// `logit_pairs` must contain exactly one two-value pair per criterion, in
/// caller order. Each pair's evidence is `true - false` in f64; a stable f64
/// softmax over all evidence values yields the probabilities, the first
/// strict evidence maximum wins, and confidence is
/// `(K * max_probability - 1) / (K - 1)` clamped to `[0, 1]`.
///
/// # Errors
///
/// Returns `Executor` when the pair count or a pair's length differs from
/// the prepared request and `NonFiniteLogit` when any selected value is
/// non-finite.
#[allow(clippy::needless_pass_by_value)]
pub fn finish_choice(
    prepared: PreparedChoice,
    logit_pairs: Vec<Vec<f32>>,
) -> Result<ChoiceResult, ChoiceError> {
    let PreparedChoice { criteria, .. } = prepared;
    if logit_pairs.len() != criteria.len() {
        return Err(ChoiceError::Executor(ExecutorError::InvalidShape(
            "executor returned a logit pair count different from the criterion count",
        )));
    }
    let mut evidence = Vec::with_capacity(criteria.len());
    for (choice_index, pair) in logit_pairs.iter().enumerate() {
        if pair.len() != 2 {
            return Err(ChoiceError::Executor(ExecutorError::InvalidShape(
                "executor returned a logit pair that is not [true, false]",
            )));
        }
        if !pair[0].is_finite() || !pair[1].is_finite() {
            return Err(ChoiceError::NonFiniteLogit { choice_index });
        }
        evidence.push(f64::from(pair[0]) - f64::from(pair[1]));
    }
    let selected_index = evidence
        .iter()
        .enumerate()
        .fold((0_usize, evidence[0]), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0;
    let maximum = evidence.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(criteria.len());
    let mut denominator = 0.0_f64;
    for &value in &evidence {
        let weight = (value - maximum).exp();
        weights.push(weight);
        denominator += weight;
    }
    #[allow(clippy::cast_precision_loss)]
    let count = criteria.len() as f64;
    let max_probability = weights[selected_index] / denominator;
    #[allow(clippy::cast_possible_truncation)]
    let confidence = ((count * max_probability - 1.0) / (count - 1.0)).clamp(0.0, 1.0) as f32;
    let mut probabilities = Vec::with_capacity(criteria.len());
    for (criterion, weight) in criteria.iter().zip(&weights) {
        #[allow(clippy::cast_possible_truncation)]
        let probability = (weight / denominator) as f32;
        probabilities.push(ChoiceProbability {
            name: criterion.name.clone(),
            probability,
        });
    }
    Ok(ChoiceResult {
        choice: criteria[selected_index].name.clone(),
        confidence,
        probabilities,
    })
}

/// Serial structured choice: the shared base prompt is prefilled once, then
/// each criterion tail is appended and its `[true, false]` selector logits
/// read back, completed before the next criterion starts. No output tokens
/// are generated and no full vocabulary row crosses to the host.
///
/// # Errors
///
/// Same as [`prepare_choice`] and [`finish_choice`], plus `Executor` for
/// executor failures and `Cancelled` when the caller's probe fires.
pub fn choose<E, C>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &ChoiceRequest<'_>,
    cancellation: &mut C,
) -> Result<ChoiceResult, ChoiceError>
where
    E: TokenChoiceExecutor,
    C: Cancellation,
{
    let prepared = prepare_choice(tokenizer, request)?;
    let mut base_task = executor.prefill_choice_base(TokenChunk::all(prepared.base_input_ids()))?;
    let base = complete_choice(&mut base_task, cancellation)?;
    let mut logit_pairs = Vec::with_capacity(prepared.criteria().len());
    for criterion in prepared.criteria() {
        let mut task = executor.append_choice_logits(
            &base,
            TokenChunk::all(criterion.tail_ids()),
            prepared.token_ids(),
        )?;
        logit_pairs.push(complete_choice(&mut task, cancellation)?);
    }
    finish_choice(prepared, logit_pairs)
}

fn complete_choice<T, C>(completion: &mut T, cancellation: &mut C) -> Result<T::Output, ChoiceError>
where
    T: InferenceCompletion,
    C: Cancellation,
{
    loop {
        if cancellation.is_cancelled() {
            completion.cancel()?;
            return Err(ChoiceError::Cancelled);
        }
        match completion.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(result) => return result.map_err(ChoiceError::from),
        }
    }
}

fn complete<T, C>(completion: &mut T, cancellation: &mut C) -> Result<T::Output, GenerationError>
where
    T: InferenceCompletion,
    C: Cancellation,
{
    loop {
        if cancellation.is_cancelled() {
            completion.cancel()?;
            return Err(GenerationError::Cancelled);
        }
        match completion.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(result) => return result.map_err(GenerationError::from),
        }
    }
}

#[cfg(test)]
fn choose_greedy(logits: &[f32]) -> Result<TokenId, GenerationError> {
    let expected = MODEL_VOCAB_SIZE as usize;
    if logits.len() != expected {
        return Err(GenerationError::LogitLength {
            actual: logits.len(),
            expected,
        });
    }
    let mut best_id = 0_u32;
    let mut best = logits[0];
    if !best.is_finite() {
        return Err(GenerationError::NonFiniteLogit { token_id: best_id });
    }
    for (index, &value) in logits.iter().enumerate().skip(1) {
        let token_id = u32::try_from(index).map_err(|_| GenerationError::LogitLength {
            actual: logits.len(),
            expected,
        })?;
        if !value.is_finite() {
            return Err(GenerationError::NonFiniteLogit { token_id });
        }
        if value > best {
            best = value;
            best_id = token_id;
        }
    }
    Ok(best_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(winners: &[(TokenId, f32)]) -> Vec<f32> {
        let mut values = vec![-10.0; MODEL_VOCAB_SIZE as usize];
        for &(id, value) in winners {
            values[id as usize] = value;
        }
        values
    }

    #[test]
    fn greedy_prefers_lowest_exact_tie_and_stops_before_emitting_eos() {
        assert_eq!(choose_greedy(&logits(&[(3, 2.0), (9, 2.0)])), Ok(3));
        assert_eq!(choose_greedy(&logits(&[(7, 2.0)])), Ok(7));
    }

    #[test]
    fn full_head_and_nonfinite_logits_are_required() {
        assert!(matches!(
            choose_greedy(&[0.0]),
            Err(GenerationError::LogitLength { .. })
        ));
        assert_eq!(
            choose_greedy(&logits(&[(4, f32::NAN)])),
            Err(GenerationError::NonFiniteLogit { token_id: 4 })
        );
    }
}
