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
    generate_impl(executor, tokenizer, request, None, cancellation)
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
    generate_impl(executor, tokenizer, request, Some(constraint), cancellation)
}

fn generate_impl<E, C>(
    executor: &mut E,
    tokenizer: &Tokenizer,
    request: &GenerationRequest<'_>,
    mut constraint: Option<&mut dyn DecodeConstraint>,
    cancellation: &mut C,
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

/// Caller-provided structured-choice request. The prompt is prefilled once
/// and each selector contributes exactly one continuation token whose logit
/// is scored; no output tokens are generated.
#[derive(Clone, Debug)]
pub struct ChoiceRequest<'a> {
    pub prompt: &'a str,
    pub add_bos: bool,
    pub selectors: &'a [&'a str],
    pub max_context_tokens: usize,
}

/// One selector's score within a completed choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceScore {
    pub selector: String,
    pub token_id: TokenId,
    pub probability: f32,
}

/// Completed choice: the normalized selector scores in caller order plus the
/// index of the winning selector.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceResult {
    pub input_ids: Vec<TokenId>,
    pub selected_index: usize,
    pub scores: Vec<ChoiceScore>,
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
                write!(formatter, "invalid choice selectors: {reason}")
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

/// Validated, tokenized choice state. Produced by [`prepare_choice`],
/// consumed by [`finish_choice`]; async hosts prefill `input_ids()` once and
/// pass `token_ids()` to `TokenChoiceExecutor::choice_logits` so only the
/// selector logits cross to the host.
pub struct PreparedChoice {
    input_ids: Vec<TokenId>,
    selectors: Vec<String>,
    token_ids: Vec<TokenId>,
}

impl PreparedChoice {
    /// The tokenized prompt input for the single prefill.
    #[must_use]
    pub fn input_ids(&self) -> &[TokenId] {
        &self.input_ids
    }

    /// The selector continuation token IDs in caller order.
    #[must_use]
    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }
}

/// Tokenize and validate a choice request without touching an executor.
///
/// Every selector must be nonempty, encode to exactly one token without
/// special tokens, carry a distinct token ID, and compose with the prompt:
/// encoding `prompt + selector` must equal `input_ids` followed by that one
/// token. `max_context_tokens` bounds the prompt input only.
///
/// # Errors
///
/// Returns `InvalidChoices` when any selector rule fails, `ContextLimit`
/// when the prompt exceeds the context bound, and `Tokenizer` for encoding
/// failures.
pub fn prepare_choice(
    tokenizer: &Tokenizer,
    request: &ChoiceRequest<'_>,
) -> Result<PreparedChoice, ChoiceError> {
    if request.selectors.len() < 2 {
        return Err(ChoiceError::InvalidChoices(
            "choice requires at least two selectors",
        ));
    }
    let input_ids = tokenizer.encode(
        request.prompt,
        EncodeOptions {
            add_special_tokens: request.add_bos,
        },
    )?;
    if input_ids.len() > request.max_context_tokens {
        return Err(ChoiceError::ContextLimit {
            required: input_ids.len(),
            limit: request.max_context_tokens,
        });
    }
    let mut selectors = Vec::with_capacity(request.selectors.len());
    let mut token_ids = Vec::with_capacity(request.selectors.len());
    let mut continued = String::new();
    for selector in request.selectors {
        if selector.is_empty() {
            return Err(ChoiceError::InvalidChoices("choice selector is empty"));
        }
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
        let token_id = ids[0];
        if token_ids.contains(&token_id) {
            return Err(ChoiceError::InvalidChoices(
                "choice selector token IDs must be distinct",
            ));
        }
        continued.clear();
        continued.push_str(request.prompt);
        continued.push_str(selector);
        let continued_ids = tokenizer.encode(
            &continued,
            EncodeOptions {
                add_special_tokens: request.add_bos,
            },
        )?;
        if continued_ids.len() != input_ids.len() + 1
            || continued_ids[..input_ids.len()] != input_ids[..]
            || continued_ids.last() != Some(&token_id)
        {
            return Err(ChoiceError::InvalidChoices(
                "choice selector is not a compositional continuation of the prompt",
            ));
        }
        token_ids.push(token_id);
        selectors.push((*selector).to_owned());
    }
    Ok(PreparedChoice {
        input_ids,
        selectors,
        token_ids,
    })
}

/// Normalize the selector logits of a completed choice readback.
///
/// `logits` must contain exactly one value per selector, in caller order.
/// The softmax runs in f64 over only the selected logits and emits f32
/// probabilities; the first selector wins an exact-logit tie.
///
/// # Errors
///
/// Returns `Executor` when the logit count differs from the selector count
/// and `NonFiniteLogit` when any selected value is non-finite.
#[allow(clippy::needless_pass_by_value)]
pub fn finish_choice(
    prepared: PreparedChoice,
    logits: Vec<f32>,
) -> Result<ChoiceResult, ChoiceError> {
    let PreparedChoice {
        input_ids,
        selectors,
        token_ids,
    } = prepared;
    if logits.len() != token_ids.len() {
        return Err(ChoiceError::Executor(ExecutorError::InvalidShape(
            "executor returned a logit count different from the selector count",
        )));
    }
    for (choice_index, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(ChoiceError::NonFiniteLogit { choice_index });
        }
    }
    let selected_index = logits
        .iter()
        .enumerate()
        .fold((0_usize, logits[0]), |best, (index, &logit)| {
            if logit > best.1 { (index, logit) } else { best }
        })
        .0;
    let maximum = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut denominator = 0.0_f64;
    for &logit in &logits {
        let weight = (f64::from(logit) - maximum).exp();
        weights.push(weight);
        denominator += weight;
    }
    let mut scores = Vec::with_capacity(token_ids.len());
    for ((selector, token_id), weight) in selectors.into_iter().zip(token_ids).zip(weights) {
        #[allow(clippy::cast_possible_truncation)]
        let probability = (weight / denominator) as f32;
        scores.push(ChoiceScore {
            selector,
            token_id,
            probability,
        });
    }
    Ok(ChoiceResult {
        input_ids,
        selected_index,
        scores,
    })
}

/// Single-pass structured choice: one prompt prefill plus one selector-logit
/// readback, then a candidate-only softmax. No output tokens are generated
/// and no full vocabulary row crosses to the host.
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
    let mut prefill = executor.prefill(TokenChunk::all(prepared.input_ids()))?;
    let prefix = complete_choice(&mut prefill, cancellation)?;
    let mut selected = executor.choice_logits(&prefix, prepared.token_ids())?;
    let logits = complete_choice(&mut selected, cancellation)?;
    finish_choice(prepared, logits)
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
