#![forbid(unsafe_code)]
//! Portable greedy plaintext generation over caller-provided executors and tokenizer assets.
//!
//! This crate has no filesystem, network, prompt-template, tool, routing, or model-loading
//! policy. The caller supplies a `TokenExecutor`, a loaded `Tokenizer`, prompt text, and limits.

use core::fmt;

use minifield_engine_api::{
    CompletionPoll, ExecutorError, InferenceCompletion, TokenChunk, TokenExecutor, TokenId,
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

    let mut prefill = executor.prefill(TokenChunk::all(&input_ids))?;
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
        let mut append_task = executor.append_argmax(&prefix)?;
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
