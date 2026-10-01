//! Caller-supplied pointer roles and deterministic decoding policy.

use minifield_engine_api::{EncoderSegments, ExecutorError, Result, TokenId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncoderLimits {
    pub max_tokens: u64,
    pub max_questions: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PointerQuestionKind {
    Choice,
    Ordinal,
    Binary {
        positive_option: usize,
    },
    Extract {
        absent_index: u32,
        source_start: u32,
        selectable: Vec<bool>,
        presence_threshold: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointerQuestion {
    pub query_index: u32,
    pub option_indices: Vec<u32>,
    pub kind: PointerQuestionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncoderInput {
    pub token_ids: Vec<TokenId>,
    pub segments: EncoderSegments,
    pub questions: Vec<PointerQuestion>,
}

impl EncoderInput {
    pub(crate) fn validate(&self, vocab: u32, limits: EncoderLimits) -> Result<()> {
        let tokens = u64::try_from(self.token_ids.len())
            .map_err(|_| ExecutorError::Overflow("encoder token count exceeds u64"))?;
        if tokens == 0
            || tokens > limits.max_tokens
            || self.questions.is_empty()
            || self.questions.len()
                > usize::try_from(limits.max_questions)
                    .map_err(|_| ExecutorError::Overflow("encoder question limit exceeds usize"))?
        {
            return Err(ExecutorError::OutOfBounds(
                "encoder input exceeds token or question limits",
            ));
        }
        self.segments.validate_tokens(tokens)?;
        if self.token_ids.iter().any(|&id| id >= vocab) {
            return Err(ExecutorError::OutOfBounds(
                "encoder token exceeds vocabulary",
            ));
        }
        for question in &self.questions {
            question.validate(self.segments.ids())?;
        }
        Ok(())
    }
}

impl PointerQuestion {
    fn validate(&self, segments: &[u32]) -> Result<()> {
        let query = usize::try_from(self.query_index)
            .map_err(|_| ExecutorError::Overflow("pointer query index exceeds usize"))?;
        let segment =
            *segments
                .get(query)
                .filter(|&&id| id != 0)
                .ok_or(ExecutorError::OutOfBounds(
                    "pointer query must name an active token",
                ))?;
        let check = |index: u32| -> Result<()> {
            let index = usize::try_from(index)
                .map_err(|_| ExecutorError::Overflow("pointer option index exceeds usize"))?;
            if segments.get(index) != Some(&segment) {
                return Err(ExecutorError::OutOfBounds(
                    "pointer candidates must be in the query segment",
                ));
            }
            Ok(())
        };
        if let PointerQuestionKind::Extract {
            absent_index,
            source_start,
            selectable,
            presence_threshold,
        } = &self.kind
        {
            if !presence_threshold.is_finite() || !(0.0..=1.0).contains(presence_threshold) {
                return Err(ExecutorError::InvalidArgument(
                    "pointer presence threshold must be in [0,1]",
                ));
            }
            check(*absent_index)?;
            let end = usize::try_from(*source_start)
                .map_err(|_| ExecutorError::Overflow("pointer source start exceeds usize"))?
                .checked_add(selectable.len())
                .ok_or(ExecutorError::Overflow(
                    "pointer source end overflows usize",
                ))?;
            if end > segments.len() {
                return Err(ExecutorError::OutOfBounds(
                    "pointer selectable source exceeds tokens",
                ));
            }
            for (offset, &valid) in selectable.iter().enumerate() {
                if valid {
                    let index = u32::try_from(offset)
                        .ok()
                        .and_then(|offset| source_start.checked_add(offset))
                        .ok_or(ExecutorError::Overflow(
                            "pointer source index overflows u32",
                        ))?;
                    check(index)?;
                    if index == *absent_index {
                        return Err(ExecutorError::InvalidArgument(
                            "pointer absent marker cannot be selectable",
                        ));
                    }
                }
            }
        } else {
            if self.option_indices.is_empty() {
                return Err(ExecutorError::InvalidArgument(
                    "pointer option list must be nonempty",
                ));
            }
            let mut seen = std::collections::BTreeSet::new();
            for &index in &self.option_indices {
                check(index)?;
                if !seen.insert(index) {
                    return Err(ExecutorError::InvalidArgument(
                        "pointer options must be distinct",
                    ));
                }
            }
            if let PointerQuestionKind::Binary { positive_option } = self.kind
                && (self.option_indices.len() != 2 || positive_option >= 2)
            {
                return Err(ExecutorError::InvalidArgument(
                    "binary pointer requires two options and a valid positive option",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PointerAnswer {
    Choice {
        index: usize,
        probabilities: Vec<f64>,
    },
    Ordinal {
        value: f64,
        probabilities: Vec<f64>,
    },
    Binary {
        probability: f64,
        probabilities: Vec<f64>,
    },
    /// Half-open source-relative token span. Text/character offsets belong to the host.
    Span {
        span: Option<[u32; 2]>,
        presence: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointerOutput {
    pub tokens: usize,
    pub start: Vec<f32>,
    pub end: Vec<f32>,
    pub answers: Vec<PointerAnswer>,
}

/// Decode row-major `[questions,tokens]` scores with the same stable policy as training.
pub fn decode_pointer(
    input: &EncoderInput,
    start: Vec<f32>,
    end: Vec<f32>,
) -> Result<PointerOutput> {
    let tokens = input.token_ids.len();
    let expected = tokens
        .checked_mul(input.questions.len())
        .ok_or(ExecutorError::Overflow(
            "pointer logit count overflows usize",
        ))?;
    if start.len() != expected
        || end.len() != expected
        || start.iter().chain(&end).any(|score| !score.is_finite())
    {
        return Err(ExecutorError::BackendFailure(
            "pointer scores must have the expected finite shape",
        ));
    }
    input.segments.validate_tokens(
        u64::try_from(tokens).map_err(|_| ExecutorError::Overflow("pointer tokens exceed u64"))?,
    )?;
    let mut answers = Vec::new();
    answers
        .try_reserve_exact(input.questions.len())
        .map_err(|_| ExecutorError::ResourceLimit("pointer answers allocation failed"))?;
    for (row, question) in input.questions.iter().enumerate() {
        question.validate(input.segments.ids())?;
        let left = &start[row * tokens..(row + 1) * tokens];
        let right = &end[row * tokens..(row + 1) * tokens];
        answers.push(decode_question(question, left, right)?);
    }
    Ok(PointerOutput {
        tokens,
        start,
        end,
        answers,
    })
}

fn decode_question(
    question: &PointerQuestion,
    left: &[f32],
    right: &[f32],
) -> Result<PointerAnswer> {
    Ok(match &question.kind {
        PointerQuestionKind::Extract {
            absent_index,
            source_start,
            selectable,
            presence_threshold,
        } => {
            let mut positions = vec![*absent_index];
            positions.try_reserve_exact(selectable.len()).map_err(|_| {
                ExecutorError::ResourceLimit("pointer selectable candidates allocation failed")
            })?;
            for (index, &valid) in selectable.iter().enumerate() {
                if valid {
                    positions.push(
                        source_start
                            .checked_add(u32::try_from(index).map_err(|_| {
                                ExecutorError::Overflow("pointer source index exceeds u32")
                            })?)
                            .ok_or(ExecutorError::Overflow(
                                "pointer source index overflows u32",
                            ))?,
                    );
                }
            }
            let presence = 1.0 - option_probabilities(left, right, &positions)?[0];
            let span = if presence >= *presence_threshold {
                best_span(left, right, selectable, *source_start)?
            } else {
                None
            };
            PointerAnswer::Span { span, presence }
        }
        kind => {
            let probabilities = option_probabilities(left, right, &question.option_indices)?;
            match kind {
                PointerQuestionKind::Choice => {
                    let mut index = 0;
                    for candidate in 1..probabilities.len() {
                        if probabilities[candidate] > probabilities[index] {
                            index = candidate;
                        }
                    }
                    PointerAnswer::Choice {
                        index,
                        probabilities,
                    }
                }
                PointerQuestionKind::Ordinal => {
                    #[allow(clippy::cast_precision_loss)]
                    let value = compensated_nonnegative_sum(
                        probabilities
                            .iter()
                            .enumerate()
                            .map(|(level, &probability)| level as f64 * probability),
                    );
                    PointerAnswer::Ordinal {
                        value,
                        probabilities,
                    }
                }
                PointerQuestionKind::Binary { positive_option } => PointerAnswer::Binary {
                    probability: probabilities[*positive_option],
                    probabilities,
                },
                PointerQuestionKind::Extract { .. } => {
                    return Err(ExecutorError::BackendFailure(
                        "pointer decoder branch mismatch",
                    ));
                }
            }
        }
    })
}

/// Fixed Neumaier F64 policy for finite, nonnegative decoder terms.
///
/// The pinned training decoder calls Python `sum`, whose `CPython` 3.12/3.13
/// float path compensates additions in input order. These bounded softmax and
/// ordinal streams cannot overflow; comparing magnitudes reduces to comparing
/// values. Keep each subtraction/addition separate and apply compensation once
/// at the end. This policy does not depend on future Python version changes.
fn compensated_nonnegative_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum >= value {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

fn option_probabilities(start: &[f32], end: &[f32], positions: &[u32]) -> Result<Vec<f64>> {
    let softmax = |scores: &[f32]| -> Result<Vec<f64>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(positions.len())
            .map_err(|_| ExecutorError::ResourceLimit("pointer softmax allocation failed"))?;
        for &position in positions {
            values.push(f64::from(
                *scores
                    .get(
                        usize::try_from(position)
                            .map_err(|_| ExecutorError::Overflow("pointer index exceeds usize"))?,
                    )
                    .ok_or(ExecutorError::OutOfBounds(
                        "pointer probability index exceeds logits",
                    ))?,
            ));
        }
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = compensated_nonnegative_sum(values.iter_mut().map(|value| {
            *value = (*value - maximum).exp();
            *value
        }));
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(ExecutorError::BackendFailure(
                "pointer probability denominator is invalid",
            ));
        }
        for value in &mut values {
            *value /= denominator;
        }
        Ok(values)
    };
    let mut left = softmax(start)?;
    for (left, right) in left.iter_mut().zip(softmax(end)?) {
        *left = (*left + right) * 0.5;
    }
    Ok(left)
}

fn best_span(
    start: &[f32],
    end: &[f32],
    selectable: &[bool],
    offset: u32,
) -> Result<Option<[u32; 2]>> {
    let offset = usize::try_from(offset)
        .map_err(|_| ExecutorError::Overflow("pointer source offset exceeds usize"))?;
    let mut best = None;
    let mut score = f64::NEG_INFINITY;
    let mut first = None;
    for (index, &valid) in selectable.iter().enumerate() {
        if !valid {
            first = None;
            continue;
        }
        if first.is_none_or(|first| start[offset + index] > start[offset + first]) {
            first = Some(index);
        }
        let first = first.ok_or(ExecutorError::BackendFailure(
            "pointer selectable run has no start",
        ))?;
        let candidate = f64::from(start[offset + first]) + f64::from(end[offset + index]);
        if candidate > score {
            best = Some([
                u32::try_from(first)
                    .map_err(|_| ExecutorError::Overflow("pointer span start exceeds u32"))?,
                u32::try_from(index + 1)
                    .map_err(|_| ExecutorError::Overflow("pointer span end exceeds u32"))?,
            ]);
            score = candidate;
        }
    }
    Ok(best)
}
