//! Stable discrete decisions plus all continuous values used to produce them.
use crate::HostResult;
use minifield_executor_core::{PointerAnswer, PointerOutput};
use serde_json::{Value, json};

pub(crate) type Prediction = (Vec<f64>, Value);

pub(crate) fn finite(values: &[f64]) -> HostResult<()> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        Err("prediction must contain finite nonempty outputs".into())
    } else {
        Ok(())
    }
}

pub(crate) fn argmax(values: &[f32]) -> HostResult<usize> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("classifier prediction must contain finite nonempty outputs".into());
    }
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ordinal_level(value: f64, options: usize) -> HostResult<u32> {
    let maximum = u32::try_from(options.checked_sub(1).ok_or("ordinal has no options")?)?;
    let level = value.round();
    // Compensated FP64 expectations can round just beyond an endpoint. Check
    // the integer decision's range while preserving the continuous value.
    if !value.is_finite() || level < 0.0 || level > f64::from(maximum) {
        return Err("ordinal decision exceeds its option range".into());
    }
    Ok(level as u32)
}

pub(crate) fn pointer(output: PointerOutput) -> HostResult<Prediction> {
    let mut values: Vec<f64> = output.start.into_iter().map(f64::from).collect();
    values.extend(output.end.into_iter().map(f64::from));
    let mut decisions = Vec::with_capacity(output.answers.len());
    for answer in output.answers {
        let decision = match answer {
            PointerAnswer::Choice {
                index,
                mut probabilities,
            } => {
                if index >= probabilities.len() {
                    return Err("choice decision exceeds its options".into());
                }
                values.append(&mut probabilities);
                json!({"type":"choice","index":index})
            }
            PointerAnswer::Ordinal {
                value,
                mut probabilities,
            } => {
                let level = ordinal_level(value, probabilities.len())?;
                values.push(value);
                values.append(&mut probabilities);
                json!({"type":"ordinal","level":level})
            }
            PointerAnswer::Binary {
                probability,
                mut probabilities,
            } => {
                values.push(probability);
                values.append(&mut probabilities);
                json!({"type":"binary","value":probability>=0.5})
            }
            PointerAnswer::Span { span, presence } => {
                values.push(presence);
                span.map_or_else(
                    || json!({"type":"absent"}),
                    |[start, end]| json!({"type":"span","start":start,"end":end}),
                )
            }
        };
        decisions.push(decision);
    }
    finite(&values)?;
    Ok((values, json!(decisions)))
}
