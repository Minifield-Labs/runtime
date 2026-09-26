//! Exact decimal predicates over admitted binary64 values.
//!
//! Raw source spellings only control admission. Once admitted, every numeric
//! assertion uses the rational written by RFC8785 for the binary64 value.

use crate::{AdmittedNumber, NumericKind, ProtocolError, RawJson, Result, admit_number};
use serde_json::{Number, Value};
use std::cmp::Ordering;

/// Bounded decimal coefficient/exponent form of a canonical wire number.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalDecimal {
    negative: bool,
    coefficient: u128,
    exponent: i32,
    digits: String,
}

impl CanonicalDecimal {
    /// Convert a finite admitted binary64 into its RFC8785 decimal rational.
    pub fn from_admitted(number: &AdmittedNumber) -> Result<Self> {
        let json = Number::from_f64(number.value).ok_or_else(|| {
            ProtocolError::Numeric(format!(
                "{} cannot become a finite binary64 JSON number",
                number.lexical
            ))
        })?;
        let wire = serde_jcs::to_string(&Value::Number(json))?;
        Self::from_wire(&wire)
    }

    /// Parse the bounded canonical decimal spelling emitted by RFC8785.
    pub fn from_wire(wire: &str) -> Result<Self> {
        let (negative, unsigned) = wire
            .strip_prefix('-')
            .map_or((false, wire), |value| (true, value));
        let (mantissa, exponent_text) =
            unsigned.find(['e', 'E']).map_or((unsigned, None), |index| {
                (&unsigned[..index], Some(&unsigned[index + 1..]))
            });
        let mut exponent = exponent_text.map_or(Ok(0), parse_i32_exponent)?;
        let (whole, fraction) = mantissa
            .split_once('.')
            .map_or((mantissa, ""), |parts| parts);
        if whole.is_empty()
            || !whole.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(ProtocolError::Numeric(format!(
                "{wire} is not a canonical decimal"
            )));
        }
        let mut digits = String::with_capacity(whole.len() + fraction.len());
        digits.push_str(whole);
        digits.push_str(fraction);
        let without_leading = digits.trim_start_matches('0');
        if without_leading.is_empty() {
            return Ok(Self {
                negative: false,
                coefficient: 0,
                exponent: 0,
                digits: "0".to_owned(),
            });
        }
        let significant = without_leading.trim_end_matches('0');
        let removed_trailing = without_leading.len() - significant.len();
        exponent = exponent
            .checked_sub(i32::try_from(fraction.len()).map_err(|_| {
                ProtocolError::Numeric("canonical fraction length exceeds i32".to_owned())
            })?)
            .and_then(|value| value.checked_add(i32::try_from(removed_trailing).ok()?))
            .ok_or_else(|| {
                ProtocolError::Numeric("canonical decimal exponent overflow".to_owned())
            })?;
        let coefficient = significant.parse::<u128>().map_err(|_| {
            ProtocolError::Numeric("canonical decimal coefficient exceeds u128".to_owned())
        })?;
        Ok(Self {
            negative,
            coefficient,
            exponent,
            digits: significant.to_owned(),
        })
    }

    /// Compare exact wire rationals.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        match (self.coefficient == 0, other.coefficient == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => {
                return if other.negative {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
            }
            (false, true) => {
                return if self.negative {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            (false, false) => {}
        }
        if self.negative != other.negative {
            return if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        let magnitude = self.compare_magnitude(other);
        if self.negative {
            magnitude.reverse()
        } else {
            magnitude
        }
    }

    /// Return whether the exact rational is an integer multiple of a positive divisor.
    pub fn is_multiple_of(&self, divisor: &Self) -> Result<bool> {
        if divisor.negative || divisor.coefficient == 0 {
            return Err(ProtocolError::Numeric(
                "multipleOf divisor must be strictly positive".to_owned(),
            ));
        }
        if self.coefficient == 0 {
            return Ok(true);
        }

        let (divisor_twos, after_twos) = factor_count(divisor.coefficient, 2);
        let (divisor_fives, divisor_rest) = factor_count(after_twos, 5);
        if self.coefficient % divisor_rest != 0 {
            return Ok(false);
        }
        let (instance_twos, after_instance_twos) = factor_count(self.coefficient, 2);
        let (instance_fives, _) = factor_count(after_instance_twos, 5);
        let decimal_shift = i64::from(self.exponent) - i64::from(divisor.exponent);
        let instance_twos = i64::try_from(instance_twos)
            .map_err(|_| ProtocolError::Numeric("factor count exceeds i64".to_owned()))?;
        let divisor_twos = i64::try_from(divisor_twos)
            .map_err(|_| ProtocolError::Numeric("factor count exceeds i64".to_owned()))?;
        let instance_fives = i64::try_from(instance_fives)
            .map_err(|_| ProtocolError::Numeric("factor count exceeds i64".to_owned()))?;
        let divisor_fives = i64::try_from(divisor_fives)
            .map_err(|_| ProtocolError::Numeric("factor count exceeds i64".to_owned()))?;
        Ok(instance_twos + decimal_shift >= divisor_twos
            && instance_fives + decimal_shift >= divisor_fives)
    }

    fn compare_magnitude(&self, other: &Self) -> Ordering {
        let self_order =
            i64::try_from(self.digits.len()).unwrap_or(i64::MAX) + i64::from(self.exponent);
        let other_order =
            i64::try_from(other.digits.len()).unwrap_or(i64::MAX) + i64::from(other.exponent);
        match self_order.cmp(&other_order) {
            Ordering::Equal => {
                let width = self.digits.len().max(other.digits.len());
                for index in 0..width {
                    let left = self.digits.as_bytes().get(index).copied().unwrap_or(b'0');
                    let right = other.digits.as_bytes().get(index).copied().unwrap_or(b'0');
                    match left.cmp(&right) {
                        Ordering::Equal => {}
                        ordering => return ordering,
                    }
                }
                Ordering::Equal
            }
            ordering => ordering,
        }
    }
}

fn parse_i32_exponent(text: &str) -> Result<i32> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'+') => (false, &text[1..]),
        Some(b'-') => (true, &text[1..]),
        _ => (false, text),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ProtocolError::Numeric(
            "canonical decimal exponent is invalid".to_owned(),
        ));
    }
    let magnitude = digits
        .parse::<i32>()
        .map_err(|_| ProtocolError::Numeric("canonical decimal exponent exceeds i32".to_owned()))?;
    Ok(if negative { -magnitude } else { magnitude })
}

fn factor_count(mut value: u128, factor: u128) -> (usize, u128) {
    let mut count = 0usize;
    while value % factor == 0 {
        value /= factor;
        count += 1;
    }
    (count, value)
}

/// Semantic JSON equality with declared binary64 numeric interpretation.
pub fn semantic_equal(left: &RawJson, right: &RawJson) -> Result<bool> {
    match (left, right) {
        (RawJson::Null, RawJson::Null) => Ok(true),
        (RawJson::Bool(left), RawJson::Bool(right)) => Ok(left == right),
        (RawJson::Number(left), RawJson::Number(right)) => {
            let left_bits = admit_number(left, NumericKind::Number)?.value.to_bits();
            let right_bits = admit_number(right, NumericKind::Number)?.value.to_bits();
            Ok(left_bits == right_bits || ((left_bits << 1) == 0 && (right_bits << 1) == 0))
        }
        (RawJson::String(left), RawJson::String(right)) => Ok(left == right),
        (RawJson::Array(left), RawJson::Array(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (left, right) in left.iter().zip(right) {
                if !semantic_equal(left, right)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (RawJson::Object(left), RawJson::Object(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (key, value) in left {
                let Some((_, other)) = right.iter().find(|(other_key, _)| other_key == key) else {
                    return Ok(false);
                };
                if !semantic_equal(value, other)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
