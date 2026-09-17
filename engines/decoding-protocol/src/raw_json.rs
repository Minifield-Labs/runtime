//! Strict JSON parsing that preserves semantic object order and number lexemes.
//!
//! `serde_json::Value` is deliberately downstream of this module. Schema and
//! runtime-value admission first need duplicate-key evidence and raw decimal
//! spelling, both of which a normal map/number conversion can erase.

use crate::{ProtocolError, Result};
use serde_json::{Map, Number, Value};
use std::collections::HashSet;
use std::str;

/// Explicit bounds for raw JSON parsing and every parser traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawJsonLimits {
    /// Maximum source bytes before UTF-8 validation.
    pub max_bytes: usize,
    /// Maximum nesting depth; the root has depth zero.
    pub max_depth: usize,
    /// Maximum values, including object keys' values and array items.
    pub max_nodes: usize,
    /// Maximum ASCII bytes in one JSON number token.
    pub max_number_bytes: usize,
}

impl Default for RawJsonLimits {
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024,
            max_depth: 64,
            max_nodes: 100_000,
            max_number_bytes: 1024,
        }
    }
}

impl RawJsonLimits {
    /// Draft-5 runtime-value admission bounds.
    #[must_use]
    pub const fn draft5_value() -> Self {
        Self {
            max_bytes: 65_536,
            max_depth: 16,
            max_nodes: 100_000,
            max_number_bytes: 1024,
        }
    }
}

/// Ordered JSON representation with raw number text retained until admission.
#[derive(Clone, Debug, PartialEq)]
pub enum RawJson {
    /// JSON null.
    Null,
    /// JSON Boolean.
    Bool(bool),
    /// A strict RFC8259 numeric lexical value.
    Number(RawNumber),
    /// Decoded Unicode scalar string.
    String(String),
    /// Ordered JSON array.
    Array(Vec<RawJson>),
    /// Insertion-ordered object fields. Duplicate decoded keys are rejected.
    Object(Vec<(String, RawJson)>),
}

impl RawJson {
    /// Convert only after callers have performed lexical admission checks.
    pub fn into_value(self) -> Result<Value> {
        match self {
            Self::Null => Ok(Value::Null),
            Self::Bool(value) => Ok(Value::Bool(value)),
            Self::Number(value) => {
                let admitted = admit_number(&value, NumericKind::Number)?;
                let number = Number::from_f64(admitted.value).ok_or_else(|| {
                    ProtocolError::Numeric(format!(
                        "{} cannot become a finite binary64 JSON number",
                        admitted.lexical
                    ))
                })?;
                Ok(Value::Number(number))
            }
            Self::String(value) => Ok(Value::String(value)),
            Self::Array(values) => values
                .into_iter()
                .map(Self::into_value)
                .collect::<Result<Vec<_>>>()
                .map(Value::Array),
            Self::Object(entries) => {
                let mut object = Map::with_capacity(entries.len());
                for (key, value) in entries {
                    object.insert(key, value.into_value()?);
                }
                Ok(Value::Object(object))
            }
        }
    }

    /// Ordered object entries, if this is an object.
    #[must_use]
    pub fn object_entries(&self) -> Option<&[(String, RawJson)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }
}

/// Original numeric lexical spelling, validated by the strict parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawNumber(String);

impl RawNumber {
    /// Return the exact source lexical representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Schema-admission numeric domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericKind {
    /// Integer schemas require an integral raw decimal lexical value.
    Integer,
    /// Number schemas allow nonintegral source decimals to round to binary64.
    Number,
}

/// A raw decimal after draft-4 binary64 and safe-integral-domain admission.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedNumber {
    /// Exact tokenized JSON number spelling. It must remain in committed history.
    pub lexical: String,
    /// Correctly rounded binary64 semantic value for validation and assembly.
    pub value: f64,
}

/// Parse a complete RFC8259 JSON document while retaining order and raw numbers.
pub fn parse_json_document(input: &[u8], limits: RawJsonLimits) -> Result<RawJson> {
    parse(input, limits, true)
}

/// Parse one runtime dynamic value. Whitespace outside JSON strings is forbidden.
pub fn parse_runtime_value(input: &[u8]) -> Result<RawJson> {
    parse(input, RawJsonLimits::draft5_value(), false)
}

/// Apply the draft-4 binary64 admission policy before converting away the raw
/// number spelling. Bounds and multipleOf remain schema-validator work.
pub fn admit_number(number: &RawNumber, kind: NumericKind) -> Result<AdmittedNumber> {
    let value = number.as_str().parse::<f64>().map_err(|error| {
        ProtocolError::Numeric(format!("{} is not binary64: {error}", number.as_str()))
    })?;
    if !value.is_finite() {
        return Err(ProtocolError::Numeric(format!(
            "{} overflows binary64",
            number.as_str()
        )));
    }
    if value == 0.0 && !raw_is_zero(number.as_str()) {
        return Err(ProtocolError::Numeric(format!(
            "{} underflows to zero",
            number.as_str()
        )));
    }

    let raw_integral = raw_is_integral(number.as_str())?;
    let converted_integral = value.fract() == 0.0;
    if (raw_integral || converted_integral)
        && !(-9_007_199_254_740_991.0..=9_007_199_254_740_991.0).contains(&value)
    {
        return Err(ProtocolError::Numeric(format!(
            "{} is outside the safe integral domain",
            number.as_str()
        )));
    }
    if kind == NumericKind::Integer && !raw_integral {
        return Err(ProtocolError::Numeric(format!(
            "{} is not mathematically integral before binary64 conversion",
            number.as_str()
        )));
    }
    Ok(AdmittedNumber {
        lexical: number.0.clone(),
        value,
    })
}

fn parse(input: &[u8], limits: RawJsonLimits, allow_whitespace: bool) -> Result<RawJson> {
    if input.len() > limits.max_bytes {
        return Err(ProtocolError::InputLimit(format!(
            "JSON input exceeds {} bytes",
            limits.max_bytes
        )));
    }
    let source = str::from_utf8(input)
        .map_err(|error| ProtocolError::InvalidJson(format!("invalid UTF-8: {error}")))?;
    let mut parser = Parser {
        source,
        bytes: input,
        index: 0,
        limits,
        allow_whitespace,
        nodes: 0,
    };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.index != input.len() {
        return Err(parser.error("trailing bytes after complete JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    source: &'a str,
    bytes: &'a [u8],
    index: usize,
    limits: RawJsonLimits,
    allow_whitespace: bool,
    nodes: usize,
}

impl Parser<'_> {
    fn value(&mut self, depth: usize) -> Result<RawJson> {
        if depth > self.limits.max_depth {
            return Err(ProtocolError::InputLimit(format!(
                "JSON depth exceeds {}",
                self.limits.max_depth
            )));
        }
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| ProtocolError::InputLimit("JSON node count overflow".to_owned()))?;
        if self.nodes > self.limits.max_nodes {
            return Err(ProtocolError::InputLimit(format!(
                "JSON nodes exceed {}",
                self.limits.max_nodes
            )));
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'n') => self.literal(b"null", RawJson::Null),
            Some(b't') => self.literal(b"true", RawJson::Bool(true)),
            Some(b'f') => self.literal(b"false", RawJson::Bool(false)),
            Some(b'"') => self.string().map(RawJson::String),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'-' | b'0'..=b'9') => self.number().map(RawJson::Number),
            Some(_) => Err(self.error("expected a JSON value")),
            None => Err(self.error("expected a JSON value before end of input")),
        }
    }

    fn literal(&mut self, literal: &[u8], value: RawJson) -> Result<RawJson> {
        let end = self
            .index
            .checked_add(literal.len())
            .ok_or_else(|| self.error("literal length overflow"))?;
        if self.bytes.get(self.index..end) != Some(literal) {
            return Err(self.error("invalid JSON literal"));
        }
        self.index = end;
        Ok(value)
    }

    fn object(&mut self, depth: usize) -> Result<RawJson> {
        self.index += 1;
        self.skip_whitespace();
        if self.consume(b'}') {
            return Ok(RawJson::Object(Vec::new()));
        }

        let mut entries = Vec::new();
        let mut keys = HashSet::new();
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("object key must be a JSON string"));
            }
            let key = self.string()?;
            if !keys.insert(key.clone()) {
                return Err(self.error("duplicate object key after escape decoding"));
            }
            self.skip_whitespace();
            self.expect(b':', "expected colon after object key")?;
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',', "expected comma or closing brace in object")?;
        }
        Ok(RawJson::Object(entries))
    }

    fn array(&mut self, depth: usize) -> Result<RawJson> {
        self.index += 1;
        self.skip_whitespace();
        if self.consume(b']') {
            return Ok(RawJson::Array(Vec::new()));
        }

        let mut values = Vec::new();
        loop {
            values.push(self.value(depth + 1)?);
            self.skip_whitespace();
            if self.consume(b']') {
                break;
            }
            self.expect(b',', "expected comma or closing bracket in array")?;
        }
        Ok(RawJson::Array(values))
    }

    fn number(&mut self) -> Result<RawNumber> {
        let start = self.index;
        self.consume(b'-');
        match self.peek() {
            Some(b'0') => self.index += 1,
            Some(b'1'..=b'9') => {
                self.index += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.index += 1;
                }
            }
            _ => return Err(self.error("invalid JSON number integer part")),
        }
        if self.consume(b'.') {
            let fraction_start = self.index;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
            if self.index == fraction_start {
                return Err(self.error("JSON number fraction requires a digit"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.index += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.index += 1;
            }
            let exponent_start = self.index;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
            if self.index == exponent_start {
                return Err(self.error("JSON number exponent requires a digit"));
            }
        }
        let lexical_bytes = self.index - start;
        if lexical_bytes > self.limits.max_number_bytes {
            return Err(ProtocolError::InputLimit(format!(
                "JSON number exceeds {} bytes",
                self.limits.max_number_bytes
            )));
        }
        let lexical = self
            .source
            .get(start..self.index)
            .ok_or_else(|| self.error("number is not on a UTF-8 boundary"))?
            .to_owned();
        Ok(RawNumber(lexical))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"', "expected opening quote")?;
        let mut output = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| self.error("unterminated JSON string"))?;
            match byte {
                b'"' => {
                    self.index += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.index += 1;
                    self.escape_into(&mut output)?;
                }
                0..=0x1f => return Err(self.error("control byte in JSON string")),
                b'<' | b'>' if !self.allow_whitespace => {
                    return Err(self.error(
                        "runtime JSON strings must use a Unicode escape for angle brackets",
                    ));
                }
                0x20..=0x7f => {
                    self.index += 1;
                    output.push(char::from(byte));
                }
                _ => {
                    let suffix = self
                        .source
                        .get(self.index..)
                        .ok_or_else(|| self.error("string is not on a UTF-8 boundary"))?;
                    let character = suffix
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("unterminated JSON string"))?;
                    self.index += character.len_utf8();
                    output.push(character);
                }
            }
        }
    }

    fn escape_into(&mut self, output: &mut String) -> Result<()> {
        let escape = self
            .peek()
            .ok_or_else(|| self.error("unterminated JSON escape"))?;
        self.index += 1;
        match escape {
            b'"' => output.push('"'),
            b'\\' => output.push('\\'),
            b'/' => output.push('/'),
            b'b' => output.push('\u{0008}'),
            b'f' => output.push('\u{000c}'),
            b'n' => output.push('\n'),
            b'r' => output.push('\r'),
            b't' => output.push('\t'),
            b'u' => self.unicode_escape_into(output)?,
            _ => return Err(self.error("invalid JSON escape")),
        }
        Ok(())
    }

    fn unicode_escape_into(&mut self, output: &mut String) -> Result<()> {
        let first = self.hex_code_unit()?;
        let codepoint = match first {
            0xd800..=0xdbff => {
                self.expect(b'\\', "high surrogate requires a second unicode escape")?;
                self.expect(b'u', "high surrogate requires a second unicode escape")?;
                let second = self.hex_code_unit()?;
                if !(0xdc00..=0xdfff).contains(&second) {
                    return Err(self.error("high surrogate is not followed by a low surrogate"));
                }
                0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
            }
            0xdc00..=0xdfff => return Err(self.error("unpaired low surrogate")),
            _ => u32::from(first),
        };
        let character = char::from_u32(codepoint)
            .ok_or_else(|| self.error("invalid unicode scalar in JSON escape"))?;
        output.push(character);
        Ok(())
    }

    fn hex_code_unit(&mut self) -> Result<u16> {
        let end = self
            .index
            .checked_add(4)
            .ok_or_else(|| self.error("unicode escape length overflow"))?;
        let digits = self
            .bytes
            .get(self.index..end)
            .ok_or_else(|| self.error("truncated unicode escape"))?;
        let mut value = 0u16;
        for digit in digits {
            let nibble = match digit {
                b'0'..=b'9' => u16::from(digit - b'0'),
                b'a'..=b'f' => u16::from(digit - b'a' + 10),
                b'A'..=b'F' => u16::from(digit - b'A' + 10),
                _ => return Err(self.error("invalid unicode escape hex digit")),
            };
            value = (value << 4) | nibble;
        }
        self.index = end;
        Ok(value)
    }

    fn expect(&mut self, byte: u8, message: &str) -> Result<()> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(self.error(message))
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        if self.allow_whitespace {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.index += 1;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn error(&self, message: &str) -> ProtocolError {
        ProtocolError::InvalidJson(format!("byte {}: {message}", self.index))
    }
}

fn raw_is_zero(raw: &str) -> bool {
    let unsigned = raw.strip_prefix('-').unwrap_or(raw);
    let mantissa = unsigned.split(['e', 'E']).next().unwrap_or(unsigned);
    mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .all(|digit| digit == b'0')
}

fn raw_is_integral(raw: &str) -> Result<bool> {
    let unsigned = raw.strip_prefix('-').unwrap_or(raw);
    let (mantissa, exponent_text) = match unsigned.find(['e', 'E']) {
        Some(index) => (&unsigned[..index], Some(&unsigned[index + 1..])),
        None => (unsigned, None),
    };
    let fraction_len = mantissa
        .split_once('.')
        .map_or(0usize, |(_, fraction)| fraction.len());
    let digits: String = mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(char::from)
        .collect();
    if digits.bytes().all(|digit| digit == b'0') {
        return Ok(true);
    }

    let exponent = exponent_text.map_or(Ok(0), bounded_exponent)?;
    let fraction_len = i64::try_from(fraction_len)
        .map_err(|_| ProtocolError::Numeric("fraction length overflow".to_owned()))?;
    if exponent >= fraction_len {
        return Ok(true);
    }

    let trailing_zeros = digits
        .bytes()
        .rev()
        .take_while(|digit| *digit == b'0')
        .count();
    let digit_len = i64::try_from(digits.len())
        .map_err(|_| ProtocolError::Numeric("digit length overflow".to_owned()))?;
    if exponent < -digit_len {
        return Ok(false);
    }
    let decimal_places = fraction_len
        .checked_sub(exponent)
        .ok_or_else(|| ProtocolError::Numeric("decimal exponent overflow".to_owned()))?;
    let needed = usize::try_from(decimal_places)
        .map_err(|_| ProtocolError::Numeric("decimal place count overflow".to_owned()))?;
    Ok(trailing_zeros >= needed)
}

fn bounded_exponent(text: &str) -> Result<i64> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'+') => (false, &text[1..]),
        Some(b'-') => (true, &text[1..]),
        _ => (false, text),
    };
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return Ok(0);
    }
    if significant.len() > 18 {
        return Ok(if negative { i64::MIN } else { i64::MAX });
    }
    let magnitude = significant.parse::<i64>().map_err(|error| {
        ProtocolError::Numeric(format!("{text} has an invalid finite exponent: {error}"))
    })?;
    Ok(if negative { -magnitude } else { magnitude })
}
