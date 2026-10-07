//! The admitted normalizer: ordered literal string replacements.
//!
//! Assets may declare `null` or a `Sequence` of `Replace` rules whose patterns
//! are plain strings, applied in order over the whole input like the reference
//! tokenizer. Every normalized byte boundary keeps its original byte offset,
//! so callers can map token spans back to the text the user typed.

use serde_json::Value;

use crate::TokenizerError;

/// Ordered literal replacements; empty when the asset's normalizer is `null`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Replacements(Vec<(String, String)>);

/// Normalized text and the original byte offset of each of its byte boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Normalized {
    /// The text the BPE model sees.
    pub text: String,
    /// `origins[i]` is the original byte offset of normalized byte boundary
    /// `i`, so its length is `text.len() + 1`. Boundaries inside a
    /// replacement map to the replaced span's start.
    pub origins: Vec<usize>,
}

impl Normalized {
    /// The original byte span covering normalized bytes `start..end`.
    #[must_use]
    pub fn original_span(&self, start: usize, end: usize) -> (usize, usize) {
        (self.origins[start], self.origins[end])
    }
}

impl Replacements {
    /// Admits `null` or a `Sequence` of literal `Replace` rules.
    pub(crate) fn from_value(value: &Value) -> Result<Self, TokenizerError> {
        if value.is_null() {
            return Ok(Self::default());
        }
        let rejected = || {
            TokenizerError::InvalidAsset(
                "normalizer must be null or a Sequence of literal Replace rules".into(),
            )
        };
        let object = value.as_object().ok_or_else(rejected)?;
        if object.len() != 2 || object.get("type") != Some(&Value::from("Sequence")) {
            return Err(rejected());
        }
        let rules = object
            .get("normalizers")
            .and_then(Value::as_array)
            .ok_or_else(rejected)?;
        let mut replacements = Vec::with_capacity(rules.len());
        for rule in rules {
            let rule = rule.as_object().ok_or_else(rejected)?;
            if rule.len() != 3 || rule.get("type") != Some(&Value::from("Replace")) {
                return Err(rejected());
            }
            let pattern = rule
                .get("pattern")
                .and_then(Value::as_object)
                .filter(|pattern| pattern.len() == 1)
                .and_then(|pattern| pattern.get("String"))
                .and_then(Value::as_str)
                .filter(|pattern| !pattern.is_empty())
                .ok_or_else(rejected)?;
            let content = rule
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(rejected)?;
            replacements.push((pattern.to_owned(), content.to_owned()));
        }
        Ok(Self(replacements))
    }

    /// Applies every rule in order, tracking original byte offsets.
    pub(crate) fn apply(&self, input: &str) -> Normalized {
        let mut text = input.to_owned();
        let mut origins: Vec<usize> = (0..=input.len()).collect();
        for (pattern, content) in &self.0 {
            if !text.contains(pattern.as_str()) {
                continue;
            }
            let mut next = String::with_capacity(text.len());
            let mut next_origins = Vec::with_capacity(origins.len());
            let mut cursor = 0;
            while cursor < text.len() {
                if text[cursor..].starts_with(pattern.as_str()) {
                    next.push_str(content);
                    next_origins.extend(std::iter::repeat_n(origins[cursor], content.len()));
                    cursor += pattern.len();
                } else {
                    let width = text[cursor..].chars().next().map_or(1, char::len_utf8);
                    next.push_str(&text[cursor..cursor + width]);
                    next_origins.extend_from_slice(&origins[cursor..cursor + width]);
                    cursor += width;
                }
            }
            next_origins.push(origins[text.len()]);
            text = next;
            origins = next_origins;
        }
        Normalized { text, origins }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::Replacements;

    fn quotes() -> Replacements {
        Replacements::from_value(&json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Replace", "pattern": {"String": "\u{2019}"}, "content": "'"},
                {"type": "Replace", "pattern": {"String": "\u{201c}"}, "content": "\""},
                {"type": "Replace", "pattern": {"String": "\u{2026}"}, "content": "..."}
            ]
        }))
        .unwrap_or_else(|error| panic!("literal replacements rejected: {error}"))
    }

    #[test]
    fn replacements_keep_original_offsets() {
        let input = "it\u{2019}s \u{201c}Q3\u{2026}";
        let normalized = quotes().apply(input);
        assert_eq!(normalized.text, "it's \"Q3...");
        assert_eq!(normalized.origins.len(), normalized.text.len() + 1);
        // "s" follows the 3-byte apostrophe in the original.
        assert_eq!(normalized.original_span(3, 4), (5, 6));
        // The whole of "Q3..." maps to "Q3…".
        let start = normalized.text.find("Q3").unwrap_or_default();
        let (from, to) = normalized.original_span(start, normalized.text.len());
        assert_eq!(&input[from..to], "Q3\u{2026}");
        assert_eq!(
            quotes().apply("plain"),
            super::Normalized {
                text: "plain".into(),
                origins: (0..=5).collect(),
            }
        );
    }

    #[test]
    fn only_literal_replace_sequences_are_admitted() {
        assert_eq!(
            Replacements::from_value(&json!(null)).ok(),
            Some(Replacements::default())
        );
        for value in [
            json!({"type": "Lowercase"}),
            json!({"type": "Replace", "pattern": {"String": "a"}, "content": "b"}),
            json!({"type": "Sequence", "normalizers": [{"type": "NFKC"}]}),
            json!({"type": "Sequence", "normalizers": [
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}]}),
            json!({"type": "Sequence", "normalizers": [
                {"type": "Replace", "pattern": {"String": ""}, "content": "x"}]}),
        ] {
            assert!(
                Replacements::from_value(&value).is_err(),
                "admitted {value}"
            );
        }
    }
}
