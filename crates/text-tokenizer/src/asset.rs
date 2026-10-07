use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::Deserialize;
use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde_json::{Value, json};

use crate::normalize::Replacements;
use crate::{FIRST_UNMAPPED_MODEL_TOKEN_ID, TokenizerError, byte_level};

/// Walks JSON before `serde_json::Value` can overwrite duplicate object keys.
struct UniqueJsonSeed;

impl<'de> DeserializeSeed<'de> for UniqueJsonSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("valid JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E>
    where
        E: DeError,
    {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        UniqueJsonSeed.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(UniqueJsonSeed)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            map.next_value_seed(UniqueJsonSeed)?;
        }
        Ok(())
    }
}

fn reject_duplicate_json_object_keys(bytes: &[u8]) -> Result<(), TokenizerError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    UniqueJsonSeed
        .deserialize(&mut deserializer)
        .map_err(|error| TokenizerError::Json(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| TokenizerError::Json(error.to_string()))
}
/// A parsed, profile-validated caller-supplied tokenizer asset.
#[derive(Clone, Debug)]
pub struct TokenizerAsset {
    model: AssetModel,
    added_tokens: Vec<AddedToken>,
    replacements: Replacements,
}

/// Internal BPE tables. Keys are owned to keep asset parsing independent from
/// the JSON buffer lifetime.
#[derive(Clone, Debug)]
pub(crate) struct AssetModel {
    pub(crate) vocab: HashMap<String, u32>,
    pub(crate) merge_ranks: HashMap<(String, String), usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct AddedToken {
    pub(crate) id: u32,
    pub(crate) content: String,
    pub(crate) special: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAsset {
    version: String,
    truncation: Value,
    padding: Value,
    added_tokens: Vec<RawAddedToken>,
    normalizer: Value,
    pre_tokenizer: Value,
    post_processor: Value,
    decoder: Value,
    model: RawModel,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
struct RawAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    #[serde(rename = "type")]
    model_type: String,
    dropout: Value,
    unk_token: Value,
    continuing_subword_prefix: Value,
    end_of_word_suffix: Value,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: serde_json::Map<String, Value>,
    merges: Vec<Vec<Value>>,
}

impl TokenizerAsset {
    /// Parses only the profile declared by `CUSTOM_RUNTIME_TEXT_PACKET_001`.
    ///
    /// # Errors
    ///
    /// Returns an error when JSON, profile settings, IDs, BPE merges, or
    /// added-token definitions are invalid.
    #[allow(clippy::too_many_lines)]
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        reject_duplicate_json_object_keys(bytes)?;
        let raw: RawAsset = serde_json::from_slice(bytes)
            .map_err(|error| TokenizerError::Json(error.to_string()))?;
        validate_profile(&raw)?;
        let replacements = Replacements::from_value(&raw.normalizer)?;

        let mut vocab = HashMap::with_capacity(raw.model.vocab.len());
        let mut ids = HashSet::with_capacity(raw.model.vocab.len());
        let mut highest_id = 0_u32;
        for (symbol, raw_id) in raw.model.vocab {
            if symbol.is_empty() {
                return invalid("vocabulary contains an empty symbol");
            }
            if byte_level::decode_symbol(&symbol).is_none() {
                return invalid(format!(
                    "vocabulary symbol {symbol:?} is outside the ByteLevel alphabet"
                ));
            }
            let id = raw_id
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    TokenizerError::InvalidAsset(format!(
                        "vocabulary ID for {symbol:?} is not an unsigned 32-bit integer"
                    ))
                })?;
            if id >= FIRST_UNMAPPED_MODEL_TOKEN_ID {
                return invalid(format!("vocabulary ID {id} exceeds mapped token range"));
            }
            if !ids.insert(id) {
                return invalid(format!("vocabulary repeats ID {id}"));
            }
            highest_id = highest_id.max(id);
            vocab.insert(symbol, id);
        }
        if vocab.get("<|startoftext|>") != Some(&crate::BOS_TOKEN_ID) {
            return invalid("BOS template token <|startoftext|> must resolve to ID 1");
        }
        if vocab.is_empty() {
            return invalid("vocabulary is empty");
        }
        let expected_len = usize::try_from(highest_id).map_err(|_| {
            TokenizerError::InvalidAsset("highest vocabulary ID does not fit usize".into())
        })? + 1;
        if ids.len() != expected_len {
            return invalid("vocabulary IDs must be contiguous from zero");
        }

        let mut merge_ranks = HashMap::with_capacity(raw.model.merges.len());
        for (rank, pair) in raw.model.merges.into_iter().enumerate() {
            let [left, right] = pair.as_slice() else {
                return invalid(format!("merge {rank} is not a two-string pair"));
            };
            let (Some(left), Some(right)) = (left.as_str(), right.as_str()) else {
                return invalid(format!("merge {rank} is not a two-string pair"));
            };
            if left.is_empty() || right.is_empty() {
                return invalid(format!("merge {rank} has an empty symbol"));
            }
            if !vocab.contains_key(left) || !vocab.contains_key(right) {
                return invalid(format!(
                    "merge {rank} references a symbol absent from vocabulary"
                ));
            }
            let combined = format!("{left}{right}");
            if !vocab.contains_key(&combined) {
                return invalid(format!("merge {rank} result is absent from vocabulary"));
            }
            if merge_ranks
                .insert((left.to_owned(), right.to_owned()), rank)
                .is_some()
            {
                return invalid(format!("merge {rank} repeats an earlier pair"));
            }
        }

        let mut added_ids = HashSet::with_capacity(raw.added_tokens.len());
        let mut added_contents = HashSet::with_capacity(raw.added_tokens.len());
        let mut added_tokens = Vec::with_capacity(raw.added_tokens.len());
        for token in raw.added_tokens {
            if token.id >= FIRST_UNMAPPED_MODEL_TOKEN_ID {
                return invalid(format!(
                    "added token ID {} exceeds mapped token range",
                    token.id
                ));
            }
            if token.content.is_empty() {
                return invalid("added token has empty content");
            }
            if token.single_word || token.lstrip || token.rstrip {
                return invalid(format!(
                    "added token {:?} uses unsupported matching flags",
                    token.content
                ));
            }
            if !added_ids.insert(token.id) {
                return invalid(format!("added token repeats ID {}", token.id));
            }
            if !added_contents.insert(token.content.clone()) {
                return invalid(format!("added token repeats content {:?}", token.content));
            }
            if let Some(&vocab_id) = vocab.get(&token.content)
                && vocab_id != token.id
            {
                return invalid(format!(
                    "added token {:?} contradicts its vocabulary ID",
                    token.content
                ));
            }
            // Added tokens match before normalization. The admitted literal
            // replacements never touch special-token text, so `normalized`
            // changes nothing. The pinned definitions set it true.
            let _ = token.normalized;
            added_tokens.push(AddedToken {
                id: token.id,
                content: token.content,
                special: token.special,
            });
        }

        if !added_tokens.iter().any(|token| {
            token.id == crate::BOS_TOKEN_ID && token.content == "<|startoftext|>" && token.special
        }) {
            return invalid(
                "BOS template token must have a matching special added-token definition",
            );
        }

        Ok(Self {
            model: AssetModel { vocab, merge_ranks },
            added_tokens,
            replacements,
        })
    }

    pub(crate) fn into_parts(self) -> (AssetModel, Vec<AddedToken>, Replacements) {
        (self.model, self.added_tokens, self.replacements)
    }
}

fn validate_profile(raw: &RawAsset) -> Result<(), TokenizerError> {
    if raw.version != "1.0" {
        return invalid("tokenizer version must be 1.0");
    }
    if !raw.truncation.is_null() || !raw.padding.is_null() {
        return invalid("truncation and padding must be null");
    }
    if raw.pre_tokenizer != expected_pretokenizer() {
        return invalid("pre-tokenizer is not the pinned Split then ByteLevel profile");
    }
    if raw.post_processor != expected_postprocessor() {
        return invalid("post-processor is not the pinned BOS-only template profile");
    }
    if raw.decoder != expected_decoder() {
        return invalid("decoder is not the pinned ByteLevel profile");
    }
    if raw.model.model_type != "BPE"
        || !raw.model.dropout.is_null()
        || !raw.model.unk_token.is_null()
        || !raw.model.continuing_subword_prefix.is_null()
        || !raw.model.end_of_word_suffix.is_null()
        || raw.model.fuse_unk
        || raw.model.byte_fallback
        || raw.model.ignore_merges
    {
        return invalid("BPE model settings are unsupported");
    }
    Ok(())
}

fn expected_pretokenizer() -> Value {
    json!({
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                "behavior": "Isolated",
                "invert": false
            },
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]
    })
}

fn expected_postprocessor() -> Value {
    json!({
        "type": "Sequence",
        "processors": [
            {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
            {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 0}}
                ],
                "special_tokens": {
                    "<|startoftext|>": {"id": "<|startoftext|>", "ids": [1], "tokens": ["<|startoftext|>"]}
                }
            }
        ]
    })
}

fn expected_decoder() -> Value {
    json!({
        "type": "Sequence",
        "decoders": [{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}]
    })
}

fn invalid<T>(message: impl Into<String>) -> Result<T, TokenizerError> {
    Err(TokenizerError::InvalidAsset(message.into()))
}
