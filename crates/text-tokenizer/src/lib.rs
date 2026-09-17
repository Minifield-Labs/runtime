#![forbid(unsafe_code)]
//! Backend-neutral byte-level BPE tokenization for caller-supplied assets.
//!
//! The crate deliberately has no filesystem, network, model, or prompt-policy
//! dependency. Callers supply bounded tokenizer bytes and decide when to add
//! the single BOS token. The supported profile is the observed LFM byte-BPE
//! profile described by the experiment packet.

mod asset;
mod byte_level;
mod pretokenize;

use std::collections::HashMap;
use std::sync::Arc;

pub use asset::TokenizerAsset;
pub use pretokenize::{PretokenizedPiece, TextSpan};

use crate::asset::{AddedToken, AssetModel};

/// The fixed model-head width, including intentionally unmapped token IDs.
pub const MODEL_VOCAB_SIZE: u32 = 65_536;
/// The first model token ID that has no mapping in the pinned tokenizer.
pub const FIRST_UNMAPPED_MODEL_TOKEN_ID: u32 = 64_402;
/// The BOS token inserted only when the caller explicitly requests it.
pub const BOS_TOKEN_ID: u32 = 1;

/// A caller-controlled tokenizer resource budget.
///
/// `max_merge_steps` charges every adjacent-pair rank lookup. The simple,
/// deterministic merger intentionally refuses a pathological piece before it
/// can consume unbounded quadratic work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerLimits {
    pub max_asset_bytes: usize,
    pub max_input_bytes: usize,
    pub max_output_ids: usize,
    pub max_merge_steps: usize,
    pub max_piece_bytes: usize,
}

impl Default for TokenizerLimits {
    fn default() -> Self {
        Self {
            max_asset_bytes: 8 * 1024 * 1024,
            max_input_bytes: 4 * 1024 * 1024,
            max_output_ids: 4 * 1024 * 1024,
            max_merge_steps: 16 * 1024 * 1024,
            max_piece_bytes: 64 * 1024,
        }
    }
}

/// The pinned profile name accepted by this crate.
#[must_use]
pub const fn supported_profile() -> &'static str {
    "byte-level-bpe/split-regex-bytelevel/template-bos"
}

/// Whether encoding prepends the pinned BOS token.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
}

/// A checked tokenizer error. No error case silently drops text or model IDs.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer asset is {actual} bytes, above caller limit {limit}")]
    AssetLimit { actual: usize, limit: usize },
    #[error("tokenizer input is {actual} bytes, above caller limit {limit}")]
    InputLimit { actual: usize, limit: usize },
    #[error("tokenizer output exceeds caller limit {limit}")]
    OutputLimit { limit: usize },
    #[error("pretokenized piece is {actual} bytes, above caller limit {limit}")]
    PieceLimit { actual: usize, limit: usize },
    #[error("BPE merge work exceeds caller limit {limit}")]
    MergeWorkLimit { limit: usize },
    #[error("malformed tokenizer asset: {0}")]
    InvalidAsset(String),
    #[error("tokenizer asset JSON is invalid: {0}")]
    Json(String),
    #[error("model token ID {0} has no tokenizer mapping")]
    UnmappedToken(u32),
    #[error("token {0} has no byte-level decoder mapping")]
    UndecodableToken(u32),
    #[error("decoded bytes are not valid UTF-8")]
    InvalidUtf8,
}

/// A byte-level BPE tokenizer loaded from a caller-provided JSON asset.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    limits: TokenizerLimits,
    model: Arc<AssetModel>,
    added_trie: Arc<AddedTrie>,
    token_bytes: Arc<Vec<Option<TokenBytes>>>,
    special_tokens: Arc<Vec<bool>>,
}

#[derive(Clone, Debug)]
enum TokenBytes {
    ByteLevel(Vec<u8>),
    Literal(Vec<u8>),
}

impl TokenBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::ByteLevel(bytes) | Self::Literal(bytes) => bytes,
        }
    }
}

/// A decoder retaining incomplete UTF-8 bytes between token chunks.
#[derive(Clone, Debug)]
pub struct StreamingDecoder {
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
    pending: Vec<u8>,
}

impl Tokenizer {
    /// Parses and validates an asset before any text is encoded.
    ///
    /// # Errors
    ///
    /// Returns an error when the asset exceeds its limit or is malformed.
    pub fn from_json_bytes(bytes: &[u8], limits: TokenizerLimits) -> Result<Self, TokenizerError> {
        if bytes.len() > limits.max_asset_bytes {
            return Err(TokenizerError::AssetLimit {
                actual: bytes.len(),
                limit: limits.max_asset_bytes,
            });
        }
        let asset = TokenizerAsset::from_json_bytes(bytes)?;
        Self::from_asset(asset, limits)
    }

    /// Builds a tokenizer from an already parsed and validated asset.
    ///
    /// # Errors
    ///
    /// Returns an error when its byte mapping conflicts with added-token definitions.
    pub fn from_asset(
        asset: TokenizerAsset,
        limits: TokenizerLimits,
    ) -> Result<Self, TokenizerError> {
        let (model, added_tokens) = asset.into_parts();
        let mut token_bytes = vec![None; FIRST_UNMAPPED_MODEL_TOKEN_ID as usize];
        let mut special_tokens = vec![false; FIRST_UNMAPPED_MODEL_TOKEN_ID as usize];

        for (symbol, id) in &model.vocab {
            let index = usize::try_from(*id).map_err(|_| {
                TokenizerError::InvalidAsset("vocabulary ID does not fit usize".into())
            })?;
            let bytes = byte_level::decode_symbol(symbol).ok_or_else(|| {
                TokenizerError::InvalidAsset(format!(
                    "vocabulary symbol for ID {id} is not byte-level"
                ))
            })?;
            token_bytes[index] = Some(TokenBytes::ByteLevel(bytes));
        }

        let mut trie = AddedTrie::default();
        for added in added_tokens {
            let index = usize::try_from(added.id).map_err(|_| {
                TokenizerError::InvalidAsset("added token ID does not fit usize".into())
            })?;
            if let Some(existing) = token_bytes[index].as_ref() {
                let existing = std::str::from_utf8(existing.bytes()).map_err(|_| {
                    TokenizerError::InvalidAsset(format!(
                        "added token {} conflicts with byte token",
                        added.id
                    ))
                })?;
                if existing != added.content {
                    return Err(TokenizerError::InvalidAsset(format!(
                        "added token {} conflicts with vocabulary content",
                        added.id
                    )));
                }
            } else {
                token_bytes[index] = Some(TokenBytes::Literal(added.content.as_bytes().to_vec()));
            }
            special_tokens[index] = added.special;
            trie.insert(&added)?;
        }

        Ok(Self {
            limits,
            model: Arc::new(model),
            added_trie: Arc::new(trie),
            token_bytes: Arc::new(token_bytes),
            special_tokens: Arc::new(special_tokens),
        })
    }

    /// Returns exact pretokenized byte-level pieces and character offsets.
    ///
    /// # Errors
    ///
    /// Returns an error when the input exceeds the caller limit.
    pub fn pretokenize(&self, input: &str) -> Result<Vec<PretokenizedPiece>, TokenizerError> {
        self.check_input(input)?;
        pretokenize::pretokenize(input)
    }

    /// Encodes text without implicit EOS or prompt construction.
    ///
    /// # Errors
    ///
    /// Returns an error for caller limits or an invalid BPE asset relation.
    pub fn encode(&self, input: &str, options: EncodeOptions) -> Result<Vec<u32>, TokenizerError> {
        self.check_input(input)?;
        let mut output = Vec::new();
        let mut remaining_merge_work = self.limits.max_merge_steps;
        if options.add_special_tokens {
            self.push_id(&mut output, BOS_TOKEN_ID)?;
        }

        let mut ordinary_start = 0;
        let mut cursor = 0;
        while cursor < input.len() {
            if let Some((end, id)) = self.added_trie.longest_match(input.as_bytes(), cursor) {
                self.encode_ordinary(
                    &input[ordinary_start..cursor],
                    &mut output,
                    &mut remaining_merge_work,
                )?;
                self.push_id(&mut output, id)?;
                cursor = end;
                ordinary_start = end;
                continue;
            }
            let character = input[cursor..]
                .chars()
                .next()
                .ok_or_else(|| TokenizerError::InvalidAsset("invalid UTF-8 input cursor".into()))?;
            cursor += character.len_utf8();
        }
        self.encode_ordinary(
            &input[ordinary_start..],
            &mut output,
            &mut remaining_merge_work,
        )?;
        Ok(output)
    }
    /// Returns an exact byte mapping for a mapped token ID.
    ///
    /// # Errors
    ///
    /// Returns an error for an intentionally unmapped model token ID.
    pub fn token_bytes(&self, id: u32) -> Result<&[u8], TokenizerError> {
        let index = usize::try_from(id).map_err(|_| TokenizerError::UnmappedToken(id))?;
        self.token_bytes
            .get(index)
            .and_then(Option::as_ref)
            .map(TokenBytes::bytes)
            .ok_or(TokenizerError::UnmappedToken(id))
    }

    /// Decodes IDs as UTF-8, rejecting unmapped IDs rather than omitting them.
    ///
    /// # Errors
    ///
    /// Returns an error for an unmapped ID or invalid final UTF-8 byte sequence.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let mut decoder = self.streaming_decoder(skip_special_tokens);
        let mut output = decoder.push(ids)?;
        output.push_str(&decoder.finish()?);
        Ok(output)
    }

    /// Starts a UTF-8-safe decoder for incrementally generated IDs.
    #[must_use]
    pub fn streaming_decoder(&self, skip_special_tokens: bool) -> StreamingDecoder {
        StreamingDecoder {
            tokenizer: self.clone(),
            skip_special_tokens,
            pending: Vec::new(),
        }
    }

    fn check_input(&self, input: &str) -> Result<(), TokenizerError> {
        if input.len() > self.limits.max_input_bytes {
            return Err(TokenizerError::InputLimit {
                actual: input.len(),
                limit: self.limits.max_input_bytes,
            });
        }
        Ok(())
    }

    fn encode_ordinary(
        &self,
        input: &str,
        output: &mut Vec<u32>,
        remaining_merge_work: &mut usize,
    ) -> Result<(), TokenizerError> {
        for piece in pretokenize::pretokenize(input)? {
            if piece.source.len() > self.limits.max_piece_bytes {
                return Err(TokenizerError::PieceLimit {
                    actual: piece.source.len(),
                    limit: self.limits.max_piece_bytes,
                });
            }
            let ids = self.bpe_encode(&piece.source, remaining_merge_work)?;
            for id in ids {
                self.push_id(output, id)?;
            }
        }
        Ok(())
    }

    fn bpe_encode(
        &self,
        input: &str,
        remaining_work: &mut usize,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mapped = byte_level::encode_bytes(input.as_bytes());
        let mut symbols: Vec<String> = mapped
            .chars()
            .map(|character| character.to_string())
            .collect();

        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for pair_index in 0..symbols.len() - 1 {
                *remaining_work =
                    remaining_work
                        .checked_sub(1)
                        .ok_or(TokenizerError::MergeWorkLimit {
                            limit: self.limits.max_merge_steps,
                        })?;
                let pair = (symbols[pair_index].clone(), symbols[pair_index + 1].clone());
                if let Some(&rank) = self.model.merge_ranks.get(&pair)
                    && best.is_none_or(|(_, best_rank)| rank < best_rank)
                {
                    best = Some((pair_index, rank));
                }
            }
            let Some((_, rank)) = best else {
                break;
            };

            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && self
                        .model
                        .merge_ranks
                        .get(&(symbols[index].clone(), symbols[index + 1].clone()))
                        .is_some_and(|candidate| *candidate == rank)
                {
                    merged.push(format!("{}{}", symbols[index], symbols[index + 1]));
                    index += 2;
                } else {
                    merged.push(symbols[index].clone());
                    index += 1;
                }
            }
            symbols = merged;
        }

        symbols
            .iter()
            .map(|symbol| {
                self.model.vocab.get(symbol).copied().ok_or_else(|| {
                    TokenizerError::InvalidAsset(format!(
                        "BPE result {symbol:?} is absent from vocabulary"
                    ))
                })
            })
            .collect()
    }
    fn push_id(&self, output: &mut Vec<u32>, id: u32) -> Result<(), TokenizerError> {
        if output.len() >= self.limits.max_output_ids {
            return Err(TokenizerError::OutputLimit {
                limit: self.limits.max_output_ids,
            });
        }
        output.push(id);
        Ok(())
    }
}

impl StreamingDecoder {
    /// Appends IDs and returns only the complete UTF-8 prefix newly available.
    ///
    /// # Errors
    ///
    /// Returns an error for an unmapped token or invalid byte sequence.
    pub fn push(&mut self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut appended = Vec::new();
        for &id in ids {
            let index = usize::try_from(id).map_err(|_| TokenizerError::UnmappedToken(id))?;
            if self.skip_special_tokens
                && self
                    .tokenizer
                    .special_tokens
                    .get(index)
                    .copied()
                    .unwrap_or(false)
            {
                continue;
            }
            appended.extend_from_slice(self.tokenizer.token_bytes(id)?);
        }
        let pending_len = self.pending.len();
        self.pending.extend_from_slice(&appended);
        match self.take_complete_utf8() {
            Ok(output) => Ok(output),
            Err(error) => {
                self.pending.truncate(pending_len);
                Err(error)
            }
        }
    }
    /// Completes decoding. Incomplete or invalid bytes are an explicit error.
    ///
    /// # Errors
    ///
    /// Returns an error when buffered bytes cannot complete a UTF-8 scalar.
    pub fn finish(&mut self) -> Result<String, TokenizerError> {
        let output = self.take_complete_utf8()?;
        if self.pending.is_empty() {
            Ok(output)
        } else {
            Err(TokenizerError::InvalidUtf8)
        }
    }

    fn take_complete_utf8(&mut self) -> Result<String, TokenizerError> {
        match std::str::from_utf8(&self.pending) {
            Ok(value) => {
                let output = value.to_owned();
                self.pending.clear();
                Ok(output)
            }
            Err(error) if error.error_len().is_none() => {
                let complete = error.valid_up_to();
                let output = std::str::from_utf8(&self.pending[..complete])
                    .map_err(|_| TokenizerError::InvalidUtf8)?
                    .to_owned();
                self.pending.drain(..complete);
                Ok(output)
            }
            Err(_) => Err(TokenizerError::InvalidUtf8),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AddedTrie {
    nodes: Vec<AddedTrieNode>,
}

#[derive(Clone, Debug, Default)]
struct AddedTrieNode {
    next: HashMap<u8, usize>,
    token_id: Option<u32>,
}

impl AddedTrie {
    fn insert(&mut self, token: &AddedToken) -> Result<(), TokenizerError> {
        if self.nodes.is_empty() {
            self.nodes.push(AddedTrieNode::default());
        }
        let mut node = 0;
        for byte in token.content.bytes() {
            let next = if let Some(&next) = self.nodes[node].next.get(&byte) {
                next
            } else {
                let next = self.nodes.len();
                self.nodes.push(AddedTrieNode::default());
                self.nodes[node].next.insert(byte, next);
                next
            };
            node = next;
        }
        if self.nodes[node].token_id.replace(token.id).is_some() {
            return Err(TokenizerError::InvalidAsset(format!(
                "duplicate added-token content {:?}",
                token.content
            )));
        }
        Ok(())
    }

    fn longest_match(&self, input: &[u8], start: usize) -> Option<(usize, u32)> {
        let mut node = 0;
        let mut best = None;
        for (offset, &byte) in input[start..].iter().enumerate() {
            let Some(&next) = self.nodes.get(node)?.next.get(&byte) else {
                break;
            };
            node = next;
            if let Some(id) = self.nodes[node].token_id {
                best = Some((start + offset + 1, id));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EncodeOptions, Tokenizer, TokenizerError, TokenizerLimits};

    fn compact_asset() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 1, "content": "<|startoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 7, "content": "<extra>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 8, "content": "python", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
                ]
            },
            "post_processor": {
                "type": "Sequence",
                "processors": [
                    {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
                    {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}], "special_tokens": {"<|startoftext|>": {"id": "<|startoftext|>", "ids": [1], "tokens": ["<|startoftext|>"]}}}
                ]
            },
            "decoder": {"type": "Sequence", "decoders": [{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}]},
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": {"a": 0, "<|startoftext|>": 1, "b": 2, "ab": 3, "x": 4, "Ã": 5, "©": 6},
                "merges": [["a", "b"]]
            }
        }))
        .unwrap_or_else(|error| panic!("synthetic tokenizer JSON serialization failed: {error}"))
    }

    fn compact_tokenizer() -> Tokenizer {
        Tokenizer::from_json_bytes(&compact_asset(), TokenizerLimits::default())
            .unwrap_or_else(|error| panic!("synthetic tokenizer asset admission failed: {error}"))
    }

    #[test]
    fn bpe_added_tokens_and_explicit_bos_are_exact() {
        let tokenizer = compact_tokenizer();
        assert_eq!(
            tokenizer.encode("ab", EncodeOptions::default()),
            Ok(vec![3])
        );
        assert_eq!(
            tokenizer.encode(
                "ab",
                EncodeOptions {
                    add_special_tokens: true
                }
            ),
            Ok(vec![1, 3])
        );
        assert_eq!(
            tokenizer.encode("a<extra>pythonb", EncodeOptions::default()),
            Ok(vec![0, 7, 8, 2])
        );
        assert_eq!(
            tokenizer.decode(&[0, 7, 8, 2], false),
            Ok("a<extra>pythonb".into())
        );
        assert_eq!(tokenizer.decode(&[0, 7, 8, 2], true), Ok("apythonb".into()));
    }

    #[test]
    fn decoder_buffers_incomplete_utf8_and_rejects_unmapped_ids() {
        let tokenizer = compact_tokenizer();
        let mut decoder = tokenizer.streaming_decoder(false);
        assert_eq!(decoder.push(&[5]), Ok(String::new()));
        assert_eq!(decoder.push(&[6]), Ok("é".into()));
        assert_eq!(decoder.finish(), Ok(String::new()));
        assert_eq!(
            tokenizer.decode(&[64_402], false),
            Err(TokenizerError::UnmappedToken(64_402))
        );
    }

    #[test]
    fn malformed_profile_and_bounded_merge_work_are_rejected() {
        let mut malformed: serde_json::Value = serde_json::from_slice(&compact_asset())
            .unwrap_or_else(|error| panic!("synthetic JSON parse failed: {error}"));
        malformed["normalizer"] = json!({"type": "Lowercase"});
        let malformed = serde_json::to_vec(&malformed)
            .unwrap_or_else(|error| panic!("synthetic JSON serialization failed: {error}"));
        assert!(matches!(
            Tokenizer::from_json_bytes(&malformed, TokenizerLimits::default()),
            Err(TokenizerError::InvalidAsset(_))
        ));

        let limits = TokenizerLimits {
            max_merge_steps: 0,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&compact_asset(), limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert_eq!(
            tokenizer.encode("ab", EncodeOptions::default()),
            Err(TokenizerError::MergeWorkLimit { limit: 0 })
        );
        let cumulative_limits = TokenizerLimits {
            max_merge_steps: 1,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&compact_asset(), cumulative_limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert_eq!(
            tokenizer.encode("ab<extra>ab", EncodeOptions::default()),
            Err(TokenizerError::MergeWorkLimit { limit: 1 })
        );
    }
    #[test]
    fn caps_are_enforced_before_bos_and_for_each_bounded_resource() {
        let asset = compact_asset();
        let asset_limits = TokenizerLimits {
            max_asset_bytes: 1,
            ..TokenizerLimits::default()
        };
        assert!(matches!(
            Tokenizer::from_json_bytes(&asset, asset_limits),
            Err(TokenizerError::AssetLimit { .. })
        ));

        let input_limits = TokenizerLimits {
            max_input_bytes: 1,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&asset, input_limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert!(matches!(
            tokenizer.encode("ab", EncodeOptions::default()),
            Err(TokenizerError::InputLimit { .. })
        ));

        let output_limits = TokenizerLimits {
            max_output_ids: 1,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&asset, output_limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert_eq!(
            tokenizer.encode("ax", EncodeOptions::default()),
            Err(TokenizerError::OutputLimit { limit: 1 })
        );

        let bos_limits = TokenizerLimits {
            max_output_ids: 0,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&asset, bos_limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert_eq!(
            tokenizer.encode(
                "",
                EncodeOptions {
                    add_special_tokens: true
                }
            ),
            Err(TokenizerError::OutputLimit { limit: 0 })
        );

        let piece_limits = TokenizerLimits {
            max_piece_bytes: 1,
            ..TokenizerLimits::default()
        };
        let tokenizer = Tokenizer::from_json_bytes(&asset, piece_limits)
            .unwrap_or_else(|error| panic!("synthetic asset admission failed: {error}"));
        assert_eq!(
            tokenizer.encode("ab", EncodeOptions::default()),
            Err(TokenizerError::PieceLimit {
                actual: 2,
                limit: 1
            })
        );
    }

    #[test]
    fn decoder_errors_leave_incomplete_prefix_available_for_retry() {
        let tokenizer = compact_tokenizer();
        let mut unknown_id = tokenizer.streaming_decoder(false);
        assert_eq!(unknown_id.push(&[5]), Ok(String::new()));
        assert_eq!(
            unknown_id.push(&[64_402]),
            Err(TokenizerError::UnmappedToken(64_402))
        );
        assert_eq!(unknown_id.push(&[6]), Ok("é".into()));

        let mut invalid_bytes = tokenizer.streaming_decoder(false);
        assert_eq!(invalid_bytes.push(&[5]), Ok(String::new()));
        assert_eq!(invalid_bytes.push(&[0]), Err(TokenizerError::InvalidUtf8));
        assert_eq!(invalid_bytes.push(&[6]), Ok("é".into()));
    }

    #[test]
    fn duplicate_json_keys_and_inconsistent_bos_template_are_rejected() {
        let asset = String::from_utf8(compact_asset())
            .unwrap_or_else(|error| panic!("synthetic asset is not UTF-8: {error}"));
        let duplicate_vocab = asset.replacen("\"a\":0", "\"a\":0,\"a\":0", 1);
        assert!(matches!(
            Tokenizer::from_json_bytes(duplicate_vocab.as_bytes(), TokenizerLimits::default()),
            Err(TokenizerError::Json(message)) if message.contains("duplicate object key")
        ));
        let duplicate_decoder = asset.replacen(
            "\"decoder\":{\"type\":\"Sequence\"",
            "\"decoder\":{\"type\":\"unsupported\",\"type\":\"Sequence\"",
            1,
        );
        assert!(matches!(
            Tokenizer::from_json_bytes(duplicate_decoder.as_bytes(), TokenizerLimits::default()),
            Err(TokenizerError::Json(message)) if message.contains("duplicate object key")
        ));

        let mut wrong_bos: serde_json::Value = serde_json::from_str(&asset)
            .unwrap_or_else(|error| panic!("synthetic JSON parse failed: {error}"));
        wrong_bos["model"]["vocab"]["<|startoftext|>"] = json!(0);
        wrong_bos["model"]["vocab"]["a"] = json!(1);
        let wrong_bos = serde_json::to_vec(&wrong_bos)
            .unwrap_or_else(|error| panic!("synthetic JSON serialization failed: {error}"));
        assert!(matches!(
            Tokenizer::from_json_bytes(&wrong_bos, TokenizerLimits::default()),
            Err(TokenizerError::InvalidAsset(_))
        ));

        let mut missing_added_bos: serde_json::Value = serde_json::from_str(&asset)
            .unwrap_or_else(|error| panic!("synthetic JSON parse failed: {error}"));
        missing_added_bos["added_tokens"]
            .as_array_mut()
            .unwrap_or_else(|| panic!("synthetic added_tokens is not an array"))
            .retain(|token| token["id"] != json!(1));
        let missing_added_bos = serde_json::to_vec(&missing_added_bos)
            .unwrap_or_else(|error| panic!("synthetic JSON serialization failed: {error}"));
        assert!(matches!(
            Tokenizer::from_json_bytes(&missing_added_bos, TokenizerLimits::default()),
            Err(TokenizerError::InvalidAsset(_))
        ));
    }
}
