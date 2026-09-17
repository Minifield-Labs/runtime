#![allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::only_used_in_recursion
)]
//! Draft-5 serialization primitives for argument decoding.
//!
//! This crate owns no model or tool execution. It establishes byte-exact safe
//! JSON, explicit framing, and independently tokenized segment boundaries that
//! later schema planning and trace replay can consume.

mod ecmascript;
mod effective;
mod numeric;
mod raw_json;
mod schema;
mod sha256;
mod teacher;
mod trace;
mod validation;
mod walker;

use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::str;
use thiserror::Error;

/// Pinned draft artifact schema identifier.
pub const DRAFT5_SCHEMA: &str = "minifield.decoding-protocol/draft-5";
/// SHA-256 of the vendored draft artifact.
pub const DRAFT5_ARTIFACT_SHA256: &str =
    "e78b6ecfa07a71cf21203ae697aec7b92a19768bbc02e320be58ad1452150fde";

/// A tokenizer vocabulary index.
pub type TokenId = u32;
/// Result for protocol-only operations.
pub type Result<T> = std::result::Result<T, ProtocolError>;

pub use schema::{
    PropertyOrder, SchemaLimits, SchemaPlan, normalize_schema, normalize_schema_document,
};
pub use sha256::sha256_hex;

pub use ecmascript::EcmaPattern;

pub use numeric::{CanonicalDecimal, semantic_equal};

pub use raw_json::{
    AdmittedNumber, NumericKind, RawJson, RawJsonLimits, RawNumber, admit_number,
    parse_json_document, parse_runtime_value,
};

/// Rejectable protocol boundary failures. They describe no action and contain
/// no generated-model content.
#[derive(Debug, Error)]
pub enum ProtocolError {
    /// RFC8785 serialization rejected the supplied typed JSON value.
    #[error("safe JSON canonicalization failed: {0}")]
    CanonicalJson(#[from] serde_json::Error),
    /// A strict raw JSON parser rejected UTF-8, syntax, duplicate decoded keys, or trailing bytes.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),
    /// A value violates the fixed draft-5 binary64 admission rules.
    #[error("invalid number: {0}")]
    Numeric(String),
    /// A schema cannot be normalized under the supported draft-2020-12 subset.
    #[error("schema: {0}")]
    Schema(String),
    /// Parsing or traversal exceeded a declared finite input limit.
    #[error("input limit: {0}")]
    InputLimit(String),
    /// The pinned artifact cannot be parsed or is not this draft version.
    #[error("invalid draft-5 protocol artifact: {0}")]
    Artifact(String),
    /// A normal vocabulary piece is not valid inverse-ByteLevel text.
    #[error(
        "normal vocabulary token {token_id} contains unsupported ByteLevel character {character:?}"
    )]
    InvalidByteLevel { token_id: TokenId, character: char },
    /// Normal and added-token maps cannot claim the same token ID.
    #[error("token ID {0} is declared more than once")]
    DuplicateTokenId(TokenId),
    /// A token ID has no known byte mapping.
    #[error("token ID {0} has no byte mapping")]
    UnknownToken(TokenId),
    /// An encoder returned a protocol marker in a payload segment.
    #[error("payload segment contains forbidden protocol token ID {0}")]
    ForbiddenPayloadToken(TokenId),
    /// Only declared BOS/message framing IDs can bypass the payload rule.
    #[error("token ID {0} is not an explicit draft-5 framing token")]
    InvalidFramingToken(TokenId),
    /// A caller-provided encoder did not satisfy its no-special-token contract.
    #[error("segment tokenizer failed: {0}")]
    Tokenizer(String),
    /// Safe JSON must always be UTF-8; this guards a future serializer change.
    #[error("safe JSON serializer produced invalid UTF-8")]
    InvalidCanonicalUtf8,
    /// Token offset arithmetic exceeded the host address space.
    #[error("token sequence length overflow")]
    TokenLengthOverflow,
}

/// Return the immutable bytes of the vendor-copied protocol artifact.
#[must_use]
pub fn draft5_artifact() -> &'static [u8] {
    include_bytes!("../fixtures/decoding-protocol.draft-5.json")
}

/// Serialize a typed semantic JSON value with RFC8785, then make literal angle
/// brackets safe for the tokenizer. Existing escapes are semantic input: a
/// literal backslash-u sequence remains data, while a decoded bracket is escaped.
pub fn safe_json(value: &Value) -> Result<Vec<u8>> {
    let canonical = serde_jcs::to_vec(value)?;
    let mut output = Vec::with_capacity(canonical.len());
    for byte in canonical {
        match byte {
            b'<' => output.extend_from_slice(br"\u003c"),
            b'>' => output.extend_from_slice(br"\u003e"),
            _ => output.push(byte),
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramingIds {
    pub bos: TokenId,
    pub message_start: TokenId,
    pub message_end: TokenId,
    pub value_end: TokenId,
}

/// Draft-5 marker and payload-rejection inventory loaded from the vendored
/// artifact. Framing is intentionally separate from tokenizer-produced payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenPolicy {
    framing: BTreeSet<TokenId>,
    forbidden_payload: BTreeSet<TokenId>,
    framing_ids: FramingIds,
}

impl TokenPolicy {
    /// Parse the immutable draft artifact and return its marker policy.
    pub fn draft5() -> Result<Self> {
        let artifact: RawProtocol = serde_json::from_slice(draft5_artifact())?;
        if artifact.schema != DRAFT5_SCHEMA {
            return Err(ProtocolError::Artifact(format!(
                "expected {DRAFT5_SCHEMA}, got {}",
                artifact.schema
            )));
        }
        let framing = [
            artifact.tokenization.bos_id,
            artifact.tokenization.message_start_id,
            artifact.tokenization.message_end_id,
            artifact.tokenization.value_end_id,
        ]
        .into_iter()
        .collect();
        Ok(Self {
            framing,
            framing_ids: FramingIds {
                bos: artifact.tokenization.bos_id,
                message_start: artifact.tokenization.message_start_id,
                message_end: artifact.tokenization.message_end_id,
                value_end: artifact.tokenization.value_end_id,
            },
            forbidden_payload: artifact
                .tokenization
                .forbidden_payload_token_ids
                .into_iter()
                .collect(),
        })
    }

    /// Return the explicit draft framing IDs.
    #[must_use]
    pub const fn framing_ids(&self) -> FramingIds {
        self.framing_ids
    }

    /// Verify an explicitly supplied framing token.
    pub fn validate_framing(&self, token_id: TokenId) -> Result<()> {
        if self.framing.contains(&token_id) {
            Ok(())
        } else {
            Err(ProtocolError::InvalidFramingToken(token_id))
        }
    }

    /// Reject marker tokens produced while encoding a payload segment.
    pub fn validate_payload(&self, token_ids: &[TokenId]) -> Result<()> {
        for token_id in token_ids {
            if self.forbidden_payload.contains(token_id) {
                return Err(ProtocolError::ForbiddenPayloadToken(*token_id));
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct RawProtocol {
    schema: String,
    tokenization: RawTokenization,
}

#[derive(Deserialize)]
struct RawTokenization {
    bos_id: TokenId,
    message_start_id: TokenId,
    message_end_id: TokenId,
    value_end_id: TokenId,
    forbidden_payload_token_ids: Vec<TokenId>,
}

/// Byte-exact inverse map for tokenizer vocabulary pieces. Normal vocabulary
/// pieces use GPT-2 ByteLevel's reversible unicode alphabet. Added tokens are
/// literal UTF-8 by the draft-5 contract and are never passed through ByteLevel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenByteMap {
    bytes_by_id: BTreeMap<TokenId, Vec<u8>>,
}

impl TokenByteMap {
    /// Build a map from normal ByteLevel pieces and explicitly added literals.
    pub fn new(
        normal: impl IntoIterator<Item = (TokenId, String)>,
        added_literals: impl IntoIterator<Item = (TokenId, String)>,
    ) -> Result<Self> {
        let mut bytes_by_id = BTreeMap::new();
        for (token_id, piece) in normal {
            let bytes = decode_byte_level_piece(token_id, &piece)?;
            if bytes_by_id.insert(token_id, bytes).is_some() {
                return Err(ProtocolError::DuplicateTokenId(token_id));
            }
        }
        for (token_id, literal) in added_literals {
            if bytes_by_id.insert(token_id, literal.into_bytes()).is_some() {
                return Err(ProtocolError::DuplicateTokenId(token_id));
            }
        }
        Ok(Self { bytes_by_id })
    }

    /// Return one token's original byte contribution without per-token decoding.
    pub fn token_bytes(&self, token_id: TokenId) -> Result<&[u8]> {
        self.bytes_by_id
            .get(&token_id)
            .map(Vec::as_slice)
            .ok_or(ProtocolError::UnknownToken(token_id))
    }

    /// Concatenate original token bytes. The caller may decode the full stream
    /// once if it expects valid UTF-8; individual token pieces are not text.
    pub fn token_stream_bytes(&self, token_ids: &[TokenId]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for token_id in token_ids {
            output.extend_from_slice(self.token_bytes(*token_id)?);
        }
        Ok(output)
    }
}

fn decode_byte_level_piece(token_id: TokenId, piece: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(piece.len());
    for character in piece.chars() {
        let Some(byte) = inverse_byte_level(character) else {
            return Err(ProtocolError::InvalidByteLevel {
                token_id,
                character,
            });
        };
        bytes.push(byte);
    }
    Ok(bytes)
}

fn inverse_byte_level(character: char) -> Option<u8> {
    let codepoint = u32::from(character);
    if matches!(codepoint, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
        return u8::try_from(codepoint).ok();
    }

    let mut offset = 0u32;
    for byte in u8::MIN..=u8::MAX {
        if byte_level_visible(byte) {
            continue;
        }
        if codepoint == 256 + offset {
            return Some(byte);
        }
        offset += 1;
    }
    None
}

const fn byte_level_visible(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff)
}

/// A tokenizer boundary that must be encoded independently. Framing is supplied
/// as an ID; all textual and safe-JSON segments go through the payload guard.
#[derive(Clone, Copy, Debug)]
pub enum Segment<'a> {
    /// One explicitly allowed BOS/message/value-end token.
    Framing(TokenId),
    /// Fixed protocol text, encoded with automatic special tokens disabled.
    Text(&'a str),
    /// A complete dynamic semantic JSON value, first passed through safe JSON.
    SafeJson(&'a Value),
}

/// Tokenizer adapter used by the protocol. Implementors must encode precisely
/// one supplied segment and must never add BOS, chat, or other special tokens.
pub trait SegmentTokenizer {
    /// Encoder-specific failure.
    type Error: Display;

    /// Encode a single segment with automatic special tokens disabled.
    fn encode_without_special_tokens(
        &self,
        segment: &str,
    ) -> std::result::Result<Vec<TokenId>, Self::Error>;
}

/// The origin of an independently encoded append.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentKind {
    /// Explicit framing ID, with no text encoding.
    Framing,
    /// Fixed protocol text.
    Text,
    /// Safe canonical JSON.
    SafeJson,
}

/// One append retained as an independent trace-ready segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSegment {
    /// Segment source category.
    pub kind: SegmentKind,
    /// UTF-8 bytes passed to the tokenizer, absent for explicit framing IDs.
    pub source: Option<Vec<u8>>,
    /// Framing ID when this is a framing segment.
    pub special_id: Option<TokenId>,
    /// IDs emitted for this one segment only.
    pub token_ids: Vec<TokenId>,
    /// Inclusive start in the assembled main token vector.
    pub start: usize,
    /// Exclusive end in the assembled main token vector.
    pub end: usize,
}

/// Concatenated IDs plus the immutable independently tokenized segment record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentedTokenStream {
    /// Final concatenation in append order.
    pub token_ids: Vec<TokenId>,
    /// Independent source and offsets for each append.
    pub segments: Vec<EncodedSegment>,
}

/// Tokenize each supplied segment separately and concatenate only after payload
/// marker validation. This function does not merge adjacent dynamic JSON values.
pub fn tokenize_segments<'a, T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    segments: impl IntoIterator<Item = Segment<'a>>,
) -> Result<SegmentedTokenStream>
where
    T: SegmentTokenizer,
{
    let mut token_ids = Vec::new();
    let mut encoded = Vec::new();
    for segment in segments {
        let start = token_ids.len();
        let (kind, source, special_id, segment_ids) = match segment {
            Segment::Framing(token_id) => {
                policy.validate_framing(token_id)?;
                (SegmentKind::Framing, None, Some(token_id), vec![token_id])
            }
            Segment::Text(text) => {
                let source = text.as_bytes().to_vec();
                let segment_ids = encode_payload(tokenizer, policy, &source)?;
                (SegmentKind::Text, Some(source), None, segment_ids)
            }
            Segment::SafeJson(value) => {
                let source = safe_json(value)?;
                let segment_ids = encode_payload(tokenizer, policy, &source)?;
                (SegmentKind::SafeJson, Some(source), None, segment_ids)
            }
        };
        let end = start
            .checked_add(segment_ids.len())
            .ok_or(ProtocolError::TokenLengthOverflow)?;
        token_ids.extend_from_slice(&segment_ids);
        encoded.push(EncodedSegment {
            kind,
            source,
            special_id,
            token_ids: segment_ids,
            start,
            end,
        });
    }
    Ok(SegmentedTokenStream {
        token_ids,
        segments: encoded,
    })
}

fn encode_payload<T>(tokenizer: &T, policy: &TokenPolicy, bytes: &[u8]) -> Result<Vec<TokenId>>
where
    T: SegmentTokenizer,
{
    let text = str::from_utf8(bytes).map_err(|_| ProtocolError::InvalidCanonicalUtf8)?;
    let token_ids = tokenizer
        .encode_without_special_tokens(text)
        .map_err(|error| ProtocolError::Tokenizer(error.to_string()))?;
    policy.validate_payload(&token_ids)?;
    Ok(token_ids)
}

pub use validation::{ValidationFailure, ValidationFailureKind, ValidationLimits};

pub use effective::{
    ChoiceKind, EffectiveAlternative, EffectiveLimits, EffectiveProvenance, EffectiveSelection,
    EffectiveTree, SchemaPathOrigin,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PublicEvent {
    System {
        policy: String,
        observation: RawJson,
    },
    User {
        content: String,
    },
    ToolCall {
        tool_call_id: String,
        tool: String,
        arguments: RawJson,
    },
    ToolResult {
        tool_call_id: String,
        result: RawJson,
    },
    AssistantText {
        content: String,
    },
}
impl PublicEvent {
    #[must_use]
    pub fn role(&self) -> &'static str {
        match self {
            Self::System { .. } => "system",
            Self::User { .. } => "user",
            Self::ToolCall { .. } | Self::AssistantText { .. } => "assistant",
            Self::ToolResult { .. } => "tool",
        }
    }
    #[must_use]
    pub fn as_raw(&self) -> RawJson {
        match self {
            Self::System {
                policy,
                observation,
            } => RawJson::Object(vec![
                ("type".to_owned(), RawJson::String("system".to_owned())),
                ("policy".to_owned(), RawJson::String(policy.clone())),
                ("observation".to_owned(), observation.clone()),
            ]),
            Self::User { content } => RawJson::Object(vec![
                ("type".to_owned(), RawJson::String("user".to_owned())),
                ("content".to_owned(), RawJson::String(content.clone())),
            ]),
            Self::ToolCall {
                tool_call_id,
                tool,
                arguments,
            } => RawJson::Object(vec![
                ("type".to_owned(), RawJson::String("tool_call".to_owned())),
                (
                    "tool_call_id".to_owned(),
                    RawJson::String(tool_call_id.clone()),
                ),
                ("tool".to_owned(), RawJson::String(tool.clone())),
                ("arguments".to_owned(), arguments.clone()),
            ]),
            Self::ToolResult {
                tool_call_id,
                result,
            } => RawJson::Object(vec![
                ("type".to_owned(), RawJson::String("tool_result".to_owned())),
                (
                    "tool_call_id".to_owned(),
                    RawJson::String(tool_call_id.clone()),
                ),
                ("result".to_owned(), result.clone()),
            ]),
            Self::AssistantText { content } => RawJson::Object(vec![
                (
                    "type".to_owned(),
                    RawJson::String("assistant_text".to_owned()),
                ),
                ("content".to_owned(), RawJson::String(content.clone())),
            ]),
        }
    }
}

pub use trace::{MainAppend, TeacherTraceInput, TraceOwnership, TracePrefix, TracePrefixBuilder};
pub use walker::{
    BranchAppend, FiniteCandidate, FiniteChoice, ProbeCandidate, ProbeOperation, ProbeTrace,
    TeacherLexicalValue, TeacherTrace, TeacherTraceBuilder,
};

pub use teacher::{TeacherPlan, TeacherUnionChoice, discover_teacher_unions, plan_teacher};
