//! Bounded `SafeTensors` header parsing, duplicate detection, and exact payload coverage.

use std::collections::{BTreeMap, BTreeSet};

use minifield_engine_api::{ByteRange, ExecutorError, Result, Shape};
use serde::{
    Deserializer as _,
    de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};

use super::{LoaderLimits, ParsedAsset, ParsedTensor, StorageDType};

/// Parse a bounded `SafeTensors` header after the eight-byte length prefix has been read.
///
/// `header_bytes` excludes the prefix. `declared_asset_bytes` is caller-bound and must describe
/// the whole original asset, including the prefix, JSON header, whitespace padding, and payload.
#[allow(clippy::too_many_lines)] // Header validation is intentionally co-located for auditability.
pub fn parse_safetensors_header(
    header: &[u8],
    header_bytes: u64,
    declared_asset_bytes: u64,
    limits: LoaderLimits,
) -> Result<ParsedAsset> {
    limits.validate()?;
    if header_bytes
        != u64::try_from(header.len())
            .map_err(|_| ExecutorError::Overflow("header length exceeds u64"))?
    {
        return Err(ExecutorError::InvalidArgument(
            "header completion length differs from declared header length",
        ));
    }
    if header_bytes == 0 || header_bytes > limits.max_header_bytes {
        return Err(ExecutorError::ResourceLimit(
            "header exceeds configured byte limit",
        ));
    }
    if header.first().copied() != Some(b'{') {
        return Err(ExecutorError::InvalidArgument(
            "safetensors header must begin with an object brace",
        ));
    }
    let payload_start = 8_u64
        .checked_add(header_bytes)
        .ok_or(ExecutorError::Overflow("payload start overflows u64"))?;
    if declared_asset_bytes > limits.max_asset_bytes {
        return Err(ExecutorError::ResourceLimit(
            "asset exceeds configured byte limit",
        ));
    }
    if payload_start > declared_asset_bytes {
        return Err(ExecutorError::OutOfBounds(
            "header extends beyond declared asset bytes",
        ));
    }

    let document = parse_header_document(header)?;
    if document.tensors.len() > limits.max_tensors {
        return Err(ExecutorError::ResourceLimit(
            "tensor count exceeds configured loader limit",
        ));
    }
    let payload_bytes = declared_asset_bytes - payload_start;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(document.tensors.len())
        .map_err(|_| ExecutorError::ResourceLimit("tensor inventory allocation failed"))?;
    for (name, entry) in document.tensors {
        if name.is_empty() || name.len() > limits.max_tensor_name_bytes {
            return Err(ExecutorError::InvalidArgument(
                "tensor name exceeds configured loader limit",
            ));
        }
        if entry.shape.len() > limits.max_rank {
            return Err(ExecutorError::InvalidShape(
                "tensor rank exceeds configured loader limit",
            ));
        }
        let shape = Shape::new(&entry.shape)?;
        let elements = shape.element_count()?;
        let data_start = entry.offsets[0];
        let data_end = entry.offsets[1];
        if data_end < data_start {
            return Err(ExecutorError::InvalidLayout(
                "tensor data offsets are reversed",
            ));
        }
        let source_len = data_end - data_start;
        if source_len > limits.max_source_tensor_bytes {
            return Err(ExecutorError::ResourceLimit(
                "tensor exceeds configured source byte limit",
            ));
        }
        let expected_len = elements
            .checked_mul(entry.dtype.byte_width())
            .ok_or(ExecutorError::Overflow("tensor source bytes overflow u64"))?;
        if source_len != expected_len {
            return Err(ExecutorError::InvalidLayout(
                "tensor source range differs from shape and storage dtype",
            ));
        }
        if data_end > payload_bytes {
            return Err(ExecutorError::OutOfBounds(
                "tensor source range exceeds payload bytes",
            ));
        }
        let offset = payload_start
            .checked_add(data_start)
            .ok_or(ExecutorError::Overflow("tensor asset offset overflows u64"))?;
        tensors.push(ParsedTensor {
            name,
            storage_dtype: entry.dtype,
            source_shape: shape,
            bytes: ByteRange {
                offset,
                len: source_len,
            },
        });
    }
    validate_payload_coverage(&tensors, payload_start, payload_bytes)?;
    Ok(ParsedAsset {
        header_bytes,
        payload_start,
        asset_bytes: declared_asset_bytes,
        metadata: document.metadata,
        tensors,
    })
}

fn validate_payload_coverage(
    tensors: &[ParsedTensor],
    payload_start: u64,
    payload_bytes: u64,
) -> Result<()> {
    let mut ordered: Vec<&ParsedTensor> = tensors.iter().collect();
    // Empty tensors precede nonempty peers at the same start. Length orders
    // equal-start ranges by end without another offset addition.
    ordered.sort_unstable_by_key(|tensor| (tensor.bytes.offset, tensor.bytes.len));
    let mut expected_offset = payload_start;
    for tensor in ordered {
        if tensor.bytes.offset != expected_offset {
            return Err(ExecutorError::InvalidLayout(
                "tensor ranges do not cover payload without holes or overlap",
            ));
        }
        expected_offset = tensor.bytes.end()?;
    }
    let expected_end = payload_start
        .checked_add(payload_bytes)
        .ok_or(ExecutorError::Overflow("payload end overflows u64"))?;
    if expected_offset != expected_end {
        return Err(ExecutorError::InvalidLayout(
            "tensor ranges do not cover complete payload",
        ));
    }
    Ok(())
}

struct HeaderDocument {
    metadata: BTreeMap<String, String>,
    tensors: Vec<(String, HeaderTensor)>,
}

struct HeaderTensor {
    dtype: StorageDType,
    shape: Vec<u64>,
    offsets: [u64; 2],
}

fn parse_header_document(bytes: &[u8]) -> Result<HeaderDocument> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = deserializer
        .deserialize_any(HeaderDocumentVisitor)
        .map_err(|error| map_header_error(&error.to_string()))?;
    deserializer
        .end()
        .map_err(|_| ExecutorError::InvalidArgument("safetensors header has trailing data"))?;
    Ok(parsed)
}

fn map_header_error(message: &str) -> ExecutorError {
    if message.starts_with("duplicate JSON key") {
        ExecutorError::DuplicateName
    } else if message.starts_with("unsupported safetensors storage") {
        ExecutorError::Unsupported("safetensors storage dtype is unsupported by this loader")
    } else if message.starts_with("unknown safetensors storage") {
        ExecutorError::InvalidDType("unknown safetensors storage dtype")
    } else {
        ExecutorError::InvalidArgument("invalid safetensors header JSON")
    }
}

struct HeaderDocumentVisitor;

impl<'de> Visitor<'de> for HeaderDocumentVisitor {
    type Value = HeaderDocument;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a `SafeTensors` header object")
    }

    fn visit_map<A>(self, mut map: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut names = BTreeSet::new();
        let mut metadata = BTreeMap::new();
        let mut tensors = Vec::new();
        while let Some(name) = map.next_key::<String>()? {
            if !names.insert(name.clone()) {
                return Err(de::Error::custom("duplicate JSON key"));
            }
            if name == "__metadata__" {
                metadata = map.next_value_seed(MetadataSeed)?;
            } else {
                tensors.push((name, map.next_value_seed(TensorSeed)?));
            }
        }
        Ok(HeaderDocument { metadata, tensors })
    }
}

struct MetadataSeed;

impl<'de> DeserializeSeed<'de> for MetadataSeed {
    type Value = BTreeMap<String, String>;

    fn deserialize<D>(self, deserializer: D) -> core::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(MetadataVisitor)
    }
}

struct MetadataVisitor;

impl<'de> Visitor<'de> for MetadataVisitor {
    type Value = BTreeMap<String, String>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a string-to-string metadata object")
    }

    fn visit_map<A>(self, mut map: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut metadata = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if metadata.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON key"));
            }
            metadata.insert(key, map.next_value::<String>()?);
        }
        Ok(metadata)
    }
}

struct TensorSeed;

impl<'de> DeserializeSeed<'de> for TensorSeed {
    type Value = HeaderTensor;

    fn deserialize<D>(self, deserializer: D) -> core::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(TensorVisitor)
    }
}

struct TensorVisitor;

impl<'de> Visitor<'de> for TensorVisitor {
    type Value = HeaderTensor;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a `SafeTensors` tensor metadata object")
    }

    fn visit_map<A>(self, mut map: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = BTreeSet::new();
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        while let Some(key) = map.next_key::<String>()? {
            if !fields.insert(key.clone()) {
                return Err(de::Error::custom("duplicate JSON key"));
            }
            match key.as_str() {
                "dtype" => {
                    let text = map.next_value::<String>()?;
                    dtype = Some(StorageDType::parse(&text).map_err(|error| match error {
                        ExecutorError::Unsupported(_) => {
                            de::Error::custom("unsupported safetensors storage")
                        }
                        _ => de::Error::custom("unknown safetensors storage"),
                    })?);
                }
                "shape" => shape = Some(map.next_value_seed(U64VectorSeed)?),
                "data_offsets" => {
                    let parsed = map.next_value_seed(U64VectorSeed)?;
                    let [start, end] = <[u64; 2]>::try_from(parsed.as_slice())
                        .map_err(|_| de::Error::custom("tensor data_offsets needs two integers"))?;
                    offsets = Some([start, end]);
                }
                _ => {
                    let _: IgnoredAny = map.next_value()?;
                    return Err(de::Error::custom("unknown safetensors tensor field"));
                }
            }
        }
        Ok(HeaderTensor {
            dtype: dtype.ok_or_else(|| de::Error::custom("missing tensor dtype"))?,
            shape: shape.ok_or_else(|| de::Error::custom("missing tensor shape"))?,
            offsets: offsets.ok_or_else(|| de::Error::custom("missing tensor data_offsets"))?,
        })
    }
}

struct U64VectorSeed;

impl<'de> DeserializeSeed<'de> for U64VectorSeed {
    type Value = Vec<u64>;

    fn deserialize<D>(self, deserializer: D) -> core::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(U64VectorVisitor)
    }
}

struct U64VectorVisitor;

impl<'de> Visitor<'de> for U64VectorVisitor {
    type Value = Vec<u64>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("an array of unsigned integer dimensions or offsets")
    }

    fn visit_seq<A>(self, mut sequence: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<u64>()? {
            values.push(value);
        }
        Ok(values)
    }
}
