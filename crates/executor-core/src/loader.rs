//! Bounded, portable `SafeTensors` ingestion and completion-driven typed weight publication.
//!
//! This module owns format parsing and model inventory policy only. It has no filesystem,
//! networking, tokenizer, model-forward, or device-specific code.

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    rc::Rc,
};

use minifield_engine_api::{
    AllocationClass, AssetBytes, AssetManifest, AssetProvider, BackendIdentity, BackendLease,
    ByteRange, CompletionPoll, DType, ExecutorError, FenceRetirement, InferenceCompletion,
    InferenceOps, Result, Shape, TensorBinding, TensorRecord, TensorRequirement,
};
use serde::{
    Deserializer as _,
    de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};

/// Explicit caller bounds for one model asset loader task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoaderLimits {
    pub max_asset_bytes: u64,
    pub max_header_bytes: u64,
    pub max_source_tensor_bytes: u64,
    pub max_retained_host_bytes: u64,
    pub max_tensor_name_bytes: usize,
    pub max_tensors: usize,
    pub max_rank: usize,
}

impl LoaderLimits {
    pub fn validate(self) -> Result<()> {
        if self.max_asset_bytes < 8
            || self.max_header_bytes == 0
            || self.max_source_tensor_bytes == 0
            || self.max_retained_host_bytes == 0
            || self.max_tensor_name_bytes == 0
            || self.max_tensors == 0
            || self.max_rank > 4
        {
            return Err(ExecutorError::InvalidArgument(
                "loader limits must be nonzero and fit portable shapes",
            ));
        }
        Ok(())
    }
}

/// Source storage accepted by this F32-compute loader. BF16 is expanded before upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageDType {
    F32,
    BF16,
}

impl StorageDType {
    fn parse(text: &str) -> Result<Self> {
        match text {
            "F32" => Ok(Self::F32),
            "BF16" => Ok(Self::BF16),
            "F64" => Err(ExecutorError::Unsupported(
                "F64 safetensors storage is unsupported by the F32 loader",
            )),
            _ => Err(ExecutorError::InvalidDType(
                "unknown or unsupported safetensors storage dtype",
            )),
        }
    }

    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::BF16 => 2,
        }
    }

    #[must_use]
    pub const fn as_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::BF16 => DType::BF16,
        }
    }
}

/// Explicit conversion from source tensor layout to the first model operator layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightLayout {
    /// Stored row-major dimensions are already the backend operator dimensions.
    Identity,
    /// Stored convolution `[hidden, 1, width]` becomes packed operator `[hidden, width]`.
    ConvHiddenSingletonWidth,
}

impl WeightLayout {
    fn output_shape(self, source: Shape) -> Result<Shape> {
        match self {
            Self::Identity => Ok(source),
            Self::ConvHiddenSingletonWidth => {
                if source.rank() != 3 || source.dim(1)? != 1 {
                    return Err(ExecutorError::InvalidShape(
                        "convolution source weight must be [hidden, 1, width]",
                    ));
                }
                Shape::new(&[source.dim(0)?, source.dim(2)?])
            }
        }
    }
}

/// One logical model role bound to a physical stored tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightRequirement {
    pub role: String,
    pub tensor_name: String,
    pub storage_dtype: StorageDType,
    pub source_shape: Shape,
    pub layout: WeightLayout,
    pub tied_to_role: Option<String>,
}

/// Exact model inventory supplied by a checked model/config module before loading begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlan {
    pub config_name: String,
    pub requirements: Vec<WeightRequirement>,
}

impl WeightPlan {
    pub(crate) fn manifest_requirements(&self) -> Vec<TensorRequirement> {
        self.requirements
            .iter()
            .map(|requirement| TensorRequirement {
                role: requirement.role.clone(),
                tensor_name: requirement.tensor_name.clone(),
                dtype: requirement.storage_dtype.as_dtype(),
                shape: requirement.source_shape,
                tied_to_role: requirement.tied_to_role.clone(),
            })
            .collect()
    }

    pub(crate) fn validate(&self, limits: LoaderLimits) -> Result<()> {
        if self.config_name.is_empty() || self.requirements.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "weight plan requires a config name and at least one role",
            ));
        }
        if self.requirements.len() > limits.max_tensors {
            return Err(ExecutorError::ResourceLimit(
                "weight plan exceeds configured tensor count",
            ));
        }
        for requirement in &self.requirements {
            if requirement.role.is_empty()
                || requirement.tensor_name.is_empty()
                || requirement.role.len() > limits.max_tensor_name_bytes
                || requirement.tensor_name.len() > limits.max_tensor_name_bytes
                || requirement.source_shape.rank() as usize > limits.max_rank
            {
                return Err(ExecutorError::InvalidArgument(
                    "weight role name or rank exceeds configured loader limit",
                ));
            }
            let _ = requirement.layout.output_shape(requirement.source_shape)?;
        }
        Ok(())
    }
}

/// Generic format-loader request.
///
/// Production LFM2 bundles must use `Lfm2LoadRequest`, which derives and binds this plan from
/// checked config bytes. This lower-level request remains for parser and format-policy diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadRequest {
    pub config_name: String,
    pub config_bytes: Vec<u8>,
    pub expected_config_sha256: [u8; 32],
    pub declared_asset_bytes: u64,
    pub expected_asset_sha256: [u8; 32],
    pub plan: WeightPlan,
    pub limits: LoaderLimits,
}

/// Lifecycle phase reported with loader failures. A failure never publishes partial weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoaderStage {
    Begin,
    Backend,
    HeaderPrefix,
    Header,
    Inventory,
    TensorRead,
    TensorDecode,
    Upload,
    Identity,
    FinalFence,
    Cancelled,
}

/// Typed loader error with a bounded stage and a shared portable cause category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoaderError {
    pub stage: LoaderStage,
    pub cause: ExecutorError,
}

impl core::fmt::Display for LoaderError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "loader {:?}: {}", self.stage, self.cause)
    }
}

impl std::error::Error for LoaderError {}

/// Nonblocking observation of a model-weight loading task.
#[derive(Debug)]
pub enum LoaderPoll<T> {
    Pending,
    Ready(core::result::Result<T, LoaderError>),
}

/// Observable logical loader-owned bytes. These are not allocator/RSS measurements.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LoaderResourceReport {
    pub retained_config_bytes: u64,
    pub retained_header_bytes: u64,
    pub retained_source_bytes: u64,
    pub decoded_f32_bytes: u64,
    pub pending_requested_bytes: u64,
}

impl LoaderResourceReport {
    pub fn total_loader_bytes(self) -> Result<u64> {
        self.retained_config_bytes
            .checked_add(self.retained_header_bytes)
            .and_then(|value| value.checked_add(self.retained_source_bytes))
            .and_then(|value| value.checked_add(self.decoded_f32_bytes))
            .and_then(|value| value.checked_add(self.pending_requested_bytes))
            .ok_or(ExecutorError::Overflow(
                "loader resource report total overflows u64",
            ))
    }
}

/// One physical tensor parsed from a `SafeTensors` header. Ranges are absolute asset ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedTensor {
    pub name: String,
    pub storage_dtype: StorageDType,
    pub source_shape: Shape,
    pub bytes: ByteRange,
}

/// A checked `SafeTensors` container header and exact physical payload inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedAsset {
    pub header_bytes: u64,
    pub payload_start: u64,
    pub asset_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<ParsedTensor>,
}

impl ParsedAsset {
    fn sorted_tensor_indexes(&self) -> Vec<usize> {
        let mut indexes: Vec<usize> = (0..self.tensors.len()).collect();
        indexes.sort_unstable_by_key(|index| self.tensors[*index].bytes.offset);
        indexes
    }

    pub(crate) fn as_manifest(&self, config_name: &str) -> AssetManifest {
        AssetManifest {
            config_name: config_name.to_owned(),
            asset_bytes: self.asset_bytes,
            tensors: self
                .tensors
                .iter()
                .map(|tensor| TensorRecord {
                    name: tensor.name.clone(),
                    dtype: tensor.storage_dtype.as_dtype(),
                    shape: tensor.source_shape,
                    bytes: tensor.bytes,
                })
                .collect(),
        }
    }
}

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
    ordered.sort_unstable_by_key(|tensor| tensor.bytes.offset);
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

/// F32 decoded values and their explicitly converted operator shape.
fn decode_tensor(
    bytes: &[u8],
    tensor: &ParsedTensor,
    layout: WeightLayout,
) -> Result<(Shape, Vec<f32>)> {
    let expected = usize::try_from(tensor.bytes.len)
        .map_err(|_| ExecutorError::Overflow("tensor byte length exceeds usize"))?;
    if bytes.len() != expected {
        return Err(ExecutorError::InvalidArgument(
            "tensor completion length differs from requested range",
        ));
    }
    let output_shape = layout.output_shape(tensor.source_shape)?;
    let element_count = usize::try_from(tensor.source_shape.element_count()?)
        .map_err(|_| ExecutorError::Overflow("tensor element count exceeds usize"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .map_err(|_| ExecutorError::ResourceLimit("decoded f32 allocation failed"))?;
    match tensor.storage_dtype {
        StorageDType::F32 => {
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_bits(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        StorageDType::BF16 => {
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                values.push(f32::from_bits(u32::from(bits) << 16));
            }
        }
    }
    if values.len() != element_count || !values.iter().all(|value| value.is_finite()) {
        return Err(ExecutorError::BackendFailure(
            "decoded model tensor contains non-finite values",
        ));
    }
    if output_shape.element_count()?
        != u64::try_from(values.len())
            .map_err(|_| ExecutorError::Overflow("decoded value count exceeds u64"))?
    {
        return Err(ExecutorError::InvalidShape(
            "owned tensor layout conversion changes element count",
        ));
    }
    Ok((output_shape, values))
}

/// One owned physical F32 backend buffer with its source storage provenance.
#[derive(Debug)]
pub struct LoadedTensor<Buffer> {
    pub name: String,
    pub source_dtype: StorageDType,
    pub source_shape: Shape,
    pub operator_shape: Shape,
    buffer: Buffer,
}

impl<Buffer> LoadedTensor<Buffer> {
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// Fully validated and published typed physical weights. Role aliases point at one physical tensor.
#[derive(Debug)]
pub struct TypedWeights<Buffer> {
    owner: Rc<()>,
    backend: BackendIdentity,
    config_sha256: [u8; 32],
    asset_sha256: [u8; 32],
    tensors: Vec<LoadedTensor<Buffer>>,
    roles: Vec<(String, usize)>,
    owned_f32_bytes: u64,
}

impl<Buffer> TypedWeights<Buffer> {
    #[must_use]
    pub fn backend(&self) -> BackendIdentity {
        self.backend
    }
    #[must_use]
    pub const fn config_sha256(&self) -> [u8; 32] {
        self.config_sha256
    }
    #[must_use]
    pub const fn asset_sha256(&self) -> [u8; 32] {
        self.asset_sha256
    }
    #[must_use]
    pub const fn owned_f32_bytes(&self) -> u64 {
        self.owned_f32_bytes
    }
    #[must_use]
    pub fn tensors(&self) -> &[LoadedTensor<Buffer>] {
        &self.tensors
    }

    pub fn buffer_for_role(&self, role: &str) -> Result<&Buffer> {
        let index = self
            .roles
            .iter()
            .find_map(|(candidate, index)| (candidate == role).then_some(*index))
            .ok_or(ExecutorError::InvalidArgument("unknown loaded weight role"))?;
        self.tensors
            .get(index)
            .map(LoadedTensor::buffer)
            .ok_or(ExecutorError::BackendFailure(
                "loaded role points outside physical tensor inventory",
            ))
    }

    #[must_use]
    pub fn same_actual_instance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.owner, &other.owner)
    }
}

#[derive(Debug)]
struct StagedWeight<Buffer> {
    manifest_index: usize,
    tensor: LoadedTensor<Buffer>,
}

enum LoadPhase<Read, Fence> {
    NeedPrefix,
    Prefix(Read),
    NeedHeader,
    Header(Read),
    NeedTensor,
    Tensor {
        manifest_index: usize,
        read: Read,
    },
    NeedFence,
    Fence(Fence),
    /// A fence whose cancellation could not prove queued work is no longer using staged buffers.
    DrainingFence(Fence),
    Finished,
    Failed,
}

/// A portable, poll-driven loader task. It owns staged weights until its final identity and fence
/// checks complete, so an error cannot expose a partially usable model.
pub struct WeightLoadTask<Read, Fence, Buffer> {
    request: LoadRequest,
    phase: LoadPhase<Read, Fence>,
    header_bytes: Option<u64>,
    parsed: Option<ParsedAsset>,
    bindings: Vec<TensorBinding>,
    sorted_indexes: Vec<usize>,
    next_tensor: usize,
    hasher: Sha256,
    staged: Vec<StagedWeight<Buffer>>,
    report: LoaderResourceReport,
    backend: Option<BackendLease>,
    retirement: Option<Rc<dyn FenceRetirement<Fence, Buffer>>>,
    terminal_error: Option<LoaderError>,
    marker: PhantomData<Buffer>,
}

impl<Read, Fence, Buffer> WeightLoadTask<Read, Fence, Buffer>
where
    Read: InferenceCompletion<Output = AssetBytes>,
    Fence: InferenceCompletion<Output = ()>,
{
    pub fn begin(request: LoadRequest) -> core::result::Result<Self, LoaderError> {
        request.limits.validate().map_err(|cause| LoaderError {
            stage: LoaderStage::Begin,
            cause,
        })?;
        request
            .plan
            .validate(request.limits)
            .map_err(|cause| LoaderError {
                stage: LoaderStage::Begin,
                cause,
            })?;
        if request.config_name != request.plan.config_name {
            return Err(LoaderError {
                stage: LoaderStage::Begin,
                cause: ExecutorError::InvalidArgument(
                    "load request config name differs from weight plan",
                ),
            });
        }
        if request.declared_asset_bytes > request.limits.max_asset_bytes {
            return Err(LoaderError {
                stage: LoaderStage::Begin,
                cause: ExecutorError::ResourceLimit("asset exceeds configured byte limit"),
            });
        }
        let config_len = u64::try_from(request.config_bytes.len()).map_err(|_| LoaderError {
            stage: LoaderStage::Begin,
            cause: ExecutorError::Overflow("config bytes length exceeds u64"),
        })?;
        if config_len > request.limits.max_retained_host_bytes {
            return Err(LoaderError {
                stage: LoaderStage::Begin,
                cause: ExecutorError::ResourceLimit("config bytes exceed retained host byte limit"),
            });
        }
        if Sha256::digest(&request.config_bytes).as_slice() != request.expected_config_sha256 {
            return Err(LoaderError {
                stage: LoaderStage::Identity,
                cause: ExecutorError::InvalidArgument("config content digest mismatch"),
            });
        }
        Ok(Self {
            request,
            phase: LoadPhase::NeedPrefix,
            header_bytes: None,
            parsed: None,
            bindings: Vec::new(),
            sorted_indexes: Vec::new(),
            next_tensor: 0,
            hasher: Sha256::new(),
            staged: Vec::new(),
            report: LoaderResourceReport {
                retained_config_bytes: config_len,
                ..LoaderResourceReport::default()
            },
            backend: None,
            retirement: None,
            terminal_error: None,
            marker: PhantomData,
        })
    }

    #[must_use]
    pub const fn resource_report(&self) -> LoaderResourceReport {
        self.report
    }

    fn release_transient_host(&mut self) {
        self.staged.clear();
        self.request.config_bytes.clear();
        self.request.config_bytes.shrink_to_fit();
        self.report = LoaderResourceReport::default();
    }

    fn bind_backend<Backend>(&mut self, backend: &Backend) -> Result<()>
    where
        Backend: InferenceOps<Buffer = Buffer, Fence = Fence>,
    {
        let observed = backend.lease();
        match &self.backend {
            None => {
                let retirement: Rc<dyn FenceRetirement<Fence, Buffer>> = backend.fence_retirement();
                self.backend = Some(observed);
                self.retirement = Some(retirement);
                Ok(())
            }
            Some(bound) if !bound.same_actual_instance(&observed) => {
                Err(ExecutorError::WrongBackend)
            }
            Some(bound) if bound.identity() != observed.identity() => {
                Err(ExecutorError::StaleBuffer)
            }
            Some(_) => Ok(()),
        }
    }

    pub fn cancel(&mut self) -> core::result::Result<(), LoaderError> {
        let phase = core::mem::replace(&mut self.phase, LoadPhase::Failed);
        match phase {
            LoadPhase::Fence(mut fence) => match fence.cancel() {
                Ok(()) => {
                    self.release_transient_host();
                    Ok(())
                }
                Err(cause) => {
                    // The backend could still own queued work. Keep every staged buffer until the
                    // same fence becomes ready through the normal poll path.
                    self.phase = LoadPhase::DrainingFence(fence);
                    self.terminal_error = Some(LoaderError {
                        stage: LoaderStage::Cancelled,
                        cause: ExecutorError::Cancelled,
                    });
                    Err(LoaderError {
                        stage: LoaderStage::Cancelled,
                        cause,
                    })
                }
            },
            LoadPhase::Prefix(mut read)
            | LoadPhase::Header(mut read)
            | LoadPhase::Tensor { mut read, .. } => {
                let result = read.cancel();
                self.release_transient_host();
                result.map_err(|cause| LoaderError {
                    stage: LoaderStage::Cancelled,
                    cause,
                })
            }
            LoadPhase::NeedPrefix
            | LoadPhase::NeedHeader
            | LoadPhase::NeedTensor
            | LoadPhase::NeedFence => {
                self.release_transient_host();
                Ok(())
            }
            LoadPhase::DrainingFence(fence) => {
                self.phase = LoadPhase::DrainingFence(fence);
                Err(LoaderError {
                    stage: LoaderStage::Cancelled,
                    cause: ExecutorError::CompletionConsumed,
                })
            }
            LoadPhase::Finished => {
                self.phase = LoadPhase::Finished;
                Err(LoaderError {
                    stage: LoaderStage::Cancelled,
                    cause: ExecutorError::CompletionConsumed,
                })
            }
            LoadPhase::Failed => Err(LoaderError {
                stage: LoaderStage::Cancelled,
                cause: ExecutorError::CompletionConsumed,
            }),
        }
    }

    fn finish_error_after_fence(&mut self, error: LoaderError) -> LoaderPoll<TypedWeights<Buffer>> {
        let phase = core::mem::replace(&mut self.phase, LoadPhase::Failed);
        match phase {
            LoadPhase::Fence(mut fence) => match fence.cancel() {
                Ok(()) | Err(ExecutorError::CompletionConsumed) => {
                    self.release_transient_host();
                    LoaderPoll::Ready(Err(error))
                }
                Err(_) => {
                    // A late backend or generation rejection must not drop staged buffers while
                    // this fence could still reference them. Preserve the original failure until
                    // normal polling establishes that the submission is terminal.
                    self.phase = LoadPhase::DrainingFence(fence);
                    self.terminal_error = Some(error);
                    LoaderPoll::Pending
                }
            },
            LoadPhase::DrainingFence(fence) => {
                self.phase = LoadPhase::DrainingFence(fence);
                if self.terminal_error.is_none() {
                    self.terminal_error = Some(error);
                }
                LoaderPoll::Pending
            }
            _ => {
                self.release_transient_host();
                LoaderPoll::Ready(Err(error))
            }
        }
    }

    pub fn poll_step<P, Backend>(
        &mut self,
        provider: &mut P,
        backend: &mut Backend,
    ) -> LoaderPoll<TypedWeights<Buffer>>
    where
        P: AssetProvider<Read = Read>,
        Backend: InferenceOps<Buffer = Buffer, Fence = Fence>,
    {
        let result = if matches!(self.phase, LoadPhase::DrainingFence(_)) {
            self.step_inner(provider, backend)
        } else {
            match self.bind_backend(backend) {
                Ok(()) => self.step_inner(provider, backend),
                Err(cause) => {
                    return self.finish_error_after_fence(LoaderError {
                        stage: LoaderStage::Backend,
                        cause,
                    });
                }
            }
        };
        match result {
            Ok(poll) => poll,
            Err(error) => self.finish_error_after_fence(error),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn step_inner<P, Backend>(
        &mut self,
        provider: &mut P,
        backend: &mut Backend,
    ) -> core::result::Result<LoaderPoll<TypedWeights<Buffer>>, LoaderError>
    where
        P: AssetProvider<Read = Read>,
        Backend: InferenceOps<Buffer = Buffer, Fence = Fence>,
    {
        match &mut self.phase {
            LoadPhase::NeedPrefix => {
                if self.request.declared_asset_bytes < 8 {
                    return Err(LoaderError {
                        stage: LoaderStage::HeaderPrefix,
                        cause: ExecutorError::OutOfBounds(
                            "asset is shorter than a safetensors prefix",
                        ),
                    });
                }
                let range = ByteRange { offset: 0, len: 8 };
                let retained = self
                    .report
                    .retained_config_bytes
                    .checked_add(range.len)
                    .ok_or(LoaderError {
                        stage: LoaderStage::HeaderPrefix,
                        cause: ExecutorError::Overflow(
                            "config and prefix retained bytes overflow u64",
                        ),
                    })?;
                if retained > self.request.limits.max_retained_host_bytes {
                    return Err(LoaderError {
                        stage: LoaderStage::HeaderPrefix,
                        cause: ExecutorError::ResourceLimit(
                            "config and prefix exceed retained host byte limit",
                        ),
                    });
                }
                let read = provider.read_range(range).map_err(|cause| LoaderError {
                    stage: LoaderStage::HeaderPrefix,
                    cause,
                })?;
                self.report.pending_requested_bytes = range.len;
                self.phase = LoadPhase::Prefix(read);
                Ok(LoaderPoll::Pending)
            }
            LoadPhase::Prefix(read) => match read.poll_step() {
                CompletionPoll::Pending => Ok(LoaderPoll::Pending),
                CompletionPoll::Ready(Err(cause)) => Err(LoaderError {
                    stage: LoaderStage::HeaderPrefix,
                    cause,
                }),
                CompletionPoll::Ready(Ok(bytes)) => {
                    self.report.pending_requested_bytes = 0;
                    if bytes.len() != 8 {
                        return Err(LoaderError {
                            stage: LoaderStage::HeaderPrefix,
                            cause: ExecutorError::InvalidArgument(
                                "header prefix completion length differs from request",
                            ),
                        });
                    }
                    let prefix: [u8; 8] = bytes.as_slice().try_into().map_err(|_| LoaderError {
                        stage: LoaderStage::HeaderPrefix,
                        cause: ExecutorError::InvalidArgument("header prefix has invalid length"),
                    })?;
                    let header_bytes = u64::from_le_bytes(prefix);
                    if header_bytes == 0 || header_bytes > self.request.limits.max_header_bytes {
                        return Err(LoaderError {
                            stage: LoaderStage::HeaderPrefix,
                            cause: ExecutorError::ResourceLimit(
                                "header exceeds configured byte limit",
                            ),
                        });
                    }
                    let end = 8_u64.checked_add(header_bytes).ok_or(LoaderError {
                        stage: LoaderStage::HeaderPrefix,
                        cause: ExecutorError::Overflow("header end overflows u64"),
                    })?;
                    if end > self.request.declared_asset_bytes {
                        return Err(LoaderError {
                            stage: LoaderStage::HeaderPrefix,
                            cause: ExecutorError::OutOfBounds(
                                "header extends beyond declared asset bytes",
                            ),
                        });
                    }
                    self.hasher.update(bytes.as_slice());
                    self.header_bytes = Some(header_bytes);
                    self.phase = LoadPhase::NeedHeader;
                    Ok(LoaderPoll::Pending)
                }
            },
            LoadPhase::NeedHeader => {
                let header_bytes = self.header_bytes.ok_or(LoaderError {
                    stage: LoaderStage::Header,
                    cause: ExecutorError::BackendFailure("header length is unavailable"),
                })?;
                let range = ByteRange {
                    offset: 8,
                    len: header_bytes,
                };
                let retained = self
                    .report
                    .retained_config_bytes
                    .checked_add(range.len)
                    .ok_or(LoaderError {
                        stage: LoaderStage::Header,
                        cause: ExecutorError::Overflow("retained header bytes overflow u64"),
                    })?;
                if retained > self.request.limits.max_retained_host_bytes {
                    return Err(LoaderError {
                        stage: LoaderStage::Header,
                        cause: ExecutorError::ResourceLimit(
                            "config and header exceed retained host byte limit",
                        ),
                    });
                }
                let read = provider.read_range(range).map_err(|cause| LoaderError {
                    stage: LoaderStage::Header,
                    cause,
                })?;
                self.report.pending_requested_bytes = range.len;
                self.phase = LoadPhase::Header(read);
                Ok(LoaderPoll::Pending)
            }
            LoadPhase::Header(read) => match read.poll_step() {
                CompletionPoll::Pending => Ok(LoaderPoll::Pending),
                CompletionPoll::Ready(Err(cause)) => Err(LoaderError {
                    stage: LoaderStage::Header,
                    cause,
                }),
                CompletionPoll::Ready(Ok(bytes)) => {
                    self.report.pending_requested_bytes = 0;
                    let header_bytes = self.header_bytes.ok_or(LoaderError {
                        stage: LoaderStage::Header,
                        cause: ExecutorError::BackendFailure("header length is unavailable"),
                    })?;
                    if bytes.len()
                        != usize::try_from(header_bytes).map_err(|_| LoaderError {
                            stage: LoaderStage::Header,
                            cause: ExecutorError::Overflow("header length exceeds usize"),
                        })?
                    {
                        return Err(LoaderError {
                            stage: LoaderStage::Header,
                            cause: ExecutorError::InvalidArgument(
                                "header completion length differs from request",
                            ),
                        });
                    }
                    self.report.retained_header_bytes = header_bytes;
                    let parsed = parse_safetensors_header(
                        bytes.as_slice(),
                        header_bytes,
                        self.request.declared_asset_bytes,
                        self.request.limits,
                    )
                    .map_err(|cause| LoaderError {
                        stage: LoaderStage::Inventory,
                        cause,
                    })?;
                    self.hasher.update(bytes.as_slice());
                    let manifest = parsed.as_manifest(&self.request.config_name);
                    self.bindings = minifield_engine_api::validate_asset_manifest(
                        &manifest,
                        &self.request.config_name,
                        &self.request.plan.manifest_requirements(),
                        minifield_engine_api::AssetLimits {
                            max_asset_bytes: self.request.limits.max_asset_bytes,
                            max_tensor_bytes: self.request.limits.max_source_tensor_bytes,
                            max_tensors: self.request.limits.max_tensors,
                        },
                    )
                    .map_err(|cause| LoaderError {
                        stage: LoaderStage::Inventory,
                        cause,
                    })?;
                    self.sorted_indexes = parsed.sorted_tensor_indexes();
                    self.staged
                        .try_reserve_exact(parsed.tensors.len())
                        .map_err(|_| LoaderError {
                            stage: LoaderStage::Inventory,
                            cause: ExecutorError::ResourceLimit(
                                "staged tensor list allocation failed",
                            ),
                        })?;
                    self.parsed = Some(parsed);
                    self.report.retained_header_bytes = 0;
                    self.phase = LoadPhase::NeedTensor;
                    Ok(LoaderPoll::Pending)
                }
            },
            LoadPhase::NeedTensor => {
                if self.next_tensor == self.sorted_indexes.len() {
                    self.phase = LoadPhase::NeedFence;
                    return Ok(LoaderPoll::Pending);
                }
                let manifest_index = self.sorted_indexes[self.next_tensor];
                let tensor = self
                    .parsed
                    .as_ref()
                    .and_then(|parsed| parsed.tensors.get(manifest_index))
                    .ok_or(LoaderError {
                        stage: LoaderStage::TensorRead,
                        cause: ExecutorError::BackendFailure(
                            "tensor inventory index is unavailable",
                        ),
                    })?;
                let decoded_bytes = tensor
                    .source_shape
                    .element_count()
                    .and_then(|count| {
                        count
                            .checked_mul(DType::F32.byte_width())
                            .ok_or(ExecutorError::Overflow("decoded f32 bytes overflow u64"))
                    })
                    .map_err(|cause| LoaderError {
                        stage: LoaderStage::TensorDecode,
                        cause,
                    })?;
                let retained = self
                    .report
                    .retained_config_bytes
                    .checked_add(tensor.bytes.len)
                    .and_then(|value| value.checked_add(decoded_bytes))
                    .ok_or(LoaderError {
                        stage: LoaderStage::TensorDecode,
                        cause: ExecutorError::Overflow("retained tensor bytes overflow u64"),
                    })?;
                if retained > self.request.limits.max_retained_host_bytes {
                    return Err(LoaderError {
                        stage: LoaderStage::TensorDecode,
                        cause: ExecutorError::ResourceLimit(
                            "source and decoded tensor exceed retained host byte limit",
                        ),
                    });
                }
                let read = provider
                    .read_range(tensor.bytes)
                    .map_err(|cause| LoaderError {
                        stage: LoaderStage::TensorRead,
                        cause,
                    })?;
                self.report.pending_requested_bytes = tensor.bytes.len;
                self.phase = LoadPhase::Tensor {
                    manifest_index,
                    read,
                };
                Ok(LoaderPoll::Pending)
            }
            LoadPhase::Tensor {
                manifest_index,
                read,
            } => match read.poll_step() {
                CompletionPoll::Pending => Ok(LoaderPoll::Pending),
                CompletionPoll::Ready(Err(cause)) => Err(LoaderError {
                    stage: LoaderStage::TensorRead,
                    cause,
                }),
                CompletionPoll::Ready(Ok(bytes)) => {
                    self.report.pending_requested_bytes = 0;
                    let index = *manifest_index;
                    let tensor = self
                        .parsed
                        .as_ref()
                        .and_then(|parsed| parsed.tensors.get(index))
                        .ok_or(LoaderError {
                            stage: LoaderStage::TensorRead,
                            cause: ExecutorError::BackendFailure(
                                "tensor inventory index is unavailable",
                            ),
                        })?;
                    if bytes.len()
                        != usize::try_from(tensor.bytes.len).map_err(|_| LoaderError {
                            stage: LoaderStage::TensorRead,
                            cause: ExecutorError::Overflow("tensor byte length exceeds usize"),
                        })?
                    {
                        return Err(LoaderError {
                            stage: LoaderStage::TensorRead,
                            cause: ExecutorError::InvalidArgument(
                                "tensor completion length differs from request",
                            ),
                        });
                    }
                    let requirement = self
                        .request
                        .plan
                        .requirements
                        .iter()
                        .find(|requirement| requirement.tensor_name == tensor.name)
                        .ok_or(LoaderError {
                            stage: LoaderStage::Inventory,
                            cause: ExecutorError::MissingRequiredTensor,
                        })?;
                    let decoded_bytes = tensor
                        .source_shape
                        .element_count()
                        .and_then(|count| {
                            count
                                .checked_mul(DType::F32.byte_width())
                                .ok_or(ExecutorError::Overflow("decoded f32 bytes overflow u64"))
                        })
                        .map_err(|cause| LoaderError {
                            stage: LoaderStage::TensorDecode,
                            cause,
                        })?;
                    let retained = self
                        .report
                        .retained_config_bytes
                        .checked_add(tensor.bytes.len)
                        .and_then(|value| value.checked_add(decoded_bytes))
                        .ok_or(LoaderError {
                            stage: LoaderStage::TensorDecode,
                            cause: ExecutorError::Overflow("retained tensor bytes overflow u64"),
                        })?;
                    if retained > self.request.limits.max_retained_host_bytes {
                        return Err(LoaderError {
                            stage: LoaderStage::TensorDecode,
                            cause: ExecutorError::ResourceLimit(
                                "source and decoded tensor exceed retained host byte limit",
                            ),
                        });
                    }
                    self.report.retained_source_bytes = tensor.bytes.len;
                    self.report.decoded_f32_bytes = decoded_bytes;
                    self.hasher.update(bytes.as_slice());
                    let (operator_shape, values) =
                        decode_tensor(bytes.as_slice(), tensor, requirement.layout).map_err(
                            |cause| LoaderError {
                                stage: LoaderStage::TensorDecode,
                                cause,
                            },
                        )?;
                    let buffer = backend
                        .upload_f32_classified(operator_shape, &values, AllocationClass::Weight)
                        .map_err(|cause| LoaderError {
                            stage: LoaderStage::Upload,
                            cause,
                        })?;
                    self.staged.push(StagedWeight {
                        manifest_index: index,
                        tensor: LoadedTensor {
                            name: tensor.name.clone(),
                            source_dtype: tensor.storage_dtype,
                            source_shape: tensor.source_shape,
                            operator_shape,
                            buffer,
                        },
                    });
                    self.report.retained_source_bytes = 0;
                    self.report.decoded_f32_bytes = 0;
                    self.next_tensor = self.next_tensor.checked_add(1).ok_or(LoaderError {
                        stage: LoaderStage::Upload,
                        cause: ExecutorError::Overflow("tensor progress counter overflows usize"),
                    })?;
                    self.phase = LoadPhase::NeedTensor;
                    Ok(LoaderPoll::Pending)
                }
            },
            LoadPhase::NeedFence => {
                let fence = backend.fence().map_err(|cause| LoaderError {
                    stage: LoaderStage::FinalFence,
                    cause,
                })?;
                self.phase = LoadPhase::Fence(fence);
                Ok(LoaderPoll::Pending)
            }
            LoadPhase::DrainingFence(fence) => match fence.poll_step() {
                CompletionPoll::Pending => Ok(LoaderPoll::Pending),
                CompletionPoll::Ready(_) => {
                    self.release_transient_host();
                    self.phase = LoadPhase::Failed;
                    Ok(LoaderPoll::Ready(Err(self
                        .terminal_error
                        .take()
                        .unwrap_or(LoaderError {
                            stage: LoaderStage::Cancelled,
                            cause: ExecutorError::Cancelled,
                        }))))
                }
            },
            LoadPhase::Fence(fence) => match fence.poll_step() {
                CompletionPoll::Pending => Ok(LoaderPoll::Pending),
                CompletionPoll::Ready(Err(cause)) => Err(LoaderError {
                    stage: LoaderStage::FinalFence,
                    cause,
                }),
                CompletionPoll::Ready(Ok(())) => {
                    let observed: [u8; 32] = self.hasher.clone().finalize().into();
                    if observed != self.request.expected_asset_sha256 {
                        return Err(LoaderError {
                            stage: LoaderStage::Identity,
                            cause: ExecutorError::InvalidArgument("asset content digest mismatch"),
                        });
                    }
                    let parsed = self.parsed.as_ref().ok_or(LoaderError {
                        stage: LoaderStage::FinalFence,
                        cause: ExecutorError::BackendFailure("parsed asset is unavailable"),
                    })?;
                    let mut slots: Vec<Option<LoadedTensor<Buffer>>> = Vec::new();
                    slots
                        .try_reserve_exact(parsed.tensors.len())
                        .map_err(|_| LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause: ExecutorError::ResourceLimit(
                                "published tensor list allocation failed",
                            ),
                        })?;
                    slots.resize_with(parsed.tensors.len(), || None);
                    for staged in self.staged.drain(..) {
                        let slot = slots.get_mut(staged.manifest_index).ok_or(LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause: ExecutorError::BackendFailure(
                                "staged tensor index is unavailable",
                            ),
                        })?;
                        if slot.replace(staged.tensor).is_some() {
                            return Err(LoaderError {
                                stage: LoaderStage::FinalFence,
                                cause: ExecutorError::DuplicateName,
                            });
                        }
                    }
                    let mut tensors = Vec::new();
                    tensors
                        .try_reserve_exact(slots.len())
                        .map_err(|_| LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause: ExecutorError::ResourceLimit(
                                "published tensor vector allocation failed",
                            ),
                        })?;
                    for slot in slots {
                        tensors.push(slot.ok_or(LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause: ExecutorError::MissingRequiredTensor,
                        })?);
                    }
                    let mut roles = Vec::new();
                    roles
                        .try_reserve_exact(self.bindings.len())
                        .map_err(|_| LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause: ExecutorError::ResourceLimit(
                                "published role list allocation failed",
                            ),
                        })?;
                    for binding in &self.bindings {
                        roles.push((binding.role.clone(), binding.tensor_index));
                    }
                    let owned_f32_bytes = tensors
                        .iter()
                        .try_fold(0_u64, |total, tensor| {
                            total
                                .checked_add(
                                    tensor
                                        .operator_shape
                                        .element_count()?
                                        .checked_mul(4)
                                        .ok_or(ExecutorError::Overflow(
                                            "owned f32 weight bytes overflow u64",
                                        ))?,
                                )
                                .ok_or(ExecutorError::Overflow(
                                    "owned f32 weight bytes overflow u64",
                                ))
                        })
                        .map_err(|cause| LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause,
                        })?;
                    self.report = LoaderResourceReport::default();
                    self.request.config_bytes.clear();
                    self.request.config_bytes.shrink_to_fit();
                    self.phase = LoadPhase::Finished;
                    let backend = self.backend.as_ref().ok_or(LoaderError {
                        stage: LoaderStage::Backend,
                        cause: ExecutorError::BackendFailure("loader backend lease is unavailable"),
                    })?;
                    Ok(LoaderPoll::Ready(Ok(TypedWeights {
                        owner: Rc::new(()),
                        backend: backend.identity(),
                        config_sha256: self.request.expected_config_sha256,
                        asset_sha256: observed,
                        tensors,
                        roles,
                        owned_f32_bytes,
                    })))
                }
            },
            LoadPhase::Finished | LoadPhase::Failed => Err(LoaderError {
                stage: LoaderStage::FinalFence,
                cause: ExecutorError::CompletionConsumed,
            }),
        }
    }
}

impl<Read, Fence, Buffer> Drop for WeightLoadTask<Read, Fence, Buffer> {
    fn drop(&mut self) {
        let phase = core::mem::replace(&mut self.phase, LoadPhase::Failed);
        let (LoadPhase::Fence(fence) | LoadPhase::DrainingFence(fence)) = phase else {
            return;
        };
        let retained = core::mem::take(&mut self.staged)
            .into_iter()
            .map(|staged| staged.tensor.buffer)
            .collect();
        if let Some(retirement) = self.retirement.take()
            && let Err(rejected) = retirement.retire(fence, retained)
        {
            // Loader staging is bound to this exact backend before a fence can exist. If a
            // backend nevertheless rejects during Drop, the task cannot return ownership, so
            // retain the complete payload conservatively until its carried fence is terminal.
            retirement.quarantine_rejected(rejected);
        }
    }
}
