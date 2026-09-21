//! Portable, backend-neutral contracts for the Minifield inference executor.
//!
//! This crate owns descriptors, bounded asset access, lifecycle/completion semantics,
//! and token-level interfaces. It deliberately has no tensor engine, model equation,
//! filesystem, thread, network, or GPU dependency.

#![forbid(unsafe_code)]
// CR01 uses compact checked descriptors whose Result errors are documented by the public
// ExecutorError taxonomy. Model-specific APIs add operation-level error details later.
#![allow(clippy::missing_errors_doc)]

use core::fmt;
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

/// Executor result type.
pub type Result<T> = std::result::Result<T, ExecutorError>;

/// Typed failures exposed by portable executor and backend APIs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorError {
    InvalidArgument(&'static str),
    InvalidShape(&'static str),
    InvalidLayout(&'static str),
    InvalidDType(&'static str),
    Overflow(&'static str),
    OutOfBounds(&'static str),
    Unsupported(&'static str),
    ResourceLimit(&'static str),
    WrongBackend,
    StaleBuffer,
    DuplicateName,
    MissingRequiredTensor,
    UnexpectedTensor,
    InvalidTie,
    Cancelled,
    CompletionConsumed,
    BackendFailure(&'static str),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidArgument(message)
            | Self::InvalidShape(message)
            | Self::InvalidLayout(message)
            | Self::InvalidDType(message)
            | Self::Overflow(message)
            | Self::OutOfBounds(message)
            | Self::Unsupported(message)
            | Self::ResourceLimit(message)
            | Self::BackendFailure(message) => message,
            Self::WrongBackend => "buffer belongs to another backend",
            Self::StaleBuffer => "buffer belongs to an earlier backend generation",
            Self::DuplicateName => "duplicate name",
            Self::MissingRequiredTensor => "required tensor is missing",
            Self::UnexpectedTensor => "undeclared tensor is present",
            Self::InvalidTie => "invalid tied tensor declaration",
            Self::Cancelled => "operation cancelled",
            Self::CompletionConsumed => "completion result was already consumed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ExecutorError {}

/// Scalar formats that can occur in buffers or imported tensors.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I32,
    U32,
}

impl DType {
    /// Width of one logical element in bytes.
    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    const fn bit(self) -> u16 {
        1_u16 << (self as u8)
    }
}

/// A bitset of explicitly supported scalar formats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DTypeSet(u16);

impl DTypeSet {
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn only(dtype: DType) -> Self {
        Self(dtype.bit())
    }

    #[must_use]
    pub const fn with(self, dtype: DType) -> Self {
        Self(self.0 | dtype.bit())
    }

    #[must_use]
    pub const fn contains(self, dtype: DType) -> bool {
        (self.0 & dtype.bit()) != 0
    }
}

/// Backend family is separate from numerical precision and operation support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Cpu,
    Wasm,
    Cuda,
    Metal,
    Other(u16),
}

/// Identity for one backend instance and lifetime generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendIdentity {
    pub kind: BackendKind,
    pub ordinal: u32,
    pub owner: u64,
    pub generation: u64,
}

/// Opaque actual-instance lease returned only by a backend implementation.
///
/// Public diagnostic identity fields may collide across independently created backends. The private
/// token distinguishes those instances while preserving the backend's own generation semantics.
#[derive(Clone, Debug)]
pub struct BackendLease {
    identity: BackendIdentity,
    actual: Rc<()>,
}

impl BackendLease {
    #[must_use]
    pub fn new(identity: BackendIdentity) -> Self {
        Self {
            identity,
            actual: Rc::new(()),
        }
    }

    #[must_use]
    pub fn with_identity(&self, identity: BackendIdentity) -> Self {
        Self {
            identity,
            actual: Rc::clone(&self.actual),
        }
    }

    #[must_use]
    pub const fn identity(&self) -> BackendIdentity {
        self.identity
    }

    #[must_use]
    pub fn same_actual_instance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.actual, &other.actual)
    }
}

/// Explicit numerical role precision. No backend name implies a precision choice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrecisionPolicy {
    pub weights: DType,
    pub activations: DType,
    pub cache: DType,
    pub accumulation: DType,
}

/// Finite inference operations. This is intentionally not a general tensor graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OperationKind {
    Copy,
    RectCopy2d,
    GatherRows,
    Add,
    Multiply,
    Linear,
    RowRmsNorm,
    Embedding,
    Rotary,
    GroupedQueryAttention,
    GatedShortConvolution,
    SwiGlu,
    LmProjection,
    PackedGatherRows,
    PackedLinear,
    AddRowRmsNorm,
    PackedLinearPair,
    PackedSwigluLinear,
    QkNormRope,
    Argmax,
    GatherColumns,
}

impl OperationKind {
    const fn bit(self) -> u64 {
        1_u64 << (self as u8)
    }
}

/// A bitset of operations whose semantics a backend has implemented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationSet(u64);

impl OperationSet {
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn with(self, operation: OperationKind) -> Self {
        Self(self.0 | operation.bit())
    }

    #[must_use]
    pub const fn contains(self, operation: OperationKind) -> bool {
        (self.0 & operation.bit()) != 0
    }
}

/// Checked rectangle for a row-major two-dimensional range copy.
///
/// The descriptor carries coordinates only. Backends still validate ownership,
/// generation, dtype, layout, bounds, destination access, and alias policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RectCopy2d {
    source_row: u64,
    source_column: u64,
    destination_row: u64,
    destination_column: u64,
    rows: u64,
    columns: u64,
}

impl RectCopy2d {
    /// Construct a finite rectangle. Zero rows or columns are valid no-op ranges.
    #[must_use]
    pub const fn new(
        source_row: u64,
        source_column: u64,
        destination_row: u64,
        destination_column: u64,
        rows: u64,
        columns: u64,
    ) -> Self {
        Self {
            source_row,
            source_column,
            destination_row,
            destination_column,
            rows,
            columns,
        }
    }

    #[must_use]
    pub const fn source_row(self) -> u64 {
        self.source_row
    }
    #[must_use]
    pub const fn source_column(self) -> u64 {
        self.source_column
    }
    #[must_use]
    pub const fn destination_row(self) -> u64 {
        self.destination_row
    }
    #[must_use]
    pub const fn destination_column(self) -> u64 {
        self.destination_column
    }
    #[must_use]
    pub const fn rows(self) -> u64 {
        self.rows
    }
    #[must_use]
    pub const fn columns(self) -> u64 {
        self.columns
    }

    /// Validate the rectangle against source and destination two-dimensional shapes.
    pub fn validate(self, source: Shape, destination: Shape) -> Result<()> {
        if source.rank() != 2 || destination.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "rectangular copy requires rank-two source and destination",
            ));
        }
        let source_row_end =
            self.source_row
                .checked_add(self.rows)
                .ok_or(ExecutorError::Overflow(
                    "rectangular source row end overflows u64",
                ))?;
        let source_column_end =
            self.source_column
                .checked_add(self.columns)
                .ok_or(ExecutorError::Overflow(
                    "rectangular source column end overflows u64",
                ))?;
        let destination_row_end =
            self.destination_row
                .checked_add(self.rows)
                .ok_or(ExecutorError::Overflow(
                    "rectangular destination row end overflows u64",
                ))?;
        let destination_column_end =
            self.destination_column
                .checked_add(self.columns)
                .ok_or(ExecutorError::Overflow(
                    "rectangular destination column end overflows u64",
                ))?;
        if source_row_end > source.dim(0)?
            || source_column_end > source.dim(1)?
            || destination_row_end > destination.dim(0)?
            || destination_column_end > destination.dim(1)?
        {
            return Err(ExecutorError::OutOfBounds(
                "rectangular copy range exceeds source or destination shape",
            ));
        }
        Ok(())
    }
}

/// Packed `[tokens, heads * head_dim]` descriptor for finite attention and rotary kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedHeadSpec {
    heads: u32,
    head_dim: u32,
}

impl PackedHeadSpec {
    pub fn new(heads: u32, head_dim: u32) -> Result<Self> {
        if heads == 0 || head_dim == 0 {
            return Err(ExecutorError::InvalidArgument(
                "head count and head dimension must be nonzero",
            ));
        }
        Ok(Self { heads, head_dim })
    }

    #[must_use]
    pub const fn heads(self) -> u32 {
        self.heads
    }
    #[must_use]
    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    pub fn packed_width(self) -> Result<u64> {
        u64::from(self.heads)
            .checked_mul(u64::from(self.head_dim))
            .ok_or(ExecutorError::Overflow("packed head width overflows u64"))
    }

    pub fn validate_packed(self, shape: Shape) -> Result<u64> {
        if shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed head tensor must be rank two",
            ));
        }
        if shape.dim(1)? != self.packed_width()? {
            return Err(ExecutorError::InvalidShape(
                "packed head tensor width differs from descriptor",
            ));
        }
        shape.dim(0)
    }
}

/// Explicit split-half `RoPE` descriptor. The first and second contiguous halves rotate together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotarySpec {
    heads: PackedHeadSpec,
    theta: f32,
}

impl RotarySpec {
    pub fn new(heads: PackedHeadSpec, theta: f32) -> Result<Self> {
        if heads.head_dim() % 2 != 0 {
            return Err(ExecutorError::InvalidShape(
                "split-half RoPE requires an even head dimension",
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RoPE theta must be finite and positive",
            ));
        }
        Ok(Self { heads, theta })
    }

    #[must_use]
    pub const fn heads(self) -> PackedHeadSpec {
        self.heads
    }
    #[must_use]
    pub const fn theta(self) -> f32 {
        self.theta
    }
}

/// Causal grouped-query attention descriptor. Each key/value head serves a contiguous group
/// of query heads; repeated physical KV buffers are deliberately unsupported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GqaSpec {
    query_heads: PackedHeadSpec,
    key_value_heads: PackedHeadSpec,
}

impl GqaSpec {
    pub fn new(query_heads: u32, key_value_heads: u32, head_dim: u32) -> Result<Self> {
        if key_value_heads == 0 || query_heads == 0 || query_heads % key_value_heads != 0 {
            return Err(ExecutorError::InvalidShape(
                "query heads must be a nonzero multiple of key/value heads",
            ));
        }
        let query_heads = PackedHeadSpec::new(query_heads, head_dim)?;
        let key_value_heads = PackedHeadSpec::new(key_value_heads, head_dim)?;
        Ok(Self {
            query_heads,
            key_value_heads,
        })
    }

    #[must_use]
    pub const fn query_heads(self) -> PackedHeadSpec {
        self.query_heads
    }
    #[must_use]
    pub const fn key_value_heads(self) -> PackedHeadSpec {
        self.key_value_heads
    }
    #[must_use]
    pub const fn group_size(self) -> u32 {
        self.query_heads.heads() / self.key_value_heads.heads()
    }
}

/// Checked gated short-convolution descriptor. Kernel taps are oldest-to-current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedShortConvSpec {
    hidden: u32,
    width: u32,
}

impl GatedShortConvSpec {
    pub fn new(hidden: u32, width: u32) -> Result<Self> {
        if hidden == 0 || width == 0 {
            return Err(ExecutorError::InvalidArgument(
                "short convolution hidden width and kernel width must be nonzero",
            ));
        }
        Ok(Self { hidden, width })
    }

    #[must_use]
    pub const fn hidden(self) -> u32 {
        self.hidden
    }
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    pub fn history_rows(self) -> Result<u64> {
        u64::from(self.width)
            .checked_sub(1)
            .ok_or(ExecutorError::Overflow(
                "short convolution history underflows",
            ))
    }
}

/// Backend capability declaration resolved once at model load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub dtypes: DTypeSet,
    pub operations: OperationSet,
    pub precision: PrecisionPolicy,
    pub max_rank: u8,
    pub max_elements: u64,
    pub max_allocation_bytes: u64,
    pub supports_nonblocking_completion: bool,
}

impl BackendCapabilities {
    /// Reject a format, operation, rank, or allocation unsupported by this backend.
    pub fn validate(
        self,
        dtype: DType,
        operation: OperationKind,
        rank: u8,
        elements: u64,
        byte_len: u64,
    ) -> Result<()> {
        if !self.dtypes.contains(dtype) {
            return Err(ExecutorError::InvalidDType(
                "dtype is unsupported by backend",
            ));
        }
        if !self.operations.contains(operation) {
            return Err(ExecutorError::Unsupported(
                "operation is unsupported by backend",
            ));
        }
        if rank > self.max_rank {
            return Err(ExecutorError::InvalidShape(
                "rank exceeds backend capability",
            ));
        }
        if elements > self.max_elements {
            return Err(ExecutorError::ResourceLimit(
                "element count exceeds backend capability",
            ));
        }
        if byte_len > self.max_allocation_bytes {
            return Err(ExecutorError::ResourceLimit(
                "allocation exceeds backend capability",
            ));
        }
        Ok(())
    }
}

/// Maximum supported rank for the initial inference-specific layouts.
pub const MAX_RANK: usize = 4;

/// Checked, fixed-rank shape descriptor. A zero dimension denotes an empty tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    rank: u8,
    dims: [u64; MAX_RANK],
}

impl Shape {
    pub fn new(dims: &[u64]) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(ExecutorError::InvalidShape(
                "rank exceeds portable descriptor limit",
            ));
        }
        let mut stored = [0_u64; MAX_RANK];
        for (index, dimension) in dims.iter().copied().enumerate() {
            stored[index] = dimension;
        }
        Ok(Self {
            rank: u8::try_from(dims.len()).map_err(|_| {
                ExecutorError::InvalidShape("rank exceeds portable descriptor limit")
            })?,
            dims: stored,
        })
    }

    #[must_use]
    pub const fn rank(self) -> u8 {
        self.rank
    }

    pub fn dim(self, index: usize) -> Result<u64> {
        if index >= self.rank as usize {
            return Err(ExecutorError::OutOfBounds("dimension index exceeds rank"));
        }
        Ok(self.dims[index])
    }

    #[must_use]
    pub fn dims(self) -> [u64; MAX_RANK] {
        self.dims
    }

    pub fn element_count(self) -> Result<u64> {
        let mut count = 1_u64;
        for index in 0..self.rank as usize {
            count = count
                .checked_mul(self.dims[index])
                .ok_or(ExecutorError::Overflow("shape element count overflows u64"))?;
        }
        Ok(count)
    }
}

/// Checked byte range within a caller-provided asset or storage object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

impl ByteRange {
    pub fn end(self) -> Result<u64> {
        self.offset
            .checked_add(self.len)
            .ok_or(ExecutorError::Overflow("byte range end overflows u64"))
    }

    pub fn validate_within(self, total_bytes: u64) -> Result<()> {
        if self.end()? > total_bytes {
            return Err(ExecutorError::OutOfBounds(
                "byte range exceeds source bytes",
            ));
        }
        Ok(())
    }
}

/// Strided tensor layout whose extent and all arithmetic are independently checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorLayout {
    dtype: DType,
    shape: Shape,
    strides: [u64; MAX_RANK],
    offset_bytes: u64,
    byte_extent: u64,
}

impl TensorLayout {
    pub fn contiguous(dtype: DType, shape: Shape) -> Result<Self> {
        let mut strides = [0_u64; MAX_RANK];
        let mut stride = 1_u64;
        for index in (0..shape.rank() as usize).rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(shape.dim(index)?)
                .ok_or(ExecutorError::Overflow("contiguous stride overflows u64"))?;
        }
        Self::strided(dtype, shape, &strides[..shape.rank() as usize], 0)
    }

    pub fn strided(dtype: DType, shape: Shape, strides: &[u64], offset_bytes: u64) -> Result<Self> {
        if strides.len() != shape.rank() as usize {
            return Err(ExecutorError::InvalidLayout(
                "stride count differs from rank",
            ));
        }
        if offset_bytes % dtype.byte_width() != 0 {
            return Err(ExecutorError::InvalidLayout(
                "byte offset is not dtype aligned",
            ));
        }
        let mut stored = [0_u64; MAX_RANK];
        for (index, stride) in strides.iter().copied().enumerate() {
            stored[index] = stride;
        }
        let element_count = shape.element_count()?;
        let byte_extent = if element_count == 0 {
            0
        } else {
            let mut greatest_element_offset = 0_u64;
            for (index, stride) in stored.iter().enumerate().take(shape.rank() as usize) {
                let extent =
                    shape
                        .dim(index)?
                        .checked_sub(1)
                        .ok_or(ExecutorError::InvalidLayout(
                            "nonempty layout has a zero dimension",
                        ))?;
                greatest_element_offset = greatest_element_offset
                    .checked_add(
                        extent
                            .checked_mul(*stride)
                            .ok_or(ExecutorError::Overflow("strided offset overflows u64"))?,
                    )
                    .ok_or(ExecutorError::Overflow("strided offset overflows u64"))?;
            }
            greatest_element_offset
                .checked_add(1)
                .and_then(|elements| elements.checked_mul(dtype.byte_width()))
                .ok_or(ExecutorError::Overflow("strided byte extent overflows u64"))?
        };
        offset_bytes
            .checked_add(byte_extent)
            .ok_or(ExecutorError::Overflow("layout end overflows u64"))?;
        Ok(Self {
            dtype,
            shape,
            strides: stored,
            offset_bytes,
            byte_extent,
        })
    }

    #[must_use]
    pub fn strides(self) -> [u64; MAX_RANK] {
        self.strides
    }

    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    #[must_use]
    pub const fn shape(self) -> Shape {
        self.shape
    }

    #[must_use]
    pub const fn offset_bytes(self) -> u64 {
        self.offset_bytes
    }

    #[must_use]
    pub const fn byte_extent(self) -> u64 {
        self.byte_extent
    }

    pub fn end_byte(self) -> Result<u64> {
        self.offset_bytes
            .checked_add(self.byte_extent)
            .ok_or(ExecutorError::Overflow("layout end overflows u64"))
    }

    pub fn validate_within(self, storage_bytes: u64) -> Result<()> {
        if self.end_byte()? > storage_bytes {
            return Err(ExecutorError::OutOfBounds("layout exceeds storage bytes"));
        }
        Ok(())
    }

    pub fn is_contiguous(self) -> Result<bool> {
        let expected = Self::contiguous(self.dtype, self.shape)?;
        Ok(self.offset_bytes == 0 && self.strides == expected.strides)
    }

    pub fn element_byte_offset(self, indices: &[u64]) -> Result<u64> {
        if indices.len() != self.shape.rank() as usize {
            return Err(ExecutorError::InvalidLayout(
                "index rank differs from layout rank",
            ));
        }
        let mut elements = 0_u64;
        for (index, coordinate) in indices.iter().copied().enumerate() {
            if coordinate >= self.shape.dim(index)? {
                return Err(ExecutorError::OutOfBounds(
                    "tensor coordinate exceeds dimension",
                ));
            }
            elements = elements
                .checked_add(
                    coordinate
                        .checked_mul(self.strides[index])
                        .ok_or(ExecutorError::Overflow("element offset overflows u64"))?,
                )
                .ok_or(ExecutorError::Overflow("element offset overflows u64"))?;
        }
        self.offset_bytes
            .checked_add(
                elements
                    .checked_mul(self.dtype.byte_width())
                    .ok_or(ExecutorError::Overflow("element byte offset overflows u64"))?,
            )
            .ok_or(ExecutorError::Overflow("element byte offset overflows u64"))
    }
}

/// Access intent for a buffer descriptor. Backends reject unsupported write aliasing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferAccess {
    ReadOnly,
    ReadWrite,
}

/// Portable buffer metadata. Raw host or device pointers never appear in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferDescriptor {
    pub backend: BackendIdentity,
    pub allocation: u64,
    pub layout: TensorLayout,
    pub access: BufferAccess,
}

impl BufferDescriptor {
    pub fn validate_for(self, backend: BackendIdentity) -> Result<()> {
        if self.backend.kind != backend.kind
            || self.backend.ordinal != backend.ordinal
            || self.backend.owner != backend.owner
        {
            return Err(ExecutorError::WrongBackend);
        }
        if self.backend.generation != backend.generation {
            return Err(ExecutorError::StaleBuffer);
        }
        Ok(())
    }
}

/// Limits counted by a backend-owned resource arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceLimits {
    pub max_allocation_bytes: u64,
    pub max_total_bytes: u64,
    pub max_pending_operations: u32,
}

/// Lifetime/accounting class for backend-owned buffers.
///
/// The class describes the caller's logical ownership purpose; it does not loosen a backend's
/// aggregate allocation limit. Existing unclassified allocation helpers deliberately default to
/// `Scratch` so older callers cannot accidentally publish mutable model state as weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationClass {
    Weight,
    Cache,
    Scratch,
    Branch,
}

impl ResourceLimits {
    pub fn validate_allocation(self, allocation_bytes: u64, current_bytes: u64) -> Result<()> {
        if allocation_bytes > self.max_allocation_bytes {
            return Err(ExecutorError::ResourceLimit(
                "allocation exceeds configured per-buffer limit",
            ));
        }
        let total = current_bytes
            .checked_add(allocation_bytes)
            .ok_or(ExecutorError::Overflow("resource total overflows u64"))?;
        if total > self.max_total_bytes {
            return Err(ExecutorError::ResourceLimit(
                "allocation exceeds configured total resource limit",
            ));
        }
        Ok(())
    }
}

/// Resource accounting includes owned resident, cache, scratch, branch, and pending bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceReport {
    pub resident_weight_bytes: u64,
    pub cache_bytes: u64,
    pub scratch_bytes: u64,
    pub staged_branch_bytes: u64,
    pub pending_operation_bytes: u64,
    pub pending_operations: u32,
}

impl ResourceReport {
    pub fn total_owned_bytes(self) -> Result<u64> {
        self.resident_weight_bytes
            .checked_add(self.cache_bytes)
            .and_then(|value| value.checked_add(self.scratch_bytes))
            .and_then(|value| value.checked_add(self.staged_branch_bytes))
            .and_then(|value| value.checked_add(self.pending_operation_bytes))
            .ok_or(ExecutorError::Overflow(
                "resource report total overflows u64",
            ))
    }

    pub fn validate(self, limits: ResourceLimits) -> Result<()> {
        if self.pending_operations > limits.max_pending_operations {
            return Err(ExecutorError::ResourceLimit(
                "pending operation count exceeds configured limit",
            ));
        }
        if self.total_owned_bytes()? > limits.max_total_bytes {
            return Err(ExecutorError::ResourceLimit(
                "owned resource bytes exceed configured limit",
            ));
        }
        Ok(())
    }
}

/// One nonblocking observation of an inference completion.
#[derive(Debug, PartialEq)]
pub enum CompletionPoll<T> {
    Pending,
    Ready(Result<T>),
}

/// A portable poll/step boundary. Implementations need not be Send, Sync, threaded, or blocking.
pub trait InferenceCompletion {
    type Output;

    /// Advance or observe one bounded completion step.
    fn poll_step(&mut self) -> CompletionPoll<Self::Output>;

    /// Request cancellation before an operation is irrevocably submitted.
    fn cancel(&mut self) -> Result<()>;
}

/// A retirement admission rejection that preserves the complete unresolved payload.
///
/// A queue must return this value before changing any ownership or accounting when the fence or
/// any retained buffer belongs to another actual backend instance. The caller can inspect the
/// cause and route or retain the original fence and buffers without dropping them.
#[derive(Debug)]
pub struct RetirementRejection<Fence, Buffer> {
    cause: ExecutorError,
    fence: Fence,
    retained: Vec<Buffer>,
}

impl<Fence, Buffer> RetirementRejection<Fence, Buffer> {
    /// Build an ownership-preserving rejection before retirement admission mutates state.
    #[must_use]
    pub const fn new(cause: ExecutorError, fence: Fence, retained: Vec<Buffer>) -> Self {
        Self {
            cause,
            fence,
            retained,
        }
    }

    #[must_use]
    pub const fn cause(&self) -> &ExecutorError {
        &self.cause
    }

    /// Recover the cause and every input exactly as supplied to retirement admission.
    #[must_use]
    pub fn into_parts(self) -> (ExecutorError, Fence, Vec<Buffer>) {
        (self.cause, self.fence, self.retained)
    }
}

/// Backend-owned quarantine for an unresolved submission fence and the buffers the submission
/// still references.
///
/// A caller may abandon a higher-level task after cancellation cannot be confirmed. In that
/// case, the task transfers the fence and buffers here instead of dropping either one. The
/// backend keeps them alive until `poll_retired` observes a terminal completion result. This
/// trait deliberately does not require threads, Send, or Sync.
pub trait FenceRetirement<Fence, Buffer> {
    /// Admit one unresolved fence and every buffer retained by that submission.
    ///
    /// Admission checks actual backend ownership before changing accounting. A rejection returns
    /// every consumed input so a normal caller can route it safely.
    fn retire(
        &self,
        fence: Fence,
        retained: Vec<Buffer>,
    ) -> core::result::Result<(), RetirementRejection<Fence, Buffer>>;

    /// Conservatively retain a payload that a task cannot return because it is being dropped.
    ///
    /// The payload must have come from this queue's `retire` rejection. Implementations retain it
    /// without cross-instance accounting and release it only after the carried fence is terminal.
    fn quarantine_rejected(&self, rejected: RetirementRejection<Fence, Buffer>);

    /// Advance all abandoned fences once. A ready success or error releases its retained buffers.
    fn poll_retired(&self);

    /// Return whether any fence may still reference an abandoned task's source snapshots.
    ///
    /// Backends that cannot expose this information must keep the conservative default. Shared
    /// callers then retain source snapshots until the executor itself is dropped or reloaded.
    fn has_unresolved(&self) -> bool {
        true
    }
}

/// Row selector source for gather operations.
///
/// `Host` carries caller-held token ids. `Device` points at a backend-resident
/// f32 `[T]` buffer of exact integer row indices, typically produced by
/// [`InferenceOps::argmax`], so a sampled token can feed an embedding gather
/// without a host roundtrip. A non-finite device id produces a non-finite
/// output row on deferred backends; backends that validate operands eagerly
/// may reject it at call time instead.
#[derive(Clone, Copy)]
pub enum TokenIds<'a, B: InferenceOps + ?Sized> {
    Host(&'a [u32]),
    Device(&'a B::Buffer),
}

/// Finite, backend-neutral inference operation contract.
///
/// Implementors own buffer storage and expose pollable fence/readback completion types.
/// A device backend may enqueue kernel methods; it retains every queued source, destination, and
/// staging buffer until the next submitted fence or readback completion releases it. Shared model
/// code must poll those completions and must not require a backend-wide blocking read or sync.
/// This intentionally names only model-execution primitives; it is not a tensor graph or general
/// expression system.
pub trait InferenceOps {
    type Buffer;
    type Fence: InferenceCompletion<Output = ()>;
    type Readback: InferenceCompletion<Output = Vec<f32>>;
    type FenceRetirement: FenceRetirement<Self::Fence, Self::Buffer> + 'static;

    fn identity(&self) -> BackendIdentity;

    /// Return this backend's non-forgeable actual-instance lease and current generation.
    fn lease(&self) -> BackendLease;
    /// Return the backend-owned retirement queue for unresolved submission fences.
    fn fence_retirement(&self) -> Rc<Self::FenceRetirement>;
    /// Advance backend-owned abandoned fence retirement without a blocking synchronization.
    fn poll_retired_fences(&self) -> Result<()>;
    fn capabilities(&self) -> BackendCapabilities;
    fn resource_report(&self) -> ResourceReport;
    fn advance_generation(&mut self) -> Result<()>;

    /// Allocate an f32 buffer with an explicit lifetime/accounting class.
    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Upload f32 values into an explicitly classified backend-owned buffer.
    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Upload raw bytes into an explicitly classified backend-owned buffer.
    /// Used for opaque packed payloads such as ternary code streams; the bytes
    /// are not interpreted as scalars by the buffer contract.
    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Allocate transient work storage. Model/prefix loaders must use the classified form.
    fn allocate_f32(&mut self, shape: Shape) -> Result<Self::Buffer> {
        self.allocate_f32_classified(shape, AllocationClass::Scratch)
    }

    /// Allocate f32 storage whose initial contents are unspecified. Callers
    /// must fully overwrite the buffer before any consumer reads it. The
    /// default forwards to the zero-initializing classified allocation;
    /// backends that pay a per-allocation zeroing cost can override it.
    fn allocate_f32_uninit(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        self.allocate_f32_classified(shape, class)
    }

    /// Upload transient work storage. Model/prefix loaders must use the classified form.
    fn upload_f32(&mut self, shape: Shape, values: &[f32]) -> Result<Self::Buffer> {
        self.upload_f32_classified(shape, values, AllocationClass::Scratch)
    }

    /// Submit a completion boundary without forcing a device-wide synchronous wait.
    fn fence(&self) -> Result<Self::Fence>;

    /// Submit host-visible f32 readback. The completion owns output until polling is ready;
    /// a device backend retains source storage and its staging buffer until that point.
    fn read_f32_async(&self, buffer: &Self::Buffer) -> Result<Self::Readback>;

    fn copy(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()>;

    /// Copy one checked rectangular range between distinct packed rank-two buffers.
    /// Overlap on the same physical allocation is unsupported until a separately qualified
    /// view/alias contract exists.
    fn copy_rect_2d(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        rectangle: RectCopy2d,
    ) -> Result<()>;

    /// Row-wise argmax over f32 `[T, V]` logits: `output[t]` is the index of
    /// the first strict maximum in row `t`, written as an exact f32 integer,
    /// or NaN when the row contains any non-finite element. `output` is f32
    /// `[T]`. `V` must be at most `1 << 24` so indices stay exactly
    /// representable. The NaN marker lets a tiny readback double as the
    /// finiteness check for the whole logits row.
    fn argmax(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()>;

    /// `argmax` with a per-element candidate mask: only positions whose bit
    /// is set in `mask` may win. `mask` is one bit per element, LSB-first
    /// inside each u64 word (`mask[i / 64]` bit `i % 64`); its length must be
    /// `ceil(width / 64)`. Non-finite values at allowed positions still
    /// poison the row to NaN; masked-out values are skipped entirely. A row
    /// with no allowed candidate produces NaN. The default implementation
    /// reports `Unsupported` rather than silently ignoring the constraint.
    fn argmax_masked(
        &self,
        _output: &mut Self::Buffer,
        _input: &Self::Buffer,
        _mask: &[u64],
    ) -> Result<()> {
        Err(ExecutorError::Unsupported(
            "masked argmax is not implemented for this backend",
        ))
    }

    fn gather_rows(
        &self,
        output: &mut Self::Buffer,
        table: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()>;

    /// Column-wise gather over a contiguous f32 `[rows, width]` input:
    /// `output[r, k] = input[r, columns[k]]`. `output` is f32
    /// `[rows, columns.len()]`. Caller order and duplicate column ids are
    /// preserved, so `columns` acts as a typed selector list. An empty
    /// `columns` is a valid no-op producing `[rows, 0]` output. Out-of-range
    /// ids are rejected. Unlike [`InferenceOps::gather_rows`], ids always
    /// arrive as host `TokenId`s; callers needing only a few columns can use
    /// this to avoid a full-width host readback.
    fn gather_columns(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        columns: &[TokenId],
    ) -> Result<()>;

    /// Gather packed ternary rows and dequantize them into an f32 `[ids, K]` output.
    ///
    /// `codes` is a U8 `[rows, K/4]` buffer in `minifield.ternary.v1` layout: each
    /// group of 128 weights occupies 32 consecutive bytes, and weight `j` of a
    /// group sits at byte `j / 4`, bits `2 * (j % 4)`. `scales` is the f32
    /// `[rows, K/128]` group-scale stream decoded from the packed file's FP16
    /// scales. Decoded weight `w = (code - 1) * scale`.
    fn packed_gather_rows(
        &self,
        output: &mut Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()>;

    /// Packed ternary linear: `output[t, r] = sum_k input[t, k] * w[r, k]` where
    /// `w` is the `minifield.ternary.v1` dequantization of `codes`/`scales` as
    /// documented on [`InferenceOps::packed_gather_rows`]. `input` is f32
    /// `[T, K]`, `codes` is U8 `[R, K/4]`, `scales` is f32 `[R, K/128]`, and
    /// `output` is f32 `[T, R]`. Group scales apply inside each 128-weight
    /// group, matching the reference dequantized matvec.
    fn packed_linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()>;

    /// Paired packed ternary linear over one shared input:
    /// `out_a[t, r] = sum_k input[t, k] * wa[r, k]` and
    /// `out_b[t, r] = sum_k input[t, k] * wb[r, k]`. Both weight sets share the
    /// `minifield.ternary.v1` layout and must have identical `[R, K]` shapes, so
    /// `out_a` and `out_b` have identical `[T, R]` shapes. Semantically equal to
    /// two `packed_linear` calls; backends may issue them as one dispatch.
    #[allow(clippy::too_many_arguments)]
    fn packed_linear_pair(
        &self,
        out_a: &mut Self::Buffer,
        out_b: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()>;

    /// Packed ternary linear over an on-the-fly `SiLU(gate) * up` activation:
    /// `output[t, r] = sum_k (silu(gate[t,k]) * up[t,k]) * w[r, k]` where `w` is
    /// the `minifield.ternary.v1` dequantization of `codes`/`scales`. `gate` and
    /// `up` are f32 `[T, K]`, `codes` is U8 `[R, K/4]`, `scales` is f32
    /// `[R, K/128]`, and `output` is f32 `[T, R]`. Semantically equal to a
    /// `swiglu` into scratch followed by `packed_linear`.
    fn packed_swiglu_linear(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()>;

    /// Fused residual add plus row RMS norm: `sum = left + right` and
    /// `normed` is the row RMS norm of `sum` scaled by `weight`, with `sum`
    /// and `normed` as distinct `[T, C]` outputs. `left`, `right` are
    /// `[T, C]` inputs and `weight` is `[C]`. Semantically equal to `add`
    /// followed by `row_rms_norm` over the sum.
    fn add_row_rms_norm(
        &self,
        sum: &mut Self::Buffer,
        normed: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()>;

    /// Fused per-head RMS norm plus split-half rotary for query and key rows:
    /// `query_out` is `rope(head_rms_norm(query, query_weight))` and `key_out`
    /// is `rope(head_rms_norm(key, key_weight))`, evaluated per `[token, head]`
    /// row with `positions` per token. `rope` carries the query head geometry
    /// and theta; `key_value_heads` carries the key head geometry. Semantically
    /// equal to `head_rms_norm` then `split_half_rotary` on each tensor.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        &self,
        query_out: &mut Self::Buffer,
        key_out: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        query_weight: &Self::Buffer,
        key_weight: &Self::Buffer,
        positions: &[u64],
        rope: RotarySpec,
        key_value_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()>;
    fn add(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()>;
    fn multiply(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()>;
    fn linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
    ) -> Result<()>;
    fn row_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()>;

    /// Apply RMS normalization independently to every [token, head] row.
    fn head_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()>;

    /// Split-half rotary embedding for packed `[tokens, heads * head_dim]` values.
    fn split_half_rotary(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()>;

    /// Append packed K/V rows to the supplied caches and calculate causal GQA output.
    /// The cache length changes only after all shapes and input bounds have been accepted.
    #[allow(clippy::too_many_arguments)]
    fn causal_gqa(
        &self,
        output: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()>;

    /// Apply B*V gated short convolution and update a [width-1, hidden] rolling
    /// U history. `projection` is the fused `[tokens, 3 * hidden]` in-projection
    /// output: token `t`'s B, C, and V rows sit at offsets `t * 3h + {0, h, 2h}`.
    fn gated_short_convolution(
        &self,
        output: &mut Self::Buffer,
        projection: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()>;

    /// Apply SiLU(gate) * up elementwise over equal contiguous layouts.
    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()>;
}

/// Immutable bytes delivered from an asset provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetBytes {
    bytes: Vec<u8>,
}

impl AssetBytes {
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Filesystem-free input boundary. Native files and WASM asset buffers adapt to this trait.
pub trait AssetProvider {
    type Read: InferenceCompletion<Output = AssetBytes>;

    fn read_range(&mut self, range: ByteRange) -> Result<Self::Read>;
}

/// Asset limits checked before any loader allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetLimits {
    pub max_asset_bytes: u64,
    pub max_tensor_bytes: u64,
    pub max_tensors: usize,
}

/// Tensor record supplied by a format-specific importer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRecord {
    pub name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: ByteRange,
}

impl TensorRecord {
    pub fn validate(&self, asset_bytes: u64, limits: AssetLimits) -> Result<()> {
        if self.name.is_empty() {
            return Err(ExecutorError::InvalidArgument("tensor name is empty"));
        }
        self.bytes.validate_within(asset_bytes)?;
        let expected_bytes = self
            .shape
            .element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or(ExecutorError::Overflow("tensor byte length overflows u64"))?;
        if expected_bytes != self.bytes.len {
            return Err(ExecutorError::InvalidLayout(
                "tensor range differs from shape and dtype byte length",
            ));
        }
        if self.bytes.len > limits.max_tensor_bytes {
            return Err(ExecutorError::ResourceLimit(
                "tensor exceeds configured byte limit",
            ));
        }
        Ok(())
    }
}

/// Explicit model role requirement. Ties use another role and must name the same source tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRequirement {
    pub role: String,
    pub tensor_name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub tied_to_role: Option<String>,
}

/// Portable imported-asset manifest. Parsing of concrete formats belongs in executor core later.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetManifest {
    pub config_name: String,
    pub asset_bytes: u64,
    pub tensors: Vec<TensorRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorBinding {
    pub role: String,
    pub tensor_index: usize,
}

/// Maximum requirement roles accepted by this foundation before a concrete model importer
/// applies its tighter, model-specific inventory.
pub const MAX_TENSOR_REQUIREMENTS: usize = 4096;

/// Validate an exact config and physical tensor inventory.
///
/// Requirements are an internal model declaration. A canonical physical role must appear before
/// each direct alias. Alias chains, forward references, self references, and duplicate canonical
/// roles are invalid. Each nonempty tensor byte range must be disjoint from every other physical
/// tensor record; tied roles share one record rather than overlapping records.
#[allow(clippy::too_many_lines)]
pub fn validate_asset_manifest(
    manifest: &AssetManifest,
    expected_config_name: &str,
    requirements: &[TensorRequirement],
    limits: AssetLimits,
) -> Result<Vec<TensorBinding>> {
    if manifest.config_name != expected_config_name {
        return Err(ExecutorError::InvalidArgument(
            "unexpected model config name",
        ));
    }
    if manifest.asset_bytes > limits.max_asset_bytes {
        return Err(ExecutorError::ResourceLimit(
            "asset exceeds configured byte limit",
        ));
    }
    if manifest.tensors.len() > limits.max_tensors {
        return Err(ExecutorError::ResourceLimit(
            "tensor count exceeds configured limit",
        ));
    }
    if requirements.len() > MAX_TENSOR_REQUIREMENTS {
        return Err(ExecutorError::ResourceLimit(
            "tensor requirement count exceeds portable limit",
        ));
    }

    let mut requirement_roles = BTreeMap::new();
    let mut canonical_roles_by_tensor = BTreeMap::new();
    let mut required_tensor_names = BTreeSet::new();
    for (index, requirement) in requirements.iter().enumerate() {
        if requirement.role.is_empty() || requirement.tensor_name.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "tensor requirement name is empty",
            ));
        }
        if requirement_roles
            .insert(requirement.role.as_str(), index)
            .is_some()
        {
            return Err(ExecutorError::DuplicateName);
        }
        match requirement.tied_to_role.as_deref() {
            None => {
                if canonical_roles_by_tensor
                    .insert(requirement.tensor_name.as_str(), requirement.role.as_str())
                    .is_some()
                {
                    return Err(ExecutorError::InvalidTie);
                }
            }
            Some(tied_to_role) => {
                if tied_to_role == requirement.role {
                    return Err(ExecutorError::InvalidTie);
                }
                let Some(target_index) = requirement_roles.get(tied_to_role).copied() else {
                    return Err(ExecutorError::InvalidTie);
                };
                let target = &requirements[target_index];
                if target.tied_to_role.is_some()
                    || target.tensor_name != requirement.tensor_name
                    || target.dtype != requirement.dtype
                    || target.shape != requirement.shape
                    || canonical_roles_by_tensor.get(requirement.tensor_name.as_str())
                        != Some(&target.role.as_str())
                {
                    return Err(ExecutorError::InvalidTie);
                }
            }
        }
        required_tensor_names.insert(requirement.tensor_name.as_str());
    }

    let mut tensor_indexes = BTreeMap::new();
    for (index, tensor) in manifest.tensors.iter().enumerate() {
        tensor.validate(manifest.asset_bytes, limits)?;
        if !required_tensor_names.contains(tensor.name.as_str()) {
            return Err(ExecutorError::UnexpectedTensor);
        }
        if tensor_indexes.insert(tensor.name.as_str(), index).is_some() {
            return Err(ExecutorError::DuplicateName);
        }
    }
    if manifest.tensors.len() != required_tensor_names.len() {
        return Err(ExecutorError::UnexpectedTensor);
    }
    for tensor_name in &required_tensor_names {
        if !tensor_indexes.contains_key(tensor_name) {
            return Err(ExecutorError::MissingRequiredTensor);
        }
    }
    for (left_index, left) in manifest.tensors.iter().enumerate() {
        if left.bytes.len == 0 {
            continue;
        }
        let left_end = left.bytes.end()?;
        for right in manifest.tensors.iter().skip(left_index + 1) {
            if right.bytes.len == 0 {
                continue;
            }
            let right_end = right.bytes.end()?;
            if left.bytes.offset < right_end && right.bytes.offset < left_end {
                return Err(ExecutorError::InvalidLayout(
                    "physical tensor byte ranges overlap",
                ));
            }
        }
    }

    let mut bindings = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let tensor_index = tensor_indexes
            .get(requirement.tensor_name.as_str())
            .copied()
            .ok_or(ExecutorError::MissingRequiredTensor)?;
        let tensor = &manifest.tensors[tensor_index];
        if tensor.dtype != requirement.dtype || tensor.shape != requirement.shape {
            return Err(ExecutorError::InvalidShape(
                "tensor does not match required shape or dtype",
            ));
        }
        bindings.push(TensorBinding {
            role: requirement.role.clone(),
            tensor_index,
        });
    }
    Ok(bindings)
}

/// A vocabulary index accepted by a loaded decoder.
pub type TokenId = u32;

/// A physical token chunk plus an optional logical-validity mask.
#[derive(Clone, Copy, Debug)]
pub struct TokenChunk<'a> {
    pub ids: &'a [TokenId],
    pub valid: Option<&'a [bool]>,
}

impl<'a> TokenChunk<'a> {
    #[must_use]
    pub const fn all(ids: &'a [TokenId]) -> Self {
        Self { ids, valid: None }
    }

    #[must_use]
    pub const fn masked(ids: &'a [TokenId], valid: &'a [bool]) -> Self {
        Self {
            ids,
            valid: Some(valid),
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.valid.is_some_and(|mask| mask.len() != self.ids.len()) {
            return Err(ExecutorError::InvalidArgument(
                "token validity mask length differs from token IDs",
            ));
        }
        Ok(())
    }

    pub fn is_valid(self, index: usize) -> Result<bool> {
        if index >= self.ids.len() {
            return Err(ExecutorError::OutOfBounds(
                "token index exceeds physical chunk",
            ));
        }
        match self.valid {
            None => Ok(true),
            Some(mask) => mask
                .get(index)
                .copied()
                .ok_or(ExecutorError::InvalidArgument(
                    "token validity mask length differs from token IDs",
                )),
        }
    }
}

/// A complete conditional candidate score.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScore {
    pub candidate_index: usize,
    pub token_count: usize,
    /// Finite f32 logits normalize and accumulate in f64 before this final f32 transport cast.
    pub log_probability: f32,
}

/// Async-compatible token inference contract. No universal blocking, Send, or Sync bound applies.
pub trait TokenExecutor {
    type Prefix;
    type Prefill: InferenceCompletion<Output = Self::Prefix>;
    /// A successful append publishes a new immutable prefix. The input snapshot is never
    /// mutated: callers replace it only after this completion is ready.
    type Append: InferenceCompletion<Output = Self::Prefix>;
    /// A fork copies backend-resident state asynchronously before publishing an independent
    /// prefix snapshot.
    type Fork: InferenceCompletion<Output = Self::Prefix>;
    type Scores: InferenceCompletion<Output = Vec<CandidateScore>>;
    type Logits: InferenceCompletion<Output = Vec<f32>>;

    fn prefill(&mut self, input: TokenChunk<'_>) -> Result<Self::Prefill>;
    fn append_known(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
    ) -> Result<Self::Append>;
    /// The greedy next-token id resolved when `prefix` was published, if it
    /// carries a logits boundary. Implementations resolve the argmax during
    /// the publish completion itself, so this accessor costs no readback and
    /// lets callers inspect the sampled token before deciding to append it.
    fn sampled_token(&mut self, prefix: &Self::Prefix) -> Result<Option<TokenId>>;
    /// Append the token currently reported by `sampled_token`. The sampled id
    /// stays backend-resident through the embedding gather, so greedy decode
    /// never reads a full logits row to the host. Takes the prefix by value:
    /// when the caller hands over the last reference, implementations may
    /// reuse its cache storage in place instead of deep-copying it. Returns
    /// `InvalidArgument` when the prefix has no resolved greedy sample.
    fn append_argmax(&mut self, prefix: Self::Prefix) -> Result<Self::Append>;
    /// `prefill` whose greedy sample is constrained to `mask`: bit `i` of the
    /// bitset (LSB-first u64 words, `ceil(vocab / 64)` long) marks an allowed
    /// token id. Implementations that cannot constrain report `Unsupported`
    /// instead of silently dropping the mask.
    fn prefill_masked(
        &mut self,
        _input: TokenChunk<'_>,
        _mask: Rc<[u64]>,
    ) -> Result<Self::Prefill> {
        Err(ExecutorError::Unsupported(
            "masked prefill is not implemented for this executor",
        ))
    }
    /// `append_argmax` whose next greedy sample is constrained to `mask`
    /// (same bitset layout as `prefill_masked`). The appended token itself
    /// was already sampled under the previous step's mask.
    fn append_argmax_masked(
        &mut self,
        _prefix: Self::Prefix,
        _mask: Rc<[u64]>,
    ) -> Result<Self::Append> {
        Err(ExecutorError::Unsupported(
            "masked append_argmax is not implemented for this executor",
        ))
    }
    fn fork(&mut self, prefix: &Self::Prefix) -> Result<Self::Fork>;
    fn next_logits(&mut self, prefix: &Self::Prefix) -> Result<Self::Logits>;
    fn score_candidates(
        &mut self,
        prefix: &Self::Prefix,
        candidates: &[&[TokenId]],
    ) -> Result<Self::Scores>;
}

/// Single-pass structured choice extension: reads back only the logits of
/// caller-selected one-token ids instead of the full vocabulary row. A
/// successful `choice_logits` completion returns the selected logits in
/// caller order, preserving duplicates, with exactly `token_ids.len()`
/// values. Implementations reject an empty selector list, ids outside the
/// vocabulary, and prefixes that carry no logits boundary.
pub trait TokenChoiceExecutor: TokenExecutor {
    type ChoiceLogits: InferenceCompletion<Output = Vec<f32>>;
    type ChoicePrefill: InferenceCompletion<Output = Vec<f32>>;

    fn choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoiceLogits>;

    /// Prefill `input` as one single-sequence pass and resolve to the
    /// `token_ids` logits at the final position in caller order, preserving
    /// duplicates, with exactly `token_ids.len()` finite values. No prefix is
    /// published and no greedy sample is computed. Implementations reject an
    /// empty accepted prompt, an empty selector list, and ids outside the
    /// vocabulary.
    fn prefill_choice_logits(
        &mut self,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> Result<Self::ChoicePrefill>;
}

/// Per-step decode constraint driven by a generation loop: supplies the
/// allowed-token bitset before each sampled step and observes each emitted
/// id so the constraint can advance its own state. Bitsets are one bit per
/// token id, LSB-first u64 words (`ceil(vocab / 64)` long), matching the
/// `argmax_masked`/`prefill_masked`/`append_argmax_masked` layout.
pub trait DecodeConstraint {
    /// Allowed ids for the upcoming sample.
    fn allowed(&mut self) -> Rc<[u64]>;
    /// Records an emitted token. Called when the id is committed, after the
    /// decode checks pass and before the append that produced it publishes.
    fn advance(&mut self, token: TokenId);
}

/// Common explicit completion state used by immediate portable adapters and tests.
#[derive(Debug)]
pub struct ReadyCompletion<T> {
    result: Option<Result<T>>,
}

impl<T> ReadyCompletion<T> {
    #[must_use]
    pub fn new(result: Result<T>) -> Self {
        Self {
            result: Some(result),
        }
    }
}

impl<T> InferenceCompletion for ReadyCompletion<T> {
    type Output = T;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.result.is_some() {
            self.result = Some(Err(ExecutorError::Cancelled));
            Ok(())
        } else {
            Err(ExecutorError::CompletionConsumed)
        }
    }
}

/// In-memory portable asset provider suitable for embedded and WASM callers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAssetProvider {
    bytes: Vec<u8>,
    max_read_bytes: u64,
}

impl MemoryAssetProvider {
    #[must_use]
    pub fn new(bytes: Vec<u8>, max_read_bytes: u64) -> Self {
        Self {
            bytes,
            max_read_bytes,
        }
    }
}

/// Immediate completion returned by the in-memory asset provider.
#[derive(Debug)]
pub struct MemoryAssetRead {
    result: Option<Result<AssetBytes>>,
}

impl AssetProvider for MemoryAssetProvider {
    type Read = MemoryAssetRead;

    fn read_range(&mut self, range: ByteRange) -> Result<Self::Read> {
        range.validate_within(
            u64::try_from(self.bytes.len())
                .map_err(|_| ExecutorError::Overflow("asset length exceeds u64"))?,
        )?;
        if range.len > self.max_read_bytes {
            return Err(ExecutorError::ResourceLimit(
                "asset read exceeds configured byte limit",
            ));
        }
        let start = usize::try_from(range.offset)
            .map_err(|_| ExecutorError::Overflow("asset offset exceeds usize"))?;
        let end = usize::try_from(range.end()?)
            .map_err(|_| ExecutorError::Overflow("asset end exceeds usize"))?;
        Ok(MemoryAssetRead {
            result: Some(Ok(AssetBytes::new(self.bytes[start..end].to_vec()))),
        })
    }
}

impl InferenceCompletion for MemoryAssetRead {
    type Output = AssetBytes;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.result.is_some() {
            self.result = Some(Err(ExecutorError::Cancelled));
            Ok(())
        } else {
            Err(ExecutorError::CompletionConsumed)
        }
    }
}
