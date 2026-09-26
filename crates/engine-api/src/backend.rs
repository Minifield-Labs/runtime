//! Backend identity, capability declarations, and finite operation descriptors.

use std::rc::Rc;

use crate::{DType, DTypeSet, ExecutorError, Result, Shape};

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
    PackedSwigluPair,
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
