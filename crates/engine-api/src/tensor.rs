//! Checked scalar formats, shapes, layouts, and buffer metadata.

use crate::{BackendIdentity, ExecutorError, Result};

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
