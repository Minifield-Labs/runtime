// Dense copies, gathers, arithmetic, activations, and selectors.

use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{
    BufferAccess, ExecutorError, OperationKind, RectCopy2d, Result, Shape, TokenId, TokenIds,
};

impl CpuBackend {
    pub fn copy(&self, output: &mut CpuBuffer, input: &CpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::Copy)?;
        self.check_f32_buffer(input)?;
        self.check_output_shape(output, input.descriptor.layout.shape())?;
        output.values.copy_from_slice(&input.values);
        Ok(())
    }

    /// Resolve gather row selectors into per-row `Option<u32>` values.
    ///
    /// Host ids pass through unchanged. Device ids are f32 buffer values
    /// produced by `argmax`: each must be finite, integral, and within
    /// `[0, 2^24)`; anything else resolves to `None`, which callers turn into
    /// a NaN-filled output row so invalid device ids propagate as poisoned
    /// activations rather than host-side panics.
    pub(super) fn resolve_token_ids(&self, ids: &TokenIds<'_, Self>) -> Result<Vec<Option<u32>>> {
        match ids {
            TokenIds::Host(ids) => Ok(ids.iter().map(|id| Some(*id)).collect()),
            TokenIds::Device(buffer) => {
                self.check_f32_buffer(buffer)?;
                let shape = buffer.descriptor.layout.shape();
                if shape.rank() != 1 {
                    return Err(ExecutorError::InvalidShape(
                        "device token-id buffer must be rank one",
                    ));
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Ok(buffer
                    .values
                    .iter()
                    .map(|value| {
                        if value.is_finite()
                            && *value >= 0.0
                            && value.fract() == 0.0
                            && *value < 16_777_216.0
                        {
                            Some(*value as u32)
                        } else {
                            None
                        }
                    })
                    .collect())
            }
        }
    }

    /// Gather selected rows from a contiguous [rows, columns] f32 table.
    #[allow(clippy::needless_pass_by_value)]
    pub fn gather_rows(
        &self,
        output: &mut CpuBuffer,
        table: &CpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherRows)?;
        self.check_f32_buffer(table)?;
        let table_shape = table.descriptor.layout.shape();
        if table_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("gather table must be rank two"));
        }
        let rows = usize::try_from(table_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("row count exceeds usize"))?;
        let columns = usize::try_from(table_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("column count exceeds usize"))?;
        let ids = self.resolve_token_ids(&ids)?;
        let output_shape = Shape::new(&[
            u64::try_from(ids.len())
                .map_err(|_| ExecutorError::Overflow("id count overflows u64"))?,
            u64::try_from(columns)
                .map_err(|_| ExecutorError::Overflow("column count overflows u64"))?,
        ])?;
        self.check_output_shape(output, output_shape)?;
        for (destination_row, id) in ids.iter().copied().enumerate() {
            let Some(id) = id else {
                output.values[destination_row * columns..(destination_row + 1) * columns]
                    .fill(f32::NAN);
                continue;
            };
            let source_row = usize::try_from(id)
                .map_err(|_| ExecutorError::OutOfBounds("gather identifier exceeds usize"))?;
            if source_row >= rows {
                return Err(ExecutorError::OutOfBounds(
                    "gather identifier exceeds row count",
                ));
            }
            let source_start = source_row
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "gather source offset overflows usize",
                ))?;
            let destination_start =
                destination_row
                    .checked_mul(columns)
                    .ok_or(ExecutorError::Overflow(
                        "gather destination offset overflows usize",
                    ))?;
            output.values[destination_start..destination_start + columns]
                .copy_from_slice(&table.values[source_start..source_start + columns]);
        }
        Ok(())
    }

    /// Gather selected columns from a contiguous [rows, width] f32 input:
    /// `output[r, k] = input[r, columns[k]]`. Caller order and duplicates are
    /// preserved; an empty `columns` is a valid no-op.
    pub fn gather_columns(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        columns: &[TokenId],
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherColumns)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "column gather input must be rank two",
            ));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("row count exceeds usize"))?;
        let width = input_shape.dim(1)?;
        for id in columns {
            if u64::from(*id) >= width {
                return Err(ExecutorError::OutOfBounds(
                    "column gather identifier exceeds input width",
                ));
            }
        }
        let count = u64::try_from(columns.len())
            .map_err(|_| ExecutorError::Overflow("column count overflows u64"))?;
        self.check_output_shape(output, Shape::new(&[input_shape.dim(0)?, count])?)?;
        let width = usize::try_from(width)
            .map_err(|_| ExecutorError::Overflow("input width exceeds usize"))?;
        for (k, id) in columns.iter().copied().enumerate() {
            let column = usize::try_from(id)
                .map_err(|_| ExecutorError::OutOfBounds("gather identifier exceeds usize"))?;
            for row in 0..rows {
                output.values[row * columns.len() + k] = input.values[row * width + column];
            }
        }
        Ok(())
    }

    pub fn add(&self, output: &mut CpuBuffer, left: &CpuBuffer, right: &CpuBuffer) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Add, |a, b| a + b)
    }

    pub fn multiply(
        &self,
        output: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
    ) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Multiply, |a, b| a * b)
    }

    fn elementwise(
        &self,
        output: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
        operation: OperationKind,
        function: impl Fn(f32, f32) -> f32,
    ) -> Result<()> {
        self.check_operation(operation)?;
        self.check_f32_buffer(left)?;
        self.check_f32_buffer(right)?;
        if left.descriptor.layout != right.descriptor.layout {
            return Err(ExecutorError::InvalidShape(
                "elementwise operands have different layouts",
            ));
        }
        self.check_output_shape(output, left.descriptor.layout.shape())?;
        for ((destination, lhs), rhs) in output
            .values
            .iter_mut()
            .zip(left.values.iter())
            .zip(right.values.iter())
        {
            let value = function(*lhs, *rhs);
            if !value.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "elementwise operation produced a non-finite value",
                ));
            }
            *destination = value;
        }
        Ok(())
    }

    /// Row-major linear projection: input [m, k] times weight [n, k] yields output [m, n].
    /// Accumulation is sequential f32 in increasing k order.
    pub fn linear(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::Linear)?;
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let weight_shape = weight.descriptor.layout.shape();
        if input_shape.rank() != 2 || weight_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "linear input and weight must be rank two",
            ));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("linear row count exceeds usize"))?;
        let inner = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("linear inner width exceeds usize"))?;
        let output_width = usize::try_from(weight_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("linear output width exceeds usize"))?;
        let weight_inner = usize::try_from(weight_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("linear weight width exceeds usize"))?;
        if inner != weight_inner {
            return Err(ExecutorError::InvalidShape(
                "linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[
            u64::try_from(rows).map_err(|_| ExecutorError::Overflow("linear rows overflow u64"))?,
            u64::try_from(output_width)
                .map_err(|_| ExecutorError::Overflow("linear width overflows u64"))?,
        ])?;
        self.check_output_shape(output, output_shape)?;
        for row in 0..rows {
            for column in 0..output_width {
                let mut accumulator = 0.0_f32;
                for index in 0..inner {
                    let input_offset = row
                        .checked_mul(inner)
                        .and_then(|value| value.checked_add(index))
                        .ok_or(ExecutorError::Overflow(
                            "linear input offset overflows usize",
                        ))?;
                    let weight_offset = column
                        .checked_mul(inner)
                        .and_then(|value| value.checked_add(index))
                        .ok_or(ExecutorError::Overflow(
                            "linear weight offset overflows usize",
                        ))?;
                    accumulator += input.values[input_offset] * weight.values[weight_offset];
                }
                if !accumulator.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "linear projection produced a non-finite value",
                    ));
                }
                let output_offset = row
                    .checked_mul(output_width)
                    .and_then(|value| value.checked_add(column))
                    .ok_or(ExecutorError::Overflow(
                        "linear output offset overflows usize",
                    ))?;
                output.values[output_offset] = accumulator;
            }
        }
        Ok(())
    }

    /// Copy a checked row-major rectangle between distinct rank-two CPU buffers.
    pub fn copy_rect_2d(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        self.check_operation(OperationKind::RectCopy2d)?;
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(output)?;
        if input.descriptor.allocation == output.descriptor.allocation {
            return Err(ExecutorError::Unsupported(
                "rectangular copy does not permit overlapping source and destination allocation",
            ));
        }
        if output.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument(
                "rectangular copy destination is read-only",
            ));
        }
        let source_shape = input.descriptor.layout.shape();
        let destination_shape = output.descriptor.layout.shape();
        rectangle.validate(source_shape, destination_shape)?;
        let source_width = usize::try_from(source_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("rectangular source width exceeds usize"))?;
        let destination_width = usize::try_from(destination_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("rectangular destination width exceeds usize"))?;
        let source_row = usize::try_from(rectangle.source_row())
            .map_err(|_| ExecutorError::Overflow("rectangular source row exceeds usize"))?;
        let source_column = usize::try_from(rectangle.source_column())
            .map_err(|_| ExecutorError::Overflow("rectangular source column exceeds usize"))?;
        let destination_row = usize::try_from(rectangle.destination_row())
            .map_err(|_| ExecutorError::Overflow("rectangular destination row exceeds usize"))?;
        let destination_column = usize::try_from(rectangle.destination_column())
            .map_err(|_| ExecutorError::Overflow("rectangular destination column exceeds usize"))?;
        let rows = usize::try_from(rectangle.rows())
            .map_err(|_| ExecutorError::Overflow("rectangular row count exceeds usize"))?;
        let columns = usize::try_from(rectangle.columns())
            .map_err(|_| ExecutorError::Overflow("rectangular column count exceeds usize"))?;
        for row in 0..rows {
            let source_start = source_row
                .checked_add(row)
                .and_then(|value| value.checked_mul(source_width))
                .and_then(|value| value.checked_add(source_column))
                .ok_or(ExecutorError::Overflow(
                    "rectangular source offset overflows usize",
                ))?;
            let destination_start = destination_row
                .checked_add(row)
                .and_then(|value| value.checked_mul(destination_width))
                .and_then(|value| value.checked_add(destination_column))
                .ok_or(ExecutorError::Overflow(
                    "rectangular destination offset overflows usize",
                ))?;
            let source_end = source_start
                .checked_add(columns)
                .ok_or(ExecutorError::Overflow(
                    "rectangular source end overflows usize",
                ))?;
            let destination_end =
                destination_start
                    .checked_add(columns)
                    .ok_or(ExecutorError::Overflow(
                        "rectangular destination end overflows usize",
                    ))?;
            output.values[destination_start..destination_end]
                .copy_from_slice(&input.values[source_start..source_end]);
        }
        Ok(())
    }

    /// Apply SiLU(gate) * up over matching contiguous f32 layouts.
    pub fn swiglu(&self, output: &mut CpuBuffer, gate: &CpuBuffer, up: &CpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::SwiGlu)?;
        self.check_f32_buffer(gate)?;
        self.check_f32_buffer(up)?;
        if gate.descriptor.layout != up.descriptor.layout {
            return Err(ExecutorError::InvalidShape(
                "SwiGLU gate and up layouts differ",
            ));
        }
        self.check_output_shape(output, gate.descriptor.layout.shape())?;
        for ((destination, gate_value), up_value) in
            output.values.iter_mut().zip(&gate.values).zip(&up.values)
        {
            let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate_value).exp());
            let result = (*gate_value * sigmoid) * *up_value;
            if !result.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "SwiGLU produced a non-finite value",
                ));
            }
            *destination = result;
        }
        Ok(())
    }

    /// Row-wise argmax over f32 `[T, V]` logits: `output[t]` is the index of the
    /// first strict maximum in row `t` as an exact f32 integer, or NaN when the
    /// row contains any non-finite element. `V` must be at most `1 << 24` so
    /// indices stay exactly representable in f32.
    pub fn argmax(&self, output: &mut CpuBuffer, input: &CpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::Argmax)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("argmax input must be rank two"));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("argmax row count exceeds usize"))?;
        let width = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("argmax width exceeds usize"))?;
        if width == 0 || width > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "argmax width must be in [1, 2^24] for exact f32 indices",
            ));
        }
        self.check_output_shape(output, Shape::new(&[input_shape.dim(0)?])?)?;
        #[allow(clippy::cast_precision_loss)]
        for row in 0..rows {
            let base = row * width;
            let mut best = f32::NEG_INFINITY;
            let mut best_index = 0_usize;
            let mut poisoned = false;
            for (index, value) in input.values[base..base + width].iter().enumerate() {
                if !value.is_finite() {
                    poisoned = true;
                } else if *value > best {
                    best = *value;
                    best_index = index;
                }
            }
            output.values[row] = if poisoned {
                f32::NAN
            } else {
                best_index as f32
            };
        }
        Ok(())
    }

    /// Masked row-wise argmax: only positions whose bit is set in `mask`
    /// (LSB-first u64 words, `ceil(width / 64)` long) are candidates.
    /// Masked-out values are skipped, so their NaN or infinity cannot poison
    /// the row; a non-finite value at an allowed position still does. A row
    /// with no allowed candidate yields NaN.
    pub fn argmax_masked(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        mask: &[u64],
    ) -> Result<()> {
        self.check_operation(OperationKind::Argmax)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "masked argmax input must be rank two",
            ));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("masked argmax row count exceeds usize"))?;
        let width = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("masked argmax width exceeds usize"))?;
        if width == 0 || width > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "masked argmax width must be in [1, 2^24] for exact f32 indices",
            ));
        }
        let words = width.div_ceil(64);
        if mask.len() != words {
            return Err(ExecutorError::InvalidArgument(
                "masked argmax mask length must be ceil(width / 64)",
            ));
        }
        self.check_output_shape(output, Shape::new(&[input_shape.dim(0)?])?)?;
        // Exact ties retain the first allowed index; approximate comparison would change ranking.
        #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
        for row in 0..rows {
            let base = row * width;
            let mut best = f32::NEG_INFINITY;
            let mut best_index = usize::MAX;
            let mut poisoned = false;
            for (index, value) in input.values[base..base + width].iter().enumerate() {
                if mask[index / 64] & (1_u64 << (index % 64)) == 0 {
                    continue;
                }
                if !value.is_finite() {
                    poisoned = true;
                } else if *value > best || (*value == best && index < best_index) {
                    best = *value;
                    best_index = index;
                }
            }
            output.values[row] = if poisoned || best_index == usize::MAX {
                f32::NAN
            } else {
                best_index as f32
            };
        }
        Ok(())
    }
}
