// Packed weight decoding and projection operations.

use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{ExecutorError, OperationKind, Result, Shape, TokenIds};

impl CpuBackend {
    /// Gather canonical ternary, NF4, or signed INT8 rows into F32 [ids, k].
    /// Every format applies one scale per 128 weights; codes width chooses
    /// the 2-bit, 4-bit, or signed-byte decoder.
    #[allow(clippy::needless_pass_by_value)]
    pub fn packed_gather_rows(
        &self,
        output: &mut CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedGatherRows)?;
        let (rows, inner, format) = self.check_packed_operands(codes, scales)?;
        self.check_f32_buffer(output)?;
        let ids = self.resolve_token_ids(&ids)?;
        let output_shape = Shape::new(&[
            u64::try_from(ids.len())
                .map_err(|_| ExecutorError::Overflow("id count overflows u64"))?,
            u64::try_from(inner).map_err(|_| ExecutorError::Overflow("width overflows u64"))?,
        ])?;
        if output.descriptor.layout.shape() != output_shape {
            return Err(ExecutorError::InvalidShape(
                "packed gather output layout differs from required shape",
            ));
        }
        let code_width = inner / format.weights_per_byte();
        for (destination_row, id) in ids.iter().copied().enumerate() {
            let Some(id) = id else {
                output.values[destination_row * inner..(destination_row + 1) * inner]
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
            let code_start = source_row
                .checked_mul(code_width)
                .ok_or(ExecutorError::Overflow(
                    "gather code offset overflows usize",
                ))?;
            let scale_start =
                source_row
                    .checked_mul(inner / 128)
                    .ok_or(ExecutorError::Overflow(
                        "gather scale offset overflows usize",
                    ))?;
            let destination_start =
                destination_row
                    .checked_mul(inner)
                    .ok_or(ExecutorError::Overflow(
                        "gather destination offset overflows usize",
                    ))?;
            for index in 0..inner {
                let group = index / 128;
                let within = index % 128;
                let weight = match format {
                    minifield_kernels_simd::PackedWeightFormat::TernaryV1 => {
                        let code = (codes.bytes[code_start + group * 32 + within / 4]
                            >> (2 * (within % 4)))
                            & 0x3;
                        f32::from(code) - 1.0
                    }
                    minifield_kernels_simd::PackedWeightFormat::Int8V1 => {
                        f32::from(i8::from_ne_bytes([codes.bytes[code_start + index]]))
                    }
                    minifield_kernels_simd::PackedWeightFormat::Nf4V1 => {
                        let byte = codes.bytes[code_start + group * 64 + within / 2];
                        let nibble = if within % 2 == 0 {
                            byte & 0x0F
                        } else {
                            byte >> 4
                        };
                        minifield_kernels_simd::NF4_LEVELS[usize::from(nibble)]
                    }
                };
                output.values[destination_start + index] =
                    weight * scales.values[scale_start + group];
            }
        }
        Ok(())
    }

    /// Packed linear: input [m, k] times decoded weight [n, k].
    /// Group-dot kernels accumulate products within each 128-weight group,
    /// then apply the scale. F32 reduction order can differ from dense linear.
    pub fn packed_linear(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinear)?;
        self.check_f32_buffer(input)?;
        let (output_width, inner, format) = self.check_packed_operands(codes, scales)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed linear input must be rank two",
            ));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("packed linear row count exceeds usize"))?;
        let input_inner = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("packed linear width exceeds usize"))?;
        if input_inner != inner {
            return Err(ExecutorError::InvalidShape(
                "packed linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[
            u64::try_from(rows)
                .map_err(|_| ExecutorError::Overflow("packed linear rows overflow u64"))?,
            u64::try_from(output_width)
                .map_err(|_| ExecutorError::Overflow("packed linear width overflows u64"))?,
        ])?;
        self.check_output_shape(output, output_shape)?;
        let code_width = inner / format.weights_per_byte();
        let groups = inner / 128;
        let row_dot = match format {
            minifield_kernels_simd::PackedWeightFormat::TernaryV1 => {
                minifield_kernels_simd::ternary_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Nf4V1 => {
                minifield_kernels_simd::nf4_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Int8V1 => {
                minifield_kernels_simd::int8_row_dot
            }
        };
        for row in 0..rows {
            for column in 0..output_width {
                let code_start = column
                    .checked_mul(code_width)
                    .ok_or(ExecutorError::Overflow(
                        "packed weight offset overflows usize",
                    ))?;
                let scale_start = column.checked_mul(groups).ok_or(ExecutorError::Overflow(
                    "packed scale offset overflows usize",
                ))?;
                let input_start = row.checked_mul(inner).ok_or(ExecutorError::Overflow(
                    "packed input offset overflows usize",
                ))?;
                // Packed group dots apply scales after their within-group reduction.
                // Their F32 rounding can differ from decoded dense arithmetic.
                let accumulator = row_dot(
                    &codes.bytes[code_start..code_start + code_width],
                    &scales.values[scale_start..scale_start + groups],
                    &input.values[input_start..input_start + inner],
                );
                if !accumulator.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "packed linear projection produced a non-finite value",
                    ));
                }
                output.values[row * output_width + column] = accumulator;
            }
        }
        Ok(())
    }

    /// Validate packed weight operands and return (rows, inner weight width,
    /// stream format). `scales` must be f32 [rows, k/128] for every packed
    /// format, so `k` comes from the scales width; the codes width then picks
    /// the decode unambiguously: `k/4` bytes is `minifield.ternary.v1`, `k/2`
    /// bytes is `minifield.nf4.v1`, and `k` bytes is signed INT8.
    fn check_packed_operands(
        &self,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
    ) -> Result<(usize, usize, minifield_kernels_simd::PackedWeightFormat)> {
        self.check_u8_buffer(codes)?;
        self.check_f32_buffer(scales)?;
        let codes_shape = codes.descriptor.layout.shape();
        let scales_shape = scales.descriptor.layout.shape();
        if codes_shape.rank() != 2 || scales_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed weight streams must be rank two",
            ));
        }
        let rows = usize::try_from(codes_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("packed row count exceeds usize"))?;
        let code_width = usize::try_from(codes_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("packed code width exceeds usize"))?;
        let groups = usize::try_from(scales_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("packed scale width exceeds usize"))?;
        let inner = groups.checked_mul(128).ok_or(ExecutorError::Overflow(
            "packed inner width overflows usize",
        ))?;
        let format = if code_width == inner / 4 {
            minifield_kernels_simd::PackedWeightFormat::TernaryV1
        } else if code_width == inner / 2 {
            minifield_kernels_simd::PackedWeightFormat::Nf4V1
        } else if code_width == inner {
            if codes.bytes.contains(&128) {
                return Err(ExecutorError::InvalidArgument("INT8 -128 code is reserved"));
            }
            minifield_kernels_simd::PackedWeightFormat::Int8V1
        } else {
            return Err(ExecutorError::InvalidShape(
                "packed code width must be inner/4 (ternary), inner/2 (nf4), or inner (int8)",
            ));
        };
        if scales_shape.dim(0)? != codes_shape.dim(0)? {
            return Err(ExecutorError::InvalidShape(
                "packed codes and scales disagree on row count",
            ));
        }
        Ok((rows, inner, format))
    }

    /// Two packed projections sharing one input, issued as two
    /// sequential `packed_linear` passes. Both weight sets must share one
    /// `[R, K]` shape so the outputs share `[T, R]`.
    #[allow(clippy::too_many_arguments)]
    pub fn packed_linear_pair(
        &self,
        out_a: &mut CpuBuffer,
        out_b: &mut CpuBuffer,
        input: &CpuBuffer,
        codes_a: &CpuBuffer,
        scales_a: &CpuBuffer,
        codes_b: &CpuBuffer,
        scales_b: &CpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinearPair)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
            || out_a.descriptor.layout.shape() != out_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair requires equal weight and output shapes",
            ));
        }
        // Both projections must pass admission before either output is written.
        self.check_f32_buffer(input)?;
        let (output_width, inner, _) = self.check_packed_operands(codes_a, scales_a)?;
        self.check_packed_operands(codes_b, scales_b)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed linear input must be rank two",
            ));
        }
        if usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("packed linear width exceeds usize"))?
            != inner
        {
            return Err(ExecutorError::InvalidShape(
                "packed linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[
            input_shape.dim(0)?,
            u64::try_from(output_width)
                .map_err(|_| ExecutorError::Overflow("packed linear width overflows u64"))?,
        ])?;
        self.check_output_shape(out_a, output_shape)?;
        self.check_output_shape(out_b, output_shape)?;
        self.packed_linear(out_a, input, codes_a, scales_a)?;
        self.packed_linear(out_b, input, codes_b, scales_b)
    }

    /// Paired packed projection with a fused `SwiGLU` epilogue:
    /// `silu(a[t, r]) * b[t, r]` where `a` and `b` are the two packed
    /// row dots against the shared input. The activation is applied only
    /// after both reductions complete.
    pub fn packed_swiglu_pair(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        codes_a: &CpuBuffer,
        scales_a: &CpuBuffer,
        codes_b: &CpuBuffer,
        scales_b: &CpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedSwigluPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_operands(codes_a, scales_a)?;
        self.check_packed_operands(codes_b, scales_b)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair input must be rank two",
            ));
        }
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("packed SwiGLU pair rows exceed usize"))?;
        if usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("packed SwiGLU pair width exceeds usize"))?
            != inner
        {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair input width differs from weight width",
            ));
        }
        self.check_output_shape(
            output,
            Shape::new(&[
                u64::try_from(rows)
                    .map_err(|_| ExecutorError::Overflow("packed SwiGLU pair rows overflow u64"))?,
                u64::try_from(output_width).map_err(|_| {
                    ExecutorError::Overflow("packed SwiGLU pair width overflows u64")
                })?,
            ])?,
        )?;
        let code_width = inner / format.weights_per_byte();
        let groups = inner / 128;
        let row_dot = match format {
            minifield_kernels_simd::PackedWeightFormat::TernaryV1 => {
                minifield_kernels_simd::ternary_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Nf4V1 => {
                minifield_kernels_simd::nf4_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Int8V1 => {
                minifield_kernels_simd::int8_row_dot
            }
        };
        for row in 0..rows {
            let input_start = row.checked_mul(inner).ok_or(ExecutorError::Overflow(
                "packed SwiGLU pair input offset overflows usize",
            ))?;
            let x = &input.values[input_start..input_start + inner];
            for column in 0..output_width {
                let code_start = column
                    .checked_mul(code_width)
                    .ok_or(ExecutorError::Overflow(
                        "packed SwiGLU pair code offset overflows usize",
                    ))?;
                let scale_start = column.checked_mul(groups).ok_or(ExecutorError::Overflow(
                    "packed SwiGLU pair scale offset overflows usize",
                ))?;
                let gate = row_dot(
                    &codes_a.bytes[code_start..code_start + code_width],
                    &scales_a.values[scale_start..scale_start + groups],
                    x,
                );
                let up = row_dot(
                    &codes_b.bytes[code_start..code_start + code_width],
                    &scales_b.values[scale_start..scale_start + groups],
                    x,
                );
                let sigmoid = 1.0_f32 / (1.0_f32 + (-gate).exp());
                let value = (gate * sigmoid) * up;
                if !value.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "packed SwiGLU pair produced a non-finite value",
                    ));
                }
                output.values[row * output_width + column] = value;
            }
        }
        Ok(())
    }

    /// Packed projection over an on-the-fly `SiLU(gate) * up`
    /// activation. The staged activation feeds the same SIMD row dot as
    /// `packed_linear`.
    pub fn packed_swiglu_linear(
        &self,
        output: &mut CpuBuffer,
        gate: &CpuBuffer,
        up: &CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedSwigluLinear)?;
        self.check_f32_buffer(gate)?;
        self.check_f32_buffer(up)?;
        let (output_width, inner, format) = self.check_packed_operands(codes, scales)?;
        let gate_shape = gate.descriptor.layout.shape();
        if gate_shape.rank() != 2 || up.descriptor.layout.shape() != gate_shape {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU gate and up layouts must match and be rank two",
            ));
        }
        let inner_u64 = u64::try_from(inner)
            .map_err(|_| ExecutorError::Overflow("packed SwiGLU inner width overflows u64"))?;
        if gate_shape.dim(1)? != inner_u64 {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU input width differs from weight width",
            ));
        }
        let rows = usize::try_from(gate_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("packed SwiGLU row count exceeds usize"))?;
        self.check_output_shape(
            output,
            Shape::new(&[
                gate_shape.dim(0)?,
                u64::try_from(output_width)
                    .map_err(|_| ExecutorError::Overflow("packed SwiGLU width overflows u64"))?,
            ])?,
        )?;
        let code_width = inner / format.weights_per_byte();
        let groups = inner / 128;
        let row_dot = match format {
            minifield_kernels_simd::PackedWeightFormat::TernaryV1 => {
                minifield_kernels_simd::ternary_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Nf4V1 => {
                minifield_kernels_simd::nf4_row_dot
            }
            minifield_kernels_simd::PackedWeightFormat::Int8V1 => {
                minifield_kernels_simd::int8_row_dot
            }
        };
        let mut activated = self.stage_f32(inner)?;
        activated.resize(inner, 0.0);
        for row in 0..rows {
            let base = row.checked_mul(inner).ok_or(ExecutorError::Overflow(
                "packed SwiGLU input offset overflows usize",
            ))?;
            for (index, cell) in activated.iter_mut().enumerate() {
                let g = gate.values[base + index];
                let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
                let value = (g * sigmoid) * up.values[base + index];
                if !value.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "packed SwiGLU activation is non-finite",
                    ));
                }
                *cell = value;
            }
            for column in 0..output_width {
                let code_start = column
                    .checked_mul(code_width)
                    .ok_or(ExecutorError::Overflow(
                        "packed weight offset overflows usize",
                    ))?;
                let scale_start = column.checked_mul(groups).ok_or(ExecutorError::Overflow(
                    "packed scale offset overflows usize",
                ))?;
                let accumulator = row_dot(
                    &codes.bytes[code_start..code_start + code_width],
                    &scales.values[scale_start..scale_start + groups],
                    &activated,
                );
                if !accumulator.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "packed SwiGLU projection produced a non-finite value",
                    ));
                }
                output.values[row * output_width + column] = accumulator;
            }
        }
        Ok(())
    }
}
