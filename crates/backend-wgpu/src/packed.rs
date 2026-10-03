//! Packed operation admission and dispatch.

use super::{
    CodeLayout, DType, ExecutorError, Kernel, OperationKind, PackedStreamFormat, Result, Shape,
    TokenIds, WgpuBackend, WgpuBuffer, element_groups, flat_grid, gemv_lanes, gemv_tiles, param32,
    params,
};

impl WgpuBackend {
    /// Validate packed weight operands and return (rows, inner weight width,
    /// stream format). `scales` is always `[rows, k/128]`, so `k` comes from
    /// the scales width; the codes width then selects the decode
    /// unambiguously: `k/4` bytes is `minifield.ternary.v1`, `k/2` bytes is
    /// `minifield.nf4.v1`, and `k` bytes is signed `minifield.int8.v1`.
    pub(super) fn check_packed_operands(
        &self,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<(u64, u64, PackedStreamFormat)> {
        if codes.code_layout != CodeLayout::Canonical {
            return Err(ExecutorError::InvalidArgument(
                "repacked codes cannot be consumed by a canonical packed kernel",
            ));
        }
        self.check_packed_layout(codes, scales)
    }

    pub(super) fn check_packed_layout(
        &self,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<(u64, u64, PackedStreamFormat)> {
        self.check_u8_buffer(codes)?;
        self.check_f32_buffer(scales)?;
        let codes_shape = codes.descriptor.layout.shape();
        let scales_shape = scales.descriptor.layout.shape();
        if codes_shape.rank() != 2 || scales_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed weight streams must be rank two",
            ));
        }
        let code_width = codes_shape.dim(1)?;
        let inner = scales_shape
            .dim(1)?
            .checked_mul(128)
            .ok_or(ExecutorError::Overflow("packed inner width overflows u64"))?;
        let format = if code_width == inner / 4 {
            PackedStreamFormat::TernaryV1
        } else if code_width == inner / 2 {
            PackedStreamFormat::Nf4V1
        } else if code_width == inner {
            PackedStreamFormat::Int8V1
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
        Ok((codes_shape.dim(0)?, inner, format))
    }

    /// Packed linear: input [m, k] times the dequantized weight [n, k]
    /// carried as canonical ternary, NF4, or signed INT8 codes and group-128
    /// scales. Dispatch selects GEMV or a shared-memory GEMM tile.
    #[allow(clippy::too_many_lines)] // Keep admission and the 3 dispatch parameter layouts together.
    pub fn packed_linear(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
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
        let rows = input_shape.dim(0)?;
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[rows, output_width])?;
        self.check_output_shape(output, output_shape)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            // sum over an empty axis is zero.
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let x = input.wgpu_buffer()?.clone();
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed linear output count overflows u64",
        ))?;
        // `x4` rebinds the activation as vec4 for 128-bit loads when the inner
        // dimension is 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        // Multi-token tiles amortize each decode across the input tile;
        // ternary/NF4 keep their qualified short-row kernels below 96 rows.
        // INT8 uses the 64x32 K16 GEMM baseline for every row count.
        if rows >= 96 || format == PackedStreamFormat::Int8V1 {
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedGemmNf4,
                PackedStreamFormat::Int8V1 => Kernel::PackedGemmInt8,
                PackedStreamFormat::TernaryV1 => Kernel::PackedGemmTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow("packed prefill grid overflows u64"))?;
            return self.device.dispatch(
                kernel,
                &[&destination, &x, &c, &s, &x],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed linear workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedGemvMtNf4,
                &[&destination, &x, &c, &s, &x],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemv,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemvNf4,
            PackedStreamFormat::Int8V1 => {
                return Err(ExecutorError::BackendFailure(
                    "signed INT8 must dispatch through its GEMM parameter layout",
                ));
            }
        };
        self.device.dispatch(
            kernel,
            &[&destination, &x, &c, &s, &x],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Packed gather: dequantize selected ternary, NF4, or signed INT8 rows (format
    /// inferred from the codes width) into an f32 [ids, k] output.
    #[allow(clippy::needless_pass_by_value)]
    pub fn packed_gather_rows(
        &self,
        output: &mut WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedGatherRows)?;
        let (rows, inner, format) = self.check_packed_operands(codes, scales)?;
        let (id_buffer, id_count, staged) = self.stage_token_ids(&ids, rows)?;
        let output_shape = Shape::new(&[id_count, inner])?;
        self.check_output_shape(output, output_shape)?;
        if id_count == 0 || inner == 0 {
            return Ok(());
        }
        let elements = id_count.checked_mul(inner).ok_or(ExecutorError::Overflow(
            "gather element count overflows u64",
        ))?;
        let groups = element_groups(if format == PackedStreamFormat::Int8V1 {
            elements / 4
        } else {
            elements
        });
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGather,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGatherNf4,
            PackedStreamFormat::Int8V1 => Kernel::PackedGatherInt8,
        };
        let result = self.device.dispatch(
            kernel,
            &[&destination, &id_buffer, &c, &s],
            &params(&[
                param32(id_count)?,
                param32(inner)?,
                param32(rows)?,
                f32::NAN.to_bits(),
            ]),
            flat_grid(groups)?,
        );
        if let Some(scratch) = staged {
            drop(scratch);
        }
        result
    }

    /// Paired packed linear over one shared input: workgroup (i, j)
    /// computes both output elements, halving dispatches for projection pairs
    /// that share an activation (K/V, gate/up).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn packed_linear_pair(
        &self,
        out_a: &mut WgpuBuffer,
        out_b: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinearPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_operands(codes_a, scales_a)?;
        // Equal code and scale widths imply the same packed stream format.
        self.check_packed_operands(codes_b, scales_b)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair input must be rank two",
            ));
        }
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        let output_shape = Shape::new(&[rows, output_width])?;
        self.check_output_shape(out_a, output_shape)?;
        self.check_output_shape(out_b, output_shape)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(out_a.wgpu_buffer()?, out_a.byte_len());
            self.device
                .record_clear(out_b.wgpu_buffer()?, out_b.byte_len());
            return Ok(());
        }
        let da = out_a.wgpu_buffer()?.clone();
        let db = out_b.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed linear pair output count overflows u64",
        ))?;
        // `x4` rebinds the activation as vec4 for 128-bit loads when the inner
        // dimension is 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        if rows >= 96 || format == PackedStreamFormat::Int8V1 {
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedGemmPairNf4,
                PackedStreamFormat::Int8V1 => Kernel::PackedGemmPairInt8,
                PackedStreamFormat::TernaryV1 => Kernel::PackedGemmPairTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "packed pair prefill grid overflows u64",
                ))?;
            return self.device.dispatch(
                kernel,
                &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed linear pair workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedGemvPairMtNf4,
                &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemvPair,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemvPairNf4,
            PackedStreamFormat::Int8V1 => {
                return Err(ExecutorError::BackendFailure(
                    "signed INT8 must dispatch through its GEMM parameter layout",
                ));
            }
        };
        self.device.dispatch(
            kernel,
            &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Packed linear over an on-the-fly `SiLU(gate) * up` activation:
    /// ternary, NF4, and signed INT8 decode from codes width. The down projection consumes
    /// the activation without a materialized intermediate tensor.
    #[allow(clippy::too_many_lines)] // Keep admission and finite dispatch choices together.
    pub fn packed_swiglu_linear(
        &self,
        output: &mut WgpuBuffer,
        gate: &WgpuBuffer,
        up: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
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
        if gate_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU input width differs from weight width",
            ));
        }
        let rows = gate_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let g = gate.wgpu_buffer()?.clone();
        let u = up.wgpu_buffer()?.clone();
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed SwiGLU output count overflows u64",
        ))?;
        // `gate4`/`up4` rebind the operands as vec4 when the inner dimension is
        // 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        if rows >= 96 || format == PackedStreamFormat::Int8V1 {
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedSwigluGemmNf4,
                PackedStreamFormat::Int8V1 => Kernel::PackedSwigluGemmInt8,
                PackedStreamFormat::TernaryV1 => Kernel::PackedSwigluGemmTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "packed SwiGLU prefill grid overflows u64",
                ))?;
            return self.device.dispatch(
                kernel,
                &[&destination, &g, &u, &c, &s, &g, &u],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed SwiGLU workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedSwigluGemvMtNf4,
                &[&destination, &g, &u, &c, &s, &g, &u],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedSwigluGemv,
            PackedStreamFormat::Nf4V1 => Kernel::PackedSwigluGemvNf4,
            PackedStreamFormat::Int8V1 => {
                return Err(ExecutorError::BackendFailure(
                    "signed INT8 must dispatch through its GEMM parameter layout",
                ));
            }
        };
        self.device.dispatch(
            kernel,
            &[&destination, &g, &u, &c, &s, &g, &u],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Paired packed projection with the `SwiGLU` epilogue inside the GEMM:
    /// `output[t, r] = silu(a[t, r]) * b[t, r]`. Each invocation already owns
    /// the finished gate and up fragments, so one hidden buffer replaces the
    /// separate gate/up materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn packed_swiglu_pair(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
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
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemmPairSwiglu,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemmPairSwigluNf4,
            PackedStreamFormat::Int8V1 => Kernel::PackedGemmPairSwigluInt8,
        };
        let columns = output_width.div_ceil(32);
        let groups = rows
            .div_ceil(64)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow("SwiGLU pair grid overflows u64"))?;
        self.device.dispatch(
            kernel,
            &[&destination, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }

    /// Whether this backend consumes LUT2-repacked ternary code streams via
    /// `packed_linear_lut2` and `packed_swiglu_pair_lut2`.
    #[must_use]
    pub fn supports_ternary_lut2(&self) -> bool {
        true
    }

    /// Rearrange a `minifield.ternary.v1` code stream into the LUT2 pair
    /// nibble layout, once at load. The returned buffer is only meaningful
    /// to the `*_lut2` ops; the raw stream stays resident for the short-row
    /// fallbacks that decode it directly.
    pub fn repack_ternary_lut2(&mut self, codes: &WgpuBuffer) -> Result<WgpuBuffer> {
        self.check_u8_buffer(codes)?;
        if codes.code_layout != CodeLayout::Canonical {
            return Err(ExecutorError::InvalidArgument(
                "LUT2 repack requires canonical codes",
            ));
        }
        let shape = codes.descriptor.layout.shape();
        if shape.rank() != 2 || shape.dim(1)? % 4 != 0 {
            return Err(ExecutorError::InvalidShape(
                "lut2 repack expects rank-two u8 codes with a word-aligned width",
            ));
        }
        let mut out = self.allocate_inner(shape, DType::U8, codes.class, false)?;
        let source = codes.wgpu_buffer()?.clone();
        let destination = out.wgpu_buffer()?.clone();
        let words = shape
            .element_count()?
            .checked_div(4)
            .ok_or(ExecutorError::Overflow("lut2 repack word count overflows"))?;
        self.device.dispatch(
            Kernel::RepackTernaryLut2,
            &[&source, &destination],
            &params(&[param32(words)?, 0, 0, 0]),
            flat_grid(element_groups(words))?,
        )?;
        out.code_layout = CodeLayout::TernaryLut2;
        Ok(out)
    }

    /// Packed linear over LUT2-repacked ternary codes through the 32x64
    /// grouped-lookup tile. Same operand contract as `packed_linear`; the
    /// codes buffer must be the `repack_ternary_lut2` image of a ternary
    /// stream, not the raw stream.
    pub fn packed_linear_lut2(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<()> {
        if codes.code_layout != CodeLayout::TernaryLut2 {
            return Err(ExecutorError::InvalidArgument(
                "LUT2 linear requires repacked codes",
            ));
        }
        self.check_operation(OperationKind::PackedLinear)?;
        self.check_f32_buffer(input)?;
        let (output_width, inner, format) = self.check_packed_layout(codes, scales)?;
        if format != PackedStreamFormat::TernaryV1 {
            return Err(ExecutorError::InvalidShape(
                "lut2 linear requires a ternary-width code stream",
            ));
        }
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 linear input must be rank two",
            ));
        }
        let rows = input_shape.dim(0)?;
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 linear input width differs from weight width",
            ));
        }
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let x = input.wgpu_buffer()?.clone();
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let columns = output_width.div_ceil(64);
        let groups = rows
            .div_ceil(32)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow("lut2 grid overflows u64"))?;
        self.device.dispatch(
            Kernel::PackedGemmTernaryLut2,
            &[&destination, &x, &c, &s, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }

    /// Paired LUT2 projection with the `SwiGLU` epilogue: one activation table
    /// per K16 tile feeds both the gate and up streams. Codes carry LUT2
    /// pair nibbles; scales remain per-128-group f32.
    #[allow(clippy::too_many_arguments)]
    pub fn packed_swiglu_pair_lut2(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
    ) -> Result<()> {
        if codes_a.code_layout != CodeLayout::TernaryLut2
            || codes_b.code_layout != CodeLayout::TernaryLut2
        {
            return Err(ExecutorError::InvalidArgument(
                "LUT2 pair requires two repacked streams",
            ));
        }
        self.check_operation(OperationKind::PackedSwigluPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_layout(codes_a, scales_a)?;
        self.check_packed_layout(codes_b, scales_b)?;
        if format != PackedStreamFormat::TernaryV1 {
            return Err(ExecutorError::InvalidShape(
                "lut2 swiglu pair requires ternary-width code streams",
            ));
        }
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair input must be rank two",
            ));
        }
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let columns = output_width.div_ceil(64);
        let groups = rows
            .div_ceil(32)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow(
                "lut2 SwiGLU pair grid overflows u64",
            ))?;
        self.device.dispatch(
            Kernel::PackedGemmPairSwigluLut2,
            &[&destination, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }
}
