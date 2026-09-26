//! Normalization operation admission and dispatch.

use super::{
    ExecutorError, Kernel, OperationKind, PackedHeadSpec, PooledBuf, Result, RotarySpec, Shape,
    WgpuBackend, WgpuBuffer, element_groups, flat_grid, param32, params,
};

impl WgpuBackend {
    /// Fused residual add plus row RMS norm: `sum = left + right` written
    /// beside `normed = rmsnorm(sum) * weight` in one workgroup per row.
    pub fn add_row_rms_norm(
        &self,
        sum: &mut WgpuBuffer,
        normed: &mut WgpuBuffer,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
        weight: &WgpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::AddRowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(left)?;
        self.check_f32_buffer(right)?;
        self.check_f32_buffer(weight)?;
        let input_shape = left.descriptor.layout.shape();
        if input_shape.rank() != 2 || right.descriptor.layout.shape() != input_shape {
            return Err(ExecutorError::InvalidShape(
                "add-norm inputs must share a rank-two layout",
            ));
        }
        let ncols = input_shape.dim(1)?;
        if weight.descriptor.layout.shape() != Shape::new(&[ncols])? {
            return Err(ExecutorError::InvalidShape(
                "add-norm weight must have one entry per column",
            ));
        }
        self.check_output_shape(sum, input_shape)?;
        self.check_output_shape(normed, input_shape)?;
        let nrows = input_shape.dim(0)?;
        if nrows == 0 || ncols == 0 {
            return Ok(());
        }
        let sum_buf = sum.wgpu_buffer()?.clone();
        let normed_buf = normed.wgpu_buffer()?.clone();
        let left_buf = left.wgpu_buffer()?.clone();
        let right_buf = right.wgpu_buffer()?.clone();
        let weight_buf = weight.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::AddNorm,
            &[&sum_buf, &normed_buf, &left_buf, &right_buf, &weight_buf],
            &params(&[param32(ncols)?, param32(nrows)?, epsilon.to_bits()]),
            flat_grid(nrows)?,
        )
    }

    /// Fused per-head RMS norm plus split-half rotary for query and key rows:
    /// one workgroup per [token, head] slice. The cos/sin table is the same
    /// host-computed layout `split_half_rotary` stages.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        query_out: &mut WgpuBuffer,
        key_out: &mut WgpuBuffer,
        query: &WgpuBuffer,
        key: &WgpuBuffer,
        query_weight: &WgpuBuffer,
        key_weight: &WgpuBuffer,
        positions: &[u64],
        rope: RotarySpec,
        key_value_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::QkNormRope)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(query)?;
        self.check_f32_buffer(key)?;
        self.check_f32_buffer(query_weight)?;
        self.check_f32_buffer(key_weight)?;
        if key_value_heads.head_dim() != rope.heads().head_dim() {
            return Err(ExecutorError::InvalidShape(
                "RoPE head dimensions must match for query and key",
            ));
        }
        let query_tokens = rope
            .heads()
            .validate_packed(query.descriptor.layout.shape())?;
        let key_tokens = key_value_heads.validate_packed(key.descriptor.layout.shape())?;
        if query_tokens != key_tokens {
            return Err(ExecutorError::InvalidShape(
                "RoPE query and key token counts differ",
            ));
        }
        let token_count = usize::try_from(query_tokens)
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds usize"))?;
        if positions.len() != token_count {
            return Err(ExecutorError::InvalidShape(
                "RoPE position count differs from token count",
            ));
        }
        self.check_output_shape(query_out, query.descriptor.layout.shape())?;
        self.check_output_shape(key_out, key.descriptor.layout.shape())?;
        let head_dim = u64::from(rope.heads().head_dim());
        if head_dim == 0 || head_dim > 512 {
            return Err(ExecutorError::Unsupported(
                "RoPE head dimension exceeds shared-memory bound",
            ));
        }
        let head_dim_u64 = u64::from(rope.heads().head_dim());
        for weight in [query_weight, key_weight] {
            if weight.descriptor.layout.shape() != Shape::new(&[head_dim_u64])? {
                return Err(ExecutorError::InvalidShape(
                    "RoPE norm weight must have head_dim entries",
                ));
            }
        }
        if query_tokens == 0 {
            return Ok(());
        }
        let table = self.upload_rope_table(positions, rope.heads().head_dim(), rope.theta())?;
        let q_heads = u64::from(rope.heads().heads());
        let kv_heads = u64::from(key_value_heads.heads());
        let heads_total = q_heads
            .checked_add(kv_heads)
            .ok_or(ExecutorError::Overflow("RoPE head count overflows u64"))?;
        let q_width = rope.heads().packed_width()?;
        let kv_width = key_value_heads.packed_width()?;
        let workgroups = query_tokens
            .checked_mul(heads_total)
            .ok_or(ExecutorError::Overflow(
                "RoPE workgroup count overflows u64",
            ))?;
        let t = table.buffer.clone();
        let qo = query_out.wgpu_buffer()?.clone();
        let ko = key_out.wgpu_buffer()?.clone();
        let q = query.wgpu_buffer()?.clone();
        let k = key.wgpu_buffer()?.clone();
        let qw = query_weight.wgpu_buffer()?.clone();
        let kw = key_weight.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::QkNormRope,
            &[&t, &qo, &ko, &q, &k, &qw, &kw],
            &params(&[
                param32(q_heads)?,
                param32(heads_total)?,
                param32(head_dim)?,
                param32(q_width)?,
                param32(kv_width)?,
                param32(query_tokens)?,
                epsilon.to_bits(),
            ]),
            flat_grid(workgroups)?,
        );
        drop(table);
        result
    }

    /// Host-computed split-half cos/sin table staged into a pooled scratch
    /// buffer. Layout: `[tokens * half]` cosines then `[tokens * half]` sines,
    /// evaluated at the same f64-to-f32 boundaries as the scalar reference.
    pub(super) fn upload_rope_table(
        &self,
        positions: &[u64],
        head_dim: u32,
        theta: f32,
    ) -> Result<PooledBuf> {
        let tokens = u64::try_from(positions.len())
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds u64"))?;
        let half = u64::from(head_dim) / 2;
        let half_usize = usize::try_from(half)
            .map_err(|_| ExecutorError::Overflow("RoPE half width exceeds usize"))?;
        let table_len = usize::try_from(
            tokens
                .checked_mul(half)
                .ok_or(ExecutorError::Overflow("RoPE table size overflows u64"))?,
        )
        .map_err(|_| ExecutorError::Overflow("RoPE table size exceeds usize"))?;
        let mut table = Vec::new();
        table
            .try_reserve_exact(
                table_len
                    .checked_mul(2)
                    .ok_or(ExecutorError::Overflow("RoPE table length overflows usize"))?,
            )
            .map_err(|_| ExecutorError::ResourceLimit("RoPE table allocation failed"))?;
        table.resize(table_len * 2, 0.0_f32);
        for (token, position) in positions.iter().copied().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_f32 = position as f32;
            if !position_f32.is_finite() {
                return Err(ExecutorError::InvalidArgument(
                    "RoPE position is not representable as finite f32",
                ));
            }
            for column in 0..half_usize {
                #[allow(clippy::cast_precision_loss)]
                let exponent = -2.0_f64 * (column as f64) / f64::from(head_dim);
                #[allow(clippy::cast_possible_truncation)]
                let frequency = (f64::from(theta).powf(exponent)) as f32;
                let angle = position_f32 * frequency;
                #[allow(clippy::cast_possible_truncation)]
                let cos = f64::from(angle).cos() as f32;
                #[allow(clippy::cast_possible_truncation)]
                let sin = f64::from(angle).sin() as f32;
                table[token * half_usize + column] = cos;
                table[table_len + token * half_usize + column] = sin;
            }
        }
        let scratch = self.device.alloc_storage(
            u64::try_from(table.len())
                .map_err(|_| ExecutorError::Overflow("RoPE table length overflows u64"))?
                .checked_mul(4)
                .ok_or(ExecutorError::Overflow("RoPE table bytes overflow u64"))?,
        )?;
        let mut bytes = Vec::with_capacity(table.len() * 4);
        for value in &table {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.device.queue.write_buffer(&scratch.buffer, 0, &bytes);
        Ok(scratch)
    }

    /// Row RMS norm over a `[rows, width]` input and a `[width]` weight.
    pub fn row_rms_norm(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::RowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let weight_shape = weight.descriptor.layout.shape();
        if input_shape.rank() != 2 || weight_shape.rank() != 1 {
            return Err(ExecutorError::InvalidShape(
                "RMS input must be rank two and weight rank one",
            ));
        }
        let rows = input_shape.dim(0)?;
        let width = input_shape.dim(1)?;
        if width != weight_shape.dim(0)? {
            return Err(ExecutorError::InvalidShape(
                "RMS input width differs from weight width",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        self.rms_norm_dispatch(output, input, weight, rows, width, epsilon)
    }

    /// Head-local RMS normalization over packed `[tokens, heads * head_dim]`.
    pub fn head_rms_norm(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::RowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let tokens = heads.validate_packed(input_shape)?;
        if weight.descriptor.layout.shape() != Shape::new(&[u64::from(heads.head_dim())])? {
            return Err(ExecutorError::InvalidShape(
                "head RMS weight must have head_dim entries",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let rows = tokens
            .checked_mul(u64::from(heads.heads()))
            .ok_or(ExecutorError::Overflow("head RMS row count overflows u64"))?;
        self.rms_norm_dispatch(
            output,
            input,
            weight,
            rows,
            u64::from(heads.head_dim()),
            epsilon,
        )
    }

    pub(super) fn rms_norm_dispatch(
        &self,
        output: &WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        rows: u64,
        width: u64,
        epsilon: f32,
    ) -> Result<()> {
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let source = input.wgpu_buffer()?.clone();
        let alpha = weight.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::RmsNorm,
            &[&source, &destination, &alpha],
            &params(&[param32(width)?, param32(rows)?, epsilon.to_bits()]),
            flat_grid(rows)?,
        )
    }

    /// Split-half `RoPE` for packed `[tokens, heads * head_dim]` rows. The
    /// cos/sin table is computed on the host with the same f64-to-f32
    /// boundaries as the scalar reference and staged into a scratch buffer.
    pub fn split_half_rotary(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::Rotary)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        let tokens = spec.heads().validate_packed(input_shape)?;
        let token_count = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds usize"))?;
        if positions.len() != token_count {
            return Err(ExecutorError::InvalidShape(
                "RoPE position count differs from token count",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let heads = spec.heads().heads();
        let head_dim = spec.heads().head_dim();
        let half = u64::from(head_dim) / 2;
        if tokens == 0 || half == 0 {
            return Ok(());
        }
        let scratch = self.upload_rope_table(positions, head_dim, spec.theta())?;
        let groups = element_groups(
            tokens
                .checked_mul(u64::from(heads))
                .and_then(|value| value.checked_mul(half))
                .ok_or(ExecutorError::Overflow("RoPE element count overflows u64"))?,
        );
        let source = input.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::Rotary,
            &[&scratch.buffer, &source, &destination],
            &params(&[heads, param32(half)?, param32(tokens)?]),
            flat_grid(groups)?,
        );
        drop(scratch);
        result
    }
}
