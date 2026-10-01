//! Segment-isolated, complete-sequence encoder operations.
use super::{
    Kernel, WgpuBackend, WgpuBuffer, device::PooledBuf, element_groups, flat_grid, param32, params,
};
use minifield_engine_api::{
    EncoderOps, EncoderSegments, ExecutorError, GatedShortConvSpec, GqaSpec, Result, Shape,
};

impl WgpuBackend {
    fn stage_segments(&self, segments: &EncoderSegments) -> Result<PooledBuf> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                segments
                    .ids()
                    .len()
                    .checked_mul(4)
                    .ok_or(ExecutorError::Overflow("segment bytes overflow usize"))?,
            )
            .map_err(|_| ExecutorError::ResourceLimit("segment staging allocation failed"))?;
        for id in segments.ids() {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        let buffer = self.device.alloc_storage(
            u64::try_from(bytes.len())
                .map_err(|_| ExecutorError::Overflow("segment bytes overflow u64"))?,
        )?;
        self.device.queue.write_buffer(&buffer.buffer, 0, &bytes);
        Ok(buffer)
    }
}

impl EncoderOps for WgpuBackend {
    fn bidirectional_gqa(
        &self,
        output: &mut WgpuBuffer,
        query: &WgpuBuffer,
        key: &WgpuBuffer,
        value: &WgpuBuffer,
        segments: &EncoderSegments,
        spec: GqaSpec,
    ) -> Result<()> {
        for buffer in [query, key, value] {
            self.check_f32_buffer(buffer)?;
        }
        let tokens = spec
            .query_heads()
            .validate_packed(query.descriptor.layout.shape())?;
        for buffer in [key, value] {
            if spec
                .key_value_heads()
                .validate_packed(buffer.descriptor.layout.shape())?
                != tokens
            {
                return Err(ExecutorError::InvalidShape(
                    "encoder GQA token counts differ",
                ));
            }
        }
        segments.validate_tokens(tokens)?;
        self.check_output_shape(output, query.descriptor.layout.shape())?;
        if tokens == 0 {
            return Ok(());
        }
        let heads = u64::from(spec.query_heads().heads());
        let dim = u64::from(spec.query_heads().head_dim());
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (dim as f32).sqrt();
        // Bound score storage independently of context length. Every block writes disjoint rows.
        let row_elements = heads
            .checked_mul(tokens)
            .ok_or(ExecutorError::Overflow("encoder score row size overflows"))?;
        let block = (16 * 1024 * 1024 / row_elements).clamp(1, tokens);
        let score_bytes = block
            .checked_mul(row_elements)
            .and_then(|n| n.checked_mul(4))
            .ok_or(ExecutorError::Overflow("encoder score bytes overflow"))?;
        let scores = self.device.alloc_storage(score_bytes)?;
        let segment_buffer = self.stage_segments(segments)?;
        let mut base = 0;
        while base < tokens {
            let count = block.min(tokens - base);
            let groups = count
                .checked_mul(heads)
                .ok_or(ExecutorError::Overflow("encoder GQA grid overflows"))?;
            self.device.dispatch(
                Kernel::EncoderGqa,
                &[
                    query.wgpu_buffer()?,
                    key.wgpu_buffer()?,
                    value.wgpu_buffer()?,
                    &segment_buffer.buffer,
                    &scores.buffer,
                    output.wgpu_buffer()?,
                ],
                &params(&[
                    param32(heads)?,
                    param32(dim)?,
                    param32(tokens)?,
                    spec.group_size(),
                    param32(base)?,
                    scale.to_bits(),
                    param32(groups)?,
                    0,
                ]),
                flat_grid(groups)?,
            )?;
            base += count;
        }
        Ok(())
    }

    fn centered_gated_convolution(
        &self,
        output: &mut WgpuBuffer,
        projection: &WgpuBuffer,
        kernel: &WgpuBuffer,
        segments: &EncoderSegments,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_f32_buffer(projection)?;
        self.check_f32_buffer(kernel)?;
        let hidden = u64::from(spec.hidden());
        let width = u64::from(spec.width());
        let shape = projection.descriptor.layout.shape();
        if shape.rank() != 2
            || shape.dim(1)?
                != hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
                    "encoder projection width overflows",
                ))?
        {
            return Err(ExecutorError::InvalidShape(
                "encoder projection must have shape [tokens,3*hidden]",
            ));
        }
        let tokens = shape.dim(0)?;
        segments.validate_tokens(tokens)?;
        self.check_output_shape(output, Shape::new(&[tokens, hidden])?)?;
        if kernel.descriptor.layout.shape() != Shape::new(&[hidden, width])? {
            return Err(ExecutorError::InvalidShape(
                "encoder convolution kernel has wrong shape",
            ));
        }
        if tokens == 0 {
            return Ok(());
        }
        let segment_buffer = self.stage_segments(segments)?;
        self.device.dispatch(
            Kernel::EncoderConv,
            &[
                projection.wgpu_buffer()?,
                kernel.wgpu_buffer()?,
                &segment_buffer.buffer,
                output.wgpu_buffer()?,
            ],
            &params(&[param32(tokens)?, param32(hidden)?, param32(width)?, 0]),
            flat_grid(element_groups(tokens.checked_mul(hidden).ok_or(
                ExecutorError::Overflow("encoder convolution grid overflows"),
            )?))?,
        )
    }
}
