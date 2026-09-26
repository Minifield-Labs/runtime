//! Attention convolution operation admission and dispatch.

use super::{
    BufferAccess, ExecutorError, GatedShortConvSpec, GqaSpec, Kernel, MAX_WGS_PER_DIM,
    OperationKind, Result, Shape, WgpuBackend, WgpuBuffer, element_groups, flat_grid, param32,
    params,
};

impl WgpuBackend {
    /// Append packed K/V rows to the caches and calculate causal GQA output.
    /// One workgroup handles one query head per appended token; the cache
    /// append is a recorded copy inside the same batch.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn causal_gqa(
        &self,
        output: &mut WgpuBuffer,
        query: &WgpuBuffer,
        key: &WgpuBuffer,
        value: &WgpuBuffer,
        key_cache: &mut WgpuBuffer,
        value_cache: &mut WgpuBuffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GroupedQueryAttention)?;
        self.check_f32_buffer(query)?;
        self.check_f32_buffer(key)?;
        self.check_f32_buffer(value)?;
        self.check_f32_buffer(key_cache)?;
        self.check_f32_buffer(value_cache)?;
        if key_cache.descriptor.allocation == value_cache.descriptor.allocation {
            return Err(ExecutorError::InvalidArgument(
                "key and value caches must have distinct storage",
            ));
        }
        if key_cache.descriptor.access != BufferAccess::ReadWrite
            || value_cache.descriptor.access != BufferAccess::ReadWrite
        {
            return Err(ExecutorError::InvalidArgument("GQA caches are read-only"));
        }
        let query_shape = query.descriptor.layout.shape();
        let tokens = spec.query_heads().validate_packed(query_shape)?;
        if spec
            .key_value_heads()
            .validate_packed(key.descriptor.layout.shape())?
            != tokens
            || spec
                .key_value_heads()
                .validate_packed(value.descriptor.layout.shape())?
                != tokens
        {
            return Err(ExecutorError::InvalidShape(
                "GQA query, key, and value token counts differ",
            ));
        }
        self.check_output_shape(output, query_shape)?;
        let cache_shape = key_cache.descriptor.layout.shape();
        if cache_shape.rank() != 2
            || cache_shape.dim(1)? != spec.key_value_heads().packed_width()?
            || value_cache.descriptor.layout.shape() != cache_shape
        {
            return Err(ExecutorError::InvalidShape(
                "GQA caches must have matching [capacity, kv_heads * head_dim] shape",
            ));
        }
        let capacity = cache_shape.dim(0)?;
        let new_cache_len = cache_len
            .checked_add(tokens)
            .ok_or(ExecutorError::Overflow("GQA cache length overflows u64"))?;
        if *cache_len > capacity || new_cache_len > capacity {
            return Err(ExecutorError::OutOfBounds(
                "GQA append exceeds cache capacity",
            ));
        }
        if tokens == 0 {
            *cache_len = new_cache_len;
            return Ok(());
        }
        let query_heads = u64::from(spec.query_heads().heads());
        let head_dim = u64::from(spec.query_heads().head_dim());
        let group_size = u64::from(spec.group_size());
        let query_width = spec.query_heads().packed_width()?;
        let kv_width = spec.key_value_heads().packed_width()?;
        if query_heads > u64::from(MAX_WGS_PER_DIM) {
            return Err(ExecutorError::ResourceLimit(
                "GQA query head count exceeds dispatch grid",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        if !scale.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA scale is non-finite"));
        }

        let q = query.wgpu_buffer()?.clone();
        let k = key.wgpu_buffer()?.clone();
        let v = value.wgpu_buffer()?.clone();
        let kc = key_cache.wgpu_buffer()?.clone();
        let vc = value_cache.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        if tokens == 1 {
            // Score scratch: one row per query head, wide enough for the
            // visible length this token sees.
            let scores = self.device.alloc_storage(
                query_heads
                    .checked_mul(new_cache_len)
                    .and_then(|e| e.checked_mul(4))
                    .ok_or(ExecutorError::Overflow("GQA scratch bytes overflow u64"))?,
            )?;
            self.device.dispatch(
                Kernel::Gqa,
                &[&q, &k, &v, &kc, &vc, &scores.buffer, &destination],
                &params(&[
                    param32(group_size)?,
                    param32(head_dim)?,
                    param32(kv_width)?,
                    param32(query_width)?,
                    param32(*cache_len)?,
                    param32(0)?,
                    param32(new_cache_len)?,
                    scale.to_bits(),
                ]),
                (
                    u32::try_from(query_heads).map_err(|_| {
                        ExecutorError::ResourceLimit("GQA heads exceed dispatch grid")
                    })?,
                    1,
                    1,
                ),
            )?;
            drop(scores);
        } else {
            // Batched path: one dispatch covers a block of tokens, one score
            // row per (head, token) pair. The block caps the scratch at
            // ~64 MiB so very long prompts dispatch a few blocks instead of
            // one huge temporary.
            let budget = 16 * 1024 * 1024_u64; // score elements
            let block = (budget / (query_heads * new_cache_len)).clamp(1, tokens);
            let scores = self.device.alloc_storage(
                block
                    .checked_mul(query_heads)
                    .and_then(|e| e.checked_mul(new_cache_len))
                    .and_then(|e| e.checked_mul(4))
                    .ok_or(ExecutorError::Overflow("GQA scratch bytes overflow u64"))?,
            )?;
            let mut base = 0_u64;
            while base < tokens {
                let block_tokens = (tokens - base).min(block);
                self.device.dispatch(
                    Kernel::GqaBatch,
                    &[&q, &k, &v, &kc, &vc, &scores.buffer, &destination],
                    &params(&[
                        param32(group_size)?,
                        param32(head_dim)?,
                        param32(kv_width)?,
                        param32(query_width)?,
                        param32(*cache_len)?,
                        0,
                        param32(new_cache_len)?,
                        scale.to_bits(),
                        param32(query_heads)?,
                        param32(block_tokens)?,
                        param32(base)?,
                        0,
                    ]),
                    flat_grid(
                        block_tokens
                            .checked_mul(query_heads)
                            .ok_or(ExecutorError::Overflow("GQA batch grid overflows u64"))?,
                    )?,
                )?;
                base += block_tokens;
            }
            drop(scores);
        }
        // Append the new K/V rows to the caches inside the same batch. The GQA
        // kernel reads appended rows from k/v, not the cache tail, so ordering
        // of these copies is unobservable.
        let append_bytes = tokens
            .checked_mul(kv_width)
            .and_then(|value| value.checked_mul(4))
            .ok_or(ExecutorError::Overflow("GQA append bytes overflow u64"))?;
        let cache_offset = cache_len
            .checked_mul(kv_width)
            .and_then(|value| value.checked_mul(4))
            .ok_or(ExecutorError::Overflow("GQA cache offset overflows u64"))?;
        self.device
            .record_copy(&k, 0, &kc, cache_offset, append_bytes);
        self.device
            .record_copy(&v, 0, &vc, cache_offset, append_bytes);
        *cache_len = new_cache_len;
        Ok(())
    }

    /// Apply B*V gated short convolution and update the [width-1, hidden]
    /// rolling U history. The kernel pair stages `history ∥ (B*V)` in scratch,
    /// convolves depthwise, then assembles the new history in the tail of the
    /// same scratch so every buffer copy stays non-overlapping.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn gated_short_convolution(
        &self,
        output: &mut WgpuBuffer,
        projection: &WgpuBuffer,
        kernel: &WgpuBuffer,
        history: &mut WgpuBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatedShortConvolution)?;
        self.check_f32_buffer(projection)?;
        self.check_f32_buffer(kernel)?;
        self.check_f32_buffer(history)?;
        let hidden = u64::from(spec.hidden());
        let projection_width = hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
            "short convolution projection width overflows u64",
        ))?;
        let projection_shape = projection.descriptor.layout.shape();
        if projection_shape.rank() != 2 || projection_shape.dim(1)? != projection_width {
            return Err(ExecutorError::InvalidShape(
                "short convolution projection must be [tokens, 3 * hidden]",
            ));
        }
        let token_shape = Shape::new(&[projection_shape.dim(0)?, hidden])?;
        self.check_output_shape(output, token_shape)?;
        if kernel.descriptor.layout.shape()
            != Shape::new(&[u64::from(spec.hidden()), u64::from(spec.width())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution kernel must be [hidden, width]",
            ));
        }
        let history_rows = spec.history_rows()?;
        if history.descriptor.layout.shape()
            != Shape::new(&[history_rows, u64::from(spec.hidden())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution history must be [width - 1, hidden]",
            ));
        }
        if history.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument(
                "short convolution history is read-only",
            ));
        }
        let tokens = token_shape.dim(0)?;
        let width = u64::from(spec.width());
        if tokens == 0 || hidden == 0 {
            return Ok(());
        }
        let hist_elems = history_rows
            .checked_mul(hidden)
            .ok_or(ExecutorError::Overflow(
                "short conv history size overflows u64",
            ))?;
        if tokens == 1 && history_rows > 0 {
            // Decode step: one fused dispatch computes the gate, conv, and the
            // shifted history in a fresh buffer, then the history buffer swaps
            // storage. This avoids the staged same-buffer copies the general
            // path needs.
            let new_hist = self.device.alloc_storage(hist_elems.checked_mul(4).ok_or(
                ExecutorError::Overflow("short conv history bytes overflow u64"),
            )?)?;
            let hp = projection.wgpu_buffer()?.clone();
            let hk = kernel.wgpu_buffer()?.clone();
            let hh = history.wgpu_buffer()?.clone();
            let destination = output.wgpu_buffer()?.clone();
            if let Err(err) = self.device.dispatch(
                Kernel::ConvStep,
                &[&hh, &hp, &hk, &new_hist.buffer, &destination],
                &params(&[
                    param32(hidden)?,
                    param32(history_rows)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(hidden))?,
            ) {
                drop(new_hist);
                return Err(err);
            }
            if let Some(old) = history.storage.replace(new_hist) {
                drop(old);
            }
            return Ok(());
        }
        let token_elems = tokens.checked_mul(hidden).ok_or(ExecutorError::Overflow(
            "short conv element count overflows u64",
        ))?;
        let ext_elems = hist_elems
            .checked_add(token_elems)
            .ok_or(ExecutorError::Overflow("short conv extent overflows u64"))?;
        // u_ext layout: [history | u]. The assembled replacement history
        // stages in `tail` because wgpu forbids same-buffer copies.
        let u_ext = self.device.alloc_storage(ext_elems.checked_mul(4).ok_or(
            ExecutorError::Overflow("short conv scratch bytes overflow u64"),
        )?)?;
        let tail = if history_rows > 0 && tokens < history_rows {
            Some(self.device.alloc_storage(hist_elems.checked_mul(4).ok_or(
                ExecutorError::Overflow("short conv tail bytes overflow u64"),
            )?)?)
        } else {
            None
        };
        let hp = projection.wgpu_buffer()?.clone();
        let hk = kernel.wgpu_buffer()?.clone();
        let hh = history.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = (|| {
            self.device.dispatch(
                Kernel::ConvGate,
                &[&hh, &hp, &u_ext.buffer],
                &params(&[
                    param32(hist_elems)?,
                    param32(ext_elems)?,
                    param32(hidden)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(ext_elems))?,
            )?;
            self.device.dispatch(
                Kernel::Conv,
                &[&u_ext.buffer, &hk, &hp, &destination],
                &params(&[
                    param32(tokens)?,
                    param32(hidden)?,
                    param32(width)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(token_elems))?,
            )
        })();
        if result.is_err() {
            drop(u_ext);
            return result;
        }
        // Assemble the new history. u row r lives at u_ext[(hist_rows + r) *
        // hidden + c]. The tokens < history_rows path stages [keep | u] in
        // `tail` because wgpu forbids same-buffer copies.
        if history_rows > 0 {
            let hist_bytes = hist_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                "short conv history bytes overflow u64",
            ))?;
            if tokens >= history_rows {
                self.device.record_copy(
                    &u_ext.buffer,
                    token_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                        "short conv history offset overflows u64",
                    ))?,
                    &hh,
                    0,
                    hist_bytes,
                );
            } else {
                let Some(tail) = &tail else {
                    return Err(ExecutorError::BackendFailure(
                        "short conv tail scratch missing",
                    ));
                };
                let keep = history_rows - tokens;
                let keep_bytes = keep
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(4))
                    .ok_or(ExecutorError::Overflow(
                        "short conv shift bytes overflow u64",
                    ))?;
                self.device.record_copy(
                    &hh,
                    token_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                        "short conv shift offset overflows u64",
                    ))?,
                    &tail.buffer,
                    0,
                    keep_bytes,
                );
                self.device.record_copy(
                    &u_ext.buffer,
                    hist_bytes,
                    &tail.buffer,
                    keep_bytes,
                    token_elems
                        .checked_mul(4)
                        .ok_or(ExecutorError::Overflow("short conv u bytes overflow u64"))?,
                );
                self.device.record_copy(&tail.buffer, 0, &hh, 0, hist_bytes);
            }
        }
        drop(u_ext);
        if let Some(tail) = tail {
            drop(tail);
        }
        Ok(())
    }
}
