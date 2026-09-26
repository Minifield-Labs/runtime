// Attention and gated short convolution operations.

use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{
    ExecutorError, GatedShortConvSpec, GqaSpec, OperationKind, Result, Shape,
};

impl CpuBackend {
    /// Causal grouped-query attention over packed rows and explicit non-repeated KV caches.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn causal_gqa(
        &self,
        output: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        value: &CpuBuffer,
        key_cache: &mut CpuBuffer,
        value_cache: &mut CpuBuffer,
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
        let query_shape = query.descriptor.layout.shape();
        let tokens = spec.query_heads().validate_packed(query_shape)?;
        let key_shape = key.descriptor.layout.shape();
        let value_shape = value.descriptor.layout.shape();
        if spec.key_value_heads().validate_packed(key_shape)? != tokens
            || spec.key_value_heads().validate_packed(value_shape)? != tokens
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
        let token_count = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("GQA token count exceeds usize"))?;
        let query_heads = usize::try_from(spec.query_heads().heads())
            .map_err(|_| ExecutorError::Overflow("GQA query head count exceeds usize"))?;
        let head_dim = usize::try_from(spec.query_heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("GQA head dimension exceeds usize"))?;
        let group_size = usize::try_from(spec.group_size())
            .map_err(|_| ExecutorError::Overflow("GQA group size exceeds usize"))?;
        let query_width = usize::try_from(spec.query_heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("GQA query width exceeds usize"))?;
        let kv_width = usize::try_from(spec.key_value_heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("GQA key/value width exceeds usize"))?;
        let initial_len = usize::try_from(*cache_len)
            .map_err(|_| ExecutorError::Overflow("GQA cache length exceeds usize"))?;
        let staged_elements =
            token_count
                .checked_mul(query_width)
                .ok_or(ExecutorError::Overflow(
                    "GQA staging element count overflows usize",
                ))?;
        let mut staged = self.stage_f32(staged_elements)?;
        staged.resize(staged_elements, 0.0);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        if !scale.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA scale is non-finite"));
        }
        for token in 0..token_count {
            let visible = initial_len
                .checked_add(token)
                .and_then(|count| count.checked_add(1))
                .ok_or(ExecutorError::Overflow(
                    "GQA visible length overflows usize",
                ))?;
            for query_head in 0..query_heads {
                let kv_head = query_head / group_size;
                let query_start = token
                    .checked_mul(query_width)
                    .and_then(|offset| offset.checked_add(query_head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("GQA query offset overflows usize"))?;
                let mut maximum = f32::NEG_INFINITY;
                for key_index in 0..visible {
                    maximum = maximum.max(Self::gqa_score(
                        query,
                        key,
                        key_cache,
                        initial_len,
                        key_index,
                        query_start,
                        kv_head,
                        kv_width,
                        head_dim,
                        scale,
                    )?);
                }
                let mut denominator = 0.0_f32;
                for key_index in 0..visible {
                    let score = Self::gqa_score(
                        query,
                        key,
                        key_cache,
                        initial_len,
                        key_index,
                        query_start,
                        kv_head,
                        kv_width,
                        head_dim,
                        scale,
                    )?;
                    denominator += (score - maximum).exp();
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(ExecutorError::BackendFailure(
                        "GQA softmax denominator is invalid",
                    ));
                }
                for dimension in 0..head_dim {
                    let mut weighted = 0.0_f32;
                    for key_index in 0..visible {
                        let score = Self::gqa_score(
                            query,
                            key,
                            key_cache,
                            initial_len,
                            key_index,
                            query_start,
                            kv_head,
                            kv_width,
                            head_dim,
                            scale,
                        )?;
                        let probability = (score - maximum).exp() / denominator;
                        let value_at_key = if key_index < initial_len {
                            value_cache.values
                                [key_index * kv_width + kv_head * head_dim + dimension]
                        } else {
                            value.values[(key_index - initial_len) * kv_width
                                + kv_head * head_dim
                                + dimension]
                        };
                        weighted += probability * value_at_key;
                    }
                    if !weighted.is_finite() {
                        return Err(ExecutorError::BackendFailure("GQA output is non-finite"));
                    }
                    staged[token * query_width + query_head * head_dim + dimension] = weighted;
                }
            }
        }
        let appended = token_count
            .checked_mul(kv_width)
            .ok_or(ExecutorError::Overflow(
                "GQA cache copy count overflows usize",
            ))?;
        let cache_offset = initial_len
            .checked_mul(kv_width)
            .ok_or(ExecutorError::Overflow("GQA cache offset overflows usize"))?;
        key_cache.values[cache_offset..cache_offset + appended]
            .copy_from_slice(&key.values[..appended]);
        value_cache.values[cache_offset..cache_offset + appended]
            .copy_from_slice(&value.values[..appended]);
        output.values.copy_from_slice(&staged);
        *cache_len = new_cache_len;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_score(
        query: &CpuBuffer,
        key: &CpuBuffer,
        key_cache: &CpuBuffer,
        initial_len: usize,
        key_index: usize,
        query_start: usize,
        kv_head: usize,
        kv_width: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<f32> {
        let mut dot = 0.0_f32;
        for dimension in 0..head_dim {
            let key_value = if key_index < initial_len {
                key_cache.values[key_index * kv_width + kv_head * head_dim + dimension]
            } else {
                key.values[(key_index - initial_len) * kv_width + kv_head * head_dim + dimension]
            };
            dot += query.values[query_start + dimension] * key_value;
        }
        let score = dot * scale;
        if !score.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA score is non-finite"));
        }
        Ok(score)
    }

    /// Apply B*V gated short convolution and update its old-to-new rolling U history.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn gated_short_convolution(
        &self,
        output: &mut CpuBuffer,
        projection: &CpuBuffer,
        kernel: &CpuBuffer,
        history: &mut CpuBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatedShortConvolution)?;
        self.check_f32_buffer(projection)?;
        self.check_f32_buffer(kernel)?;
        self.check_f32_buffer(history)?;
        let hidden_u64 = u64::from(spec.hidden());
        let projection_shape = projection.descriptor.layout.shape();
        if projection_shape.rank() != 2
            || projection_shape.dim(1)?
                != hidden_u64.checked_mul(3).ok_or(ExecutorError::Overflow(
                    "short convolution projection width overflows u64",
                ))?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution projection must be [tokens, 3 * hidden]",
            ));
        }
        let tokens_u64 = projection_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[tokens_u64, hidden_u64])?)?;
        if kernel.descriptor.layout.shape()
            != Shape::new(&[u64::from(spec.hidden()), u64::from(spec.width())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution kernel must be [hidden, width]",
            ));
        }
        if history.descriptor.layout.shape()
            != Shape::new(&[spec.history_rows()?, u64::from(spec.hidden())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution history must be [width - 1, hidden]",
            ));
        }
        let tokens = usize::try_from(tokens_u64)
            .map_err(|_| ExecutorError::Overflow("short convolution token count exceeds usize"))?;
        let hidden = usize::try_from(spec.hidden())
            .map_err(|_| ExecutorError::Overflow("short convolution hidden width exceeds usize"))?;
        let width = usize::try_from(spec.width())
            .map_err(|_| ExecutorError::Overflow("short convolution kernel width exceeds usize"))?;
        let history_rows = width.checked_sub(1).ok_or(ExecutorError::Overflow(
            "short convolution history underflows",
        ))?;
        let row_elements = tokens.checked_mul(hidden).ok_or(ExecutorError::Overflow(
            "short convolution element count overflows usize",
        ))?;
        let staged_elements = row_elements.checked_mul(2).ok_or(ExecutorError::Overflow(
            "short convolution staging count overflows usize",
        ))?;
        let mut staged = self.stage_f32(staged_elements)?;
        staged.resize(staged_elements, 0.0);
        let (u, produced) = staged.split_at_mut(row_elements);
        // Fused projection layout: token t's B, C, V rows sit at
        // projection[t * 3h + {0, h, 2h} + channel].
        let projection_width = 3 * hidden;
        for (index, destination) in u.iter_mut().enumerate() {
            let token = index / hidden;
            let channel = index - token * hidden;
            let base = token * projection_width + channel;
            let gate = projection.values[base] * projection.values[base + 2 * hidden];
            if !gate.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "short convolution gate product is non-finite",
                ));
            }
            *destination = gate;
        }
        for token in 0..tokens {
            for channel in 0..hidden {
                let mut total = 0.0_f32;
                for tap in 0..width {
                    let relative = token as i128 + tap as i128 - history_rows as i128;
                    let u_value = if relative < 0 {
                        let history_row = usize::try_from(relative + history_rows as i128)
                            .map_err(|_| {
                                ExecutorError::Overflow(
                                    "short convolution history index exceeds usize",
                                )
                            })?;
                        history.values[history_row * hidden + channel]
                    } else {
                        let current = usize::try_from(relative).map_err(|_| {
                            ExecutorError::Overflow("short convolution token index exceeds usize")
                        })?;
                        u[current * hidden + channel]
                    };
                    total += kernel.values[channel * width + tap] * u_value;
                }
                let result = total * projection.values[token * projection_width + hidden + channel];
                if !result.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "short convolution output is non-finite",
                    ));
                }
                produced[token * hidden + channel] = result;
            }
        }
        if history_rows > 0 {
            if tokens >= history_rows {
                let source_start = (tokens - history_rows) * hidden;
                history
                    .values
                    .copy_from_slice(&u[source_start..source_start + history_rows * hidden]);
            } else {
                let keep_rows = history_rows - tokens;
                history.values.copy_within(
                    (history_rows - keep_rows) * hidden..history_rows * hidden,
                    0,
                );
                history.values[keep_rows * hidden..].copy_from_slice(u);
            }
        }
        output.values.copy_from_slice(produced);
        Ok(())
    }
}
