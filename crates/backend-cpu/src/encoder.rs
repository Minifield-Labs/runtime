//! Scalar complete-sequence encoder reference equations.

use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{
    EncoderOps, EncoderSegments, ExecutorError, GatedShortConvSpec, GqaSpec, OperationKind, Result,
    Shape,
};

impl EncoderOps for CpuBackend {
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn bidirectional_gqa(
        &self,
        output: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        value: &CpuBuffer,
        segments: &EncoderSegments,
        spec: GqaSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GroupedQueryAttention)?;
        self.check_f32_buffer(query)?;
        self.check_f32_buffer(key)?;
        self.check_f32_buffer(value)?;
        let shape = query.descriptor.layout.shape();
        let rows = spec.query_heads().validate_packed(shape)?;
        segments.validate_tokens(rows)?;
        if spec
            .key_value_heads()
            .validate_packed(key.descriptor.layout.shape())?
            != rows
            || key.descriptor.layout.shape() != value.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "encoder GQA token counts differ",
            ));
        }
        self.check_output_shape(output, shape)?;
        let tokens = usize::try_from(rows)
            .map_err(|_| ExecutorError::Overflow("encoder rows exceed usize"))?;
        let heads = usize::try_from(spec.query_heads().heads())
            .map_err(|_| ExecutorError::Overflow("encoder heads exceed usize"))?;
        let dim = usize::try_from(spec.query_heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("encoder head dimension exceeds usize"))?;
        let group = usize::try_from(spec.group_size())
            .map_err(|_| ExecutorError::Overflow("encoder GQA group exceeds usize"))?;
        let kv_width = usize::try_from(spec.key_value_heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("encoder KV width exceeds usize"))?;
        let query_width = heads * dim;
        let scale = 1.0 / (dim as f32).sqrt();
        // Both temporary regions are simultaneously live. Admit their total
        // in one checked staging allocation rather than checking each in isolation.
        let staging_elements =
            query
                .values
                .len()
                .checked_add(tokens)
                .ok_or(ExecutorError::Overflow(
                    "encoder GQA staging elements overflow usize",
                ))?;
        let mut storage = self.stage_f32(staging_elements)?;
        storage.resize(staging_elements, 0.0);
        let (staged, scores) = storage.split_at_mut(query.values.len());
        for token in 0..tokens {
            let segment = segments.ids()[token];
            if segment == 0 {
                continue;
            }
            for head in 0..heads {
                let q_offset = token * query_width + head * dim;
                let kv_head = head / group;
                let mut maximum = f32::NEG_INFINITY;
                for (other, score) in scores.iter_mut().enumerate() {
                    *score = f32::NEG_INFINITY;
                    if segments.ids()[other] != segment {
                        continue;
                    }
                    let k_offset = other * kv_width + kv_head * dim;
                    let mut dot = 0.0;
                    for channel in 0..dim {
                        dot += query.values[q_offset + channel] * key.values[k_offset + channel];
                    }
                    *score = dot * scale;
                    if !score.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "encoder GQA score is non-finite",
                        ));
                    }
                    maximum = maximum.max(*score);
                }
                let mut denominator = 0.0;
                for (other, score) in scores.iter_mut().enumerate() {
                    *score = if segments.ids()[other] == segment {
                        (*score - maximum).exp()
                    } else {
                        0.0
                    };
                    denominator += *score;
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(ExecutorError::BackendFailure(
                        "encoder GQA softmax denominator is invalid",
                    ));
                }
                for channel in 0..dim {
                    let mut weighted = 0.0;
                    for (other, &score) in scores.iter().enumerate() {
                        // Skip invalid rows rather than multiplying NaN padding by zero.
                        if segments.ids()[other] == segment {
                            weighted += (score / denominator)
                                * value.values[other * kv_width + kv_head * dim + channel];
                        }
                    }
                    if !weighted.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "encoder GQA output is non-finite",
                        ));
                    }
                    staged[q_offset + channel] = weighted;
                }
            }
        }
        output.values.copy_from_slice(staged);
        Ok(())
    }

    fn centered_gated_convolution(
        &self,
        output: &mut CpuBuffer,
        projection: &CpuBuffer,
        kernel: &CpuBuffer,
        segments: &EncoderSegments,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatedShortConvolution)?;
        self.check_f32_buffer(projection)?;
        self.check_f32_buffer(kernel)?;
        let hidden = u64::from(spec.hidden());
        let shape = projection.descriptor.layout.shape();
        if shape.rank() != 2 || shape.dim(1)? != hidden * 3 {
            return Err(ExecutorError::InvalidShape(
                "centered convolution projection must be [tokens,3*hidden]",
            ));
        }
        segments.validate_tokens(shape.dim(0)?)?;
        self.check_output_shape(output, Shape::new(&[shape.dim(0)?, hidden])?)?;
        if kernel.descriptor.layout.shape() != Shape::new(&[hidden, u64::from(spec.width())])? {
            return Err(ExecutorError::InvalidShape(
                "centered convolution kernel must be [hidden,width]",
            ));
        }
        let channels = usize::try_from(hidden)
            .map_err(|_| ExecutorError::Overflow("convolution channels exceed usize"))?;
        let width = usize::try_from(spec.width())
            .map_err(|_| ExecutorError::Overflow("convolution width exceeds usize"))?;
        let mut staged = self.stage_f32(output.values.len())?;
        staged.resize(output.values.len(), 0.0);
        for (token, &segment) in segments.ids().iter().enumerate() {
            if segment == 0 {
                continue;
            }
            for channel in 0..channels {
                let mut total = 0.0;
                for tap in 0..width {
                    let other = (token + tap).checked_sub(width / 2);
                    if let Some(other) = other.filter(|&other| {
                        other < segments.ids().len() && segments.ids()[other] == segment
                    }) {
                        let base = other * 3 * channels + channel;
                        total += kernel.values[channel * width + tap]
                            * (projection.values[base] * projection.values[base + 2 * channels]);
                    }
                }
                let value = projection.values[token * 3 * channels + channels + channel] * total;
                if !value.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "centered convolution output is non-finite",
                    ));
                }
                staged[token * channels + channel] = value;
            }
        }
        output.values.copy_from_slice(&staged);
        Ok(())
    }
}
