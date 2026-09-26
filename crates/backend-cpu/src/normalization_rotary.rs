// Normalization and rotary position operations.

use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{
    ExecutorError, OperationKind, PackedHeadSpec, Result, RotarySpec, Shape,
};

impl CpuBackend {
    /// Row RMS norm over a `[rows, width]` input and a `[width]` weight.
    /// The sum and square root use f32 to define the scalar reference rounding path.
    pub fn row_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
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
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("RMS row count exceeds usize"))?;
        let width = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("RMS width exceeds usize"))?;
        let weight_width = usize::try_from(weight_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("RMS weight width exceeds usize"))?;
        if width != weight_width {
            return Err(ExecutorError::InvalidShape(
                "RMS input width differs from weight width",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        if width == 0 {
            return Ok(());
        }
        if width > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "RMS width exceeds exact f32 divisor range",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let width_as_f32 = width as f32;
        for row in 0..rows {
            let start = row
                .checked_mul(width)
                .ok_or(ExecutorError::Overflow("RMS row offset overflows usize"))?;
            let mut squared_sum = 0.0_f32;
            for value in &input.values[start..start + width] {
                squared_sum += value * value;
            }
            let reciprocal = (squared_sum / width_as_f32 + epsilon).sqrt().recip();
            if !reciprocal.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "RMS normalization reciprocal is non-finite",
                ));
            }
            for column in 0..width {
                let normalized = input.values[start + column] * reciprocal;
                let value = normalized * weight.values[column];
                if !value.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "RMS normalization produced a non-finite value",
                    ));
                }
                output.values[start + column] = value;
            }
        }
        Ok(())
    }

    /// Apply head-local RMS normalization to packed `[tokens, heads * head_dim]` rows.
    pub fn head_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
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
        let weight_shape = weight.descriptor.layout.shape();
        if weight_shape != Shape::new(&[u64::from(heads.head_dim())])? {
            return Err(ExecutorError::InvalidShape(
                "head RMS weight must have head_dim entries",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let tokens = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("head RMS token count exceeds usize"))?;
        let head_count = usize::try_from(heads.heads())
            .map_err(|_| ExecutorError::Overflow("head RMS head count exceeds usize"))?;
        let head_dim = usize::try_from(heads.head_dim())
            .map_err(|_| ExecutorError::Overflow("head RMS head dimension exceeds usize"))?;
        let packed_width = usize::try_from(heads.packed_width()?)
            .map_err(|_| ExecutorError::Overflow("head RMS packed width exceeds usize"))?;
        if head_dim > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "RMS width exceeds exact f32 divisor range",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let head_dim_f32 = head_dim as f32;
        for token in 0..tokens {
            for head in 0..head_count {
                let start = token
                    .checked_mul(packed_width)
                    .and_then(|value| value.checked_add(head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("head RMS offset overflows usize"))?;
                let mut squared_sum = 0.0_f32;
                for value in &input.values[start..start + head_dim] {
                    squared_sum += value * value;
                }
                let reciprocal = (squared_sum / head_dim_f32 + epsilon).sqrt().recip();
                if !reciprocal.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "head RMS reciprocal is non-finite",
                    ));
                }
                for column in 0..head_dim {
                    let normalized = input.values[start + column] * reciprocal;
                    let value = normalized * weight.values[column];
                    if !value.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "head RMS normalization produced a non-finite value",
                        ));
                    }
                    output.values[start + column] = value;
                }
            }
        }
        Ok(())
    }

    /// Apply split-half `RoPE` to packed `[tokens, heads * head_dim]` rows.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn split_half_rotary(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
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
        let head_count = usize::try_from(spec.heads().heads())
            .map_err(|_| ExecutorError::Overflow("RoPE head count exceeds usize"))?;
        let head_dim = usize::try_from(spec.heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("RoPE head dimension exceeds usize"))?;
        let half = head_dim / 2;
        let packed_width = usize::try_from(spec.heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("RoPE packed width exceeds usize"))?;
        for (token, position) in positions.iter().copied().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_f32 = position as f32;
            if !position_f32.is_finite() {
                return Err(ExecutorError::InvalidArgument(
                    "RoPE position is not representable as finite f32",
                ));
            }
            for head in 0..head_count {
                let head_start = token
                    .checked_mul(packed_width)
                    .and_then(|value| value.checked_add(head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("RoPE offset overflows usize"))?;
                for column in 0..half {
                    let exponent = -2.0_f64 * (column as f64) / (head_dim as f64);
                    let frequency = (f64::from(spec.theta()).powf(exponent)) as f32;
                    let angle = position_f32 * frequency;
                    let sine = f64::from(angle).sin() as f32;
                    let cosine = f64::from(angle).cos() as f32;
                    let first = input.values[head_start + column];
                    let second = input.values[head_start + half + column];
                    // Frequency and trig use the explicitly documented f64-to-f32 boundaries;
                    // vector products and combinations are sequential f32 model arithmetic.
                    let rotated_first = first * cosine - second * sine;
                    let rotated_second = second * cosine + first * sine;
                    if !rotated_first.is_finite() || !rotated_second.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "RoPE produced a non-finite value",
                        ));
                    }
                    output.values[head_start + column] = rotated_first;
                    output.values[head_start + half + column] = rotated_second;
                }
            }
        }
        Ok(())
    }

    /// Fused residual add plus row RMS norm, decomposed into the existing
    /// `add` and `row_rms_norm` passes.
    pub fn add_row_rms_norm(
        &self,
        sum: &mut CpuBuffer,
        normed: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
        weight: &CpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::AddRowRmsNorm)?;
        self.add(sum, left, right)?;
        self.row_rms_norm(normed, sum, weight, epsilon)
    }

    /// Per-head RMS norm followed by split-half rotary on the query and key
    /// rows, sharing the scalar reference math of `head_rms_norm` and
    /// `split_half_rotary`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    pub fn qk_norm_rope(
        &self,
        query_out: &mut CpuBuffer,
        key_out: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        query_weight: &CpuBuffer,
        key_weight: &CpuBuffer,
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
        let head_dim = usize::try_from(rope.heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("RoPE head dimension exceeds usize"))?;
        let half = head_dim / 2;
        let weight_width = u64::from(rope.heads().head_dim());
        for (label, weight) in [("query", query_weight), ("key", key_weight)] {
            if weight.descriptor.layout.shape() != Shape::new(&[weight_width])? {
                return Err(ExecutorError::InvalidShape(label));
            }
        }
        if head_dim > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "RMS width exceeds exact f32 divisor range",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let head_dim_f32 = head_dim as f32;
        let mut normed = self.stage_f32(head_dim)?;
        normed.resize(head_dim, 0.0);
        let mut apply = |input: &CpuBuffer,
                         output: &mut CpuBuffer,
                         weight: &CpuBuffer,
                         heads: PackedHeadSpec|
         -> Result<()> {
            let packed_width = usize::try_from(heads.packed_width()?)
                .map_err(|_| ExecutorError::Overflow("RoPE packed width exceeds usize"))?;
            let head_count = usize::try_from(heads.heads())
                .map_err(|_| ExecutorError::Overflow("RoPE head count exceeds usize"))?;
            for (token, position) in positions.iter().copied().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let position_f32 = position as f32;
                if !position_f32.is_finite() {
                    return Err(ExecutorError::InvalidArgument(
                        "RoPE position is not representable as finite f32",
                    ));
                }
                for head in 0..head_count {
                    let start = token
                        .checked_mul(packed_width)
                        .and_then(|value| value.checked_add(head.checked_mul(head_dim)?))
                        .ok_or(ExecutorError::Overflow("RoPE offset overflows usize"))?;
                    let mut squared_sum = 0.0_f32;
                    for value in &input.values[start..start + head_dim] {
                        squared_sum += value * value;
                    }
                    let reciprocal = (squared_sum / head_dim_f32 + epsilon).sqrt().recip();
                    if !reciprocal.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "head RMS reciprocal is non-finite",
                        ));
                    }
                    for (index, cell) in normed.iter_mut().enumerate() {
                        *cell = input.values[start + index] * reciprocal * weight.values[index];
                    }
                    for column in 0..half {
                        let exponent = -2.0_f64 * (column as f64) / (head_dim as f64);
                        let frequency = (f64::from(rope.theta()).powf(exponent)) as f32;
                        let angle = position_f32 * frequency;
                        let sine = f64::from(angle).sin() as f32;
                        let cosine = f64::from(angle).cos() as f32;
                        let first = normed[column];
                        let second = normed[half + column];
                        let rotated_first = first * cosine - second * sine;
                        let rotated_second = second * cosine + first * sine;
                        if !rotated_first.is_finite() || !rotated_second.is_finite() {
                            return Err(ExecutorError::BackendFailure(
                                "RoPE produced a non-finite value",
                            ));
                        }
                        output.values[start + column] = rotated_first;
                        output.values[start + half + column] = rotated_second;
                    }
                }
            }
            Ok(())
        };
        apply(query, query_out, query_weight, rope.heads())?;
        apply(key, key_out, key_weight, key_value_heads)
    }
}
