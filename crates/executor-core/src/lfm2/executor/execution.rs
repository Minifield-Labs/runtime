//! Ordered model pass and backend projection dispatch.

use super::{
    AllocationClass, ExecutorError, GatedShortConvSpec, GqaSpec, InferenceOps, LayerCache,
    LayerKind, Lfm2LayerWeightRole, Lfm2Lut2Mode, Lfm2ResolvedWeight, Lfm2TypedWeights,
    Lfm2WeightRole, ModelContext, PackedHeadSpec, PrefixStorage, RectCopy2d, Result, RotarySpec,
    Shape, TokenId, TokenIds, allocate, layer_role, shape,
};

/// Linear projection through whichever operand set the role is stored as:
/// dense f32 weights or `minifield.ternary.v1` packed code/scale streams.
pub(super) fn weight_linear<B: InferenceOps>(
    backend: &B,
    output: &mut B::Buffer,
    input: &B::Buffer,
    weights: &Lfm2TypedWeights<B::Buffer>,
    role: Lfm2WeightRole,
) -> Result<()> {
    match weights.resolve(role)? {
        Lfm2ResolvedWeight::Dense(weight) => backend.linear(output, input, weight),
        Lfm2ResolvedWeight::Packed { codes, scales } => {
            backend.packed_linear(output, input, codes, scales)
        }
    }
}

pub(super) fn push_scratch<B: InferenceOps>(
    scratch: &mut Vec<B::Buffer>,
    values: impl IntoIterator<Item = B::Buffer>,
) {
    scratch.extend(values);
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub(super) fn append_tokens<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    state: &mut PrefixStorage<B>,
    tokens: &[TokenId],
    ids: TokenIds<'_, B>,
    produce_logits: bool,
    scratch: &mut Vec<B::Buffer>,
) -> Result<()> {
    let config = context.config();
    if tokens.is_empty() {
        return Err(ExecutorError::InvalidArgument(
            "token pass requires at least one token",
        ));
    }
    for token in tokens {
        if *token >= config.vocab_size {
            return Err(ExecutorError::OutOfBounds(
                "valid token ID exceeds loaded model vocabulary",
            ));
        }
    }
    let rows = u64::try_from(tokens.len())
        .map_err(|_| ExecutorError::Overflow("token count exceeds u64"))?;
    let base = state.length;
    let length = context.checked_length(base, rows)?;
    let positions: Vec<u64> = (0..rows).map(|offset| base + offset).collect();
    let hidden = u64::from(config.hidden_size);
    let intermediate = u64::from(config.effective_intermediate_size);
    let kv_width = u64::from(config.key_value_heads)
        .checked_mul(u64::from(config.head_dim))
        .ok_or(ExecutorError::Overflow(
            "LFM2 key/value width overflows u64",
        ))?;
    let mut x = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
    match context.weights.resolve(Lfm2WeightRole::TokenEmbedding)? {
        Lfm2ResolvedWeight::Dense(embedding) => {
            backend.gather_rows(&mut x.buffer, embedding, ids)?;
        }
        Lfm2ResolvedWeight::Packed { codes, scales } => {
            backend.packed_gather_rows(&mut x.buffer, codes, scales, ids)?;
        }
    }

    // Decode-fusion layout: every residual add is fused with the RMS norm that
    // consumes its sum, so the loop carries `x` (the residual base) alongside
    // `u` (the already-normed input for the layer about to run). The final
    // iteration's fused norm uses the embedding norm weight, producing the
    // lm_head input without a separate pass.
    let mut u = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
    if !config.layers.is_empty() {
        backend.row_rms_norm(
            &mut u.buffer,
            &x.buffer,
            context
                .weights
                .buffer_for(layer_role(0, Lfm2LayerWeightRole::OperatorNorm))?,
            config.block_norm_epsilon,
        )?;
    }
    for (index, kind) in config.layers.iter().copied().enumerate() {
        let mut operator = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
        match kind {
            LayerKind::Conv => {
                let mut projection = allocate(
                    backend,
                    shape(
                        rows,
                        hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
                            "LFM2 convolution projection width overflows u64",
                        ))?,
                    )?,
                    AllocationClass::Scratch,
                )?;
                weight_linear(
                    backend,
                    &mut projection.buffer,
                    &u.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::ConvInProjection),
                )?;
                let mut convolved =
                    allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
                let Some(LayerCache::Conv { history }) = state.layers.get_mut(index) else {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix convolution cache does not match model layer",
                    ));
                };
                backend.gated_short_convolution(
                    &mut convolved.buffer,
                    &projection.buffer,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::ConvKernel))?,
                    &mut history.buffer,
                    GatedShortConvSpec::new(config.hidden_size, config.conv_width)?,
                )?;
                weight_linear(
                    backend,
                    &mut operator.buffer,
                    &convolved.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::ConvOutProjection),
                )?;
                push_scratch::<B>(scratch, [projection.buffer, convolved.buffer]);
            }
            LayerKind::FullAttention => {
                let mut q = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
                let mut k = allocate(backend, shape(rows, kv_width)?, AllocationClass::Scratch)?;
                let mut v = allocate(backend, shape(rows, kv_width)?, AllocationClass::Scratch)?;
                weight_linear(
                    backend,
                    &mut q.buffer,
                    &u.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::QueryProjection),
                )?;
                if context
                    .weights
                    .role_quant(layer_role(index, Lfm2LayerWeightRole::KeyProjection))
                    == context
                        .weights
                        .role_quant(layer_role(index, Lfm2LayerWeightRole::ValueProjection))
                    && let (
                        Lfm2ResolvedWeight::Packed {
                            codes: k_codes,
                            scales: k_scales,
                        },
                        Lfm2ResolvedWeight::Packed {
                            codes: v_codes,
                            scales: v_scales,
                        },
                    ) = (
                        context
                            .weights
                            .resolve(layer_role(index, Lfm2LayerWeightRole::KeyProjection))?,
                        context
                            .weights
                            .resolve(layer_role(index, Lfm2LayerWeightRole::ValueProjection))?,
                    )
                {
                    backend.packed_linear_pair(
                        &mut k.buffer,
                        &mut v.buffer,
                        &u.buffer,
                        k_codes,
                        k_scales,
                        v_codes,
                        v_scales,
                    )?;
                } else {
                    weight_linear(
                        backend,
                        &mut k.buffer,
                        &u.buffer,
                        &context.weights,
                        layer_role(index, Lfm2LayerWeightRole::KeyProjection),
                    )?;
                    weight_linear(
                        backend,
                        &mut v.buffer,
                        &u.buffer,
                        &context.weights,
                        layer_role(index, Lfm2LayerWeightRole::ValueProjection),
                    )?;
                }
                let query_heads = PackedHeadSpec::new(config.attention_heads, config.head_dim)?;
                let key_value_heads = PackedHeadSpec::new(config.key_value_heads, config.head_dim)?;
                let mut q_rope = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
                let mut k_rope =
                    allocate(backend, shape(rows, kv_width)?, AllocationClass::Scratch)?;
                backend.qk_norm_rope(
                    &mut q_rope.buffer,
                    &mut k_rope.buffer,
                    &q.buffer,
                    &k.buffer,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::QueryNorm))?,
                    context
                        .weights
                        .buffer_for(layer_role(index, Lfm2LayerWeightRole::KeyNorm))?,
                    &positions,
                    RotarySpec::new(query_heads, config.rope_theta)?,
                    key_value_heads,
                    config.block_norm_epsilon,
                )?;
                let mut attention =
                    allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
                let Some(LayerCache::Attention { key, value, length }) =
                    state.layers.get_mut(index)
                else {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix attention cache does not match model layer",
                    ));
                };
                backend.causal_gqa(
                    &mut attention.buffer,
                    &q_rope.buffer,
                    &k_rope.buffer,
                    &v.buffer,
                    &mut key.buffer,
                    &mut value.buffer,
                    length,
                    GqaSpec::new(
                        config.attention_heads,
                        config.key_value_heads,
                        config.head_dim,
                    )?,
                )?;
                weight_linear(
                    backend,
                    &mut operator.buffer,
                    &attention.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::OutputProjection),
                )?;
                push_scratch::<B>(
                    scratch,
                    [
                        q.buffer,
                        k.buffer,
                        v.buffer,
                        q_rope.buffer,
                        k_rope.buffer,
                        attention.buffer,
                    ],
                );
            }
        }
        let mut residual = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
        let mut ffn_input = allocate(backend, shape(rows, hidden)?, AllocationClass::Scratch)?;
        backend.add_row_rms_norm(
            &mut residual.buffer,
            &mut ffn_input.buffer,
            &x.buffer,
            &operator.buffer,
            context
                .weights
                .buffer_for(layer_role(index, Lfm2LayerWeightRole::FfnNorm))?,
            config.block_norm_epsilon,
        )?;
        // After the final layer's sequence mixing, only the last token row is
        // consumed (logits read row `rows - 1` alone). Run that FFN at m = 1
        // and drop the other rows' gate/up/down work entirely.
        let last_layer = index + 1 == config.layers.len();
        let ffn_rows = if last_layer { 1 } else { rows };
        if ffn_rows != rows {
            let mut sliced_input = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
            backend.copy_rect_2d(
                &mut sliced_input.buffer,
                &ffn_input.buffer,
                RectCopy2d::new(rows - 1, 0, 0, 0, 1, hidden),
            )?;
            let mut sliced_residual =
                allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
            backend.copy_rect_2d(
                &mut sliced_residual.buffer,
                &residual.buffer,
                RectCopy2d::new(rows - 1, 0, 0, 0, 1, hidden),
            )?;
            push_scratch::<B>(scratch, [ffn_input.buffer, residual.buffer]);
            ffn_input = sliced_input;
            residual = sliced_residual;
        }
        let mut down = allocate(backend, shape(ffn_rows, hidden)?, AllocationClass::Scratch)?;
        // Prefill on a fully packed FFN applies SwiGLU inside the gate/up
        // projection's epilogue, so the down projection consumes one hidden
        // buffer instead of recomputing the activation per output tile. The
        // fused op only exists as a tiled GEMM, so below the backend's
        // multi-token crossover the pair + SwiGLU-linear path (which has
        // short-row kernels) is the faster route.
        let fused = ffn_rows >= 96
            && if context
                .weights
                .role_quant(layer_role(index, Lfm2LayerWeightRole::FfnW1))
                == context
                    .weights
                    .role_quant(layer_role(index, Lfm2LayerWeightRole::FfnW3))
                && let (
                    Lfm2ResolvedWeight::Packed {
                        codes: gate_codes,
                        scales: gate_scales,
                    },
                    Lfm2ResolvedWeight::Packed {
                        codes: up_codes,
                        scales: up_scales,
                    },
                    Lfm2ResolvedWeight::Packed {
                        codes: down_codes,
                        scales: down_scales,
                    },
                ) = (
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW1))?,
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW3))?,
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW2))?,
                )
            {
                let mut hidden = allocate(
                    backend,
                    shape(ffn_rows, intermediate)?,
                    AllocationClass::Scratch,
                )?;
                // LUT2 paths run only when the backend repacked this role at
                // load; raw streams cover every other consumer unchanged.
                let mode = context.lut2_mode.get();
                let pair_lut2 = if mode == Lfm2Lut2Mode::Auto {
                    match (
                        context
                            .lut2_codes
                            .get(&layer_role(index, Lfm2LayerWeightRole::FfnW1)),
                        context
                            .lut2_codes
                            .get(&layer_role(index, Lfm2LayerWeightRole::FfnW3)),
                    ) {
                        (Some(gate_lut2), Some(up_lut2)) => Some((gate_lut2, up_lut2)),
                        _ => None,
                    }
                } else {
                    None
                };
                match pair_lut2 {
                    Some((gate_lut2, up_lut2)) => backend.packed_swiglu_pair_lut2(
                        &mut hidden.buffer,
                        &ffn_input.buffer,
                        gate_lut2,
                        gate_scales,
                        up_lut2,
                        up_scales,
                    )?,
                    None => backend.packed_swiglu_pair(
                        &mut hidden.buffer,
                        &ffn_input.buffer,
                        gate_codes,
                        gate_scales,
                        up_codes,
                        up_scales,
                    )?,
                }
                let down_lut2 = if mode == Lfm2Lut2Mode::Auto || mode == Lfm2Lut2Mode::DownOnly {
                    context
                        .lut2_codes
                        .get(&layer_role(index, Lfm2LayerWeightRole::FfnW2))
                } else {
                    None
                };
                match down_lut2 {
                    Some(down_lut2) => backend.packed_linear_lut2(
                        &mut down.buffer,
                        &hidden.buffer,
                        down_lut2,
                        down_scales,
                    )?,
                    None => backend.packed_linear(
                        &mut down.buffer,
                        &hidden.buffer,
                        down_codes,
                        down_scales,
                    )?,
                }
                push_scratch::<B>(scratch, [hidden.buffer]);
                true
            } else {
                false
            };
        if !fused {
            let mut gate = allocate(
                backend,
                shape(ffn_rows, intermediate)?,
                AllocationClass::Scratch,
            )?;
            let mut up = allocate(
                backend,
                shape(ffn_rows, intermediate)?,
                AllocationClass::Scratch,
            )?;
            if context
                .weights
                .role_quant(layer_role(index, Lfm2LayerWeightRole::FfnW1))
                == context
                    .weights
                    .role_quant(layer_role(index, Lfm2LayerWeightRole::FfnW3))
                && let (
                    Lfm2ResolvedWeight::Packed {
                        codes: gate_codes,
                        scales: gate_scales,
                    },
                    Lfm2ResolvedWeight::Packed {
                        codes: up_codes,
                        scales: up_scales,
                    },
                ) = (
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW1))?,
                    context
                        .weights
                        .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW3))?,
                )
            {
                backend.packed_linear_pair(
                    &mut gate.buffer,
                    &mut up.buffer,
                    &ffn_input.buffer,
                    gate_codes,
                    gate_scales,
                    up_codes,
                    up_scales,
                )?;
            } else {
                weight_linear(
                    backend,
                    &mut gate.buffer,
                    &ffn_input.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::FfnW1),
                )?;
                weight_linear(
                    backend,
                    &mut up.buffer,
                    &ffn_input.buffer,
                    &context.weights,
                    layer_role(index, Lfm2LayerWeightRole::FfnW3),
                )?;
            }
            match context
                .weights
                .resolve(layer_role(index, Lfm2LayerWeightRole::FfnW2))?
            {
                Lfm2ResolvedWeight::Packed { codes, scales } => {
                    backend.packed_swiglu_linear(
                        &mut down.buffer,
                        &gate.buffer,
                        &up.buffer,
                        codes,
                        scales,
                    )?;
                }
                Lfm2ResolvedWeight::Dense(_) => {
                    let mut activated = allocate(
                        backend,
                        shape(ffn_rows, intermediate)?,
                        AllocationClass::Scratch,
                    )?;
                    backend.swiglu(&mut activated.buffer, &gate.buffer, &up.buffer)?;
                    weight_linear(
                        backend,
                        &mut down.buffer,
                        &activated.buffer,
                        &context.weights,
                        layer_role(index, Lfm2LayerWeightRole::FfnW2),
                    )?;
                    push_scratch::<B>(scratch, [activated.buffer]);
                }
            }
            push_scratch::<B>(scratch, [gate.buffer, up.buffer]);
        }
        // The next layer's operator norm (or the final embedding norm) rides
        // on the same fused add+norm pass that produces the residual base.
        let mut next_x = allocate(backend, shape(ffn_rows, hidden)?, AllocationClass::Scratch)?;
        let mut next_u = allocate(backend, shape(ffn_rows, hidden)?, AllocationClass::Scratch)?;
        let (next_norm, next_epsilon) = if index + 1 < config.layers.len() {
            (
                layer_role(index + 1, Lfm2LayerWeightRole::OperatorNorm),
                config.block_norm_epsilon,
            )
        } else {
            (Lfm2WeightRole::EmbeddingNorm, config.norm_epsilon)
        };
        backend.add_row_rms_norm(
            &mut next_x.buffer,
            &mut next_u.buffer,
            &residual.buffer,
            &down.buffer,
            context.weights.buffer_for(next_norm)?,
            next_epsilon,
        )?;
        push_scratch::<B>(
            scratch,
            [
                x.buffer,
                u.buffer,
                operator.buffer,
                residual.buffer,
                ffn_input.buffer,
                down.buffer,
            ],
        );
        x = next_x;
        u = next_u;
    }
    if config.layers.is_empty() {
        backend.row_rms_norm(
            &mut u.buffer,
            &x.buffer,
            context.weights.buffer_for(Lfm2WeightRole::EmbeddingNorm)?,
            config.norm_epsilon,
        )?;
    }
    if produce_logits {
        let mut logits = allocate(
            backend,
            shape(1, u64::from(context.weights.output_width()))?,
            AllocationClass::Cache,
        )?;
        // A non-empty layer stack already reduced `u` to its last row.
        if rows == 1 || !config.layers.is_empty() {
            weight_linear(
                backend,
                &mut logits.buffer,
                &u.buffer,
                &context.weights,
                context.weights.output_role(),
            )?;
        } else {
            let mut last = allocate(backend, shape(1, hidden)?, AllocationClass::Scratch)?;
            backend.copy_rect_2d(
                &mut last.buffer,
                &u.buffer,
                RectCopy2d::new(rows - 1, 0, 0, 0, 1, hidden),
            )?;
            weight_linear(
                backend,
                &mut logits.buffer,
                &last.buffer,
                &context.weights,
                context.weights.output_role(),
            )?;
            push_scratch::<B>(scratch, [last.buffer]);
        }
        state.next_logits = Some(logits);
    }
    push_scratch::<B>(scratch, [x.buffer, u.buffer]);
    state.history.extend_from_slice(tokens);
    state.length = length;
    Ok(())
}

/// Greedy-sample epilogue: reduce the staged `[1, V]` logits to a
/// device-resident token id. The publish readback then only needs this one
/// f32: NaN means the logits row held a non-finite value, and a finite value
/// doubles as the next append's embedding-gather selector.
pub(super) fn sample_epilogue<B: InferenceOps>(
    backend: &mut B,
    state: &mut PrefixStorage<B>,
    mask: Option<&[u64]>,
) -> Result<()> {
    if state.sampled.is_none() {
        state.sampled = Some(allocate(
            backend,
            Shape::new(&[1])?,
            AllocationClass::Cache,
        )?);
    }
    let (Some(logits), Some(sampled)) = (state.next_logits.as_ref(), state.sampled.as_mut()) else {
        return Err(ExecutorError::BackendFailure(
            "token pass produced no logits boundary",
        ));
    };
    match mask {
        Some(mask) => backend.argmax_masked(&mut sampled.buffer, &logits.buffer, mask)?,
        None => backend.argmax(&mut sampled.buffer, &logits.buffer)?,
    }
    state.sampled_id = None;
    Ok(())
}

pub(super) fn append_token<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    state: &mut PrefixStorage<B>,
    token: TokenId,
    ids: TokenIds<'_, B>,
    mask: Option<&[u64]>,
    scratch: &mut Vec<B::Buffer>,
) -> Result<()> {
    append_tokens(context, backend, state, &[token], ids, true, scratch)?;
    sample_epilogue(backend, state, mask)
}
