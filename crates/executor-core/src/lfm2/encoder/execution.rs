//! Complete encoder equations. Every output is retained in the task arena.

use super::{EncoderInput, EncoderTypedWeights, PointerWeightRole};
use crate::lfm2::{
    LayerKind, Lfm2LayerWeightRole as LayerRole, Lfm2ResolvedWeight, Lfm2WeightRole,
};
use minifield_engine_api::{
    AllocationClass, EncoderOps, ExecutorError, GatedShortConvSpec, GqaSpec, PackedHeadSpec,
    RectCopy2d, Result, RotarySpec, Shape, TokenIds,
};

fn layer(index: usize, role: LayerRole) -> Lfm2WeightRole {
    Lfm2WeightRole::Layer { index, role }
}

/// Each operation writes a newly allocated final slot. Earlier slots are immutable
/// inputs, so safe slice splitting makes alias policy visible without raw pointers.
fn compute<B: EncoderOps>(
    backend: &mut B,
    scratch: &mut Vec<B::Buffer>,
    shape: Shape,
    apply: impl FnOnce(&B, &mut B::Buffer, &[B::Buffer]) -> Result<()>,
) -> Result<usize> {
    let index = scratch.len();
    scratch
        .try_reserve(1)
        .map_err(|_| ExecutorError::ResourceLimit("encoder scratch slots allocation failed"))?;
    scratch.push(backend.allocate_f32_uninit(shape, AllocationClass::Scratch)?);
    let (inputs, outputs) = scratch.split_at_mut(index);
    apply(backend, &mut outputs[0], inputs)?;
    Ok(index)
}

fn linear<B: EncoderOps>(
    backend: &mut B,
    scratch: &mut Vec<B::Buffer>,
    weights: &EncoderTypedWeights<B::Buffer>,
    input: usize,
    rows: u64,
    width: u64,
    role: Lfm2WeightRole,
) -> Result<usize> {
    compute(
        backend,
        scratch,
        Shape::new(&[rows, width])?,
        |backend, output, buffers| match weights.resolve(role)? {
            Lfm2ResolvedWeight::Dense(weight) => backend.linear(output, &buffers[input], weight),
            Lfm2ResolvedWeight::Packed { codes, scales } => {
                backend.packed_linear(output, &buffers[input], codes, scales)
            }
        },
    )
}

const LAYERS_PER_STEP: usize = 4;

#[derive(Default)]
pub(super) struct EncodeCursor {
    next_layer: usize,
    residual: Option<usize>,
}

/// Record at most four layers, then the pointer head after the final layer.
/// Return the retained `[2*questions,tokens]` score index only when complete.
#[allow(clippy::too_many_lines)]
pub(super) fn encode<B: EncoderOps>(
    backend: &mut B,
    weights: &EncoderTypedWeights<B::Buffer>,
    input: &EncoderInput,
    scratch: &mut Vec<B::Buffer>,
    cursor: &mut EncodeCursor,
) -> Result<Option<usize>> {
    let cfg = &weights.config().backbone;
    let rows = u64::try_from(input.token_ids.len())
        .map_err(|_| ExecutorError::Overflow("encoder rows exceed u64"))?;
    let hidden = u64::from(cfg.hidden_size);
    let intermediate = u64::from(cfg.effective_intermediate_size);
    let kv_width = u64::from(cfg.key_value_heads) * u64::from(cfg.head_dim);
    let hidden_shape = Shape::new(&[rows, hidden])?;
    let mut residual = match cursor.residual {
        Some(index) => index,
        None => compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, _| match weights.resolve(Lfm2WeightRole::TokenEmbedding)? {
                Lfm2ResolvedWeight::Dense(table) => {
                    backend.gather_rows(output, table, TokenIds::Host(&input.token_ids))
                }
                Lfm2ResolvedWeight::Packed { codes, scales } => backend.packed_gather_rows(
                    output,
                    codes,
                    scales,
                    TokenIds::Host(&input.token_ids),
                ),
            },
        )?,
    };
    let end_layer = cfg
        .layers
        .len()
        .min(cursor.next_layer.saturating_add(LAYERS_PER_STEP));
    for (index, &kind) in cfg
        .layers
        .iter()
        .enumerate()
        .take(end_layer)
        .skip(cursor.next_layer)
    {
        let normalized = compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, buffers| {
                backend.row_rms_norm(
                    output,
                    &buffers[residual],
                    weights.dense(layer(index, LayerRole::OperatorNorm))?,
                    cfg.block_norm_epsilon,
                )
            },
        )?;
        let operator = match kind {
            LayerKind::Conv => {
                let projection = linear(
                    backend,
                    scratch,
                    weights,
                    normalized,
                    rows,
                    hidden * 3,
                    layer(index, LayerRole::ConvInProjection),
                )?;
                let mixed = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.centered_gated_convolution(
                            output,
                            &buffers[projection],
                            weights.dense(layer(index, LayerRole::ConvKernel))?,
                            &input.segments,
                            GatedShortConvSpec::new(cfg.hidden_size, cfg.conv_width)?,
                        )
                    },
                )?;
                linear(
                    backend,
                    scratch,
                    weights,
                    mixed,
                    rows,
                    hidden,
                    layer(index, LayerRole::ConvOutProjection),
                )?
            }
            LayerKind::FullAttention => {
                let query = linear(
                    backend,
                    scratch,
                    weights,
                    normalized,
                    rows,
                    hidden,
                    layer(index, LayerRole::QueryProjection),
                )?;
                let key = linear(
                    backend,
                    scratch,
                    weights,
                    normalized,
                    rows,
                    kv_width,
                    layer(index, LayerRole::KeyProjection),
                )?;
                let value = linear(
                    backend,
                    scratch,
                    weights,
                    normalized,
                    rows,
                    kv_width,
                    layer(index, LayerRole::ValueProjection),
                )?;
                let query_norm = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.head_rms_norm(
                            output,
                            &buffers[query],
                            weights.dense(layer(index, LayerRole::QueryNorm))?,
                            PackedHeadSpec::new(cfg.attention_heads, cfg.head_dim)?,
                            cfg.block_norm_epsilon,
                        )
                    },
                )?;
                let key_norm = compute(
                    backend,
                    scratch,
                    Shape::new(&[rows, kv_width])?,
                    |backend, output, buffers| {
                        backend.head_rms_norm(
                            output,
                            &buffers[key],
                            weights.dense(layer(index, LayerRole::KeyNorm))?,
                            PackedHeadSpec::new(cfg.key_value_heads, cfg.head_dim)?,
                            cfg.block_norm_epsilon,
                        )
                    },
                )?;
                let query_rope = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.split_half_rotary(
                            output,
                            &buffers[query_norm],
                            input.segments.positions(),
                            RotarySpec::new(
                                PackedHeadSpec::new(cfg.attention_heads, cfg.head_dim)?,
                                cfg.rope_theta,
                            )?,
                        )
                    },
                )?;
                let key_rope = compute(
                    backend,
                    scratch,
                    Shape::new(&[rows, kv_width])?,
                    |backend, output, buffers| {
                        backend.split_half_rotary(
                            output,
                            &buffers[key_norm],
                            input.segments.positions(),
                            RotarySpec::new(
                                PackedHeadSpec::new(cfg.key_value_heads, cfg.head_dim)?,
                                cfg.rope_theta,
                            )?,
                        )
                    },
                )?;
                let attended = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.bidirectional_gqa(
                            output,
                            &buffers[query_rope],
                            &buffers[key_rope],
                            &buffers[value],
                            &input.segments,
                            GqaSpec::new(cfg.attention_heads, cfg.key_value_heads, cfg.head_dim)?,
                        )
                    },
                )?;
                linear(
                    backend,
                    scratch,
                    weights,
                    attended,
                    rows,
                    hidden,
                    layer(index, LayerRole::OutputProjection),
                )?
            }
        };
        let sum = compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, buffers| backend.add(output, &buffers[residual], &buffers[operator]),
        )?;
        let ffn_input = compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, buffers| {
                backend.row_rms_norm(
                    output,
                    &buffers[sum],
                    weights.dense(layer(index, LayerRole::FfnNorm))?,
                    cfg.block_norm_epsilon,
                )
            },
        )?;
        let gate = linear(
            backend,
            scratch,
            weights,
            ffn_input,
            rows,
            intermediate,
            layer(index, LayerRole::FfnW1),
        )?;
        let up = linear(
            backend,
            scratch,
            weights,
            ffn_input,
            rows,
            intermediate,
            layer(index, LayerRole::FfnW3),
        )?;
        let activated = compute(
            backend,
            scratch,
            Shape::new(&[rows, intermediate])?,
            |backend, output, buffers| backend.swiglu(output, &buffers[gate], &buffers[up]),
        )?;
        let down = linear(
            backend,
            scratch,
            weights,
            activated,
            rows,
            hidden,
            layer(index, LayerRole::FfnW2),
        )?;
        // Every final-layer token survives: pointer keys use the whole sequence.
        residual = compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, buffers| backend.add(output, &buffers[sum], &buffers[down]),
        )?;
    }
    cursor.next_layer = end_layer;
    cursor.residual = Some(residual);
    if end_layer < cfg.layers.len() {
        return Ok(None);
    }
    let normalized = compute(
        backend,
        scratch,
        hidden_shape,
        |backend, output, buffers| {
            backend.row_rms_norm(
                output,
                &buffers[residual],
                weights.dense(Lfm2WeightRole::EmbeddingNorm)?,
                cfg.norm_epsilon,
            )
        },
    )?;
    let channels = usize::try_from(hidden)
        .map_err(|_| ExecutorError::Overflow("encoder hidden width exceeds usize"))?;
    let mut mask = Vec::new();
    let elements = usize::try_from(hidden_shape.element_count()?)
        .map_err(|_| ExecutorError::Overflow("encoder mask size exceeds usize"))?;
    mask.try_reserve_exact(elements)
        .map_err(|_| ExecutorError::ResourceLimit("encoder final mask allocation failed"))?;
    for &segment in input.segments.ids() {
        mask.extend(std::iter::repeat_n(
            if segment == 0 { 0.0 } else { 1.0 },
            channels,
        ));
    }
    let mask_index = scratch.len();
    scratch.push(backend.upload_f32_classified(hidden_shape, &mask, AllocationClass::Scratch)?);
    let encoded = compute(
        backend,
        scratch,
        hidden_shape,
        |backend, output, buffers| {
            backend.multiply(output, &buffers[normalized], &buffers[mask_index])
        },
    )?;
    let queries: Vec<u32> = input
        .questions
        .iter()
        .map(|question| question.query_index)
        .collect();
    let questions = u64::try_from(queries.len())
        .map_err(|_| ExecutorError::Overflow("pointer questions exceed u64"))?;
    let asked = compute(
        backend,
        scratch,
        Shape::new(&[questions, hidden])?,
        |backend, output, buffers| {
            backend.gather_rows(output, &buffers[encoded], TokenIds::Host(&queries))
        },
    )?;
    let pointer = u64::from(weights.config().pointer_width);
    let mut scores = Vec::with_capacity(2);
    for (query_role, key_role) in [
        (PointerWeightRole::StartQuery, PointerWeightRole::StartKey),
        (PointerWeightRole::EndQuery, PointerWeightRole::EndKey),
    ] {
        let query = compute(
            backend,
            scratch,
            Shape::new(&[questions, pointer])?,
            |backend, output, buffers| {
                backend.linear(output, &buffers[asked], weights.pointer(query_role)?)
            },
        )?;
        let key = compute(
            backend,
            scratch,
            Shape::new(&[rows, pointer])?,
            |backend, output, buffers| {
                backend.linear(output, &buffers[encoded], weights.pointer(key_role)?)
            },
        )?;
        scores.push(compute(
            backend,
            scratch,
            Shape::new(&[questions, rows])?,
            |backend, output, buffers| backend.linear(output, &buffers[query], &buffers[key]),
        )?);
    }
    compute(
        backend,
        scratch,
        Shape::new(&[questions * 2, rows])?,
        |backend, output, buffers| {
            backend.copy_rect_2d(
                output,
                &buffers[scores[0]],
                RectCopy2d::new(0, 0, 0, 0, questions, rows),
            )?;
            backend.copy_rect_2d(
                output,
                &buffers[scores[1]],
                RectCopy2d::new(0, 0, questions, 0, questions, rows),
            )
        },
    )
    .map(Some)
}
