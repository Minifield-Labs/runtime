//! Complete encoder equations with same-shape scratch reuse after each last use.

use super::{EncoderInput, EncoderTypedWeights, PointerWeightRole};
use crate::lfm2::{
    LayerKind, Lfm2LayerWeightRole as LayerRole, Lfm2ResolvedWeight, Lfm2WeightRole,
};
use minifield_engine_api::{
    AllocationClass, EncoderOps, ExecutorError, GatedShortConvSpec, GqaSpec, OperationKind,
    PackedHeadSpec, RectCopy2d, Result, RotarySpec, Shape, TokenIds,
};

fn layer(index: usize, role: LayerRole) -> Lfm2WeightRole {
    Lfm2WeightRole::Layer { index, role }
}

/// Released buffers remain owned until the task's completion boundary. Reuse only
/// records later writes after all consumers of the previous value were recorded.
pub(super) struct Scratch<Buffer> {
    slots: Vec<Option<(Shape, Buffer)>>,
    reusable: Vec<(Shape, Buffer)>,
}

impl<Buffer> Default for Scratch<Buffer> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            reusable: Vec::new(),
        }
    }
}

impl<Buffer> Scratch<Buffer> {
    pub(super) fn get(&self, index: usize) -> Option<&Buffer> {
        self.slots.get(index)?.as_ref().map(|(_, buffer)| buffer)
    }

    fn buffer(&self, index: usize) -> Result<&Buffer> {
        self.get(index).ok_or(ExecutorError::BackendFailure(
            "encoder scratch value is unavailable",
        ))
    }

    fn vacant_slot(&mut self) -> Result<usize> {
        if let Some(index) = self.slots.iter().position(Option::is_none) {
            return Ok(index);
        }
        self.slots
            .try_reserve(1)
            .map_err(|_| ExecutorError::ResourceLimit("encoder scratch slots allocation failed"))?;
        let index = self.slots.len();
        self.slots.push(None);
        Ok(index)
    }

    fn insert(&mut self, shape: Shape, buffer: Buffer) -> Result<usize> {
        let index = self.vacant_slot()?;
        self.slots[index] = Some((shape, buffer));
        Ok(index)
    }

    fn release(&mut self, index: usize) -> Result<()> {
        self.reusable.try_reserve(1).map_err(|_| {
            ExecutorError::ResourceLimit("encoder reusable slots allocation failed")
        })?;
        let buffer = self.slots.get_mut(index).and_then(Option::take).ok_or(
            ExecutorError::BackendFailure("encoder scratch value was already released"),
        )?;
        self.reusable.push(buffer);
        Ok(())
    }

    pub(super) fn clear(&mut self) {
        self.slots.clear();
        self.reusable.clear();
    }

    pub(super) fn into_buffers(self) -> Vec<Buffer> {
        self.slots
            .into_iter()
            .flatten()
            .chain(self.reusable)
            .map(|(_, buffer)| buffer)
            .collect()
    }
}

/// The output is removed from reusable storage before inputs are borrowed.
/// It returns to the arena even if recording fails, preserving retirement.
fn compute<B: EncoderOps>(
    backend: &mut B,
    scratch: &mut Scratch<B::Buffer>,
    shape: Shape,
    apply: impl FnOnce(&B, &mut B::Buffer, &Scratch<B::Buffer>) -> Result<()>,
) -> Result<usize> {
    let index = scratch.vacant_slot()?;
    let mut output = match scratch
        .reusable
        .iter()
        .position(|(candidate, _)| *candidate == shape)
    {
        Some(index) => scratch.reusable.swap_remove(index).1,
        None => backend.allocate_f32_uninit(shape, AllocationClass::Scratch)?,
    };
    let result = apply(backend, &mut output, scratch);
    scratch.slots[index] = Some((shape, output));
    result?;
    Ok(index)
}

/// Both outputs return to their distinct slots before a recording error escapes.
fn compute_pair<B: EncoderOps>(
    backend: &mut B,
    scratch: &mut Scratch<B::Buffer>,
    shapes: [Shape; 2],
    apply: impl FnOnce(&B, &mut B::Buffer, &mut B::Buffer, &Scratch<B::Buffer>) -> Result<()>,
) -> Result<(usize, usize)> {
    let first = compute(backend, scratch, shapes[0], |_, _, _| Ok(()))?;
    let second = compute(backend, scratch, shapes[1], |_, _, _| Ok(()))?;
    let mut first_output = scratch.slots[first]
        .take()
        .ok_or(ExecutorError::BackendFailure(
            "encoder scratch value is unavailable",
        ))?;
    let Some(mut second_output) = scratch.slots[second].take() else {
        scratch.slots[first] = Some(first_output);
        return Err(ExecutorError::BackendFailure(
            "encoder scratch value is unavailable",
        ));
    };
    let result = apply(backend, &mut first_output.1, &mut second_output.1, scratch);
    scratch.slots[first] = Some(first_output);
    scratch.slots[second] = Some(second_output);
    result?;
    Ok((first, second))
}

fn linear<B: EncoderOps>(
    backend: &mut B,
    scratch: &mut Scratch<B::Buffer>,
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
            Lfm2ResolvedWeight::Dense(weight) => {
                backend.linear(output, buffers.buffer(input)?, weight)
            }
            Lfm2ResolvedWeight::Packed { codes, scales } => {
                backend.packed_linear(output, buffers.buffer(input)?, codes, scales)
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
    scratch: &mut Scratch<B::Buffer>,
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
                    buffers.buffer(residual)?,
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
                scratch.release(normalized)?;
                let mixed = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.centered_gated_convolution(
                            output,
                            buffers.buffer(projection)?,
                            weights.dense(layer(index, LayerRole::ConvKernel))?,
                            &input.segments,
                            GatedShortConvSpec::new(cfg.hidden_size, cfg.conv_width)?,
                        )
                    },
                )?;
                scratch.release(projection)?;
                let operator = linear(
                    backend,
                    scratch,
                    weights,
                    mixed,
                    rows,
                    hidden,
                    layer(index, LayerRole::ConvOutProjection),
                )?;
                scratch.release(mixed)?;
                operator
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
                scratch.release(normalized)?;
                let query_norm = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.head_rms_norm(
                            output,
                            buffers.buffer(query)?,
                            weights.dense(layer(index, LayerRole::QueryNorm))?,
                            PackedHeadSpec::new(cfg.attention_heads, cfg.head_dim)?,
                            cfg.block_norm_epsilon,
                        )
                    },
                )?;
                scratch.release(query)?;
                let key_norm = compute(
                    backend,
                    scratch,
                    Shape::new(&[rows, kv_width])?,
                    |backend, output, buffers| {
                        backend.head_rms_norm(
                            output,
                            buffers.buffer(key)?,
                            weights.dense(layer(index, LayerRole::KeyNorm))?,
                            PackedHeadSpec::new(cfg.key_value_heads, cfg.head_dim)?,
                            cfg.block_norm_epsilon,
                        )
                    },
                )?;
                scratch.release(key)?;
                let query_rope = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.split_half_rotary(
                            output,
                            buffers.buffer(query_norm)?,
                            input.segments.positions(),
                            RotarySpec::new(
                                PackedHeadSpec::new(cfg.attention_heads, cfg.head_dim)?,
                                cfg.rope_theta,
                            )?,
                        )
                    },
                )?;
                scratch.release(query_norm)?;
                let key_rope = compute(
                    backend,
                    scratch,
                    Shape::new(&[rows, kv_width])?,
                    |backend, output, buffers| {
                        backend.split_half_rotary(
                            output,
                            buffers.buffer(key_norm)?,
                            input.segments.positions(),
                            RotarySpec::new(
                                PackedHeadSpec::new(cfg.key_value_heads, cfg.head_dim)?,
                                cfg.rope_theta,
                            )?,
                        )
                    },
                )?;
                scratch.release(key_norm)?;
                let attended = compute(
                    backend,
                    scratch,
                    hidden_shape,
                    |backend, output, buffers| {
                        backend.bidirectional_gqa(
                            output,
                            buffers.buffer(query_rope)?,
                            buffers.buffer(key_rope)?,
                            buffers.buffer(value)?,
                            &input.segments,
                            GqaSpec::new(cfg.attention_heads, cfg.key_value_heads, cfg.head_dim)?,
                        )
                    },
                )?;
                scratch.release(query_rope)?;
                scratch.release(key_rope)?;
                scratch.release(value)?;
                let operator = linear(
                    backend,
                    scratch,
                    weights,
                    attended,
                    rows,
                    hidden,
                    layer(index, LayerRole::OutputProjection),
                )?;
                scratch.release(attended)?;
                operator
            }
        };
        let (sum, ffn_input) = if backend
            .capabilities()
            .operations
            .contains(OperationKind::AddRowRmsNorm)
        {
            let outputs = compute_pair(
                backend,
                scratch,
                [hidden_shape; 2],
                |backend, sum, normed, buffers| {
                    backend.add_row_rms_norm(
                        sum,
                        normed,
                        buffers.buffer(residual)?,
                        buffers.buffer(operator)?,
                        weights.dense(layer(index, LayerRole::FfnNorm))?,
                        cfg.block_norm_epsilon,
                    )
                },
            )?;
            scratch.release(residual)?;
            scratch.release(operator)?;
            outputs
        } else {
            let sum = compute(
                backend,
                scratch,
                hidden_shape,
                |backend, output, buffers| {
                    backend.add(output, buffers.buffer(residual)?, buffers.buffer(operator)?)
                },
            )?;
            scratch.release(residual)?;
            scratch.release(operator)?;
            let ffn_input = compute(
                backend,
                scratch,
                hidden_shape,
                |backend, output, buffers| {
                    backend.row_rms_norm(
                        output,
                        buffers.buffer(sum)?,
                        weights.dense(layer(index, LayerRole::FfnNorm))?,
                        cfg.block_norm_epsilon,
                    )
                },
            )?;
            (sum, ffn_input)
        };
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
        scratch.release(ffn_input)?;
        let activated = compute(
            backend,
            scratch,
            Shape::new(&[rows, intermediate])?,
            |backend, output, buffers| {
                backend.swiglu(output, buffers.buffer(gate)?, buffers.buffer(up)?)
            },
        )?;
        scratch.release(gate)?;
        scratch.release(up)?;
        let down = linear(
            backend,
            scratch,
            weights,
            activated,
            rows,
            hidden,
            layer(index, LayerRole::FfnW2),
        )?;
        scratch.release(activated)?;
        // Every final-layer token survives: pointer keys use the whole sequence.
        residual = compute(
            backend,
            scratch,
            hidden_shape,
            |backend, output, buffers| {
                backend.add(output, buffers.buffer(sum)?, buffers.buffer(down)?)
            },
        )?;
        scratch.release(sum)?;
        scratch.release(down)?;
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
                buffers.buffer(residual)?,
                weights.dense(Lfm2WeightRole::EmbeddingNorm)?,
                cfg.norm_epsilon,
            )
        },
    )?;
    scratch.release(residual)?;
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
    // Host uploads precede pending GPU dispatches, so the mask needs fresh storage.
    let mask_index = scratch.insert(
        hidden_shape,
        backend.upload_f32_classified(hidden_shape, &mask, AllocationClass::Scratch)?,
    )?;
    let encoded = compute(
        backend,
        scratch,
        hidden_shape,
        |backend, output, buffers| {
            backend.multiply(
                output,
                buffers.buffer(normalized)?,
                buffers.buffer(mask_index)?,
            )
        },
    )?;
    scratch.release(normalized)?;
    scratch.release(mask_index)?;
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
            backend.gather_rows(output, buffers.buffer(encoded)?, TokenIds::Host(&queries))
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
                backend.linear(output, buffers.buffer(asked)?, weights.pointer(query_role)?)
            },
        )?;
        let key = compute(
            backend,
            scratch,
            Shape::new(&[rows, pointer])?,
            |backend, output, buffers| {
                backend.linear(output, buffers.buffer(encoded)?, weights.pointer(key_role)?)
            },
        )?;
        scores.push(compute(
            backend,
            scratch,
            Shape::new(&[questions, rows])?,
            |backend, output, buffers| {
                backend.linear(output, buffers.buffer(query)?, buffers.buffer(key)?)
            },
        )?);
        scratch.release(query)?;
        scratch.release(key)?;
    }
    scratch.release(encoded)?;
    scratch.release(asked)?;
    let output = compute(
        backend,
        scratch,
        Shape::new(&[questions * 2, rows])?,
        |backend, output, buffers| {
            backend.copy_rect_2d(
                output,
                buffers.buffer(scores[0])?,
                RectCopy2d::new(0, 0, 0, 0, questions, rows),
            )?;
            backend.copy_rect_2d(
                output,
                buffers.buffer(scores[1])?,
                RectCopy2d::new(0, 0, questions, 0, questions, rows),
            )
        },
    )?;
    scratch.release(scores[0])?;
    scratch.release(scores[1])?;
    Ok(Some(output))
}
