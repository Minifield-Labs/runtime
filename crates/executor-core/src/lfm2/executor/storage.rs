//! Prefix storage allocation, cloning, and branch cache copying.

use super::{
    AllocationClass, ExecutorError, InferenceOps, LayerCache, LayerKind, Lfm2Prefix, ModelContext,
    PrefixStorage, Rc, RectCopy2d, Result, Shape, Tensor, shape,
};

pub(super) fn allocate<B: InferenceOps>(
    backend: &mut B,
    shape: Shape,
    class: AllocationClass,
) -> Result<Tensor<B>> {
    let buffer = if class == AllocationClass::Scratch {
        backend.allocate_f32_uninit(shape, class)?
    } else {
        backend.allocate_f32_classified(shape, class)?
    };
    Ok(Tensor { buffer, shape })
}
pub(super) fn clone_tensor<B: InferenceOps>(
    backend: &mut B,
    source: &Tensor<B>,
) -> Result<Tensor<B>> {
    // The copy overwrites every byte, so a zero-initializing allocation would
    // be a wasted dispatch on backends that record one.
    let mut output = Tensor {
        buffer: backend.allocate_f32_uninit(source.shape, AllocationClass::Cache)?,
        shape: source.shape,
    };
    backend.copy(&mut output.buffer, &source.buffer)?;
    Ok(output)
}

pub(super) fn publish<B: InferenceOps>(
    context: &ModelContext<B>,
    storage: PrefixStorage<B>,
) -> Lfm2Prefix<B> {
    Lfm2Prefix {
        owner: Rc::clone(&context.owner),
        lease: context.lease.clone(),
        config_sha256: context.weights.inner().config_sha256(),
        asset_sha256: context.weights.inner().asset_sha256(),
        numerical_mode: context.config().numerical_mode,
        storage: Rc::new(storage),
    }
}
pub(super) fn allocate_empty<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
) -> Result<PrefixStorage<B>> {
    let config = context.config();
    let mut history = Vec::new();
    history
        .try_reserve_exact(
            usize::try_from(context.limits.max_logical_tokens).map_err(|_| {
                ExecutorError::Overflow("configured logical capacity exceeds usize")
            })?,
        )
        .map_err(|_| ExecutorError::ResourceLimit("prefix token-history allocation failed"))?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers.len())
        .map_err(|_| ExecutorError::ResourceLimit("prefix layer-state allocation failed"))?;
    let kv_width = u64::from(config.key_value_heads)
        .checked_mul(u64::from(config.head_dim))
        .ok_or(ExecutorError::Overflow("LFM2 cache width overflows u64"))?;
    for kind in &config.layers {
        match kind {
            LayerKind::Conv => layers.push(LayerCache::Conv {
                history: allocate(
                    backend,
                    shape(
                        u64::from(config.conv_width - 1),
                        u64::from(config.hidden_size),
                    )?,
                    AllocationClass::Cache,
                )?,
            }),
            LayerKind::FullAttention => layers.push(LayerCache::Attention {
                key: allocate(
                    backend,
                    shape(context.limits.max_logical_tokens, kv_width)?,
                    AllocationClass::Cache,
                )?,
                value: allocate(
                    backend,
                    shape(context.limits.max_logical_tokens, kv_width)?,
                    AllocationClass::Cache,
                )?,
                length: 0,
            }),
        }
    }
    Ok(PrefixStorage {
        length: 0,
        history,
        layers,
        next_logits: None,
        sampled: None,
        sampled_id: None,
    })
}

pub(super) fn clone_storage<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    source: &PrefixStorage<B>,
    copy_logits: bool,
) -> Result<PrefixStorage<B>> {
    if source.length > context.limits.max_logical_tokens
        || source.history.len() as u64 != source.length
    {
        return Err(ExecutorError::InvalidArgument(
            "prefix cache metadata differs from logical token history",
        ));
    }
    let mut history = Vec::new();
    history
        .try_reserve_exact(source.history.len())
        .map_err(|_| ExecutorError::ResourceLimit("fork token-history allocation failed"))?;
    history.extend_from_slice(&source.history);
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(source.layers.len())
        .map_err(|_| ExecutorError::ResourceLimit("fork layer-state allocation failed"))?;
    for layer in &source.layers {
        match layer {
            LayerCache::Conv { history } => layers.push(LayerCache::Conv {
                history: clone_tensor(backend, history)?,
            }),
            LayerCache::Attention { key, value, length } => layers.push(LayerCache::Attention {
                key: clone_tensor(backend, key)?,
                value: clone_tensor(backend, value)?,
                length: *length,
            }),
        }
    }
    // Appends never read staged `next_logits` (the epilogue overwrites it
    // with fresh logits), so a full-vocabulary clone there is pure waste.
    // Forks publish the snapshot itself, so they keep the copy.
    let next_logits = match &source.next_logits {
        Some(tensor) if copy_logits => Some(clone_tensor(backend, tensor)?),
        _ => None,
    };
    // `sampled` is still cloned because the embedding gather reads it before
    // the epilogue replaces it.
    let sampled = match &source.sampled {
        Some(tensor) => Some(clone_tensor(backend, tensor)?),
        None => None,
    };
    Ok(PrefixStorage {
        length: source.length,
        history,
        layers,
        next_logits,
        sampled,
        sampled_id: source.sampled_id,
    })
}

/// Allocate branch working storage for an unscored base prefix without
/// recording any device work: source metadata, layer kinds, shapes, and live
/// cache lengths are validated, every destination cache uses
/// `allocate_f32_uninit` so no clear is recorded, and only host token history
/// is copied. `next_logits`, `sampled`, and `sampled_id` are not carried over.
/// `copy_branch_storage` records the device-side copies once the caller owns
/// this storage and can route it through the partial-recording quarantine.
pub(super) fn allocate_branch_storage<B: InferenceOps>(
    context: &ModelContext<B>,
    backend: &mut B,
    source: &PrefixStorage<B>,
) -> Result<PrefixStorage<B>> {
    if source.length > context.limits.max_logical_tokens
        || source.history.len() as u64 != source.length
    {
        return Err(ExecutorError::InvalidArgument(
            "prefix cache metadata differs from logical token history",
        ));
    }
    let mut history = Vec::new();
    history
        .try_reserve_exact(
            usize::try_from(context.limits.max_logical_tokens).map_err(|_| {
                ExecutorError::Overflow("configured logical capacity exceeds usize")
            })?,
        )
        .map_err(|_| ExecutorError::ResourceLimit("branch token-history allocation failed"))?;
    history.extend_from_slice(&source.history);
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(source.layers.len())
        .map_err(|_| ExecutorError::ResourceLimit("branch layer-state allocation failed"))?;
    for layer in &source.layers {
        match layer {
            LayerCache::Conv { history } => layers.push(LayerCache::Conv {
                history: Tensor {
                    buffer: backend.allocate_f32_uninit(history.shape, AllocationClass::Cache)?,
                    shape: history.shape,
                },
            }),
            LayerCache::Attention { key, value, length } => {
                if *length > key.shape.dim(0)? || key.shape != value.shape {
                    return Err(ExecutorError::InvalidArgument(
                        "prefix attention cache length exceeds its allocation",
                    ));
                }
                layers.push(LayerCache::Attention {
                    key: Tensor {
                        buffer: backend.allocate_f32_uninit(key.shape, AllocationClass::Cache)?,
                        shape: key.shape,
                    },
                    value: Tensor {
                        buffer: backend.allocate_f32_uninit(value.shape, AllocationClass::Cache)?,
                        shape: value.shape,
                    },
                    length: *length,
                });
            }
        }
    }
    Ok(PrefixStorage {
        length: source.length,
        history,
        layers,
        next_logits: None,
        sampled: None,
        sampled_id: None,
    })
}

/// Record the device-side copies that populate branch `destination` storage
/// from `source` after validating the two layer layouts correspond:
/// convolution history is copied fully and only the live attention rows are
/// copied. Recorded buffers must already be owned by the caller's
/// partial-recording quarantine path.
pub(super) fn copy_branch_storage<B: InferenceOps>(
    backend: &mut B,
    source: &PrefixStorage<B>,
    destination: &mut PrefixStorage<B>,
) -> Result<()> {
    if source.layers.len() != destination.layers.len() || source.length != destination.length {
        return Err(ExecutorError::InvalidArgument(
            "branch storage layout differs from its source prefix",
        ));
    }
    for (source_layer, destination_layer) in source.layers.iter().zip(destination.layers.iter_mut())
    {
        match (source_layer, destination_layer) {
            (
                LayerCache::Conv {
                    history: source_history,
                },
                LayerCache::Conv {
                    history: destination_history,
                },
            ) => {
                if source_history.shape != destination_history.shape {
                    return Err(ExecutorError::InvalidArgument(
                        "branch convolution history shape differs from its source prefix",
                    ));
                }
                backend.copy(&mut destination_history.buffer, &source_history.buffer)?;
            }
            (
                LayerCache::Attention {
                    key: source_key,
                    value: source_value,
                    length,
                },
                LayerCache::Attention {
                    key: destination_key,
                    value: destination_value,
                    length: destination_length,
                },
            ) => {
                if source_key.shape != destination_key.shape
                    || source_value.shape != destination_value.shape
                    || *length != *destination_length
                    || *length > source_key.shape.dim(0)?
                {
                    return Err(ExecutorError::InvalidArgument(
                        "branch attention cache layout differs from its source prefix",
                    ));
                }
                backend.copy_rect_2d(
                    &mut destination_key.buffer,
                    &source_key.buffer,
                    RectCopy2d::new(0, 0, 0, 0, *length, source_key.shape.dim(1)?),
                )?;
                backend.copy_rect_2d(
                    &mut destination_value.buffer,
                    &source_value.buffer,
                    RectCopy2d::new(0, 0, 0, 0, *length, source_value.shape.dim(1)?),
                )?;
            }
            _ => {
                return Err(ExecutorError::InvalidArgument(
                    "branch layer kind differs from its source prefix",
                ));
            }
        }
    }
    Ok(())
}
