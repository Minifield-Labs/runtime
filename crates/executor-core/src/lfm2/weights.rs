//! Checked LFM2 weight inventory and typed role bindings.
//!
//! The generic loader can parse arbitrary bounded F32/BF16 containers for diagnostics. Delivered
//! LFM2 loading uses this module so checkpoint names, source shapes, source dtypes, and the tied
//! output head are derived from the already validated configuration before any upload starts.

use minifield_engine_api::{
    AssetLimits, ExecutorError, InferenceCompletion, InferenceOps, Result, TensorBinding,
    validate_asset_manifest,
};

use std::collections::HashMap;

use crate::{
    lfm2::{LayerKind, Lfm2Config, Lfm2StorageDType, parse_lfm2_config},
    loader::{
        CheckedHeader, LoadRequest, LoaderError, LoaderLimits, LoaderPoll, ParsedAsset,
        StorageDType, TypedWeights, WeightLayout, WeightLoadTask, WeightPlan, WeightRequirement,
    },
};

const LFM2_CONFIG_NAME: &str = "lfm2";

/// Stored weight representation for a delivered LFM2 asset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lfm2WeightFormat {
    /// Every matrix weight is a dense F32/BF16 tensor.
    Dense,
    /// `minifield.ternary.v1` split streams: every matrix weight is a
    /// `<tensor>.codes` U8 [rows, k/4] tensor plus a `<tensor>.scales` F16
    /// [rows, k/128] tensor. Norm and convolution-kernel roles stay dense.
    TernaryV1,
    /// `minifield.nf4.v1` split streams: every matrix weight is a
    /// `<tensor>.codes` U8 [rows, k/2] tensor (two NF4 level indices per byte,
    /// low nibble first) plus a `<tensor>.scales` F16 [rows, k/128] tensor.
    /// Norm and convolution-kernel roles stay dense.
    Nf4V1,
    /// Signed byte codes U8 [rows, k] in two's-complement order, with
    /// F16 [rows, k/128] scales. The value -128 is reserved.
    Int8V1,
    /// `minifield.mixed.v1`: per-tensor quantization declared in the
    /// `tensor_quantization` metadata map. Listed tensors carry their named
    /// scheme (`ternary-v1`, `nf4-v1`); unlisted matmul roles stay dense.
    MixedV1,
}

impl Lfm2WeightFormat {
    /// Whether matmul roles are stored as packed code/scale split streams.
    /// Mixed bundles answer per role instead; see
    /// [`Lfm2WeightPlan::role_quant`].
    #[must_use]
    pub const fn is_packed(self) -> bool {
        matches!(self, Self::TernaryV1 | Self::Nf4V1 | Self::Int8V1)
    }
}

/// Sniff a `SafeTensors` asset's `__metadata__.format` marker without
/// ingesting tensors. Dense remains the default only when no marker is present.
/// The known `pt` producer marker also routes to dense; the bounded loader
/// still admits tensors by their dtype, shape, and exact inventory.
/// Explicit unknown formats or versions, and nonstring markers, reject.
/// This is a routing hint for callers; the real header validation happens in
/// the bounded loader.
pub fn detect_lfm2_weight_format(asset: &[u8]) -> Result<Lfm2WeightFormat> {
    match sniff_metadata(asset)?.get("format") {
        None => format_from_marker(None),
        Some(serde_json::Value::String(marker)) => format_from_marker(Some(marker)),
        Some(_) => Err(ExecutorError::InvalidArgument(
            "LFM2 weight format marker must be a string",
        )),
    }
}

fn format_from_marker(marker: Option<&str>) -> Result<Lfm2WeightFormat> {
    match marker {
        None | Some("pt") => Ok(Lfm2WeightFormat::Dense),
        Some("minifield.ternary.v1") => Ok(Lfm2WeightFormat::TernaryV1),
        Some("minifield.nf4.v1") => Ok(Lfm2WeightFormat::Nf4V1),
        Some("minifield.int8.v1") => Ok(Lfm2WeightFormat::Int8V1),
        Some("minifield.mixed.v1") => Ok(Lfm2WeightFormat::MixedV1),
        Some(_) => Err(ExecutorError::Unsupported(
            "unrecognized LFM2 weight format",
        )),
    }
}

/// Parse the `__metadata__.tensor_quantization` map: tensor name to the
/// scheme it was packed with (`"ternary-v1"`, `"nf4-v1"`, or a dense dtype
/// name). An absent declaration yields an empty map, which callers
/// treat as "every matmul role follows the bundle format".
pub fn parse_lfm2_tensor_quantization(
    asset: &[u8],
) -> Result<std::collections::HashMap<String, Lfm2WeightFormat>> {
    let metadata = sniff_metadata(asset)?;
    // `__metadata__` values are strings per the safetensors spec, so the map
    // arrives JSON-encoded. A bare object is accepted as well for callers
    // that hand-assembled a header.
    let entries = match metadata.get("tensor_quantization") {
        None => return Ok(std::collections::HashMap::new()),
        Some(serde_json::Value::String(encoded)) => serde_json::from_str::<
            serde_json::Map<String, serde_json::Value>,
        >(encoded)
        .map_err(|_| {
            ExecutorError::InvalidArgument("tensor_quantization metadata is not valid JSON")
        })?,
        Some(serde_json::Value::Object(entries)) => entries.clone(),
        Some(_) => {
            return Err(ExecutorError::InvalidArgument(
                "tensor_quantization metadata must be a JSON map",
            ));
        }
    };
    quantization_from_entries(&entries)
}

fn quantization_from_entries(
    entries: &serde_json::Map<String, serde_json::Value>,
) -> Result<HashMap<String, Lfm2WeightFormat>> {
    entries
        .iter()
        .map(|(name, level)| {
            let level = match level.as_str() {
                Some("ternary-v1") => Lfm2WeightFormat::TernaryV1,
                Some("nf4-v1") => Lfm2WeightFormat::Nf4V1,
                Some("int8-v1") => Lfm2WeightFormat::Int8V1,
                Some("dense" | "bf16" | "f16" | "f32") => Lfm2WeightFormat::Dense,
                _ => {
                    return Err(ExecutorError::InvalidArgument(
                        "tensor_quantization level must be ternary-v1, nf4-v1, int8-v1, or a dense dtype",
                    ));
                }
            };
            Ok((name.clone(), level))
        })
        .collect()
}

/// Stored representation derived exclusively from a bounded, validated header.
pub(crate) struct StoredRepresentation {
    pub(crate) header: CheckedHeader,
    pub(crate) format: Lfm2WeightFormat,
    pub(crate) quantization: HashMap<String, Lfm2WeightFormat>,
}

impl StoredRepresentation {
    pub(crate) fn discover(
        asset: &[u8],
        config_bytes: usize,
        limits: LoaderLimits,
    ) -> Result<Self> {
        let header = CheckedHeader::from_asset(asset, config_bytes, limits)?;
        let metadata = &header.parsed.metadata;
        let format = format_from_marker(metadata.get("format").map(String::as_str))?;
        let quantization = if let Some(encoded) = metadata.get("tensor_quantization") {
            let entries = serde_json::from_str(encoded).map_err(|_| {
                ExecutorError::InvalidArgument("tensor_quantization metadata is not a JSON map")
            })?;
            quantization_from_entries(&entries)?
        } else {
            HashMap::new()
        };
        Ok(Self {
            header,
            format,
            quantization,
        })
    }
}

/// Read a `SafeTensors` header's `__metadata__` object without ingesting
/// tensors. Dense remains the default when no marker is present.
fn sniff_metadata(asset: &[u8]) -> Result<serde_json::Map<String, serde_json::Value>> {
    const MAX_SNIFF_HEADER_BYTES: usize = 8 << 20;
    let prefix = asset.get(..8).ok_or(ExecutorError::InvalidArgument(
        "asset is shorter than a safetensors prefix",
    ))?;
    let header_len = u64::from_le_bytes(
        prefix
            .try_into()
            .map_err(|_| ExecutorError::InvalidArgument("safetensors prefix is unreadable"))?,
    );
    let header_len = usize::try_from(header_len)
        .map_err(|_| ExecutorError::Overflow("header length exceeds usize"))?;
    if header_len > MAX_SNIFF_HEADER_BYTES {
        return Err(ExecutorError::ResourceLimit(
            "safetensors header exceeds sniff limit",
        ));
    }
    let header =
        asset
            .get(8..8_usize.saturating_add(header_len))
            .ok_or(ExecutorError::OutOfBounds(
                "safetensors header extends beyond asset bytes",
            ))?;
    let value: serde_json::Value = serde_json::from_slice(header)
        .map_err(|_| ExecutorError::InvalidArgument("safetensors header is not valid JSON"))?;
    let header = value.as_object().ok_or(ExecutorError::InvalidArgument(
        "safetensors header must be a JSON object",
    ))?;
    match header.get("__metadata__") {
        None => Ok(serde_json::Map::new()),
        Some(serde_json::Value::Object(metadata)) => Ok(metadata.clone()),
        Some(_) => Err(ExecutorError::InvalidArgument(
            "safetensors metadata must be a JSON object",
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Lfm2WeightRole {
    TokenEmbedding,
    TiedLmHead,
    ClassificationHead,
    EmbeddingNorm,
    Layer {
        index: usize,
        role: Lfm2LayerWeightRole,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Lfm2LayerWeightRole {
    ConvKernel,
    ConvInProjection,
    ConvOutProjection,
    QueryNorm,
    KeyNorm,
    QueryProjection,
    KeyProjection,
    ValueProjection,
    OutputProjection,
    FfnW1,
    FfnW2,
    FfnW3,
    FfnNorm,
    OperatorNorm,
}

/// Exact physical LFM2 tensor inventory derived only from a validated configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Lfm2WeightPlan {
    config: Lfm2Config,
    format: Lfm2WeightFormat,
    classes: Option<u32>,
    plan: WeightPlan,
    /// Stored quantization scheme per matmul role base name (`token_embedding`,
    /// `layer.3.ffn.w1`, ...). Roles absent from this map are dense.
    quant_by_role: HashMap<Box<str>, Lfm2WeightFormat>,
}

impl Lfm2WeightPlan {
    /// Derive the dense inventory for a validated configuration.
    pub fn from_config(config: Lfm2Config) -> Result<Self> {
        Self::from_config_with_format(config, Lfm2WeightFormat::Dense)
    }

    /// Derive all names, layouts, dimensions and source dtype before reading an asset header.
    /// This uses effective FF width, never raw `intermediate_size`.
    pub fn from_config_with_format(config: Lfm2Config, format: Lfm2WeightFormat) -> Result<Self> {
        Self::from_config_with_quantization(config, format, &HashMap::new())
    }

    /// Derive the inventory with per-tensor quantization overrides, keyed by
    /// tensor name (`model.embed_tokens.weight`, ...). Overrides apply under
    /// every packed format; under `minifield.mixed.v1` unlisted matmul roles
    /// stay dense.
    #[allow(clippy::too_many_lines)] // Exact inventory is intentionally listed together for audit.
    pub fn from_config_with_quantization(
        config: Lfm2Config,
        format: Lfm2WeightFormat,
        overrides: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        config.validate()?;
        let source_dtype = match config.weight_storage_dtype {
            Lfm2StorageDType::F32 => StorageDType::F32,
            Lfm2StorageDType::BF16 => StorageDType::BF16,
            Lfm2StorageDType::F16 => StorageDType::F16,
        };
        let hidden = u64::from(config.hidden_size);
        let intermediate = u64::from(config.effective_intermediate_size);
        let head_dim = u64::from(config.head_dim);
        let key_value = u64::from(config.key_value_heads)
            .checked_mul(head_dim)
            .ok_or(ExecutorError::Overflow(
                "LFM2 key/value projection rows overflow",
            ))?;
        let vocab = u64::from(config.vocab_size);
        let conv_width = u64::from(config.conv_width);
        let three_hidden = hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
            "LFM2 convolution projection rows overflow",
        ))?;

        let mut requirements = Vec::new();
        let mut quant_by_role = HashMap::new();
        requirements
            .try_reserve_exact(
                3_usize
                    .checked_add(
                        config
                            .layers
                            .len()
                            .checked_mul(11)
                            .ok_or(ExecutorError::Overflow("LFM2 role count overflows usize"))?,
                    )
                    .ok_or(ExecutorError::Overflow("LFM2 role count overflows usize"))?,
            )
            .map_err(|_| ExecutorError::ResourceLimit("LFM2 plan allocation failed"))?;
        push_matmul(
            &mut requirements,
            format,
            overrides,
            &mut quant_by_role,
            "token_embedding",
            "model.embed_tokens.weight",
            source_dtype,
            &[vocab, hidden],
            None,
        )?;
        push_matmul(
            &mut requirements,
            format,
            overrides,
            &mut quant_by_role,
            "tied_lm_head",
            "model.embed_tokens.weight",
            source_dtype,
            &[vocab, hidden],
            Some("token_embedding"),
        )?;
        push(
            &mut requirements,
            "embedding_norm",
            "model.embedding_norm.weight",
            source_dtype,
            &[hidden],
            WeightLayout::Identity,
            None,
        )?;

        for (index, kind) in config.layers.iter().copied().enumerate() {
            let prefix = format!("model.layers.{index}");
            match kind {
                LayerKind::Conv => {
                    push(
                        &mut requirements,
                        &format!("layer.{index}.conv.kernel"),
                        &format!("{prefix}.conv.conv.weight"),
                        source_dtype,
                        &[hidden, 1, conv_width],
                        WeightLayout::ConvHiddenSingletonWidth,
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.conv.in_projection"),
                        &format!("{prefix}.conv.in_proj.weight"),
                        source_dtype,
                        &[three_hidden, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.conv.out_projection"),
                        &format!("{prefix}.conv.out_proj.weight"),
                        source_dtype,
                        &[hidden, hidden],
                        None,
                    )?;
                }
                LayerKind::FullAttention => {
                    push(
                        &mut requirements,
                        &format!("layer.{index}.attention.q_norm"),
                        &format!("{prefix}.self_attn.q_layernorm.weight"),
                        source_dtype,
                        &[head_dim],
                        WeightLayout::Identity,
                        None,
                    )?;
                    push(
                        &mut requirements,
                        &format!("layer.{index}.attention.k_norm"),
                        &format!("{prefix}.self_attn.k_layernorm.weight"),
                        source_dtype,
                        &[head_dim],
                        WeightLayout::Identity,
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.attention.q_projection"),
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        source_dtype,
                        &[hidden, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.attention.k_projection"),
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        source_dtype,
                        &[key_value, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.attention.v_projection"),
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        source_dtype,
                        &[key_value, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        overrides,
                        &mut quant_by_role,
                        &format!("layer.{index}.attention.out_projection"),
                        &format!("{prefix}.self_attn.out_proj.weight"),
                        source_dtype,
                        &[hidden, hidden],
                        None,
                    )?;
                }
            }
            push_matmul(
                &mut requirements,
                format,
                overrides,
                &mut quant_by_role,
                &format!("layer.{index}.ffn.w1"),
                &format!("{prefix}.feed_forward.w1.weight"),
                source_dtype,
                &[intermediate, hidden],
                None,
            )?;
            push_matmul(
                &mut requirements,
                format,
                overrides,
                &mut quant_by_role,
                &format!("layer.{index}.ffn.w2"),
                &format!("{prefix}.feed_forward.w2.weight"),
                source_dtype,
                &[hidden, intermediate],
                None,
            )?;
            push_matmul(
                &mut requirements,
                format,
                overrides,
                &mut quant_by_role,
                &format!("layer.{index}.ffn.w3"),
                &format!("{prefix}.feed_forward.w3.weight"),
                source_dtype,
                &[intermediate, hidden],
                None,
            )?;
            push(
                &mut requirements,
                &format!("layer.{index}.ffn_norm"),
                &format!("{prefix}.ffn_norm.weight"),
                source_dtype,
                &[hidden],
                WeightLayout::Identity,
                None,
            )?;
            push(
                &mut requirements,
                &format!("layer.{index}.operator_norm"),
                &format!("{prefix}.operator_norm.weight"),
                source_dtype,
                &[hidden],
                WeightLayout::Identity,
                None,
            )?;
        }
        let plan = WeightPlan {
            config_name: LFM2_CONFIG_NAME.to_owned(),
            requirements,
        };
        plan.validate(LoaderLimits {
            max_asset_bytes: u64::MAX,
            max_header_bytes: 1,
            max_source_tensor_bytes: u64::MAX,
            max_retained_host_bytes: 1,
            max_tensor_name_bytes: 1_024,
            max_tensors: usize::MAX,
            max_rank: 4,
        })?;
        Ok(Self {
            config,
            format,
            classes: None,
            plan,
            quant_by_role,
        })
    }

    /// Dense backbone with an independent last-token classification projection.
    pub fn from_config_classifier(config: Lfm2Config, classes: u32) -> Result<Self> {
        Self::from_config_classifier_with_format(config, classes, Lfm2WeightFormat::Dense)
    }

    /// Classifier backbone in either stored format. The classification head
    /// always stays dense; packed formats (`minifield.ternary.v1`,
    /// `minifield.nf4.v1`) pack every rank-two matmul role including the token
    /// embedding, while `tied_lm_head` is dropped.
    pub fn from_config_classifier_with_format(
        config: Lfm2Config,
        classes: u32,
        format: Lfm2WeightFormat,
    ) -> Result<Self> {
        Self::from_config_classifier_with_quantization(config, classes, format, &HashMap::new())
    }

    /// Classifier backbone with per-tensor quantization overrides; see
    /// [`Lfm2WeightPlan::from_config_with_quantization`].
    pub fn from_config_classifier_with_quantization(
        config: Lfm2Config,
        classes: u32,
        format: Lfm2WeightFormat,
        overrides: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        if classes == 0 || classes > 65_536 {
            return Err(ExecutorError::InvalidArgument(
                "classifier needs 1..=65536 classes",
            ));
        }
        let mut result = Self::from_config_with_quantization(config, format, overrides)?;
        result
            .plan
            .requirements
            .retain(|item| item.role != "tied_lm_head" && !item.role.starts_with("tied_lm_head."));
        let dtype = match result.config.weight_storage_dtype {
            Lfm2StorageDType::F32 => StorageDType::F32,
            Lfm2StorageDType::BF16 => StorageDType::BF16,
            Lfm2StorageDType::F16 => StorageDType::F16,
        };
        push(
            &mut result.plan.requirements,
            "classification_head",
            "classification_head.weight",
            dtype,
            &[u64::from(classes), u64::from(result.config.hidden_size)],
            WeightLayout::Identity,
            None,
        )?;
        result.classes = Some(classes);
        Ok(result)
    }

    #[must_use]
    pub fn config(&self) -> &Lfm2Config {
        &self.config
    }

    /// Explicit classifier width, or no classifier head for language models.
    #[must_use]
    pub const fn classes(&self) -> Option<u32> {
        self.classes
    }

    #[must_use]
    pub const fn format(&self) -> Lfm2WeightFormat {
        self.format
    }

    /// Stored quantization scheme for one role. Dense roles (norms,
    /// convolution kernels, unlisted `mixed.v1` tensors) return
    /// [`Lfm2WeightFormat::Dense`].
    #[must_use]
    pub fn role_quant(&self, role: Lfm2WeightRole) -> Lfm2WeightFormat {
        self.role_base_name(role)
            .ok()
            .and_then(|base| self.quant_by_role.get(base.as_str()).copied())
            .unwrap_or(Lfm2WeightFormat::Dense)
    }

    /// Whether any role is stored as a packed code/scale split stream.
    #[allow(clippy::case_sensitive_file_extension_comparisons)] // role names are not paths
    #[must_use]
    pub fn has_packed(&self) -> bool {
        self.plan
            .requirements
            .iter()
            .any(|item| item.role.ends_with(".codes"))
    }

    #[must_use]
    pub fn generic_plan(&self) -> &WeightPlan {
        &self.plan
    }

    #[must_use]
    pub fn physical_tensor_count(&self) -> usize {
        self.plan
            .requirements
            .iter()
            .filter(|requirement| requirement.tied_to_role.is_none())
            .count()
    }

    pub fn physical_parameter_count(&self) -> Result<u64> {
        self.plan
            .requirements
            .iter()
            .filter(|requirement| requirement.tied_to_role.is_none())
            .try_fold(0_u64, |total, requirement| {
                total
                    .checked_add(requirement.source_shape.element_count()?)
                    .ok_or(ExecutorError::Overflow(
                        "LFM2 physical parameter count overflows u64",
                    ))
            })
    }

    pub fn validate_parsed_asset(
        &self,
        parsed: &ParsedAsset,
        limits: LoaderLimits,
    ) -> Result<Vec<TensorBinding>> {
        validate_asset_manifest(
            &parsed.as_manifest(LFM2_CONFIG_NAME),
            LFM2_CONFIG_NAME,
            &self.plan.manifest_requirements(),
            AssetLimits {
                max_asset_bytes: limits.max_asset_bytes,
                max_tensor_bytes: limits.max_source_tensor_bytes,
                max_tensors: limits.max_tensors,
            },
        )
    }

    /// Base requirement name for a role, without checking whether that role
    /// exists in this plan's stored format.
    pub(crate) fn role_base_name(&self, role: Lfm2WeightRole) -> Result<String> {
        let name = match role {
            Lfm2WeightRole::TokenEmbedding => "token_embedding".to_owned(),
            Lfm2WeightRole::TiedLmHead => "tied_lm_head".to_owned(),
            Lfm2WeightRole::ClassificationHead => "classification_head".to_owned(),
            Lfm2WeightRole::EmbeddingNorm => "embedding_norm".to_owned(),
            Lfm2WeightRole::Layer { index, role } => {
                let kind = self
                    .config
                    .layers
                    .get(index)
                    .ok_or(ExecutorError::OutOfBounds(
                        "LFM2 layer role index exceeds configuration",
                    ))?;
                let suffix = match role {
                    Lfm2LayerWeightRole::ConvKernel if *kind == LayerKind::Conv => "conv.kernel",
                    Lfm2LayerWeightRole::ConvInProjection if *kind == LayerKind::Conv => {
                        "conv.in_projection"
                    }
                    Lfm2LayerWeightRole::ConvOutProjection if *kind == LayerKind::Conv => {
                        "conv.out_projection"
                    }
                    Lfm2LayerWeightRole::QueryNorm if *kind == LayerKind::FullAttention => {
                        "attention.q_norm"
                    }
                    Lfm2LayerWeightRole::KeyNorm if *kind == LayerKind::FullAttention => {
                        "attention.k_norm"
                    }
                    Lfm2LayerWeightRole::QueryProjection if *kind == LayerKind::FullAttention => {
                        "attention.q_projection"
                    }
                    Lfm2LayerWeightRole::KeyProjection if *kind == LayerKind::FullAttention => {
                        "attention.k_projection"
                    }
                    Lfm2LayerWeightRole::ValueProjection if *kind == LayerKind::FullAttention => {
                        "attention.v_projection"
                    }
                    Lfm2LayerWeightRole::OutputProjection if *kind == LayerKind::FullAttention => {
                        "attention.out_projection"
                    }
                    Lfm2LayerWeightRole::FfnW1 => "ffn.w1",
                    Lfm2LayerWeightRole::FfnW2 => "ffn.w2",
                    Lfm2LayerWeightRole::FfnW3 => "ffn.w3",
                    Lfm2LayerWeightRole::FfnNorm => "ffn_norm",
                    Lfm2LayerWeightRole::OperatorNorm => "operator_norm",
                    _ => {
                        return Err(ExecutorError::InvalidArgument(
                            "LFM2 role is invalid for layer kind",
                        ));
                    }
                };
                format!("layer.{index}.{suffix}")
            }
        };
        Ok(name)
    }

    fn role_name(&self, role: Lfm2WeightRole) -> Result<String> {
        let name = self.role_base_name(role)?;
        if self.plan.requirements.iter().any(|item| item.role == name) {
            Ok(name)
        } else {
            Err(ExecutorError::MissingRequiredTensor)
        }
    }

    fn bind<Buffer>(&self, inner: TypedWeights<Buffer>) -> Result<Lfm2TypedWeights<Buffer>> {
        for requirement in &self.plan.requirements {
            let _ = inner.buffer_for_role(&requirement.role)?;
        }
        Ok(Lfm2TypedWeights {
            plan: self.clone(),
            inner,
        })
    }
}

fn push(
    requirements: &mut Vec<WeightRequirement>,
    role: &str,
    tensor_name: &str,
    storage_dtype: StorageDType,
    dimensions: &[u64],
    layout: WeightLayout,
    tied_to_role: Option<&str>,
) -> Result<()> {
    requirements.push(WeightRequirement {
        role: role.to_owned(),
        tensor_name: tensor_name.to_owned(),
        storage_dtype,
        source_shape: minifield_engine_api::Shape::new(dimensions)?,
        layout,
        tied_to_role: tied_to_role.map(str::to_owned),
    });
    Ok(())
}

/// Emit one dense requirement or the packed split-stream pair for the plan's
/// stored format. Packed roles keep the base name as a prefix: `<role>.codes`
/// is U8 `[rows, k/weights_per_byte]` (four ternary weights per byte for
/// `minifield.ternary.v1`, two NF4 weights per byte for `minifield.nf4.v1`)
/// and `<role>.scales` is F16 `[rows, k/128]`, where `[rows, k]` is the dense
/// operator shape.
#[allow(clippy::too_many_arguments)]
fn push_matmul(
    requirements: &mut Vec<WeightRequirement>,
    format: Lfm2WeightFormat,
    overrides: &HashMap<String, Lfm2WeightFormat>,
    quant_by_role: &mut HashMap<Box<str>, Lfm2WeightFormat>,
    role: &str,
    tensor_name: &str,
    storage_dtype: StorageDType,
    dimensions: &[u64],
    tied_to_role: Option<&str>,
) -> Result<()> {
    let quant =
        overrides
            .get(tensor_name)
            .copied()
            .unwrap_or(if format == Lfm2WeightFormat::MixedV1 {
                Lfm2WeightFormat::Dense
            } else {
                format
            });
    quant_by_role.insert(role.into(), quant);
    let weights_per_byte = match quant {
        Lfm2WeightFormat::Dense => {
            return push(
                requirements,
                role,
                tensor_name,
                storage_dtype,
                dimensions,
                WeightLayout::Identity,
                tied_to_role,
            );
        }
        Lfm2WeightFormat::TernaryV1 => 4,
        Lfm2WeightFormat::Nf4V1 => 2,
        Lfm2WeightFormat::Int8V1 => 1,
        Lfm2WeightFormat::MixedV1 => {
            return Err(ExecutorError::InvalidArgument(
                "mixed.v1 must resolve to a per-tensor scheme",
            ));
        }
    };
    if dimensions.len() != 2 {
        return Err(ExecutorError::InvalidShape(
            "packed weight requirement must be rank two",
        ));
    }
    let (rows, columns) = (dimensions[0], dimensions[1]);
    if columns % 128 != 0 {
        return Err(ExecutorError::InvalidShape(
            "packed weight input width must be a multiple of 128",
        ));
    }
    let tied_codes = tied_to_role.map(|tied| format!("{tied}.codes"));
    let tied_scales = tied_to_role.map(|tied| format!("{tied}.scales"));
    push(
        requirements,
        &format!("{role}.codes"),
        &format!("{tensor_name}.codes"),
        StorageDType::U8,
        &[rows, columns / weights_per_byte],
        if quant == Lfm2WeightFormat::Int8V1 {
            WeightLayout::SignedInt8Codes
        } else {
            WeightLayout::Identity
        },
        tied_codes.as_deref(),
    )?;
    push(
        requirements,
        &format!("{role}.scales"),
        &format!("{tensor_name}.scales"),
        StorageDType::F16,
        &[rows, columns / 128],
        WeightLayout::Identity,
        tied_scales.as_deref(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lfm2LoadRequest {
    request: LoadRequest,
    plan: Lfm2WeightPlan,
    checked_header: Option<CheckedHeader>,
    quantization: HashMap<String, Lfm2WeightFormat>,
}

impl Lfm2LoadRequest {
    /// Discover storage from one checked header. `None` selects a language-model
    /// head; `Some(classes)` selects a classifier with that output width.
    /// The supplied bytes determine the asset length. Loading still checks the
    /// provider's header and complete asset against the expected identities.
    pub fn discover(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        asset: &[u8],
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        classes: Option<u32>,
    ) -> Result<Self> {
        let stored = StoredRepresentation::discover(asset, config_bytes.len(), limits)?;
        let size = stored.header.parsed.asset_bytes;
        let mut request = match classes {
            Some(classes) => Self::new_classifier_with_quantization(
                config_bytes,
                expected_config_sha256,
                size,
                expected_asset_sha256,
                limits,
                classes,
                stored.format,
                &stored.quantization,
            )?,
            None => Self::new_with_quantization(
                config_bytes,
                expected_config_sha256,
                size,
                expected_asset_sha256,
                limits,
                stored.format,
                &stored.quantization,
            )?,
        };
        request.checked_header = Some(stored.header);
        Ok(request)
    }

    /// Per-tensor storage declarations used to construct this request's plan.
    #[must_use]
    pub fn quantization(&self) -> &HashMap<String, Lfm2WeightFormat> {
        &self.quantization
    }

    /// Build a request for a dense F32/BF16 asset.
    pub fn new(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
    ) -> Result<Self> {
        Self::new_with_format(
            config_bytes,
            expected_config_sha256,
            declared_asset_bytes,
            expected_asset_sha256,
            limits,
            Lfm2WeightFormat::Dense,
        )
    }

    /// Build a request whose requirement inventory matches the asset's stored
    /// format (dense or `minifield.ternary.v1` split streams).
    pub fn new_with_format(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        format: Lfm2WeightFormat,
    ) -> Result<Self> {
        Self::new_with_quantization(
            config_bytes,
            expected_config_sha256,
            declared_asset_bytes,
            expected_asset_sha256,
            limits,
            format,
            &HashMap::new(),
        )
    }

    /// Build a request whose inventory matches the asset's stored format plus
    /// its `tensor_quantization` overrides.
    pub fn new_with_quantization(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        format: Lfm2WeightFormat,
        quantization: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        let config = parse_lfm2_config(&config_bytes)?;
        let plan = Lfm2WeightPlan::from_config_with_quantization(config, format, quantization)?;
        Ok(Self {
            request: LoadRequest {
                config_name: LFM2_CONFIG_NAME.to_owned(),
                config_bytes,
                expected_config_sha256,
                declared_asset_bytes,
                expected_asset_sha256,
                plan: plan.plan.clone(),
                limits,
            },
            plan,
            checked_header: None,
            quantization: quantization.clone(),
        })
    }

    /// Load classifier weights without changing the input vocabulary.
    pub fn new_classifier(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        classes: u32,
    ) -> Result<Self> {
        Self::new_classifier_with_format(
            config_bytes,
            expected_config_sha256,
            declared_asset_bytes,
            expected_asset_sha256,
            limits,
            classes,
            Lfm2WeightFormat::Dense,
        )
    }

    /// Load classifier weights in the given stored format.
    pub fn new_classifier_with_format(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        classes: u32,
        format: Lfm2WeightFormat,
    ) -> Result<Self> {
        Self::new_classifier_with_quantization(
            config_bytes,
            expected_config_sha256,
            declared_asset_bytes,
            expected_asset_sha256,
            limits,
            classes,
            format,
            &HashMap::new(),
        )
    }

    /// Load classifier weights matching the asset's stored format plus
    /// `tensor_quantization` overrides.
    #[allow(clippy::too_many_arguments)]
    pub fn new_classifier_with_quantization(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        classes: u32,
        format: Lfm2WeightFormat,
        quantization: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        let config = parse_lfm2_config(&config_bytes)?;
        let plan = Lfm2WeightPlan::from_config_classifier_with_quantization(
            config,
            classes,
            format,
            quantization,
        )?;
        Ok(Self {
            request: LoadRequest {
                config_name: LFM2_CONFIG_NAME.to_owned(),
                config_bytes,
                expected_config_sha256,
                declared_asset_bytes,
                expected_asset_sha256,
                plan: plan.plan.clone(),
                limits,
            },
            plan,
            checked_header: None,
            quantization: quantization.clone(),
        })
    }

    #[must_use]
    pub fn plan(&self) -> &Lfm2WeightPlan {
        &self.plan
    }
}

pub struct Lfm2WeightLoadTask<Read, Fence, Buffer> {
    task: WeightLoadTask<Read, Fence, Buffer>,
    plan: Lfm2WeightPlan,
}

impl<Read, Fence, Buffer> Lfm2WeightLoadTask<Read, Fence, Buffer>
where
    Read: InferenceCompletion<Output = minifield_engine_api::AssetBytes>,
    Fence: InferenceCompletion<Output = ()>,
{
    pub fn begin(request: Lfm2LoadRequest) -> core::result::Result<Self, LoaderError> {
        let plan = request.plan.clone();
        let task = WeightLoadTask::begin_checked(request.request, request.checked_header)?;
        Ok(Self { task, plan })
    }

    #[must_use]
    pub const fn resource_report(&self) -> crate::loader::LoaderResourceReport {
        self.task.resource_report()
    }

    pub fn cancel(&mut self) -> core::result::Result<(), LoaderError> {
        self.task.cancel()
    }

    pub fn poll_step<P, Backend>(
        &mut self,
        provider: &mut P,
        backend: &mut Backend,
    ) -> LoaderPoll<Lfm2TypedWeights<Buffer>>
    where
        P: minifield_engine_api::AssetProvider<Read = Read>,
        Backend: InferenceOps<Buffer = Buffer, Fence = Fence>,
    {
        match self.task.poll_step(provider, backend) {
            LoaderPoll::Pending => LoaderPoll::Pending,
            LoaderPoll::Ready(Err(error)) => LoaderPoll::Ready(Err(error)),
            LoaderPoll::Ready(Ok(inner)) => {
                LoaderPoll::Ready(self.plan.bind(inner).map_err(|cause| LoaderError {
                    stage: crate::loader::LoaderStage::FinalFence,
                    cause,
                }))
            }
        }
    }
}

#[derive(Debug)]
pub struct Lfm2TypedWeights<Buffer> {
    plan: Lfm2WeightPlan,
    inner: TypedWeights<Buffer>,
}

impl<Buffer> Lfm2TypedWeights<Buffer> {
    #[must_use]
    pub const fn classes(&self) -> Option<u32> {
        self.plan.classes
    }

    pub(crate) fn output_width(&self) -> u32 {
        self.classes().unwrap_or(self.config().vocab_size)
    }

    pub(crate) fn output_role(&self) -> Lfm2WeightRole {
        if self.classes().is_some() {
            Lfm2WeightRole::ClassificationHead
        } else {
            Lfm2WeightRole::TiedLmHead
        }
    }

    #[must_use]
    pub fn config(&self) -> &Lfm2Config {
        self.plan.config()
    }

    #[must_use]
    pub fn physical_tensor_count(&self) -> usize {
        self.plan.physical_tensor_count()
    }

    #[must_use]
    pub fn inner(&self) -> &TypedWeights<Buffer> {
        &self.inner
    }

    /// Dense-role lookup. Fails for roles stored as packed split streams;
    /// `resolve` is the format-agnostic entry point.
    pub fn buffer_for(&self, role: Lfm2WeightRole) -> Result<&Buffer> {
        self.inner.buffer_for_role(&self.plan.role_name(role)?)
    }

    #[must_use]
    pub const fn format(&self) -> Lfm2WeightFormat {
        self.plan.format
    }

    /// Stored quantization scheme for one role; see
    /// [`Lfm2WeightPlan::role_quant`].
    #[must_use]
    pub fn role_quant(&self, role: Lfm2WeightRole) -> Lfm2WeightFormat {
        self.plan.role_quant(role)
    }

    /// Whether any role is stored as a packed code/scale split stream.
    #[allow(clippy::case_sensitive_file_extension_comparisons)] // role names are not paths
    #[must_use]
    pub fn has_packed(&self) -> bool {
        self.plan.has_packed()
    }

    /// Resolve a role to its stored operand set: one dense buffer, or the
    /// packed code/scale pair for packed matmul roles.
    pub fn resolve(&self, role: Lfm2WeightRole) -> Result<Lfm2ResolvedWeight<'_, Buffer>> {
        let base = self.plan.role_base_name(role)?;
        if self
            .plan
            .plan
            .requirements
            .iter()
            .any(|item| item.role == format!("{base}.codes"))
        {
            return Ok(Lfm2ResolvedWeight::Packed {
                codes: self.inner.buffer_for_role(&format!("{base}.codes"))?,
                scales: self.inner.buffer_for_role(&format!("{base}.scales"))?,
            });
        }
        Ok(Lfm2ResolvedWeight::Dense(
            self.inner.buffer_for_role(&base)?,
        ))
    }
}

/// Stored operand set for one weight role.
pub enum Lfm2ResolvedWeight<'a, Buffer> {
    /// One dense f32 buffer used by `linear`/`gather_rows`.
    Dense(&'a Buffer),
    /// Packed split streams used by `packed_linear`/`packed_gather_rows`;
    /// the code width selects the decode (`minifield.ternary.v1` vs
    /// `minifield.nf4.v1`).
    Packed {
        codes: &'a Buffer,
        scales: &'a Buffer,
    },
}
