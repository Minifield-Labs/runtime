//! Checked LFM2 weight inventory and typed role bindings.
//!
//! The generic loader can parse arbitrary bounded F32/BF16 containers for diagnostics. Delivered
//! LFM2 loading uses this module so checkpoint names, source shapes, source dtypes, and the tied
//! output head are derived from the already validated configuration before any upload starts.

use minifield_engine_api::{
    AssetLimits, ExecutorError, InferenceCompletion, InferenceOps, Result, TensorBinding,
    validate_asset_manifest,
};

use crate::{
    lfm2::{LayerKind, Lfm2Config, Lfm2StorageDType, parse_lfm2_config},
    loader::{
        LoadRequest, LoaderError, LoaderLimits, LoaderPoll, ParsedAsset, StorageDType,
        TypedWeights, WeightLayout, WeightLoadTask, WeightPlan, WeightRequirement,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lfm2WeightRole {
    TokenEmbedding,
    TiedLmHead,
    EmbeddingNorm,
    Layer {
        index: usize,
        role: Lfm2LayerWeightRole,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    plan: WeightPlan,
}

impl Lfm2WeightPlan {
    /// Derive the dense inventory for a validated configuration.
    pub fn from_config(config: Lfm2Config) -> Result<Self> {
        Self::from_config_with_format(config, Lfm2WeightFormat::Dense)
    }

    /// Derive all names, layouts, dimensions and source dtype before reading an asset header.
    /// This uses effective FF width, never raw `intermediate_size`.
    #[allow(clippy::too_many_lines)] // Exact inventory is intentionally listed together for audit.
    pub fn from_config_with_format(config: Lfm2Config, format: Lfm2WeightFormat) -> Result<Self> {
        config.validate()?;
        let source_dtype = match config.weight_storage_dtype {
            Lfm2StorageDType::F32 => StorageDType::F32,
            Lfm2StorageDType::BF16 => StorageDType::BF16,
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
            "token_embedding",
            "model.embed_tokens.weight",
            source_dtype,
            &[vocab, hidden],
            None,
        )?;
        push_matmul(
            &mut requirements,
            format,
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
                        &format!("layer.{index}.conv.in_projection"),
                        &format!("{prefix}.conv.in_proj.weight"),
                        source_dtype,
                        &[three_hidden, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
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
                        &format!("layer.{index}.attention.q_projection"),
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        source_dtype,
                        &[hidden, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        &format!("layer.{index}.attention.k_projection"),
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        source_dtype,
                        &[key_value, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
                        &format!("layer.{index}.attention.v_projection"),
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        source_dtype,
                        &[key_value, hidden],
                        None,
                    )?;
                    push_matmul(
                        &mut requirements,
                        format,
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
                &format!("layer.{index}.ffn.w1"),
                &format!("{prefix}.feed_forward.w1.weight"),
                source_dtype,
                &[intermediate, hidden],
                None,
            )?;
            push_matmul(
                &mut requirements,
                format,
                &format!("layer.{index}.ffn.w2"),
                &format!("{prefix}.feed_forward.w2.weight"),
                source_dtype,
                &[hidden, intermediate],
                None,
            )?;
            push_matmul(
                &mut requirements,
                format,
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
            plan,
        })
    }

    #[must_use]
    pub fn config(&self) -> &Lfm2Config {
        &self.config
    }

    #[must_use]
    pub const fn format(&self) -> Lfm2WeightFormat {
        self.format
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
    fn role_base_name(&self, role: Lfm2WeightRole) -> Result<String> {
        let name = match role {
            Lfm2WeightRole::TokenEmbedding => "token_embedding".to_owned(),
            Lfm2WeightRole::TiedLmHead => "tied_lm_head".to_owned(),
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

/// Emit one dense requirement or the `minifield.ternary.v1` split-stream pair,
/// depending on the plan's stored format. Packed roles keep the base name as a
/// prefix: `<role>.codes` is U8 `[rows, k/4]` and `<role>.scales` is F16
/// `[rows, k/128]`, where `[rows, k]` is the dense operator shape.
fn push_matmul(
    requirements: &mut Vec<WeightRequirement>,
    format: Lfm2WeightFormat,
    role: &str,
    tensor_name: &str,
    storage_dtype: StorageDType,
    dimensions: &[u64],
    tied_to_role: Option<&str>,
) -> Result<()> {
    match format {
        Lfm2WeightFormat::Dense => push(
            requirements,
            role,
            tensor_name,
            storage_dtype,
            dimensions,
            WeightLayout::Identity,
            tied_to_role,
        ),
        Lfm2WeightFormat::TernaryV1 => {
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
                &[rows, columns / 4],
                WeightLayout::Identity,
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
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lfm2LoadRequest {
    request: LoadRequest,
    plan: Lfm2WeightPlan,
}

impl Lfm2LoadRequest {
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
        let config = parse_lfm2_config(&config_bytes)?;
        let plan = Lfm2WeightPlan::from_config_with_format(config, format)?;
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
        let task = WeightLoadTask::begin(request.request)?;
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

    /// Resolve a role to its stored operand set: one dense buffer, or the
    /// packed ternary code/scale pair for packed matmul roles.
    pub fn resolve(&self, role: Lfm2WeightRole) -> Result<Lfm2ResolvedWeight<'_, Buffer>> {
        let base = self.plan.role_base_name(role)?;
        if self.plan.format == Lfm2WeightFormat::TernaryV1
            && self
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
    /// `minifield.ternary.v1` split streams used by `packed_linear`/`packed_gather_rows`.
    Packed {
        codes: &'a Buffer,
        scales: &'a Buffer,
    },
}
