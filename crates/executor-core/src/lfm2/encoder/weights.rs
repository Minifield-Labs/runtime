use super::{EncoderConfig, parse_encoder_config};
use crate::lfm2::{Lfm2ResolvedWeight, Lfm2WeightFormat, Lfm2WeightPlan, Lfm2WeightRole};
use crate::{
    LoadRequest, LoaderError, LoaderLimits, LoaderPoll, LoaderResourceReport, LoaderStage,
    TypedWeights, WeightLayout, WeightLoadTask, WeightPlan, WeightRequirement,
};
use crate::{lfm2::weights::StoredRepresentation, loader::CheckedHeader};
use minifield_engine_api::{
    AssetProvider, ExecutorError, InferenceCompletion, InferenceOps, Result, Shape,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointerWeightRole {
    StartQuery,
    StartKey,
    EndQuery,
    EndKey,
}

impl PointerWeightRole {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::StartQuery => "pointer.start_query",
            Self::StartKey => "pointer.start_key",
            Self::EndQuery => "pointer.end_query",
            Self::EndKey => "pointer.end_key",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncoderWeightPlan {
    config: EncoderConfig,
    backbone: Lfm2WeightPlan,
    plan: WeightPlan,
}

impl EncoderWeightPlan {
    pub fn from_config_with_quantization(
        config: EncoderConfig,
        format: Lfm2WeightFormat,
        overrides: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        let backbone = Lfm2WeightPlan::from_config_with_quantization(
            config.backbone.clone(),
            format,
            overrides,
        )?;
        let mut plan = backbone.generic_plan().clone();
        plan.requirements
            .retain(|item| !item.role.starts_with("tied_lm_head"));
        let dtype = plan
            .requirements
            .iter()
            .find(|item| item.role == "embedding_norm")
            .ok_or(ExecutorError::MissingRequiredTensor)?
            .storage_dtype;
        for role in [
            PointerWeightRole::StartQuery,
            PointerWeightRole::StartKey,
            PointerWeightRole::EndQuery,
            PointerWeightRole::EndKey,
        ] {
            plan.requirements.push(WeightRequirement {
                role: role.name().to_owned(),
                tensor_name: format!("{}.weight", role.name()),
                storage_dtype: dtype,
                source_shape: Shape::new(&[
                    u64::from(config.pointer_width),
                    u64::from(config.backbone.hidden_size),
                ])?,
                layout: WeightLayout::Identity,
                tied_to_role: None,
            });
        }
        Ok(Self {
            config,
            backbone,
            plan,
        })
    }

    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
    #[must_use]
    pub fn generic_plan(&self) -> &WeightPlan {
        &self.plan
    }
    #[must_use]
    pub fn role_quant(&self, role: Lfm2WeightRole) -> Lfm2WeightFormat {
        self.backbone.role_quant(role)
    }
    pub fn physical_parameter_count(&self) -> Result<u64> {
        self.plan
            .requirements
            .iter()
            .filter(|item| item.tied_to_role.is_none())
            .try_fold(0_u64, |total, item| {
                total.checked_add(item.source_shape.element_count()?).ok_or(
                    ExecutorError::Overflow("encoder physical parameter count overflows u64"),
                )
            })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncoderLoadRequest {
    request: LoadRequest,
    plan: EncoderWeightPlan,
    checked_header: Option<CheckedHeader>,
}

impl EncoderLoadRequest {
    /// Discover the backbone storage from a checked header and select the pointer head.
    pub fn discover(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        asset: &[u8],
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
    ) -> Result<Self> {
        let stored = StoredRepresentation::discover(asset, config_bytes.len(), limits)?;
        let mut request = Self::new_with_quantization(
            config_bytes,
            expected_config_sha256,
            stored.header.parsed.asset_bytes,
            expected_asset_sha256,
            limits,
            stored.format,
            &stored.quantization,
        )?;
        request.checked_header = Some(stored.header);
        Ok(request)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_quantization(
        config_bytes: Vec<u8>,
        expected_config_sha256: [u8; 32],
        declared_asset_bytes: u64,
        expected_asset_sha256: [u8; 32],
        limits: LoaderLimits,
        format: Lfm2WeightFormat,
        overrides: &HashMap<String, Lfm2WeightFormat>,
    ) -> Result<Self> {
        let config = parse_encoder_config(&config_bytes)?;
        let plan = EncoderWeightPlan::from_config_with_quantization(config, format, overrides)?;
        let request = LoadRequest {
            config_name: plan.plan.config_name.clone(),
            config_bytes,
            expected_config_sha256,
            declared_asset_bytes,
            expected_asset_sha256,
            plan: plan.plan.clone(),
            limits,
        };
        Ok(Self {
            request,
            plan,
            checked_header: None,
        })
    }
    #[must_use]
    pub fn plan(&self) -> &EncoderWeightPlan {
        &self.plan
    }
}

pub struct EncoderWeightLoadTask<Read, Fence, Buffer> {
    task: WeightLoadTask<Read, Fence, Buffer>,
    plan: EncoderWeightPlan,
}

impl<Read, Fence, Buffer> EncoderWeightLoadTask<Read, Fence, Buffer>
where
    Read: InferenceCompletion<Output = minifield_engine_api::AssetBytes>,
    Fence: InferenceCompletion<Output = ()>,
{
    pub fn begin(request: EncoderLoadRequest) -> core::result::Result<Self, LoaderError> {
        Ok(Self {
            plan: request.plan,
            task: WeightLoadTask::begin_checked(request.request, request.checked_header)?,
        })
    }
    #[must_use]
    pub const fn resource_report(&self) -> LoaderResourceReport {
        self.task.resource_report()
    }
    pub fn cancel(&mut self) -> core::result::Result<(), LoaderError> {
        self.task.cancel()
    }
    pub fn poll_step<P, B>(
        &mut self,
        provider: &mut P,
        backend: &mut B,
    ) -> LoaderPoll<EncoderTypedWeights<Buffer>>
    where
        P: AssetProvider<Read = Read>,
        B: InferenceOps<Buffer = Buffer, Fence = Fence>,
    {
        match self.task.poll_step(provider, backend) {
            LoaderPoll::Pending => LoaderPoll::Pending,
            LoaderPoll::Ready(Err(error)) => LoaderPoll::Ready(Err(error)),
            LoaderPoll::Ready(Ok(inner)) => {
                let validation = self
                    .plan
                    .plan
                    .requirements
                    .iter()
                    .try_for_each(|requirement| {
                        inner.buffer_for_role(&requirement.role).map(|_| ())
                    });
                LoaderPoll::Ready(
                    validation
                        .map(|()| EncoderTypedWeights {
                            plan: self.plan.clone(),
                            inner,
                        })
                        .map_err(|cause| LoaderError {
                            stage: LoaderStage::FinalFence,
                            cause,
                        }),
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct EncoderTypedWeights<Buffer> {
    plan: EncoderWeightPlan,
    inner: TypedWeights<Buffer>,
}

impl<Buffer> EncoderTypedWeights<Buffer> {
    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.plan.config
    }
    #[must_use]
    pub fn inner(&self) -> &TypedWeights<Buffer> {
        &self.inner
    }
    pub fn pointer(&self, role: PointerWeightRole) -> Result<&Buffer> {
        self.inner.buffer_for_role(role.name())
    }
    pub fn dense(&self, role: Lfm2WeightRole) -> Result<&Buffer> {
        self.inner
            .buffer_for_role(&self.plan.backbone.role_base_name(role)?)
    }
    pub fn resolve(&self, role: Lfm2WeightRole) -> Result<Lfm2ResolvedWeight<'_, Buffer>> {
        let name = self.plan.backbone.role_base_name(role)?;
        if self
            .plan
            .plan
            .requirements
            .iter()
            .any(|item| item.role == format!("{name}.codes"))
        {
            Ok(Lfm2ResolvedWeight::Packed {
                codes: self.inner.buffer_for_role(&format!("{name}.codes"))?,
                scales: self.inner.buffer_for_role(&format!("{name}.scales"))?,
            })
        } else {
            Ok(Lfm2ResolvedWeight::Dense(
                self.inner.buffer_for_role(&name)?,
            ))
        }
    }
}
