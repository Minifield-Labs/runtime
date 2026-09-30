//! Hash the exact bytes consumed by the bounded runtime loader.
use crate::{
    HostResult,
    completion::{check, wait_loader},
    request::{Inputs, Request},
};
use minifield_engine_api::{InferenceOps, MemoryAssetProvider};
use minifield_executor_core::{
    EncoderLoadRequest, EncoderTypedWeights, EncoderWeightLoadTask, Lfm2LoadRequest,
    Lfm2TypedWeights, Lfm2WeightFormat, Lfm2WeightLoadTask, LoaderLimits,
    detect_lfm2_weight_format, parse_lfm2_tensor_quantization,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, time::Instant};

pub(crate) struct ModelAsset {
    config: Vec<u8>,
    config_hash: [u8; 32],
    bytes: Vec<u8>,
    weights_hash: [u8; 32],
    format: Lfm2WeightFormat,
    quantization: HashMap<String, Lfm2WeightFormat>,
}

pub(crate) struct Assets {
    pub model: ModelAsset,
    pub tokenizer: Vec<u8>,
    pub inputs: Inputs,
    pub identity: Value,
}

pub(crate) fn assets(request: &Request, limit: Instant) -> HostResult<Assets> {
    check(limit)?;
    let config = fs::read(request.bundle.join("config.json"))?;
    let bytes = fs::read(request.bundle.join("model.safetensors"))?;
    let tokenizer = fs::read(&request.tokenizer)?;
    let inputs_bytes = fs::read(&request.inputs)?;
    check(limit)?;
    let config_hash = Sha256::digest(&config).into();
    let weights_hash = Sha256::digest(&bytes).into();
    let identity = json!({"config_sha256":format!("{:x}",Sha256::digest(&config)),"weights_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "tokenizer_sha256":format!("{:x}",Sha256::digest(&tokenizer)),"inputs_sha256":format!("{:x}",Sha256::digest(&inputs_bytes))});
    let inputs: Inputs = serde_json::from_slice(&inputs_bytes)?;
    inputs.validate(request)?;
    let format = detect_lfm2_weight_format(&bytes)?;
    let quantization = parse_lfm2_tensor_quantization(&bytes)?;
    check(limit)?;
    Ok(Assets {
        model: ModelAsset {
            config,
            config_hash,
            bytes,
            weights_hash,
            format,
            quantization,
        },
        tokenizer,
        inputs,
        identity,
    })
}

fn limits(size: u64) -> HostResult<LoaderLimits> {
    Ok(LoaderLimits {
        max_asset_bytes: size,
        max_header_bytes: 1 << 20,
        max_source_tensor_bytes: size,
        max_retained_host_bytes: size.checked_mul(6).ok_or("loader budget overflows")?,
        max_tensor_name_bytes: 1024,
        max_tensors: 4096,
        max_rank: 4,
    })
}

pub(crate) fn classifier<B: InferenceOps>(
    asset: ModelAsset,
    backend: &mut B,
    classes: u32,
    limit: Instant,
) -> HostResult<Lfm2TypedWeights<B::Buffer>> {
    let size = u64::try_from(asset.bytes.len())?;
    let request = Lfm2LoadRequest::new_classifier_with_quantization(
        asset.config,
        asset.config_hash,
        size,
        asset.weights_hash,
        limits(size)?,
        classes,
        asset.format,
        &asset.quantization,
    )?;
    let mut task = Lfm2WeightLoadTask::begin(request)?;
    let mut provider = MemoryAssetProvider::new(asset.bytes, size);
    let result = wait_loader(limit, || task.poll_step(&mut provider, backend));
    if result.is_err() {
        let _ = task.cancel();
    }
    result
}

pub(crate) fn pointer<B: InferenceOps>(
    asset: ModelAsset,
    backend: &mut B,
    limit: Instant,
) -> HostResult<EncoderTypedWeights<B::Buffer>> {
    let size = u64::try_from(asset.bytes.len())?;
    let request = EncoderLoadRequest::new_with_quantization(
        asset.config,
        asset.config_hash,
        size,
        asset.weights_hash,
        limits(size)?,
        asset.format,
        &asset.quantization,
    )?;
    let mut task = EncoderWeightLoadTask::begin(request)?;
    let mut provider = MemoryAssetProvider::new(asset.bytes, size);
    let result = wait_loader(limit, || task.poll_step(&mut provider, backend));
    if result.is_err() {
        let _ = task.cancel();
    }
    result
}
