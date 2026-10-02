//! Bundle loading into the browser's WebGPU backend.

use super::telemetry::BrowserTelemetry;
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{MemoryAssetProvider, ResourceLimits};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2TypedWeights,
    Lfm2WeightLoadTask, LoaderLimits, LoaderPoll,
};
use minifield_runtime_telemetry::Model;
use minifield_text_tokenizer::{Tokenizer, TokenizerLimits};
use sha2::{Digest, Sha256};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use super::{
    MAX_LOGICAL_TOKENS, WebClassifier, WebDemo,
    interop::{browser_yield, js_debug, js_error},
};

/// Load the packed ternary bundle from bytes the page fetched.
#[wasm_bindgen]
pub async fn load(
    config: Vec<u8>,
    weights: Vec<u8>,
    tokenizer: Vec<u8>,
) -> Result<WebDemo, JsValue> {
    let (backend, typed, model) =
        load_weights(config, weights, Sha256::digest(&tokenizer).into(), None).await?;
    let telemetry = BrowserTelemetry::new(model, &backend);
    let executor = Lfm2Executor::new(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: MAX_LOGICAL_TOKENS,
        },
    )
    .map_err(js_error)?;
    let tokenizer =
        Tokenizer::from_json_bytes(&tokenizer, TokenizerLimits::default()).map_err(js_error)?;
    Ok(WebDemo {
        telemetry,
        executor,
        tokenizer,
        tool_prefix: None,
    })
}

async fn load_weights(
    config: Vec<u8>,
    weights: Vec<u8>,
    tokenizer_hash: [u8; 32],
    classes: Option<u32>,
) -> Result<
    (
        WgpuBackend,
        Lfm2TypedWeights<minifield_backend_wgpu::WgpuBuffer>,
        Model,
    ),
    JsValue,
> {
    let limits = ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 2 << 30,
        max_pending_operations: 512,
    };
    let mut backend = WgpuBackend::new_async(0xE0_3C, limits)
        .await
        .map_err(js_error)?;

    let weights_len = weights.len() as u64;
    let loader_limits = LoaderLimits {
        max_asset_bytes: weights_len,
        max_header_bytes: 1 << 20,
        max_source_tensor_bytes: weights_len,
        max_retained_host_bytes: weights_len * 6,
        max_tensor_name_bytes: 1024,
        max_tensors: 4096,
        max_rank: 4,
    };
    let config_hash = Sha256::digest(&config).into();
    let weight_hash = Sha256::digest(&weights).into();
    let request = Lfm2LoadRequest::discover(
        config,
        config_hash,
        &weights,
        weight_hash,
        loader_limits,
        classes,
    )
    .map_err(js_debug)?;
    let model = Model::from_plan(
        request.plan(),
        request.quantization(),
        [config_hash, weight_hash, tokenizer_hash],
    );
    let mut provider = MemoryAssetProvider::new(weights, weights_len);
    let mut task = Lfm2WeightLoadTask::begin(request).map_err(js_debug)?;
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => browser_yield().await,
            LoaderPoll::Ready(result) => break result.map_err(js_debug)?,
        }
    };
    Ok((backend, typed, model))
}

#[wasm_bindgen]
pub async fn load_classifier(
    config: Vec<u8>,
    weights: Vec<u8>,
    tokenizer: Vec<u8>,
    classes: u32,
) -> Result<WebClassifier, JsValue> {
    let tokenizer_hash = Sha256::digest(&tokenizer).into();
    let tokenizer =
        Tokenizer::from_json_bytes(&tokenizer, TokenizerLimits::default()).map_err(js_error)?;
    let (backend, typed, model) =
        load_weights(config, weights, tokenizer_hash, Some(classes)).await?;
    let telemetry = BrowserTelemetry::new(model, &backend);
    let limits = Lfm2ExecutionLimits {
        max_logical_tokens: MAX_LOGICAL_TOKENS,
    };
    let classifier = Lfm2Classifier::new(backend, typed, limits).map_err(js_error)?;
    Ok(WebClassifier {
        telemetry,
        classes,
        classifier,
        tokenizer,
        max_logical_tokens: limits.max_logical_tokens,
        anchor_ids: None,
        shared_head: None,
        base: None,
    })
}
