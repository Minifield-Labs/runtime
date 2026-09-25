//! Native GPU check of the same classifier used by the browser.
//! Usage: `classify BUNDLE_DIR PROMPTS_JSON` (array of strings).
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, MemoryAssetProvider, ResourceLimits, TokenChunk,
};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2LoadRequest, Lfm2WeightLoadTask, LoaderLimits,
    LoaderPoll,
};
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerLimits};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: classify BUNDLE_DIR PROMPTS_JSON".into());
    }
    let dir = PathBuf::from(&args[1]);
    let config = fs::read(dir.join("config.json"))?;
    let weights = fs::read(dir.join("model.safetensors"))?;
    let size = u64::try_from(weights.len())?;
    let tokenizer = Tokenizer::from_json_bytes(
        &fs::read(dir.join("tokenizer.json"))?,
        TokenizerLimits::default(),
    )?;
    let prompts: Vec<String> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut backend = WgpuBackend::new(
        19,
        ResourceLimits {
            max_allocation_bytes: 1 << 30,
            max_total_bytes: 2 << 30,
            max_pending_operations: 512,
        },
    )?;
    let request = Lfm2LoadRequest::new_classifier(
        config.clone(),
        Sha256::digest(&config).into(),
        size,
        Sha256::digest(&weights).into(),
        LoaderLimits {
            max_asset_bytes: size,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: size,
            max_retained_host_bytes: size * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        8,
    )?;
    let mut provider = MemoryAssetProvider::new(weights, size);
    let mut task = Lfm2WeightLoadTask::begin(request)?;
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            LoaderPoll::Ready(result) => break result?,
        }
    };
    let mut classifier = Lfm2Classifier::new(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: 512,
        },
    )?;
    for prompt in prompts {
        let ids = tokenizer.encode(
            &prompt,
            EncodeOptions {
                add_special_tokens: false,
            },
        )?;
        let started = Instant::now();
        let mut task = classifier.classify(TokenChunk::all(&ids))?;
        let logits = loop {
            match task.poll_step() {
                CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
                CompletionPoll::Ready(result) => break result?,
            }
        };
        println!(
            "{}",
            serde_json::json!({"ids":ids,"logits":logits,"seconds":started.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}
