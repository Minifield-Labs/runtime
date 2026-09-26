//! Native GPU check of the same classifier used by the browser.
//! Usage: `classify BUNDLE_DIR PROMPTS_JSON` (array of strings).
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, MemoryAssetProvider, ResourceLimits, TokenChunk,
};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2LoadRequest, Lfm2WeightLoadTask, LoaderLimits,
    LoaderPoll, detect_lfm2_weight_format, parse_lfm2_tensor_quantization,
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
    let request = Lfm2LoadRequest::new_classifier_with_quantization(
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
        detect_lfm2_weight_format(&weights)?,
        &parse_lfm2_tensor_quantization(&weights)?,
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
    let encoded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|prompt| {
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let shared_prefix = env::var_os("CLASSIFY_PREFIX").is_some() && encoded.len() > 1;
    if shared_prefix {
        let mut common = encoded[0].len();
        for ids in &encoded[1..] {
            common = common.min(ids.len());
            while common > 0 && ids[..common] != encoded[0][..common] {
                common -= 1;
            }
        }
        let base_ids = &encoded[0][..common];
        eprintln!("shared prefix: {common} tokens");
        let mut base_task = classifier.prefill_base(TokenChunk::all(base_ids))?;
        let base = loop {
            match base_task.poll_step() {
                CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
                CompletionPoll::Ready(result) => break result?,
            }
        };
        for ids in &encoded {
            let started = Instant::now();
            let mut task = classifier.classify_tail(&base, TokenChunk::all(&ids[common..]))?;
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
        return Ok(());
    }
    for ids in &encoded {
        let started = Instant::now();
        let mut task = classifier.classify(TokenChunk::all(ids))?;
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
