//! Explicit CPU loader/inference probe for generated converter fixtures.
//! Usage: `cargo run -p minifield-executor-core --example converter_probe -- BUNDLE [--classes N]`

use std::{env, error::Error, fs::File, io::Read, path::Path};

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, ExecutorError, InferenceCompletion, MemoryAssetProvider, ResourceLimits,
    TokenChunk, TokenExecutor,
};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2TypedWeights,
    Lfm2WeightLoadTask, LoaderLimits, LoaderPoll,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const PROMPT: [u32; 2] = [1, 3];
const TAIL: [u32; 1] = [4];
const FULL: [u32; 3] = [1, 3, 4];
const MAX_POLLS: usize = 1_000_000;
type CpuWeights = Lfm2TypedWeights<<CpuBackend as minifield_engine_api::InferenceOps>::Buffer>;

fn complete<T: InferenceCompletion>(mut task: T) -> Result<T::Output, ExecutorError> {
    for _ in 0..MAX_POLLS {
        if let CompletionPoll::Ready(result) = task.poll_step() {
            return result;
        }
    }
    Err(ExecutorError::ResourceLimit("probe completion poll limit"))
}

fn bounded_read(path: &Path, maximum: u64) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > maximum {
        return Err("probe asset exceeds its explicit size limit".into());
    }
    Ok(bytes)
}

fn load(
    directory: &Path,
    classes: Option<u32>,
) -> Result<(CpuBackend, CpuWeights), Box<dyn Error>> {
    let config = bounded_read(&directory.join("config.json"), 1 << 20)?;
    let asset = bounded_read(&directory.join("model.safetensors"), 16 << 20)?;
    let config_hash = Sha256::digest(&config).into();
    let asset_hash = Sha256::digest(&asset).into();
    let size = u64::try_from(asset.len())?;
    let limits = LoaderLimits {
        max_asset_bytes: size,
        max_header_bytes: 1 << 20,
        max_source_tensor_bytes: size,
        max_retained_host_bytes: size * 6,
        max_tensor_name_bytes: 1024,
        max_tensors: 4096,
        max_rank: 4,
    };
    let request =
        Lfm2LoadRequest::discover(config, config_hash, &asset, asset_hash, limits, classes)?;
    let mut backend = CpuBackend::new(
        0xC017,
        ResourceLimits {
            max_allocation_bytes: 16 << 20,
            max_total_bytes: 64 << 20,
            max_pending_operations: 256,
        },
    );
    let mut provider = MemoryAssetProvider::new(asset, size);
    let mut task = Lfm2WeightLoadTask::begin(request)?;
    for _ in 0..MAX_POLLS {
        if let LoaderPoll::Ready(result) = task.poll_step(&mut provider, &mut backend) {
            return Ok((backend, result?));
        }
    }
    Err("probe loader poll limit".into())
}

fn lm(backend: CpuBackend, weights: CpuWeights) -> Result<Value, ExecutorError> {
    let mut executor = Lfm2Executor::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    )?;
    let base = complete(executor.prefill(TokenChunk::all(&PROMPT))?)?;
    let logits = complete(executor.next_logits(&base)?)?;
    let appended = complete(executor.append_known(&base, TokenChunk::all(&TAIL))?)?;
    let append_logits = complete(executor.next_logits(&appended)?)?;
    let fresh = complete(executor.prefill(TokenChunk::all(&FULL))?)?;
    let fresh_logits = complete(executor.next_logits(&fresh)?)?;
    Ok(json!({
        "mode": "lm", "tokens": PROMPT, "logits": logits,
        "append_tokens": TAIL, "append_logits": append_logits, "fresh_logits": fresh_logits,
        "base_state": {"logical_length": base.logical_length(), "token_history": base.token_history()},
        "append_state": {"logical_length": appended.logical_length(), "token_history": appended.token_history()},
    }))
}

fn classifier(backend: CpuBackend, weights: CpuWeights) -> Result<Value, ExecutorError> {
    let mut executor = Lfm2Classifier::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    )?;
    let logits = complete(executor.classify(TokenChunk::all(&PROMPT))?)?;
    let base = complete(executor.prefill_base(TokenChunk::all(&PROMPT))?)?;
    let append_logits = complete(executor.classify_tail(&base, TokenChunk::all(&TAIL))?)?;
    let fresh_logits = complete(executor.classify(TokenChunk::all(&FULL))?)?;
    Ok(json!({
        "mode": "classifier", "tokens": PROMPT, "logits": logits,
        "append_tokens": TAIL, "append_logits": append_logits, "fresh_logits": fresh_logits,
        // classify_tail preserves the shared base; its branch prefix is private.
        "base_state": {"logical_length": base.logical_length(), "token_history": base.token_history()},
        "append_sequence": FULL,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 && (arguments.len() != 3 || arguments[1] != "--classes") {
        return Err("usage: converter_probe BUNDLE [--classes N]".into());
    }
    let classes = if arguments.len() == 3 {
        Some(
            arguments[2]
                .to_str()
                .ok_or("classes must be UTF-8")?
                .parse::<u32>()?,
        )
    } else {
        None
    };
    let (backend, weights) = load(Path::new(&arguments[0]), classes)?;
    let output = if classes.is_some() {
        classifier(backend, weights)?
    } else {
        lm(backend, weights)?
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
