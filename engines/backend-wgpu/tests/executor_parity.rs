//! Full-model wgpu-vs-CPU parity: the real LFM2.5-230M dense bundle runs the
//! same teacher-forced token stream through `Lfm2Executor` on both backends,
//! and per-position logits are compared. Skips cleanly without
//! `MINIFIELD_LFM25_BUNDLE_DIR` or a usable GPU adapter.
#![allow(clippy::cast_possible_truncation, clippy::expect_used)]

use std::{fs, path::PathBuf};

use minifield_backend_cpu::CpuBackend;
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, InferenceOps, MemoryAssetProvider, ResourceLimits,
    TokenChunk, TokenExecutor,
};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2WeightFormat, Lfm2WeightLoadTask,
    LoaderLimits, LoaderPoll,
};
use sha2::{Digest, Sha256};

const LOGIT_TOLERANCE: f32 = 0.05;

fn ready<T, C: InferenceCompletion<Output = T>>(completion: &mut C) -> T {
    for _ in 0..1_000_000 {
        match completion.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(Ok(value)) => return value,
            CompletionPoll::Ready(Err(error)) => panic!("completion error: {error:?}"),
        }
    }
    panic!("completion did not become ready")
}

fn load<B: InferenceOps>(mut backend: B, config: &[u8], weights: &[u8]) -> Lfm2Executor<B> {
    let request = Lfm2LoadRequest::new(
        config.to_vec(),
        Sha256::digest(config).into(),
        weights.len() as u64,
        Sha256::digest(weights).into(),
        LoaderLimits {
            max_asset_bytes: weights.len() as u64,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: weights.len() as u64,
            max_retained_host_bytes: weights.len() as u64 * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
    )
    .expect("load request");
    let mut provider = MemoryAssetProvider::new(weights.to_vec(), weights.len() as u64);
    let mut task = Lfm2WeightLoadTask::begin(request).expect("load task");
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Ok(typed)) => break typed,
            LoaderPoll::Ready(Err(error)) => panic!("loader error: {error:?}"),
        }
    };
    Lfm2Executor::new(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: 64,
        },
    )
    .expect("executor")
}

fn teacher_forced_logits<B: InferenceOps>(
    executor: &mut Lfm2Executor<B>,
    tokens: &[u32],
) -> (Vec<f32>, Vec<u32>) {
    let mut task = executor
        .prefill(TokenChunk::all(&tokens[..1]))
        .expect("prefill");
    let mut prefix = ready(&mut task);
    let mut top_ids = Vec::new();
    for token in tokens.iter().skip(1) {
        let mut logits_task = executor.next_logits(&prefix).expect("logits");
        top_ids.push(argmax(&ready(&mut logits_task)));
        let mut append = executor
            .append_known(&prefix, TokenChunk::all(&[*token]))
            .expect("append");
        prefix = ready(&mut append);
    }
    let mut logits_task = executor.next_logits(&prefix).expect("logits");
    let logits = ready(&mut logits_task);
    top_ids.push(argmax(&logits));
    (logits, top_ids)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .expect("argmax")
}

#[test]
#[ignore = "requires MINIFIELD_LFM25_BUNDLE_DIR and a wgpu adapter"]
fn wgpu_executor_matches_cpu_on_real_model() {
    let bundle = PathBuf::from(
        std::env::var_os("MINIFIELD_LFM25_BUNDLE_DIR").expect("MINIFIELD_LFM25_BUNDLE_DIR"),
    );
    let config = fs::read(bundle.join("config.json")).expect("config.json");
    let weights = fs::read(bundle.join("model.safetensors")).expect("model.safetensors");

    // Oracle prompt p01, tokenized by llama.cpp (BOS included).
    let tokens: Vec<u32> = vec![
        1, 1098, 4605, 10800, 36387, 56586, 1391, 779, 46199, 4949, 3627, 779,
    ];
    let limits = ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 6 << 30,
        max_pending_operations: 512,
    };

    let gpu_start = std::time::Instant::now();
    let gpu = WgpuBackend::new(0xE0_3C, limits).expect("wgpu backend");
    let mut gpu_exec = load(gpu, &config, &weights);
    let gpu_load = gpu_start.elapsed();
    let gpu_run = std::time::Instant::now();
    let (gpu_logits, gpu_top) = teacher_forced_logits(&mut gpu_exec, &tokens);
    let gpu_run = gpu_run.elapsed();

    let cpu_start = std::time::Instant::now();
    let cpu = CpuBackend::new(0xE0_2C, limits);
    let mut cpu_exec = load(cpu, &config, &weights);
    let cpu_load = cpu_start.elapsed();
    let cpu_run = std::time::Instant::now();
    let (cpu_logits, cpu_top) = teacher_forced_logits(&mut cpu_exec, &tokens);
    let cpu_run = cpu_run.elapsed();

    assert_eq!(gpu_logits.len(), cpu_logits.len(), "logit width");
    let max_delta = gpu_logits
        .iter()
        .zip(cpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let agreements = gpu_top
        .iter()
        .zip(cpu_top.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "wgpu vs cpu: {}/{} top-1 positions agree, final-position max|delta|={max_delta:.4}",
        agreements,
        gpu_top.len()
    );
    println!(
        "wgpu: load {gpu_load:?}, 12-token run {gpu_run:?}; \
         cpu: load {cpu_load:?}, run {cpu_run:?}"
    );
    println!("wgpu top-1 sequence: {gpu_top:?}");
    println!("cpu  top-1 sequence: {cpu_top:?}");
    assert_eq!(gpu_top, cpu_top, "top-1 sequences diverged");
    assert!(
        max_delta <= LOGIT_TOLERANCE,
        "logit divergence {max_delta} exceeds {LOGIT_TOLERANCE}"
    );
}

fn load_format<B: InferenceOps>(
    mut backend: B,
    config: &[u8],
    weights: &[u8],
    format: Lfm2WeightFormat,
) -> Lfm2Executor<B> {
    let request = Lfm2LoadRequest::new_with_format(
        config.to_vec(),
        Sha256::digest(config).into(),
        weights.len() as u64,
        Sha256::digest(weights).into(),
        LoaderLimits {
            max_asset_bytes: weights.len() as u64,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: weights.len() as u64,
            max_retained_host_bytes: weights.len() as u64 * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        format,
    )
    .expect("load request");
    let mut provider = MemoryAssetProvider::new(weights.to_vec(), weights.len() as u64);
    let mut task = Lfm2WeightLoadTask::begin(request).expect("load task");
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Ok(typed)) => break typed,
            LoaderPoll::Ready(Err(error)) => panic!("loader error: {error:?}"),
        }
    };
    Lfm2Executor::new(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: 64,
        },
    )
    .expect("executor")
}

/// Packed ternary end-to-end: the real `minifield.ternary.v1` bundle runs
/// the same teacher-forced stream on wgpu and CPU, and per-position logits
/// and top-1 ids are compared within the SIMD/reorder tolerance.
#[test]
#[ignore = "requires MINIFIELD_LFM25_BUNDLE_DIR, MINIFIELD_LFM25_PACKED and a wgpu adapter"]
fn wgpu_packed_executor_matches_cpu_on_real_model() {
    let bundle = PathBuf::from(
        std::env::var_os("MINIFIELD_LFM25_BUNDLE_DIR").expect("MINIFIELD_LFM25_BUNDLE_DIR"),
    );
    let packed_path =
        PathBuf::from(std::env::var_os("MINIFIELD_LFM25_PACKED").expect("MINIFIELD_LFM25_PACKED"));
    let config = fs::read(bundle.join("config.json")).expect("config.json");
    let packed = fs::read(&packed_path).expect("packed safetensors");

    // Oracle prompt p01, tokenized by llama.cpp (BOS included).
    let tokens: Vec<u32> = vec![
        1, 1098, 4605, 10800, 36387, 56586, 1391, 779, 46199, 4949, 3627, 779,
    ];
    let limits = ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 6 << 30,
        max_pending_operations: 512,
    };

    let gpu_start = std::time::Instant::now();
    let gpu = WgpuBackend::new(0xE0_3C, limits).expect("wgpu backend");
    let mut gpu_exec = load_format(gpu, &config, &packed, Lfm2WeightFormat::TernaryV1);
    let gpu_load = gpu_start.elapsed();
    let gpu_run = std::time::Instant::now();
    let (gpu_logits, gpu_top) = teacher_forced_logits(&mut gpu_exec, &tokens);
    let gpu_run = gpu_run.elapsed();

    let cpu_start = std::time::Instant::now();
    let cpu = CpuBackend::new(0xE0_2C, limits);
    let mut cpu_exec = load_format(cpu, &config, &packed, Lfm2WeightFormat::TernaryV1);
    let cpu_load = cpu_start.elapsed();
    let cpu_run = std::time::Instant::now();
    let (cpu_logits, cpu_top) = teacher_forced_logits(&mut cpu_exec, &tokens);
    let cpu_run = cpu_run.elapsed();

    assert_eq!(gpu_logits.len(), cpu_logits.len(), "logit width");
    let max_delta = gpu_logits
        .iter()
        .zip(cpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let agreements = gpu_top
        .iter()
        .zip(cpu_top.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "wgpu packed vs cpu packed: {}/{} top-1 positions agree, \
         final-position max|delta|={max_delta:.4}",
        agreements,
        gpu_top.len()
    );
    println!(
        "wgpu packed: load {gpu_load:?}, 12-token run {gpu_run:?}; \
         cpu packed: load {cpu_load:?}, run {cpu_run:?}"
    );
    println!("wgpu top-1 sequence: {gpu_top:?}");
    println!("cpu  top-1 sequence: {cpu_top:?}");
    assert_eq!(gpu_top, cpu_top, "top-1 sequences diverged");
    assert!(
        max_delta <= LOGIT_TOLERANCE,
        "logit divergence {max_delta} exceeds {LOGIT_TOLERANCE}"
    );
}
