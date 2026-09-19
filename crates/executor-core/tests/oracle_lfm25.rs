//! Oracle gate T3 (docs/ternary-format-v1.md): the dense F32 executor on real
//! LFM2.5-230M weights must produce the same next-token decision as llama.cpp's
//! BF16 run at every captured position.
//!
//! Requires two external artifact directories, kept outside Git:
//! - `MINIFIELD_LFM25_BUNDLE_DIR`: config.json + model.safetensors (BF16).
//! - `MINIFIELD_LFM25_ORACLE_DIR`: manifest.json + out/pNN.logits.f32 produced
//!   by scripts/oracle/run-oracle.sh (CPU llama.cpp, f32 KV, temp 0).
//!
//! Run: `cargo test -p minifield-executor-core --release --test oracle_lfm25 -- --ignored --nocapture`
#![allow(
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, MemoryAssetProvider, ResourceLimits, TokenChunk,
    TokenExecutor,
};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2WeightLoadTask, LoaderLimits,
    LoaderPoll,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Provisional stop-line from the spec; tighten after measured values land.
const MAX_ABS_LOGIT_DELTA: f32 = 0.5;

/// A generated-token disagreement is only acceptable at a near-tie: our own
/// top-2 logit gap must be below this margin (~2x the observed mean|Δ|). A
/// confident wrong answer indicates a real bug, not numeric noise.
const NEAR_TIE_GAP: f32 = 0.05;

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn env_dir(variable: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(
        || panic!("{variable} must name the external artifact directory"),
        PathBuf::from,
    )
}

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

fn load(bundle: &Path) -> Lfm2Executor<CpuBackend> {
    let config = fs::read(bundle.join("config.json")).expect("config.json");
    let weights = fs::read(bundle.join("model.safetensors")).expect("model.safetensors");
    let weight_len = u64::try_from(weights.len()).expect("asset length");
    let request = Lfm2LoadRequest::new(
        config.clone(),
        digest(&config),
        weight_len,
        digest(&weights),
        LoaderLimits {
            max_asset_bytes: weight_len,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: weight_len,
            max_retained_host_bytes: weight_len.checked_mul(3).expect("host bound"),
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
    )
    .expect("checked load request");
    let mut backend = CpuBackend::new(
        0xE0_2C,
        ResourceLimits {
            max_allocation_bytes: 3 << 30,
            max_total_bytes: 3 << 30,
            max_pending_operations: 256,
        },
    );
    let mut provider = MemoryAssetProvider::new(weights, weight_len);
    let mut task = Lfm2WeightLoadTask::begin(request).expect("load task");
    let weights = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Ok(weights)) => break weights,
            LoaderPoll::Ready(Err(error)) => panic!("loader error: {error:?}"),
        }
    };
    Lfm2Executor::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 4096,
        },
    )
    .expect("executor")
}

fn oracle_logits(path: &Path) -> (u32, Vec<f32>) {
    let bytes = fs::read(path).expect("oracle logits file");
    let n_tokens = u32::from_le_bytes(bytes[..4].try_into().expect("n_tokens"));
    let n_vocab = u32::from_le_bytes(bytes[4..8].try_into().expect("n_vocab")) as usize;
    let values: Vec<f32> = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    assert_eq!(
        values.len(),
        n_tokens as usize * n_vocab,
        "logits row count"
    );
    (n_tokens, values)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .expect("nonempty logits")
}

#[test]
#[ignore = "requires MINIFIELD_LFM25_BUNDLE_DIR and MINIFIELD_LFM25_ORACLE_DIR"]
fn real_lfm25_forward_matches_llamacpp_oracle() {
    let bundle = env_dir("MINIFIELD_LFM25_BUNDLE_DIR");
    let oracle = env_dir("MINIFIELD_LFM25_ORACLE_DIR");
    let manifest: Value =
        serde_json::from_slice(&fs::read(oracle.join("manifest.json")).expect("manifest"))
            .expect("manifest JSON");
    let prompts = manifest["prompts"].as_object().expect("prompts object");

    let mut executor = load(&bundle);
    let n_vocab = 65536usize;
    let mut total_positions = 0u32;
    let mut top1_mismatches = 0u32;
    let mut generated_mismatches = 0u32;
    let mut generated_compared = 0u32;
    let mut global_max_delta = 0.0f32;
    let mut global_sum_delta = 0.0f64;
    let mut global_count = 0u64;

    let mut ids_sorted: Vec<&String> = prompts.keys().collect();
    ids_sorted.sort();
    for pid in ids_sorted {
        let entry = &prompts[pid];
        let prompt_ids: Vec<u32> = entry["prompt_token_ids_dump_logits"]
            .as_array()
            .expect("prompt ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let emitted: Vec<u32> = entry["emitted_ids"]
            .as_array()
            .expect("emitted ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let (oracle_tokens, oracle) =
            oracle_logits(&oracle.join(entry["logits_file"].as_str().expect("logits file")));
        assert_eq!(
            oracle_tokens as usize,
            prompt_ids.len(),
            "{pid}: oracle rows vs prompt ids"
        );

        // Teacher-forced per-position comparison: append one token at a time so
        // next_logits after token i lines up with oracle row i (predicts i+1).
        let mut prefill = executor
            .prefill(TokenChunk::all(&prompt_ids[..1]))
            .expect("prefill");
        let mut prefix = ready(&mut prefill);
        let mut prompt_max = 0.0f32;
        let mut prompt_mismatch = 0u32;
        for i in 0..prompt_ids.len() {
            if i > 0 {
                let mut append = executor
                    .append_known(&prefix, TokenChunk::all(&prompt_ids[i..=i]))
                    .expect("append");
                prefix = ready(&mut append);
            }
            let mut logits_task = executor.next_logits(&prefix).expect("logits");
            let ours = ready(&mut logits_task);
            assert_eq!(ours.len(), n_vocab, "logits width");
            let oracle_row = &oracle[i * n_vocab..(i + 1) * n_vocab];
            if argmax(&ours) != argmax(oracle_row) {
                prompt_mismatch += 1;
            }
            for (a, b) in ours.iter().zip(oracle_row.iter()) {
                let delta = (a - b).abs();
                prompt_max = prompt_max.max(delta);
                global_sum_delta += f64::from(delta);
                global_count += 1;
            }
        }
        total_positions += prompt_ids.len() as u32;
        top1_mismatches += prompt_mismatch;
        global_max_delta = global_max_delta.max(prompt_max);

        // Greedy continuation under matched history: each step compares our
        // argmax to the oracle's emitted id, then teacher-forces the oracle id
        // so a divergence can't cascade into meaningless later comparisons.
        let mut gen_mismatch = 0u32;
        for (step, expected) in emitted.iter().enumerate() {
            let mut logits_task = executor.next_logits(&prefix).expect("logits");
            let ours = ready(&mut logits_task);
            let next = argmax(&ours);
            if next != *expected {
                gen_mismatch += 1;
                let mut sorted = ours.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let gap = sorted[0] - sorted[1];
                println!(
                    "  {pid} step {step}: oracle {expected}, ours {next}, \
                     top2 gap {gap:.6} ({:.4} vs {:.4})",
                    sorted[0], sorted[1]
                );
                assert!(
                    gap <= NEAR_TIE_GAP,
                    "{pid} step {step}: confident wrong answer (gap {gap:.4})"
                );
            }
            let mut append = executor
                .append_known(&prefix, TokenChunk::all(&[*expected]))
                .expect("gen append");
            prefix = ready(&mut append);
        }
        generated_mismatches += gen_mismatch;
        generated_compared += emitted.len() as u32;
        println!(
            "{pid}: {} positions, top-1 mismatches {prompt_mismatch}, max|Δ| {prompt_max:.4}, \
             generated {} ids, mismatches {gen_mismatch}",
            prompt_ids.len(),
            emitted.len()
        );
    }

    let mean_delta = global_sum_delta / global_count as f64;
    println!(
        "summary: {total_positions} positions, {top1_mismatches} top-1 mismatches, \
         max|Δ| {global_max_delta:.4}, mean|Δ| {mean_delta:.6}; \
         {generated_compared} generated ids, {generated_mismatches} mismatches"
    );
    assert_eq!(top1_mismatches, 0, "top-1 must agree at every position");
    // Generated disagreements already failed inside the loop unless the gap
    // was a near-tie; reaching here means all of them were.
    assert!(
        global_max_delta <= MAX_ABS_LOGIT_DELTA,
        "max|Δ| {global_max_delta} exceeds declared stop-line {MAX_ABS_LOGIT_DELTA}"
    );
}
