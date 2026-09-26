//! T4 packed ternary path: `packed_linear`/`packed_gather_rows` must equal the
//! dequantized `minifield.ternary.v1` weights through the dense f32 ops, and a
//! fully packed LFM2 model must produce matching logits to the same
//! dequantized weights through the dense executor.
//!
//! The kernel vectors come from the committed `ternary-v1-001` fixture. The
//! end-to-end check builds a tiny packed + dequantized-dense safetensors pair
//! in memory, so no large artifacts are committed. Gather is bitwise (decode
//! only); linear uses a declared tolerance because the SIMD kernel accumulates
//! f32 in a different order than the dense matvec. All products are exact, so
//! only reorder error is possible.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::{fs, path::PathBuf};

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, ExecutorError, InferenceCompletion, MemoryAssetProvider, ResourceLimits, Shape,
    TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenIds,
};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LayerWeightRole, Lfm2LoadRequest, Lfm2WeightFormat,
    Lfm2WeightLoadTask, Lfm2WeightRole, LoaderLimits, LoaderPoll, detect_lfm2_weight_format,
    parse_lfm2_tensor_quantization,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const VECTORS: &str = include_str!("fixtures/ternary-v1-001/vectors.json");

/// SIMD accumulation reorders the f32 sum; products are exact so only reorder
/// error is possible. Generous vs observed ~1e-6, tight vs a real weight bug.
const REORDER_TOLERANCE: f32 = 1e-4;

fn assert_within_reorder(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    let max_delta = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta <= REORDER_TOLERANCE,
        "{context}: max|delta| {max_delta} exceeds {REORDER_TOLERANCE}"
    );
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn backend(total_bytes: u64) -> CpuBackend {
    CpuBackend::new(
        0xE0_2C,
        ResourceLimits {
            max_allocation_bytes: total_bytes,
            max_total_bytes: total_bytes,
            max_pending_operations: 256,
        },
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

fn shape(rows: u64, columns: u64) -> Shape {
    Shape::new(&[rows, columns]).expect("shape")
}

/// Truncating f32 -> IEEE-754 half encoder for fixture generation. Subnormal
/// and overflowing inputs flush to zero/infinity; fixture scales never reach
/// those ranges.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 112;
    let mantissa = ((bits >> 13) & 0x03FF) as u16;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exponent as u16) << 10) | mantissa
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits >> 10 & 0x1F);
    let mantissa = u32::from(bits & 0x03FF);
    let value = if exponent == 0 {
        mantissa as f32 * 2f32.powi(-24)
    } else if exponent == 31 {
        f32::from_bits(0x7F80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(((exponent + 112) << 23) | (mantissa << 13))
    };
    f32::from_bits(value.to_bits() | sign)
}

/// Pack one row of 128 weights per `minifield.ternary.v1` (absmax, byte-
/// sequential codes). Returns (code bytes, decoded f32 scale, dequantized row).
fn pack_group(row: &[f32]) -> ([u8; 32], f32, Vec<f32>) {
    let scale = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let quantized: Vec<u8> = if scale == 0.0 {
        vec![1; 128]
    } else {
        row.iter()
            .map(|w| ((w / scale).round() + 1.0).clamp(0.0, 3.0) as u8)
            .collect()
    };
    let mut codes = [0_u8; 32];
    for (j, q) in quantized.iter().enumerate() {
        codes[j / 4] |= q << (2 * (j % 4));
    }
    let decoded = f16_to_f32(f32_to_f16(scale));
    let dequantized = quantized
        .iter()
        .map(|q| (f32::from(*q) - 1.0) * decoded)
        .collect();
    (codes, decoded, dequantized)
}

#[test]
fn packed_linear_and_gather_match_dequantized_reference() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("vectors JSON");
    let rows = fixture["vectors"].as_array().expect("vectors array");
    let mut cpu = backend(1 << 20);
    let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.01 - 0.5).collect();
    let input_buffer = cpu.upload_f32(shape(1, 128), &input).expect("input buffer");

    for (row, vector) in rows.iter().enumerate() {
        let codes: Vec<u8> = vector["expected_codes"]
            .as_array()
            .expect("codes")
            .iter()
            .map(|v| v.as_u64().expect("code") as u8)
            .collect();
        assert_eq!(codes.len(), 128, "row {row} code count");
        let mut code_bytes = [0_u8; 32];
        for (j, q) in codes.iter().enumerate() {
            code_bytes[j / 4] |= q << (2 * (j % 4));
        }
        let scale = vector["scale_f16"].as_f64().expect("scale") as f32;
        let expected_row: Vec<f32> = vector["expected_dequant"]
            .as_array()
            .expect("dequantized")
            .iter()
            .map(|v| v.as_f64().expect("value") as f32)
            .collect();

        let codes_buffer = cpu
            .upload_u8_classified(
                shape(1, 32),
                &code_bytes,
                minifield_engine_api::AllocationClass::Weight,
            )
            .expect("codes buffer");
        let scales_buffer = cpu
            .upload_f32(shape(1, 1), &[scale])
            .expect("scales buffer");
        let dense_buffer = cpu
            .upload_f32(shape(1, 128), &expected_row)
            .expect("dense buffer");

        // packed_linear must equal the dense matvec on the dequantized row.
        let mut packed_out = cpu.allocate_f32(shape(1, 1)).expect("packed out");
        let mut dense_out = cpu.allocate_f32(shape(1, 1)).expect("dense out");
        cpu.packed_linear(
            &mut packed_out,
            &input_buffer,
            &codes_buffer,
            &scales_buffer,
        )
        .expect("packed linear");
        cpu.linear(&mut dense_out, &input_buffer, &dense_buffer)
            .expect("dense linear");
        assert_within_reorder(
            packed_out.as_slice(),
            dense_out.as_slice(),
            &format!("row {row} packed_linear diverged"),
        );

        // packed_gather_rows must equal the dequantized row bitwise.
        let mut gathered = cpu.allocate_f32(shape(1, 128)).expect("gather out");
        cpu.packed_gather_rows(
            &mut gathered,
            &codes_buffer,
            &scales_buffer,
            TokenIds::Host(&[0]),
        )
        .expect("packed gather");
        assert_eq!(
            gathered.as_slice(),
            expected_row.as_slice(),
            "row {row} packed_gather_rows diverged"
        );
    }
}

/// Multi-group coverage: the real model has 8-24 groups per row, the fixture
/// and tiny model only 1. Two groups with different scales, checked against
/// the dequantized dense matvec.
#[test]
fn packed_linear_multiple_groups_per_row() {
    let mut cpu = backend(1 << 20);
    let weights: Vec<f32> = (0..256)
        .map(|i| {
            if i < 128 {
                weight_at(90_000 + i) * 0.01
            } else {
                weight_at(91_000 + i)
            }
        })
        .collect();
    let mut codes = Vec::with_capacity(64);
    let mut scales = Vec::with_capacity(2);
    let mut dequant = Vec::with_capacity(256);
    for group in weights.as_chunks::<128>().0 {
        let (code_bytes, decoded, deq) = pack_group(group);
        codes.extend_from_slice(&code_bytes);
        scales.push(decoded);
        dequant.extend_from_slice(&deq);
    }
    assert!(scales[0] != scales[1], "test needs distinct scales");

    let input: Vec<f32> = (0..256).map(|i| weight_at(92_000 + i)).collect();
    let input_buffer = cpu.upload_f32(shape(1, 256), &input).expect("input");
    let codes_buffer = cpu
        .upload_u8_classified(
            shape(1, 64),
            &codes,
            minifield_engine_api::AllocationClass::Weight,
        )
        .expect("codes");
    let scales_buffer = cpu.upload_f32(shape(1, 2), &scales).expect("scales");
    let dense_buffer = cpu.upload_f32(shape(1, 256), &dequant).expect("dense");

    let mut packed_out = cpu.allocate_f32(shape(1, 1)).expect("packed out");
    let mut dense_out = cpu.allocate_f32(shape(1, 1)).expect("dense out");
    cpu.packed_linear(
        &mut packed_out,
        &input_buffer,
        &codes_buffer,
        &scales_buffer,
    )
    .expect("packed linear");
    cpu.linear(&mut dense_out, &input_buffer, &dense_buffer)
        .expect("dense linear");
    assert_within_reorder(
        packed_out.as_slice(),
        dense_out.as_slice(),
        "multi-group packed_linear diverged",
    );
}

#[test]
fn packed_ops_reject_invalid_operands() {
    let mut cpu = backend(1 << 20);
    let input = cpu.upload_f32(shape(1, 128), &[0.0; 128]).expect("input");
    let codes = cpu
        .upload_u8_classified(
            shape(1, 32),
            &[0x55; 32],
            minifield_engine_api::AllocationClass::Weight,
        )
        .expect("codes");
    let scales = cpu.upload_f32(shape(1, 1), &[1.0]).expect("scales");
    let mut out = cpu.allocate_f32(shape(1, 1)).expect("out");

    // F32 weight where U8 codes belong.
    let f32_weight = cpu
        .upload_f32(shape(1, 32), &[0.0; 32])
        .expect("f32 weight");
    assert!(
        cpu.packed_linear(&mut out, &input, &f32_weight, &scales)
            .is_err(),
        "f32 codes accepted"
    );

    // U8 buffer where F32 scales belong.
    let u8_scales = cpu
        .upload_u8_classified(
            shape(1, 1),
            &[0],
            minifield_engine_api::AllocationClass::Weight,
        )
        .expect("u8 scales");
    assert!(
        cpu.packed_linear(&mut out, &input, &codes, &u8_scales)
            .is_err(),
        "u8 scales accepted"
    );

    // Codes width not columns / 4.
    let bad_codes = cpu
        .upload_u8_classified(
            shape(1, 16),
            &[0x55; 16],
            minifield_engine_api::AllocationClass::Weight,
        )
        .expect("bad codes");
    assert!(
        cpu.packed_linear(&mut out, &input, &bad_codes, &scales)
            .is_err(),
        "truncated codes accepted"
    );

    // Output width != weight rows.
    let mut wide_out = cpu.allocate_f32(shape(1, 2)).expect("wide out");
    assert!(
        cpu.packed_linear(&mut wide_out, &input, &codes, &scales)
            .is_err(),
        "wide output accepted"
    );

    // Non-unit activation rows.
    let batch_input = cpu
        .upload_f32(shape(2, 128), &[0.0; 256])
        .expect("batch input");
    assert!(
        cpu.packed_linear(&mut out, &batch_input, &codes, &scales)
            .is_err(),
        "batched input accepted"
    );
}

/// Minimal safetensors writer: name -> (dtype, shape, little-endian bytes).
fn safetensors(tensors: &[(&str, &str, Vec<u64>, &[u8])]) -> Vec<u8> {
    safetensors_with_metadata(tensors, None)
}

/// Same writer with an optional `__metadata__` object.
fn safetensors_with_metadata(
    tensors: &[(&str, &str, Vec<u64>, &[u8])],
    metadata: Option<Value>,
) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".to_owned(), metadata);
    }
    let mut offset = 0_u64;
    for (name, dtype, dims, bytes) in tensors {
        header.insert(
            (*name).to_owned(),
            json!({
                "dtype": dtype,
                "shape": dims,
                "data_offsets": [offset, offset + bytes.len() as u64],
            }),
        );
        offset += bytes.len() as u64;
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header");
    let mut output = (header_bytes.len() as u64).to_le_bytes().to_vec();
    output.extend_from_slice(&header_bytes);
    for (_, _, _, bytes) in tensors {
        output.extend_from_slice(bytes);
    }
    output
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deterministic pseudo-random weight in [-1, 1] (Knuth multiplicative hash).
fn weight_at(index: usize) -> f32 {
    let hashed = (index as u32).wrapping_mul(2_654_435_761) >> 8;
    hashed as f32 / 16_777_216.0 * 2.0 - 1.0
}

fn load_executor(
    config: &[u8],
    weights: &[u8],
    format: Lfm2WeightFormat,
    total_bytes: u64,
) -> Lfm2Executor<CpuBackend> {
    let request = Lfm2LoadRequest::new_with_format(
        config.to_vec(),
        digest(config),
        weights.len() as u64,
        digest(weights),
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
    finish_load(request, weights, total_bytes)
}

fn finish_load(
    request: Lfm2LoadRequest,
    weights: &[u8],
    total_bytes: u64,
) -> Lfm2Executor<CpuBackend> {
    let mut cpu = backend(total_bytes);
    let mut provider = MemoryAssetProvider::new(weights.to_vec(), weights.len() as u64);
    let mut task = Lfm2WeightLoadTask::begin(request).expect("load task");
    let weights = loop {
        match task.poll_step(&mut provider, &mut cpu) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Ok(weights)) => break weights,
            LoaderPoll::Ready(Err(error)) => panic!("loader error: {error:?}"),
        }
    };
    Lfm2Executor::new(
        cpu,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 64,
        },
    )
    .expect("executor")
}

/// Deterministic tiny LFM2 fixture: (config, packed safetensors, dequantized
/// safetensors). hidden=intermediate=vocab=128 keeps every matmul input width
/// a multiple of 128 so all roles pack. One conv + one attention layer.
#[allow(clippy::too_many_lines)]
type DenseTensors = Vec<(String, Vec<u64>, Vec<f32>)>;

fn tiny_packed_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (config, dense) = tiny_dense_base();
    tiny_packed_fixture_from(&config, dense)
}

/// The deterministic dense tensor list shared by every fixture flavour.
#[allow(clippy::too_many_lines)]
fn tiny_dense_base() -> (Vec<u8>, DenseTensors) {
    let config = serde_json::to_vec(&json!({
        "block_auto_adjust_ff_dim": false,
        "bos_token_id": 1,
        "conv_L_cache": 3,
        "conv_bias": false,
        "eos_token_id": 7,
        "hidden_size": 128,
        "intermediate_size": 128,
        "layer_types": ["conv", "full_attention"],
        "model_type": "lfm2",
        "norm_eps": 1e-05,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "pad_token_id": 0,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": true,
        "vocab_size": 128
    }))
    .expect("config");

    // Dense source tensors [name -> (shape, values)], all matmul roles present.
    let mut dense: DenseTensors = Vec::new();
    let mut cursor = 0_usize;
    let mut matrix = |name: &str, rows: u64, columns: u64, dense: &mut DenseTensors| {
        let count = (rows * columns) as usize;
        let values: Vec<f32> = (0..count)
            .map(|i| {
                weight_at({
                    cursor += 1;
                    cursor + i
                })
            })
            .collect();
        dense.push((name.to_owned(), vec![rows, columns], values));
    };
    matrix("model.embed_tokens.weight", 128, 128, &mut dense);
    dense.push((
        "model.embedding_norm.weight".to_owned(),
        vec![128],
        (0..128).map(|i| 0.5 + weight_at(i) * 0.5).collect(),
    ));
    let layer_matrices: [(&str, u64, u64); 6] = [
        ("conv.in_proj.weight", 384, 128),
        ("conv.out_proj.weight", 128, 128),
        ("self_attn.q_proj.weight", 128, 128),
        ("self_attn.k_proj.weight", 64, 128),
        ("self_attn.v_proj.weight", 64, 128),
        ("self_attn.out_proj.weight", 128, 128),
    ];
    let mut layer_dense: [DenseTensors; 2] = [Vec::new(), Vec::new()];
    for (index, tensors) in layer_dense.iter_mut().enumerate() {
        let prefix = format!("model.layers.{index}");
        if index == 0 {
            matrix(&format!("{prefix}.conv.conv.weight"), 128, 3, tensors);
            for (suffix, rows, columns) in &layer_matrices[..2] {
                matrix(&format!("{prefix}.{suffix}"), *rows, *columns, tensors);
            }
        } else {
            for (suffix, rows, columns) in &layer_matrices[2..] {
                matrix(&format!("{prefix}.{suffix}"), *rows, *columns, tensors);
            }
            tensors.push((
                format!("{prefix}.self_attn.q_layernorm.weight"),
                vec![64],
                (0..64).map(|i| 0.5 + weight_at(5000 + i) * 0.5).collect(),
            ));
            tensors.push((
                format!("{prefix}.self_attn.k_layernorm.weight"),
                vec![64],
                (0..64).map(|i| 0.5 + weight_at(6000 + i) * 0.5).collect(),
            ));
        }
        for suffix in ["feed_forward.w1.weight", "feed_forward.w3.weight"] {
            matrix(&format!("{prefix}.{suffix}"), 128, 128, tensors);
        }
        matrix(
            &format!("{prefix}.feed_forward.w2.weight"),
            128,
            128,
            tensors,
        );
        for (suffix, offset) in [
            ("ffn_norm.weight", 7000_usize),
            ("operator_norm.weight", 8000),
        ] {
            tensors.push((
                format!("{prefix}.{suffix}"),
                vec![128],
                (0..128)
                    .map(|i| 0.5 + weight_at(offset + i) * 0.5)
                    .collect(),
            ));
        }
    }
    dense.append(&mut layer_dense[0]);
    dense.append(&mut layer_dense[1]);

    // Conv kernels must be stored [hidden, 1, width]; the generator above
    // produced [hidden, width], so fix the shape before writing.
    for (_, dims, _) in &mut dense {
        if dims.len() == 2 && dims[1] == 3 {
            *dims = vec![dims[0], 1, 3];
        }
    }

    (config, dense)
}

/// Pack every rank-two matmul in `dense` to ternary and write the packed +
/// dequantized-dense safetensors side by side.
fn tiny_packed_fixture_from(config: &[u8], dense: DenseTensors) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Write the packed and dequantized-dense safetensors side by side.
    let mut packed_tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut dequant_tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    for (name, dims, values) in &dense {
        if dims.len() == 2 && *dims.last().expect("dims") >= 128 && dims[1].is_multiple_of(128) {
            let rows = dims[0] as usize;
            let columns = dims[1] as usize;
            let groups = columns / 128;
            let mut codes = Vec::with_capacity(rows * columns / 4);
            let mut scales = Vec::with_capacity(rows * groups * 2);
            let mut dequant = Vec::with_capacity(values.len());
            for row in values.chunks_exact(columns) {
                for group in row.as_chunks::<128>().0 {
                    let (code_bytes, decoded, deq) = pack_group(group);
                    codes.extend_from_slice(&code_bytes);
                    scales.extend_from_slice(&f32_to_f16(decoded).to_le_bytes());
                    dequant.extend_from_slice(&deq);
                }
            }
            packed_tensors.push((
                format!("{name}.codes"),
                "U8".to_owned(),
                vec![dims[0], dims[1] / 4],
                codes,
            ));
            packed_tensors.push((
                format!("{name}.scales"),
                "F16".to_owned(),
                vec![dims[0], groups as u64],
                scales,
            ));
            dequant_tensors.push((
                name.clone(),
                "F32".to_owned(),
                dims.clone(),
                f32_bytes(&dequant),
            ));
        } else {
            let bytes = f32_bytes(values);
            packed_tensors.push((name.clone(), "F32".to_owned(), dims.clone(), bytes.clone()));
            dequant_tensors.push((name.clone(), "F32".to_owned(), dims.clone(), bytes));
        }
    }
    let packed_refs: Vec<(&str, &str, Vec<u64>, &[u8])> = packed_tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), b.as_slice()))
        .collect();
    let dequant_refs: Vec<(&str, &str, Vec<u64>, &[u8])> = dequant_tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), b.as_slice()))
        .collect();
    let packed_bytes = safetensors(&packed_refs);
    let dequant_bytes = safetensors(&dequant_refs);
    (config.to_vec(), packed_bytes, dequant_bytes)
}

#[test]
fn packed_executor_matches_dequantized_dense_executor() {
    let (config, packed_bytes, dequant_bytes) = tiny_packed_fixture();
    let mut packed_exec =
        load_executor(&config, &packed_bytes, Lfm2WeightFormat::TernaryV1, 1 << 24);
    let mut dense_exec = load_executor(&config, &dequant_bytes, Lfm2WeightFormat::Dense, 1 << 24);

    let tokens: Vec<u32> = vec![1, 5, 9, 42, 7, 100];
    let mut packed_task = packed_exec
        .prefill(TokenChunk::all(&tokens[..1]))
        .expect("prefill");
    let mut dense_task = dense_exec
        .prefill(TokenChunk::all(&tokens[..1]))
        .expect("prefill");
    let mut packed_prefix = ready(&mut packed_task);
    let mut dense_prefix = ready(&mut dense_task);
    for (step, token) in tokens.iter().enumerate().skip(1) {
        let mut pa = packed_exec
            .append_known(&packed_prefix, TokenChunk::all(&[*token]))
            .expect("append");
        let mut da = dense_exec
            .append_known(&dense_prefix, TokenChunk::all(&[*token]))
            .expect("append");
        packed_prefix = ready(&mut pa);
        dense_prefix = ready(&mut da);
        let mut pl = packed_exec.next_logits(&packed_prefix).expect("logits");
        let mut dl = dense_exec.next_logits(&dense_prefix).expect("logits");
        let packed_logits = ready(&mut pl);
        let dense_logits = ready(&mut dl);
        assert_within_reorder(
            &packed_logits,
            &dense_logits,
            &format!("step {step}: packed executor logits diverged from dequantized dense"),
        );
    }
}

/// First strict maximum, matching the executor's device argmax tie-break.
fn first_argmax(logits: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut id = 0_u32;
    for (index, &value) in logits.iter().enumerate() {
        if value.is_finite() && value > best {
            best = value;
            id = index as u32;
        }
    }
    id
}

#[test]
fn append_argmax_extends_prefix_with_resolved_greedy_token() {
    let (config, packed_bytes, _) = tiny_packed_fixture();
    let mut executor = load_executor(&config, &packed_bytes, Lfm2WeightFormat::TernaryV1, 1 << 24);
    let mut task = executor
        .prefill(TokenChunk::all(&[3, 5, 9]))
        .expect("prefill");
    let mut prefix = ready(&mut task);

    for _ in 0..4 {
        // The published sample must equal the first-max index of the
        // prefix's logits row.
        let mut logits_task = executor.next_logits(&prefix).expect("logits");
        let logits = ready(&mut logits_task);
        let expected = first_argmax(&logits);
        assert_eq!(
            executor.sampled_token(&prefix).expect("sampled"),
            Some(expected),
            "sampled token diverged from logits argmax"
        );

        // append_argmax must produce the same state as append_known of the
        // same token: identical history and bitwise-identical next logits.
        let mut greedy = executor
            .append_argmax(prefix.clone())
            .expect("greedy append");
        let greedy_prefix = ready(&mut greedy);
        let mut known = executor
            .append_known(&prefix, TokenChunk::all(&[expected]))
            .expect("known append");
        let known_prefix = ready(&mut known);
        assert_eq!(greedy_prefix.token_history(), known_prefix.token_history());
        let mut greedy_logits = executor.next_logits(&greedy_prefix).expect("logits");
        let mut known_logits = executor.next_logits(&known_prefix).expect("logits");
        assert_eq!(ready(&mut greedy_logits), ready(&mut known_logits));
        prefix = greedy_prefix;
    }
}

/// The tiny fixture's conv + `full_attention` layer pair exercises both state
/// paths in one bulk pass. On CPU every op computes each row independently,
/// so bulk prefill must equal the serial one-token reference bitwise.
#[test]
fn bulk_prefill_matches_serial_token_appends() {
    let (config, packed_bytes, dequant_bytes) = tiny_packed_fixture();
    let tokens: Vec<u32> = vec![3, 5, 9, 42, 7];
    for (weights, format, label) in [
        (&dequant_bytes, Lfm2WeightFormat::Dense, "dense"),
        (&packed_bytes, Lfm2WeightFormat::TernaryV1, "packed"),
    ] {
        let mut bulk = load_executor(&config, weights, format, 1 << 24);
        let mut serial = load_executor(&config, weights, format, 1 << 24);

        let mut bulk_task = bulk
            .prefill(TokenChunk::all(&tokens))
            .expect("bulk prefill");
        let bulk_prefix = ready(&mut bulk_task);

        let mut empty_task = serial.prefill(TokenChunk::all(&[])).expect("empty prefill");
        let mut serial_prefix = ready(&mut empty_task);
        for token in &tokens {
            let mut append = serial
                .append_known(&serial_prefix, TokenChunk::all(&[*token]))
                .expect("serial append");
            serial_prefix = ready(&mut append);
        }

        assert_eq!(
            bulk_prefix.logical_length(),
            serial_prefix.logical_length(),
            "{label}: logical length"
        );
        assert_eq!(
            bulk_prefix.token_history(),
            serial_prefix.token_history(),
            "{label}: history"
        );
        assert_eq!(
            bulk.sampled_token(&bulk_prefix).expect("bulk sampled"),
            serial
                .sampled_token(&serial_prefix)
                .expect("serial sampled"),
            "{label}: sampled token"
        );

        let mut bulk_logits = bulk.next_logits(&bulk_prefix).expect("bulk logits");
        let mut serial_logits = serial.next_logits(&serial_prefix).expect("serial logits");
        assert_eq!(
            ready(&mut bulk_logits),
            ready(&mut serial_logits),
            "{label}: final logits diverged"
        );

        let mut bulk_append = bulk
            .append_known(&bulk_prefix, TokenChunk::all(&[11]))
            .expect("bulk continuation");
        let mut serial_append = serial
            .append_known(&serial_prefix, TokenChunk::all(&[11]))
            .expect("serial continuation");
        let bulk_next = ready(&mut bulk_append);
        let serial_next = ready(&mut serial_append);
        assert_eq!(
            bulk_next.token_history(),
            serial_next.token_history(),
            "{label}: continuation history"
        );
        let mut bulk_next_logits = bulk.next_logits(&bulk_next).expect("bulk next logits");
        let mut serial_next_logits = serial
            .next_logits(&serial_next)
            .expect("serial next logits");
        assert_eq!(
            ready(&mut bulk_next_logits),
            ready(&mut serial_next_logits),
            "{label}: continuation logits diverged"
        );
    }
}

#[test]
fn prefill_choice_logits_reads_selected_columns_directly() {
    let (config, packed_bytes, dequant_bytes) = tiny_packed_fixture();
    let prompt = [3_u32, 5, 9, 42];
    let selectors = [41_u32, 7, 100, 7];
    for (weights, format, label) in [
        (&dequant_bytes, Lfm2WeightFormat::Dense, "dense"),
        (&packed_bytes, Lfm2WeightFormat::TernaryV1, "packed"),
    ] {
        let mut executor = load_executor(&config, weights, format, 1 << 24);

        let mut task = executor
            .prefill_choice_logits(TokenChunk::all(&prompt), &selectors)
            .expect("choice prefill");
        let values = ready(&mut task);
        assert_eq!(
            task.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
            "{label}: completion must be consumed"
        );

        // Same prompt through the publishing prefill path yields identical
        // final logits on CPU; the direct task must equal that selection.
        let mut prefill = executor.prefill(TokenChunk::all(&prompt)).expect("prefill");
        let prefix = ready(&mut prefill);
        let mut logits_task = executor.next_logits(&prefix).expect("logits");
        let full = ready(&mut logits_task);
        let expected: Vec<f32> = selectors.iter().map(|&id| full[id as usize]).collect();
        assert_eq!(values, expected, "{label}: selected logits diverged");

        assert!(
            executor
                .prefill_choice_logits(TokenChunk::all(&[]), &selectors)
                .is_err(),
            "{label}: empty prompt accepted"
        );
        assert!(
            executor
                .prefill_choice_logits(TokenChunk::all(&prompt), &[])
                .is_err(),
            "{label}: empty selectors accepted"
        );
        assert!(
            executor
                .prefill_choice_logits(TokenChunk::all(&prompt), &[1, 128])
                .is_err(),
            "{label}: out-of-vocab selector accepted"
        );

        let mut cancelled = executor
            .prefill_choice_logits(TokenChunk::all(&prompt), &selectors)
            .expect("choice prefill");
        cancelled.cancel().expect("cancel");
        assert_eq!(
            cancelled.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
            "{label}: cancelled task must not produce values"
        );
    }
}

/// Shared-base branches: an unscored base prefill publishes only cache state,
/// and each `append_choice_logits` tail must equal a direct full prefill of
/// base+tail. The fixture's conv + `full_attention` layers cover both cache
/// paths, and CPU rows are computed independently so equality is bitwise.
#[test]
fn append_choice_logits_branches_off_a_shared_unscored_base() {
    let (config, packed_bytes, dequant_bytes) = tiny_packed_fixture();
    let base_tokens = [3_u32, 5, 9];
    let tails: [&[u32]; 2] = [&[42, 7], &[11]];
    let selectors = [41_u32, 7, 100];
    for (weights, format, label) in [
        (&dequant_bytes, Lfm2WeightFormat::Dense, "dense"),
        (&packed_bytes, Lfm2WeightFormat::TernaryV1, "packed"),
    ] {
        let mut executor = load_executor(&config, weights, format, 1 << 24);

        let mut base_task = executor
            .prefill_choice_base(TokenChunk::all(&base_tokens))
            .expect("base prefill");
        let base = ready(&mut base_task);
        assert_eq!(base.token_history(), base_tokens, "{label}");
        assert_eq!(base.logical_length(), 3, "{label}");
        assert_eq!(
            executor.sampled_token(&base).expect("sampled"),
            None,
            "{label}: unscored base must carry no sample"
        );
        assert!(
            executor.next_logits(&base).is_err(),
            "{label}: unscored base must carry no logits"
        );

        for tail in tails {
            let mut task = executor
                .append_choice_logits(&base, TokenChunk::all(tail), &selectors)
                .expect("branch");
            let values = ready(&mut task);
            let full: Vec<u32> = base_tokens.iter().chain(tail.iter()).copied().collect();
            let mut direct = executor
                .prefill_choice_logits(TokenChunk::all(&full), &selectors)
                .expect("direct prefill");
            assert_eq!(
                values,
                ready(&mut direct),
                "{label}: branch logits diverged from direct base+tail prefill"
            );
            assert_eq!(base.token_history(), base_tokens, "{label}: base mutated");
            assert_eq!(base.logical_length(), 3, "{label}: base length mutated");
        }

        // A forked copy of the same base branches identically.
        let mut fork_task = executor.fork(&base).expect("fork base");
        let forked = ready(&mut fork_task);
        let mut forked_branch = executor
            .append_choice_logits(&forked, TokenChunk::all(tails[0]), &selectors)
            .expect("forked branch");
        let mut direct = executor
            .prefill_choice_logits(
                TokenChunk::all(
                    &base_tokens
                        .iter()
                        .chain(tails[0])
                        .copied()
                        .collect::<Vec<_>>(),
                ),
                &selectors,
            )
            .expect("direct prefill");
        assert_eq!(
            ready(&mut forked_branch),
            ready(&mut direct),
            "{label}: forked branch diverged"
        );

        assert!(
            executor.prefill_choice_base(TokenChunk::all(&[])).is_err(),
            "{label}: empty base accepted"
        );
        assert!(
            executor
                .append_choice_logits(&base, TokenChunk::all(&[]), &selectors)
                .is_err(),
            "{label}: empty tail accepted"
        );
        assert!(
            executor
                .append_choice_logits(&base, TokenChunk::all(&[7]), &[])
                .is_err(),
            "{label}: empty selectors accepted"
        );
        assert!(
            executor
                .append_choice_logits(&base, TokenChunk::all(&[7]), &[1, 128])
                .is_err(),
            "{label}: out-of-vocab selector accepted"
        );
        let oversized = vec![3_u32; 62];
        assert!(
            executor
                .append_choice_logits(&base, TokenChunk::all(&oversized), &selectors)
                .is_err(),
            "{label}: base+tail overflow accepted"
        );

        let mut task = executor
            .append_choice_logits(&base, TokenChunk::all(&[7]), &selectors)
            .expect("branch");
        let _ = ready(&mut task);
        assert_eq!(
            task.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
            "{label}: completion must be consumed"
        );

        let mut cancelled = executor
            .append_choice_logits(&base, TokenChunk::all(&[7]), &selectors)
            .expect("branch");
        cancelled.cancel().expect("cancel");
        assert_eq!(
            cancelled.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
            "{label}: cancelled branch must not produce values"
        );
        // The cancelled branch leaves the base valid for another criterion.
        let mut after = executor
            .append_choice_logits(&base, TokenChunk::all(&[7]), &selectors)
            .expect("branch after cancel");
        let _ = ready(&mut after);
        assert_eq!(base.token_history(), base_tokens, "{label}: base mutated");
    }
}

#[test]
fn choice_logits_reads_back_only_selected_columns_in_caller_order() {
    let (config, packed_bytes, _) = tiny_packed_fixture();
    let mut executor = load_executor(&config, &packed_bytes, Lfm2WeightFormat::TernaryV1, 1 << 24);
    let mut task = executor
        .prefill(TokenChunk::all(&[3, 5, 9]))
        .expect("prefill");
    let prefix = ready(&mut task);

    let selectors = [41_u32, 7, 100, 7];
    let mut logits_task = executor.next_logits(&prefix).expect("logits");
    let full = ready(&mut logits_task);
    let expected: Vec<f32> = selectors.iter().map(|&id| full[id as usize]).collect();
    let mut choice = executor
        .choice_logits(&prefix, &selectors)
        .expect("choice logits");
    assert_eq!(ready(&mut choice), expected);
    assert_eq!(
        choice.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed))
    );

    assert!(executor.choice_logits(&prefix, &[]).is_err());
    assert!(executor.choice_logits(&prefix, &[1, 128]).is_err());
    let mut empty_task = executor
        .prefill(TokenChunk::all(&[]))
        .expect("empty prefill");
    let empty_prefix = ready(&mut empty_task);
    assert!(executor.choice_logits(&empty_prefix, &[1]).is_err());
}

/// Teacher-forces `tokens` through the executor and returns the final
/// position's logits plus the top-1 token at each position.
fn teacher_forced_logits(
    executor: &mut Lfm2Executor<CpuBackend>,
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
#[ignore = "requires MINIFIELD_LFM25_BUNDLE_DIR and MINIFIELD_LFM25_PACKED"]
fn real_packed_model_loads_and_runs() {
    let bundle = PathBuf::from(
        std::env::var_os("MINIFIELD_LFM25_BUNDLE_DIR").expect("MINIFIELD_LFM25_BUNDLE_DIR"),
    );
    let packed_path =
        PathBuf::from(std::env::var_os("MINIFIELD_LFM25_PACKED").expect("MINIFIELD_LFM25_PACKED"));
    let config = fs::read(bundle.join("config.json")).expect("config.json");
    let packed_bytes = fs::read(&packed_path).expect("packed safetensors");
    let dense_bytes = fs::read(bundle.join("model.safetensors")).expect("dense safetensors");

    // Oracle prompt p01, tokenized by llama.cpp (BOS included).
    let tokens: Vec<u32> = vec![
        1, 1098, 4605, 10800, 36387, 56586, 1391, 779, 46199, 4949, 3627, 779, 8008, 18703, 963,
        28707, 521, 2158, 7766, 27870, 12098, 988, 779, 6707, 1883, 523,
    ];

    let packed_start = std::time::Instant::now();
    let mut packed_exec =
        load_executor(&config, &packed_bytes, Lfm2WeightFormat::TernaryV1, 1 << 30);
    let packed_load = packed_start.elapsed();
    let packed_run = std::time::Instant::now();
    let (packed_logits, packed_top) = teacher_forced_logits(&mut packed_exec, &tokens);
    let packed_run = packed_run.elapsed();
    assert_eq!(packed_logits.len(), 65536, "logit width");
    assert!(packed_logits.iter().all(|v| v.is_finite()), "finite logits");

    // Informational quality signal: naive absmax-RTN ternary vs the dense BF16
    // model on identical token histories. No assertion; the number belongs in
    // the task report.
    let dense_start = std::time::Instant::now();
    let mut dense_exec = load_executor(&config, &dense_bytes, Lfm2WeightFormat::Dense, 1 << 30);
    let dense_load = dense_start.elapsed();
    let dense_run = std::time::Instant::now();
    let (dense_logits, dense_top) = teacher_forced_logits(&mut dense_exec, &tokens);
    let dense_run = dense_run.elapsed();
    let agreements = packed_top
        .iter()
        .zip(dense_top.iter())
        .filter(|(a, b)| a == b)
        .count();
    let max_delta = packed_logits
        .iter()
        .zip(dense_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "packed vs dense: {}/{} top-1 positions agree, final-position max|delta|={max_delta:.3}",
        agreements,
        packed_top.len()
    );
    println!("packed top-1 sequence: {packed_top:?}");
    println!("dense  top-1 sequence: {dense_top:?}");
    println!(
        "packed: load {packed_load:?}, 26-token teacher-forced run {packed_run:?}; \
         dense: load {dense_load:?}, run {dense_run:?}"
    );
}

/// Mixed bundle: every rank-two matmul packs ternary except the token
/// embedding, which stays dense exactly as a QAT-eval scope would leave it.
/// Returns (config, mixed safetensors, expected-dense safetensors).
fn tiny_mixed_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (config, dense) = tiny_dense_base();
    let mut mixed_tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut expected_tensors: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut quantization = serde_json::Map::new();
    for (name, dims, values) in &dense {
        let is_matmul =
            dims.len() == 2 && *dims.last().expect("dims") >= 128 && dims[1].is_multiple_of(128);
        if is_matmul && name != "model.embed_tokens.weight" {
            let rows = dims[0] as usize;
            let columns = dims[1] as usize;
            let groups = columns / 128;
            let mut codes = Vec::with_capacity(rows * columns / 4);
            let mut scales = Vec::with_capacity(rows * groups * 2);
            let mut dequant = Vec::with_capacity(values.len());
            for row in values.chunks_exact(columns) {
                for group in row.as_chunks::<128>().0 {
                    let (code_bytes, decoded, deq) = pack_group(group);
                    codes.extend_from_slice(&code_bytes);
                    scales.extend_from_slice(&f32_to_f16(decoded).to_le_bytes());
                    dequant.extend_from_slice(&deq);
                }
            }
            mixed_tensors.push((
                format!("{name}.codes"),
                "U8".to_owned(),
                vec![dims[0], dims[1] / 4],
                codes,
            ));
            mixed_tensors.push((
                format!("{name}.scales"),
                "F16".to_owned(),
                vec![dims[0], groups as u64],
                scales,
            ));
            expected_tensors.push((
                name.clone(),
                "F32".to_owned(),
                dims.clone(),
                f32_bytes(&dequant),
            ));
            quantization.insert(name.clone(), Value::from("ternary-v1"));
        } else {
            let bytes = f32_bytes(values);
            mixed_tensors.push((name.clone(), "F32".to_owned(), dims.clone(), bytes.clone()));
            expected_tensors.push((name.clone(), "F32".to_owned(), dims.clone(), bytes));
            quantization.insert(name.clone(), Value::from("f32"));
        }
    }
    let mixed_refs: Vec<(&str, &str, Vec<u64>, &[u8])> = mixed_tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), b.as_slice()))
        .collect();
    let expected_refs: Vec<(&str, &str, Vec<u64>, &[u8])> = expected_tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), b.as_slice()))
        .collect();
    let metadata = json!({
        "format": "minifield.mixed.v1",
        "tensor_quantization": serde_json::to_string(&quantization).expect("quantization map"),
    });
    let mixed_bytes = safetensors_with_metadata(&mixed_refs, Some(metadata));
    let expected_bytes = safetensors(&expected_refs);
    (config, mixed_bytes, expected_bytes)
}

#[test]
fn mixed_bundle_loads_dense_embedding_and_packed_matmuls() {
    let (config, mixed_bytes, expected_bytes) = tiny_mixed_fixture();

    let format = detect_lfm2_weight_format(&mixed_bytes).expect("detect format");
    assert_eq!(format, Lfm2WeightFormat::MixedV1);
    let quantization = parse_lfm2_tensor_quantization(&mixed_bytes).expect("quantization map");
    let request = Lfm2LoadRequest::new_with_quantization(
        config.clone(),
        digest(&config),
        mixed_bytes.len() as u64,
        digest(&mixed_bytes),
        LoaderLimits {
            max_asset_bytes: mixed_bytes.len() as u64,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: mixed_bytes.len() as u64,
            max_retained_host_bytes: mixed_bytes.len() as u64 * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        format,
        &quantization,
    )
    .expect("load request");
    let plan = request.plan();
    assert_eq!(
        plan.role_quant(Lfm2WeightRole::TokenEmbedding),
        Lfm2WeightFormat::Dense
    );
    assert_eq!(
        plan.role_quant(Lfm2WeightRole::Layer {
            index: 0,
            role: Lfm2LayerWeightRole::FfnW1
        }),
        Lfm2WeightFormat::TernaryV1
    );
    assert!(plan.has_packed());

    let mut mixed_exec = finish_load(request, &mixed_bytes, 1 << 24);
    let mut dense_exec = load_executor(&config, &expected_bytes, Lfm2WeightFormat::Dense, 1 << 24);

    let tokens: Vec<u32> = vec![1, 5, 9, 42, 7, 100];
    let mut mixed_task = mixed_exec
        .prefill(TokenChunk::all(&tokens[..1]))
        .expect("prefill");
    let mut dense_task = dense_exec
        .prefill(TokenChunk::all(&tokens[..1]))
        .expect("prefill");
    let mut mixed_prefix = ready(&mut mixed_task);
    let mut dense_prefix = ready(&mut dense_task);
    for (step, token) in tokens.iter().enumerate().skip(1) {
        let mut ma = mixed_exec
            .append_known(&mixed_prefix, TokenChunk::all(&[*token]))
            .expect("append");
        let mut da = dense_exec
            .append_known(&dense_prefix, TokenChunk::all(&[*token]))
            .expect("append");
        mixed_prefix = ready(&mut ma);
        dense_prefix = ready(&mut da);
        let mut ml = mixed_exec.next_logits(&mixed_prefix).expect("logits");
        let mut dl = dense_exec.next_logits(&dense_prefix).expect("logits");
        assert_within_reorder(
            &ready(&mut ml),
            &ready(&mut dl),
            &format!("step {step}: mixed bundle diverged from expected dense"),
        );
    }
}
