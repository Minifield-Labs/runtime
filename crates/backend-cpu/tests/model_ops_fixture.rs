#![allow(clippy::expect_used, clippy::cast_possible_truncation)]
// Fixture JSON numbers parse as f64, then deliberately reconstruct the declared f32 inputs.

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    ExecutorError, GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, PackedHeadSpec,
    RectCopy2d, ResourceLimits, RotarySpec, Shape,
};
use serde_json::Value;

const CASES: &str = include_str!("fixtures/model-ops-001/cases.json");
const MANIFEST: &str = include_str!("fixtures/model-ops-001/manifest.json");
const ATOL: f32 = 2.0e-6;
const RTOL: f32 = 3.0e-5;

fn backend() -> CpuBackend {
    CpuBackend::new(
        202,
        ResourceLimits {
            max_allocation_bytes: 65_536,
            max_total_bytes: 262_144,
            max_pending_operations: 4,
        },
    )
}

fn f32s(value: &Value) -> Vec<f32> {
    match value {
        Value::Number(number) => vec![number.as_f64().expect("finite JSON number") as f32],
        Value::Array(items) => items.iter().flat_map(f32s).collect(),
        _ => panic!("expected numeric JSON array"),
    }
}

fn rows(value: &Value) -> usize {
    value.as_array().expect("array").len()
}
fn width2(value: &Value) -> usize {
    value
        .as_array()
        .expect("array")
        .first()
        .expect("nonempty row")
        .as_array()
        .expect("row")
        .len()
}
fn width3(value: &Value) -> (usize, usize) {
    let token = value
        .as_array()
        .expect("tokens")
        .first()
        .expect("token")
        .as_array()
        .expect("heads");
    (
        token.len(),
        token
            .first()
            .expect("head")
            .as_array()
            .expect("dimensions")
            .len(),
    )
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let bound = ATOL + RTOL * want.abs();
        assert!(
            (got - want).abs() <= bound,
            "{label}[{index}]: got {got:?}, expected {want:?}, bound {bound:?}"
        );
    }
}

fn case_by_id<'a>(root: &'a Value, id: &str) -> &'a Value {
    root["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["id"] == id)
        .expect("fixture case")
}

#[test]
fn independent_model_ops_fixture_hashes_and_inventory_are_pinned() {
    let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest JSON");
    assert_eq!(
        manifest["cases_sha256"],
        "c5dfba5fbd4652c9a9cb9f6ea5f86e8db582b831b46189cd719a314eb39ff33f"
    );
    assert_eq!(
        manifest["generator_sha256"],
        "2be4f1c791cabcfde1090969104c514970ac85fd913db2eea8faa63c7123a374"
    );
    assert_eq!(manifest["count"], 18);
    let root: Value = serde_json::from_str(CASES).expect("cases JSON");
    assert_eq!(root["cases"].as_array().expect("cases").len(), 18);
}

#[test]
fn independent_short_convolution_cases_cover_startup_partitions_and_asymmetry() {
    let root: Value = serde_json::from_str(CASES).expect("cases JSON");
    for id in [
        "conv_scalar_full",
        "conv_scalar_0_1",
        "conv_scalar_1_2",
        "conv_scalar_2_3",
        "conv_scalar_3_4",
        "conv_scalar_0_2",
        "conv_scalar_2_4",
        "conv_scalar_0_3",
        "conv_scalar_1_4",
        "conv_asymmetric_full",
        "conv_asymmetric_tail",
        "conv_width_one",
    ] {
        let case = case_by_id(&root, id);
        let b = &case["B"];
        let c = &case["C"];
        let v = &case["V"];
        let kernel = &case["kernel"];
        let tokens = rows(b);
        let hidden = width2(b);
        let width = width2(kernel);
        let mut runtime = backend();
        let b = runtime
            .upload_f32(
                Shape::new(&[tokens as u64, hidden as u64]).expect("shape"),
                &f32s(b),
            )
            .expect("B");
        let c = runtime
            .upload_f32(
                Shape::new(&[tokens as u64, hidden as u64]).expect("shape"),
                &f32s(c),
            )
            .expect("C");
        let v = runtime
            .upload_f32(
                Shape::new(&[tokens as u64, hidden as u64]).expect("shape"),
                &f32s(v),
            )
            .expect("V");
        let kernel = runtime
            .upload_f32(
                Shape::new(&[hidden as u64, width as u64]).expect("shape"),
                &f32s(kernel),
            )
            .expect("kernel");
        let mut history_values = vec![0.0; width.saturating_sub(1) * hidden];
        let prefix = f32s(&case["prefix_U"]);
        let prefix_offset = history_values.len() - prefix.len();
        history_values[prefix_offset..].copy_from_slice(&prefix);
        let mut history = runtime
            .upload_f32(
                Shape::new(&[width.saturating_sub(1) as u64, hidden as u64]).expect("shape"),
                &history_values,
            )
            .expect("history");
        let mut output = runtime
            .allocate_f32(Shape::new(&[tokens as u64, hidden as u64]).expect("shape"))
            .expect("output");
        runtime
            .gated_short_convolution(
                &mut output,
                &b,
                &c,
                &v,
                &kernel,
                &mut history,
                GatedShortConvSpec::new(hidden as u32, width as u32).expect("spec"),
            )
            .expect("conv");
        close(
            &runtime.read_f32(&output).expect("read output"),
            &f32s(&case["output"]),
            id,
        );
        let tail = f32s(&case["tail_U"]);
        let actual_history = runtime.read_f32(&history).expect("read history");
        close(
            &actual_history[actual_history.len() - tail.len()..],
            &tail,
            &format!("{id}: history"),
        );
    }
}

#[test]
fn independent_rotary_and_head_rms_cases_use_declared_packed_layouts() {
    let root: Value = serde_json::from_str(CASES).expect("cases JSON");
    for id in ["rotary_split_half", "rotary_dim_six"] {
        let case = case_by_id(&root, id);
        let input = &case["input"];
        let tokens = rows(input);
        let (heads, dim) = width3(input);
        let mut runtime = backend();
        let input = runtime
            .upload_f32(
                Shape::new(&[tokens as u64, (heads * dim) as u64]).expect("shape"),
                &f32s(input),
            )
            .expect("input");
        let mut output = runtime
            .allocate_f32(Shape::new(&[tokens as u64, (heads * dim) as u64]).expect("shape"))
            .expect("output");
        let positions: Vec<u64> = case["positions"]
            .as_array()
            .expect("positions")
            .iter()
            .map(|x| x.as_u64().expect("position"))
            .collect();
        runtime
            .split_half_rotary(
                &mut output,
                &input,
                &positions,
                RotarySpec::new(
                    PackedHeadSpec::new(heads as u32, dim as u32).expect("heads"),
                    case["theta"].as_f64().expect("theta") as f32,
                )
                .expect("RoPE spec"),
            )
            .expect("RoPE");
        close(
            &runtime.read_f32(&output).expect("read"),
            &f32s(&case["output"]),
            id,
        );
    }
    let case = case_by_id(&root, "head_local_qk_rms");
    let input = &case["input"];
    let tokens = rows(input);
    let (heads, dim) = width3(input);
    let mut runtime = backend();
    let input = runtime
        .upload_f32(
            Shape::new(&[tokens as u64, (heads * dim) as u64]).expect("shape"),
            &f32s(input),
        )
        .expect("input");
    let weight = runtime
        .upload_f32(
            Shape::new(&[dim as u64]).expect("shape"),
            &f32s(&case["weight"]),
        )
        .expect("weight");
    let mut output = runtime
        .allocate_f32(Shape::new(&[tokens as u64, (heads * dim) as u64]).expect("shape"))
        .expect("output");
    runtime
        .head_rms_norm(
            &mut output,
            &input,
            &weight,
            PackedHeadSpec::new(heads as u32, dim as u32).expect("heads"),
            case["epsilon"].as_f64().expect("epsilon") as f32,
        )
        .expect("head RMS");
    close(
        &runtime.read_f32(&output).expect("read"),
        &f32s(&case["output"]),
        "head_local_qk_rms",
    );
}

#[test]
fn independent_gqa_cases_cover_full_chunk_and_decode_cache_paths() {
    let root: Value = serde_json::from_str(CASES).expect("cases JSON");
    for id in ["gqa_4_to_2_full", "gqa_4_to_2_tail", "gqa_6_to_2_decode"] {
        let case = case_by_id(&root, id);
        let query = &case["Q"];
        let key = &case["K"];
        let value = &case["V"];
        let query_tokens = rows(query);
        let total = rows(key);
        let (query_heads, dim) = width3(query);
        let (kv_heads, kv_dim) = width3(key);
        assert_eq!(dim, kv_dim);
        let start = case["query_start"].as_u64().expect("query start") as usize;
        assert_eq!(query_tokens, total - start);
        let query_values = f32s(query);
        let key_values = f32s(key);
        let value_values = f32s(value);
        let query_width = query_heads * dim;
        let kv_width = kv_heads * dim;
        let mut runtime = backend();
        let query = runtime
            .upload_f32(
                Shape::new(&[query_tokens as u64, query_width as u64]).expect("shape"),
                &query_values,
            )
            .expect("Q");
        let key = runtime
            .upload_f32(
                Shape::new(&[query_tokens as u64, kv_width as u64]).expect("shape"),
                &key_values[start * kv_width..],
            )
            .expect("K tail");
        let value = runtime
            .upload_f32(
                Shape::new(&[query_tokens as u64, kv_width as u64]).expect("shape"),
                &value_values[start * kv_width..],
            )
            .expect("V tail");
        let mut key_cache_values = vec![0.0; total * kv_width];
        let mut value_cache_values = vec![0.0; total * kv_width];
        key_cache_values[..start * kv_width].copy_from_slice(&key_values[..start * kv_width]);
        value_cache_values[..start * kv_width].copy_from_slice(&value_values[..start * kv_width]);
        let mut key_cache = runtime
            .upload_f32(
                Shape::new(&[total as u64, kv_width as u64]).expect("shape"),
                &key_cache_values,
            )
            .expect("K cache");
        let mut value_cache = runtime
            .upload_f32(
                Shape::new(&[total as u64, kv_width as u64]).expect("shape"),
                &value_cache_values,
            )
            .expect("V cache");
        let mut output = runtime
            .allocate_f32(Shape::new(&[query_tokens as u64, query_width as u64]).expect("shape"))
            .expect("output");
        let mut cache_len = start as u64;
        runtime
            .causal_gqa(
                &mut output,
                &query,
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                &mut cache_len,
                GqaSpec::new(query_heads as u32, kv_heads as u32, dim as u32).expect("spec"),
            )
            .expect("GQA");
        assert_eq!(cache_len, total as u64, "{id}: cache len");
        close(
            &runtime.read_f32(&output).expect("read"),
            &f32s(&case["output"]),
            id,
        );
        close(
            &runtime.read_f32(&key_cache).expect("key cache")[..total * kv_width],
            &key_values,
            &format!("{id}: key cache"),
        );
        close(
            &runtime.read_f32(&value_cache).expect("value cache")[..total * kv_width],
            &value_values,
            &format!("{id}: value cache"),
        );
    }
}

#[test]
fn checked_rectangular_copy_rejects_overlap_and_out_of_bounds() {
    let mut runtime = backend();
    let input = runtime
        .upload_f32(
            Shape::new(&[3, 4]).expect("shape"),
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        )
        .expect("input");
    let mut output = runtime
        .upload_f32(Shape::new(&[4, 5]).expect("shape"), &[-1.; 20])
        .expect("output");
    runtime
        .copy_rect_2d(&mut output, &input, RectCopy2d::new(1, 1, 2, 2, 2, 3))
        .expect("rect copy");
    assert_eq!(
        runtime.read_f32(&output).expect("read"),
        vec![
            -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 5., 6., 7., -1., -1., 9.,
            10., 11.
        ]
    );
    assert!(
        runtime
            .copy_rect_2d(&mut output, &input, RectCopy2d::new(2, 3, 0, 0, 2, 1))
            .is_err()
    );
}

#[test]
fn gqa_rejects_over_capacity_before_mutating_cache_or_length() {
    let mut runtime = backend();
    let query = runtime
        .upload_f32(Shape::new(&[1, 4]).expect("shape"), &[0.1, 0.2, 0.3, 0.4])
        .expect("query");
    let key = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[0.5, 0.6])
        .expect("key");
    let value = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[0.7, 0.8])
        .expect("value");
    let mut key_cache = runtime
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[1., 2., 3., 4.])
        .expect("key cache");
    let mut value_cache = runtime
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[5., 6., 7., 8.])
        .expect("value cache");
    let key_before = runtime.read_f32(&key_cache).expect("key snapshot");
    let value_before = runtime.read_f32(&value_cache).expect("value snapshot");
    let mut output = runtime
        .allocate_f32(Shape::new(&[1, 4]).expect("shape"))
        .expect("output");
    let mut cache_len = 2;
    assert!(
        runtime
            .causal_gqa(
                &mut output,
                &query,
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                &mut cache_len,
                GqaSpec::new(2, 1, 2).expect("spec")
            )
            .is_err()
    );
    assert_eq!(cache_len, 2);
    assert_eq!(runtime.read_f32(&key_cache).expect("key after"), key_before);
    assert_eq!(
        runtime.read_f32(&value_cache).expect("value after"),
        value_before
    );
}

const ROUNDING_SENTINELS: &str =
    include_str!("fixtures/cr02a-rounding-sentinels-001/rounding-sentinels.json");

#[test]
fn independent_rounding_sentinels_require_sequential_f32_vector_math() {
    let sentinel: Value = serde_json::from_str(ROUNDING_SENTINELS).expect("sentinel JSON");
    let rotary = &sentinel["rotary"];
    let input_values = f32s(&rotary["input"]);
    let expected_rotary: Vec<u32> = rotary["expected_u32"]
        .as_array()
        .expect("rotary expected bits")
        .iter()
        .map(|value| value.as_u64().expect("rotary bit") as u32)
        .collect();
    let mut runtime = backend();
    let input = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &input_values)
        .expect("rotary input");
    let mut output = runtime
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("rotary output");
    runtime
        .split_half_rotary(
            &mut output,
            &input,
            &[rotary["position"].as_u64().expect("position")],
            RotarySpec::new(
                PackedHeadSpec::new(1, rotary["head_dim"].as_u64().expect("head dim") as u32)
                    .expect("heads"),
                rotary["theta"].as_f64().expect("theta") as f32,
            )
            .expect("spec"),
        )
        .expect("rotary");
    assert_eq!(
        output
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_rotary,
        "RoPE vector products and sums must stay at their declared f32 boundaries"
    );

    let convolution = &sentinel["convolution"];
    let b = runtime
        .upload_f32(
            Shape::new(&[1, 1]).expect("shape"),
            &f32s(&convolution["B"]),
        )
        .expect("B");
    let c = runtime
        .upload_f32(
            Shape::new(&[1, 1]).expect("shape"),
            &f32s(&convolution["C"]),
        )
        .expect("C");
    let v = runtime
        .upload_f32(
            Shape::new(&[1, 1]).expect("shape"),
            &f32s(&convolution["V"]),
        )
        .expect("V");
    let kernel = runtime
        .upload_f32(
            Shape::new(&[1, 3]).expect("shape"),
            &f32s(&convolution["kernel"]),
        )
        .expect("kernel");
    let mut history = runtime
        .upload_f32(
            Shape::new(&[2, 1]).expect("shape"),
            &f32s(&convolution["prefix_U"]),
        )
        .expect("history");
    let mut output = runtime
        .allocate_f32(Shape::new(&[1, 1]).expect("shape"))
        .expect("conv output");
    runtime
        .gated_short_convolution(
            &mut output,
            &b,
            &c,
            &v,
            &kernel,
            &mut history,
            GatedShortConvSpec::new(1, 3).expect("conv spec"),
        )
        .expect("conv");
    let expected_conv: Vec<u32> = convolution["expected_u32"]
        .as_array()
        .expect("conv expected rows")
        .iter()
        .flat_map(|row| row.as_array().expect("conv expected row"))
        .map(|value| value.as_u64().expect("conv expected bit") as u32)
        .collect();
    assert_eq!(
        output
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_conv,
        "short-convolution tap accumulation must be sequential f32"
    );
}

#[test]
fn generic_inference_ops_compose_packed_projection_head_rms_rotary_and_gqa() {
    let mut runtime = backend();
    let input = InferenceOps::upload_f32(
        &mut runtime,
        Shape::new(&[2, 2]).expect("shape"),
        &[1.0, 2.0, 3.0, 4.0],
    )
    .expect("input");
    let projection_weight = InferenceOps::upload_f32(
        &mut runtime,
        Shape::new(&[4, 2]).expect("shape"),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    )
    .expect("projection weight");
    let mut projected =
        InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 4]).expect("shape"))
            .expect("projection output");
    InferenceOps::linear(&runtime, &mut projected, &input, &projection_weight).expect("projection");

    let heads = PackedHeadSpec::new(2, 2).expect("heads");
    let rms_weight =
        InferenceOps::upload_f32(&mut runtime, Shape::new(&[2]).expect("shape"), &[1.0, 1.0])
            .expect("RMS weight");
    let mut normalized =
        InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 4]).expect("shape"))
            .expect("RMS output");
    InferenceOps::head_rms_norm(
        &runtime,
        &mut normalized,
        &projected,
        &rms_weight,
        heads,
        1e-5,
    )
    .expect("head RMS");

    let mut query = InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 4]).expect("shape"))
        .expect("rotary output");
    InferenceOps::split_half_rotary(
        &runtime,
        &mut query,
        &normalized,
        &[0, 1],
        RotarySpec::new(heads, 1_000_000.0).expect("rotary spec"),
    )
    .expect("rotary");

    let key = InferenceOps::upload_f32(
        &mut runtime,
        Shape::new(&[2, 2]).expect("shape"),
        &[0.25, 0.5, 0.75, 1.0],
    )
    .expect("key");
    let value = InferenceOps::upload_f32(
        &mut runtime,
        Shape::new(&[2, 2]).expect("shape"),
        &[1.0, 2.0, 3.0, 4.0],
    )
    .expect("value");
    let mut key_cache =
        InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 2]).expect("shape"))
            .expect("key cache");
    let mut value_cache =
        InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 2]).expect("shape"))
            .expect("value cache");
    let mut attended =
        InferenceOps::allocate_f32(&mut runtime, Shape::new(&[2, 4]).expect("shape"))
            .expect("attention output");
    let mut cache_len = 0;
    InferenceOps::causal_gqa(
        &runtime,
        &mut attended,
        &query,
        &key,
        &value,
        &mut key_cache,
        &mut value_cache,
        &mut cache_len,
        GqaSpec::new(2, 1, 2).expect("GQA spec"),
    )
    .expect("GQA");
    assert_eq!(cache_len, 2);
    assert!(attended.as_slice().iter().all(|value| value.is_finite()));
    assert!(key_cache.as_slice().iter().all(|value| value.is_finite()));
    assert!(value_cache.as_slice().iter().all(|value| value.is_finite()));
}

#[test]
#[allow(clippy::too_many_lines)] // Exercises independent convolution and GQA rollback/recovery cases in one resource-limited scenario.
fn held_readbacks_prevent_allocation_and_operator_staging_until_cancelled() {
    let limits = ResourceLimits {
        max_allocation_bytes: 16,
        max_total_bytes: 36,
        max_pending_operations: 2,
    };
    let mut runtime = CpuBackend::new(0xC0F0, limits);
    let b = runtime
        .upload_f32(Shape::new(&[1, 1]).expect("shape"), &[2.0])
        .expect("B");
    let c = runtime
        .upload_f32(Shape::new(&[1, 1]).expect("shape"), &[3.0])
        .expect("C");
    let v = runtime
        .upload_f32(Shape::new(&[1, 1]).expect("shape"), &[4.0])
        .expect("V");
    let kernel = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[1.0, 2.0])
        .expect("kernel");
    let mut history = runtime
        .upload_f32(Shape::new(&[1, 1]).expect("shape"), &[9.0])
        .expect("history");
    let mut output = runtime
        .allocate_f32(Shape::new(&[1, 1]).expect("shape"))
        .expect("output");
    let mut held = runtime.read_f32_async(&b).expect("held readback");
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(32));
    assert!(matches!(
        runtime.allocate_f32(Shape::new(&[2]).expect("shape")),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    ));
    assert_eq!(
        runtime.gated_short_convolution(
            &mut output,
            &b,
            &c,
            &v,
            &kernel,
            &mut history,
            GatedShortConvSpec::new(1, 2).expect("conv spec"),
        ),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    );
    assert_eq!(history.as_slice(), &[9.0]);
    assert_eq!(output.as_slice(), &[0.0]);
    held.cancel().expect("cancel readback");
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(28));
    runtime
        .gated_short_convolution(
            &mut output,
            &b,
            &c,
            &v,
            &kernel,
            &mut history,
            GatedShortConvSpec::new(1, 2).expect("conv spec"),
        )
        .expect("conv after cancel");
    assert_eq!(history.as_slice(), &[8.0]);
    assert_eq!(output.as_slice(), &[75.0]);

    let limits = ResourceLimits {
        max_allocation_bytes: 16,
        max_total_bytes: 60,
        max_pending_operations: 2,
    };
    let mut runtime = CpuBackend::new(0xC0F1, limits);
    let query = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[0.5, 0.25])
        .expect("query");
    let key = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[0.4, 0.8])
        .expect("key");
    let value = runtime
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[1.0, 3.0])
        .expect("value");
    let mut key_cache = runtime
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("key cache");
    let mut value_cache = runtime
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("value cache");
    let mut output = runtime
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("output");
    let mut held = runtime.read_f32_async(&query).expect("held readback");
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(56));
    let mut cache_len = 0;
    assert_eq!(
        runtime.causal_gqa(
            &mut output,
            &query,
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            &mut cache_len,
            GqaSpec::new(1, 1, 2).expect("GQA spec"),
        ),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    );
    assert_eq!(cache_len, 0);
    assert_eq!(key_cache.as_slice(), &[0.0, 0.0]);
    assert_eq!(value_cache.as_slice(), &[0.0, 0.0]);
    assert_eq!(output.as_slice(), &[0.0, 0.0]);
    held.cancel().expect("cancel readback");
    runtime
        .causal_gqa(
            &mut output,
            &query,
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            &mut cache_len,
            GqaSpec::new(1, 1, 2).expect("GQA spec"),
        )
        .expect("GQA after cancel");
    assert_eq!(cache_len, 1);
    assert_eq!(key_cache.as_slice(), &[0.4, 0.8]);
    assert_eq!(value_cache.as_slice(), &[1.0, 3.0]);
}
