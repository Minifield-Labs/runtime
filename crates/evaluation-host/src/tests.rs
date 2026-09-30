#![allow(clippy::expect_used, clippy::float_cmp)]
use crate::{
    HostResult, backend, completion, measurement, prediction, preparation,
    request::{Case, Request},
    run,
};
use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{CompletionPoll, EncoderSegments, InferenceCompletion, ResourceLimits};
use minifield_executor_core::{
    EncoderInput, PointerAnswer, PointerOutput, PointerQuestion, PointerQuestionKind,
    decode_pointer,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn request_value() -> Value {
    json!({"schema_version":1,"backend":"cpu_reference","arithmetic":"f32","task":"classifier","bundle":"unused","tokenizer":"unused","inputs":"unused",
        "classes":3,"context":16,"mode":"full","warmups":0,"measured_cycles":1,"phase":"correctness","deadline_seconds":10.0,"lut2_mode":"off","max_lut2_bytes":0})
}
fn request() -> Request {
    serde_json::from_value(request_value()).expect("request")
}

#[test]
fn request_rejects_unknown_fields_and_invalid_phase_bounds() {
    for (key, replacement) in [
        ("schema_version", json!(2)),
        ("backend", json!("gpu_auto")),
        ("arithmetic", json!("f16")),
        ("task", json!("generation")),
        ("context", json!(0)),
        ("measured_cycles", json!(0)),
        ("deadline_seconds", json!(-1.0)),
        ("lut2_mode", json!("default")),
        ("classes", json!(0)),
        ("mode", json!("streaming")),
        ("phase", json!("anything")),
    ] {
        let mut value = request_value();
        value[key] = replacement;
        let parsed: Request = serde_json::from_value(value).expect("parse");
        assert!(parsed.validate().is_err(), "{key}");
    }
    let mut value = request_value();
    value["unknown"] = json!(1);
    assert!(serde_json::from_value::<Request>(value).is_err());
    let mut invalid = request();
    invalid.measured_cycles = 2;
    assert!(invalid.validate().is_err());
    invalid.phase = "measure".into();
    assert!(invalid.validate().is_ok());
    invalid.deadline_seconds = f64::NAN;
    assert!(invalid.validate().is_err());
    let mut value = request_value();
    value["task"] = json!("pointer");
    value["classes"] = Value::Null;
    value["mode"] = json!("cached");
    assert!(
        serde_json::from_value::<Request>(value)
            .expect("pointer request")
            .validate()
            .is_err()
    );
}

struct SlowReady {
    cancelled: bool,
}
impl InferenceCompletion for SlowReady {
    type Output = ();
    fn poll_step(&mut self) -> CompletionPoll<()> {
        std::thread::sleep(Duration::from_millis(8));
        CompletionPoll::Ready(Ok(()))
    }
    fn cancel(&mut self) -> minifield_engine_api::Result<()> {
        self.cancelled = true;
        Ok(())
    }
}
#[test]
fn deadline_rejects_cpu_recording_that_finishes_after_the_limit() {
    let mut task = SlowReady { cancelled: false };
    assert!(completion::wait(&mut task, completion::deadline(0.001).expect("deadline")).is_err());
    assert!(task.cancelled);
}

#[test]
fn decisions_preserve_continuous_pointer_values_and_integer_ordinal_levels() {
    let output = PointerOutput {
        tokens: 1,
        start: vec![1.0],
        end: vec![2.0],
        answers: vec![
            PointerAnswer::Choice {
                index: 1,
                probabilities: vec![0.4, 0.6],
            },
            PointerAnswer::Ordinal {
                value: 0.6,
                probabilities: vec![0.4, 0.6],
            },
            PointerAnswer::Binary {
                probability: 0.5,
                probabilities: vec![0.5, 0.5],
            },
            PointerAnswer::Span {
                span: Some([2, 4]),
                presence: 0.8,
            },
            PointerAnswer::Span {
                span: None,
                presence: 0.1,
            },
        ],
    };
    let (values, decisions) = prediction::pointer(output).expect("pointer output");
    assert_eq!(
        values,
        &[1.0, 2.0, 0.4, 0.6, 0.6, 0.4, 0.6, 0.5, 0.5, 0.5, 0.8, 0.1]
    );
    assert_eq!(
        decisions,
        json!([{"type":"choice","index":1},{"type":"ordinal","level":1},{"type":"binary","value":true},{"type":"span","start":2,"end":4},{"type":"absent"}])
    );
    assert_eq!(decisions[1]["level"].as_u64(), Some(1));
    assert_eq!(prediction::argmax(&[2.0, 2.0]).expect("argmax"), 0);
    for values in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        assert!(prediction::argmax(&values).is_err());
    }
    let output = PointerOutput {
        tokens: 1,
        start: vec![0.0],
        end: vec![0.0],
        answers: vec![PointerAnswer::Ordinal {
            value: f64::NAN,
            probabilities: vec![1.0],
        }],
    };
    assert!(prediction::pointer(output).is_err());
}

#[test]
fn host_decisions_keep_f64_values_at_binary_and_ordinal_thresholds() {
    let below = 0.5 - 1e-9;
    let output = PointerOutput {
        tokens: 1,
        start: vec![0.0],
        end: vec![0.0],
        answers: vec![
            PointerAnswer::Binary {
                probability: below,
                probabilities: vec![below, 1.0 - below],
            },
            PointerAnswer::Ordinal {
                value: below,
                probabilities: vec![1.0 - below, below],
            },
        ],
    };
    let (continuous, decisions) = prediction::pointer(output).expect("prediction");
    assert_eq!(continuous[2], below);
    assert_eq!(decisions[0], json!({"type":"binary","value":false}));
    assert_eq!(decisions[1], json!({"type":"ordinal","level":0}));
}

#[test]
fn decoded_compensated_presence_and_ordinal_values_survive_host_json() {
    let input = EncoderInput {
        token_ids: vec![1; 10],
        segments: EncoderSegments::single(10).expect("segments"),
        questions: vec![
            PointerQuestion {
                query_index: 0,
                option_indices: vec![],
                kind: PointerQuestionKind::Extract {
                    absent_index: 0,
                    source_start: 1,
                    selectable: vec![true; 9],
                    presence_threshold: 0.5,
                },
            },
            PointerQuestion {
                query_index: 0,
                option_indices: vec![0, 1, 2, 3, 4],
                kind: PointerQuestionKind::Ordinal,
            },
        ],
    };
    let mut logits = vec![-37.429_947_f32; 20];
    logits[0] = 0.0;
    logits[1] = -3e-16_f32;
    logits[10] = 0.0;
    logits[11] = 0.0;
    let output = decode_pointer(&input, logits.clone(), logits.clone()).expect("decoder");
    let (continuous, decisions) = prediction::pointer(output).expect("host prediction");
    assert_eq!(
        decisions,
        json!([{"type":"span","start":0,"end":1},{"type":"ordinal","level":1}])
    );
    assert_eq!(continuous.len(), 47);
    assert_eq!(continuous[40], 0.5);
    assert_eq!(continuous[41], f64::from_bits(0x3fe0_0000_0000_0002));
    assert_eq!(
        &continuous[..20],
        logits.iter().copied().map(f64::from).collect::<Vec<_>>()
    );
    let encoded = serde_json::to_vec(&(continuous, decisions)).expect("encode JSON");
    let decoded: (Vec<f64>, Value) = serde_json::from_slice(&encoded).expect("decode JSON");
    assert_eq!(decoded.0[40], 0.5);
    assert_eq!(decoded.0[41], f64::from_bits(0x3fe0_0000_0000_0002));
    assert_eq!(decoded.1[0], json!({"type":"span","start":0,"end":1}));
}

#[test]
fn ordinal_endpoint_rounding_preserves_decoder_value_through_host_json() {
    let input = EncoderInput {
        token_ids: vec![1; 11],
        segments: EncoderSegments::single(11).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: (0..11).collect(),
            kind: PointerQuestionKind::Ordinal,
        }],
    };
    let mut logits = vec![-1000.0_f32; 11];
    logits[9] = -36.75;
    logits[10] = 0.0;
    let output = decode_pointer(&input, logits.clone(), logits).expect("decoder");
    let expected = f64::from_bits(10.0_f64.to_bits() + 1);
    let value = match &output.answers[0] {
        PointerAnswer::Ordinal { value, .. } => *value,
        _ => f64::NAN,
    };
    // The compensated expectation rounds one ULP above the endpoint. Its
    // integer decision still names the valid final option.
    assert_eq!(value, expected);
    let (continuous, decisions) = prediction::pointer(output).expect("host prediction");
    assert_eq!(continuous[22], expected);
    assert_eq!(decisions, json!([{"type":"ordinal","level":10}]));
    let encoded = serde_json::to_vec(&(continuous, decisions)).expect("encode JSON");
    let decoded: (Vec<f64>, Value) = serde_json::from_slice(&encoded).expect("decode JSON");
    assert_eq!(decoded.0[22], expected);
    assert_eq!(decoded.1[0], json!({"type":"ordinal","level":10}));
    for value in [f64::NAN, f64::INFINITY, -0.5, 10.5] {
        let output = PointerOutput {
            tokens: 1,
            start: vec![0.0],
            end: vec![0.0],
            answers: vec![PointerAnswer::Ordinal {
                value,
                probabilities: vec![0.0; 11],
            }],
        };
        assert!(prediction::pointer(output).is_err());
    }
}

#[test]
fn fixed_work_measurement_rejects_nonfinite_outputs_and_counts_completed_predictions() {
    let case: Case = serde_json::from_value(json!({"id":"test","token_ids":[1]})).expect("case");
    let mut calls = 0;
    let result = measurement::case(&case, 3, || {
        calls += 1;
        Ok((vec![1.0], json!(0)))
    })
    .expect("measure");
    assert_eq!(calls, 3);
    assert_eq!(result["completed_predictions"], 3);
    assert_eq!(
        result["latencies_seconds"]
            .as_array()
            .expect("latencies")
            .len(),
        3
    );
    assert!(measurement::case(&case, 1, || Ok((vec![f64::NAN], json!(0)))).is_err());
}

#[test]
fn prefix_selection_leaves_a_nonempty_tail_for_every_case() {
    assert_eq!(preparation::common_prefix(&[]), 0);
    assert_eq!(
        preparation::common_prefix(&[vec![1, 2, 3], vec![1, 2, 4]]),
        2
    );
    assert_eq!(preparation::common_prefix(&[vec![1, 2], vec![1, 2]]), 1);
    assert_eq!(preparation::common_prefix(&[vec![], vec![1]]), 0);
}

struct Fixture {
    directory: PathBuf,
    request: Request,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    fn classifier() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "minifield-host-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).expect("fixture directory");
        let config =
            include_bytes!("../../executor-core/tests/fixtures/numerical-lfm-001-config.json");
        let source = include_bytes!(
            "../../executor-core/tests/fixtures/numerical-lfm-001-weights.safetensors"
        );
        let length = usize::try_from(u64::from_le_bytes(source[..8].try_into().expect("length")))
            .expect("length");
        let mut header: Value = serde_json::from_slice(&source[8..8 + length]).expect("header");
        let mut payload = source[8 + length..].to_vec();
        let offset = usize::try_from(
            header["model.embed_tokens.weight"]["data_offsets"][0]
                .as_u64()
                .expect("offset"),
        )
        .expect("offset");
        let start = payload.len();
        for row in [2, 5, 7] {
            let bytes = payload[offset + row * 64..offset + (row + 1) * 64].to_vec();
            payload.extend(bytes);
        }
        header["classification_head.weight"] =
            json!({"dtype":"F32","shape":[3,16],"data_offsets":[start,payload.len()]});
        let header = serde_json::to_vec(&header).expect("header");
        let mut bytes = u64::try_from(header.len())
            .expect("header size")
            .to_le_bytes()
            .to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(directory.join("config.json"), config).expect("config");
        fs::write(directory.join("model.safetensors"), bytes).expect("weights");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer identity");
        fs::write(directory.join("inputs.json"),serde_json::to_vec(&json!({"schema_version":1,"task":"classifier","cases":[{"id":"one","token_ids":[1,3,5]},{"id":"two","token_ids":[1,3,7]}]})).expect("inputs")).expect("inputs");
        let mut request = request();
        request.bundle = directory.clone();
        request.tokenizer = directory.join("tokenizer.json");
        request.inputs = directory.join("inputs.json");
        Self { directory, request }
    }
    fn evaluate(&self) -> HostResult<Value> {
        run::evaluate(
            CpuBackend::new(
                9,
                ResourceLimits {
                    max_allocation_bytes: 1 << 24,
                    max_total_bytes: 1 << 28,
                    max_pending_operations: 4096,
                },
            ),
            &self.request,
            Instant::now(),
        )
    }
}

#[test]
fn actual_cpu_host_full_cached_parity_hashes_and_phase_work_are_stable() {
    let mut fixture = Fixture::classifier();
    let full = fixture.evaluate().expect("full");
    fixture.request.mode = "cached".into();
    let cached = fixture.evaluate().expect("cached");
    for index in 0..2 {
        let left: Vec<f32> = serde_json::from_value(full["cases"][index]["outputs"][0].clone())
            .expect("full outputs");
        let right: Vec<f32> = serde_json::from_value(cached["cases"][index]["outputs"][0].clone())
            .expect("cached outputs");
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() < 0.000_01);
        }
        assert_eq!(
            full["cases"][index]["predictions"],
            cached["cases"][index]["predictions"]
        );
    }
    assert_eq!(full["artifacts"], cached["artifacts"]);
    assert_eq!(
        full["artifacts"]["weights_sha256"],
        format!(
            "{:x}",
            Sha256::digest(fs::read(fixture.directory.join("model.safetensors")).expect("weights"))
        )
    );
    assert_eq!(full["backend"]["implementation"], "cpu_reference");
    fixture.request.phase = "measure".into();
    fixture.request.warmups = 1;
    fixture.request.measured_cycles = 2;
    let measured = fixture.evaluate().expect("measured");
    assert_eq!(measured["cases"][0]["completed_predictions"], 2);
    assert_eq!(
        measured["cases"][0]["outputs"]
            .as_array()
            .expect("outputs")
            .len(),
        2
    );
    assert!(measured["warmup_seconds"].as_f64().expect("warmup") > 0.0);
}

#[test]
fn unsupported_native_metal_selection_never_falls_back_to_cpu() {
    let mut request = request();
    request.backend = "unavailable_backend".into();
    assert!(backend::evaluate(&request, Instant::now()).is_err());
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        request.backend = "native_metal".into();
        let error =
            backend::evaluate(&request, Instant::now()).expect_err("unsupported native Metal");
        assert!(error.to_string().contains("wasn't compiled"));
    }
    #[cfg(not(feature = "wgpu"))]
    {
        request.backend = "wgpu_metal".into();
        assert!(backend::evaluate(&request, Instant::now()).is_err());
    }
}

#[test]
fn actual_pointer_host_admits_encoder_heads_and_reports_full_continuous_outputs() {
    let mut fixture = Fixture::classifier();
    let mut config: Value = serde_json::from_slice(include_bytes!(
        "../../executor-core/tests/fixtures/numerical-lfm-001-config.json"
    ))
    .expect("config");
    config["architectures"] = json!(["Lfm2BidirectionalForMaskedLM"]);
    config["use_cache"] = json!(false);
    config["minifield_pointer"] =
        json!({"format":"minifield.magicbox-joint-pointer/1","projection_dim":4});
    fs::write(
        fixture.directory.join("config.json"),
        serde_json::to_vec(&config).expect("config"),
    )
    .expect("config");
    let source =
        include_bytes!("../../executor-core/tests/fixtures/numerical-lfm-001-weights.safetensors");
    let length = usize::try_from(u64::from_le_bytes(source[..8].try_into().expect("length")))
        .expect("length");
    let mut header: Value = serde_json::from_slice(&source[8..8 + length]).expect("header");
    let mut payload = source[8 + length..].to_vec();
    for name in ["start_query", "start_key", "end_query", "end_key"] {
        let start = payload.len();
        for row in 0..4 {
            for column in 0..16 {
                payload.extend((if row == column { 1.0_f32 } else { 0.0 }).to_le_bytes());
            }
        }
        header[format!("pointer.{name}.weight")] =
            json!({"dtype":"F32","shape":[4,16],"data_offsets":[start,payload.len()]});
    }
    let header = serde_json::to_vec(&header).expect("header");
    let mut bytes = u64::try_from(header.len())
        .expect("length")
        .to_le_bytes()
        .to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(fixture.directory.join("model.safetensors"), bytes).expect("weights");
    fs::write(&fixture.request.inputs,serde_json::to_vec(&json!({"schema_version":1,"task":"pointer","cases":[{"id":"pointer","token_ids":[1,2,3,0],"segments":[1,1,1,0],"questions":[
        {"query_index":0,"option_indices":[1,2],"kind":{"type":"choice"}},
        {"query_index":0,"option_indices":[],"kind":{"type":"extract","absent_index":0,"source_start":1,"selectable":[true,true,false],"presence_threshold":0.5}}
    ]}]})).expect("inputs")).expect("inputs");
    fixture.request.task = "pointer".into();
    fixture.request.classes = None;
    let result = fixture.evaluate().expect("pointer host");
    let output: Vec<f32> =
        serde_json::from_value(result["cases"][0]["outputs"][0].clone()).expect("output");
    assert_eq!(output.len(), 19);
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(output[3], 0.0);
    assert_eq!(output[7], 0.0);
    assert_eq!(result["cases"][0]["predictions"][0][0]["type"], "choice");
    assert!(matches!(
        result["cases"][0]["predictions"][0][1]["type"].as_str(),
        Some("span" | "absent")
    ));
    assert!(
        result["resources"]["peak_accounted_bytes"]
            .as_u64()
            .expect("peak")
            >= result["resources"]["accounted_bytes"]
                .as_u64()
                .expect("accounted")
    );
}

#[test]
fn dispatch_snapshot_rejects_reset_or_missing_counters() {
    let before = measurement::Snapshot {
        evidence: json!({}),
        counts: std::collections::BTreeMap::from([("gemm".into(), 3)]),
        accounted: 0,
        peak: 0,
    };
    let mut after = measurement::Snapshot {
        evidence: json!({}),
        counts: std::collections::BTreeMap::from([("gemm".into(), 8)]),
        accounted: 0,
        peak: 0,
    };
    assert_eq!(
        measurement::dispatch_delta(&before, &after).expect("delta")["gemm"],
        5
    );
    after.counts.insert("gemm".into(), 2);
    assert!(measurement::dispatch_delta(&before, &after).is_err());
    after.counts.clear();
    assert!(measurement::dispatch_delta(&before, &after).is_err());
}

#[test]
fn gpu_dispatch_total_tracks_new_kernels_and_rejects_invalid_evidence() {
    use std::collections::BTreeMap;
    for implementation in ["wgpu_metal", "native_metal"] {
        let before = measurement::Snapshot {
            evidence: json!({"implementation":implementation}),
            counts: BTreeMap::from([("old_kernel".into(), 3)]),
            accounted: 0,
            peak: 0,
        };
        let mut after = measurement::Snapshot {
            evidence: before.evidence.clone(),
            counts: BTreeMap::from([("old_kernel".into(), 3), ("new_kernel".into(), 5)]),
            accounted: 0,
            peak: 0,
        };
        let delta = measurement::dispatch_delta(&before, &after).expect("delta");
        assert_eq!(delta["gpu_dispatches"], 5);
        assert_eq!(delta["old_kernel"], 0);
        assert_eq!(delta["new_kernel"], 5);
        after.evidence["implementation"] = json!("cpu_reference");
        assert!(measurement::dispatch_delta(&before, &after).is_err());
        after.evidence = before.evidence.clone();
        after.counts.insert("gpu_dispatches".into(), 1);
        assert!(measurement::dispatch_delta(&before, &after).is_err());
        after.counts.remove("gpu_dispatches");
        after.counts.insert("new_kernel".into(), u64::MAX);
        after.counts.insert("another_kernel".into(), 1);
        assert!(measurement::dispatch_delta(&before, &after).is_err());
    }
}
