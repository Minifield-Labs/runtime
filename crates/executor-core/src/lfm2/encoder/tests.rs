#![allow(clippy::expect_used, clippy::float_cmp)]

use super::*;
use crate::lfm2::Lfm2WeightFormat;
use crate::{LoaderLimits, LoaderPoll};
use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, InferenceOps, MemoryAssetProvider,
};
use minifield_engine_api::{
    EncoderOps, EncoderSegments, GatedShortConvSpec, GqaSpec, ResourceLimits, Shape,
};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

fn backend() -> CpuBackend {
    backend_with_pending_limit(10000)
}

fn backend_with_pending_limit(max_pending_operations: u32) -> CpuBackend {
    CpuBackend::new(
        7,
        ResourceLimits {
            max_allocation_bytes: 16 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024,
            max_pending_operations,
        },
    )
}

#[test]
fn segments_restart_positions_and_reject_reentered_labels() {
    let segments = EncoderSegments::new(vec![0, 2, 2, 0, 7, 7, 7, 0]).expect("segments");
    assert_eq!(segments.positions(), &[0, 0, 1, 0, 0, 1, 2, 0]);
    assert!(EncoderSegments::new(vec![1, 2, 1]).is_err());
    assert!(segments.validate_tokens(3).is_err());
}

#[test]
fn bidirectional_attention_sees_future_and_isolates_segments_and_padding() {
    let mut backend = backend();
    let shape = Shape::new(&[5, 2]).expect("shape");
    let query = backend.upload_f32(shape, &[0.0; 10]).expect("q");
    let key = backend.upload_f32(shape, &[0.0; 10]).expect("k");
    let value = backend
        .upload_f32(
            shape,
            &[2.0, 4.0, 6.0, 8.0, 100.0, 100.0, 20.0, 22.0, 24.0, 26.0],
        )
        .expect("v");
    let mut output = backend.allocate_f32(shape).expect("out");
    backend
        .bidirectional_gqa(
            &mut output,
            &query,
            &key,
            &value,
            &EncoderSegments::new(vec![1, 1, 0, 2, 2]).expect("segments"),
            GqaSpec::new(1, 1, 2).expect("gqa"),
        )
        .expect("attention");
    assert_eq!(
        output.as_slice(),
        &[4.0, 6.0, 4.0, 6.0, 0.0, 0.0, 22.0, 24.0, 22.0, 24.0]
    );
}

#[test]
fn centered_convolution_keeps_future_tap_and_excludes_other_segments() {
    let mut backend = backend();
    let projection = backend
        .upload_f32(
            Shape::new(&[5, 3]).expect("shape"),
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 100.0, 1.0, 1.0, 4.0, 1.0, 1.0, 5.0,
            ],
        )
        .expect("projection");
    let taps = backend
        .upload_f32(Shape::new(&[1, 3]).expect("shape"), &[1.0, 10.0, 100.0])
        .expect("taps");
    let mut output = backend
        .allocate_f32(Shape::new(&[5, 1]).expect("shape"))
        .expect("out");
    backend
        .centered_gated_convolution(
            &mut output,
            &projection,
            &taps,
            &EncoderSegments::new(vec![1, 1, 0, 2, 2]).expect("segments"),
            GatedShortConvSpec::new(1, 3).expect("conv"),
        )
        .expect("convolution");
    assert_eq!(output.as_slice(), &[210.0, 21.0, 0.0, 540.0, 54.0]);
}

#[test]
fn encoder_attention_checks_combined_temporary_storage_before_writing_output() {
    let mut backend = CpuBackend::new(
        71,
        ResourceLimits {
            max_allocation_bytes: 64,
            max_total_bytes: 84,
            max_pending_operations: 64,
        },
    );
    let shape = Shape::new(&[2, 2]).expect("shape");
    let query = backend.upload_f32(shape, &[0.0; 4]).expect("query");
    let key = backend.upload_f32(shape, &[0.0; 4]).expect("key");
    let value = backend.upload_f32(shape, &[1.0; 4]).expect("value");
    let mut output = backend.allocate_f32(shape).expect("output");
    // 64 resident bytes plus 16 output-stage bytes plus 8 score bytes exceed 84.
    // Each temporary region alone would fit, so independent admission is insufficient.
    let result = backend.bidirectional_gqa(
        &mut output,
        &query,
        &key,
        &value,
        &EncoderSegments::single(2).expect("segments"),
        GqaSpec::new(1, 1, 2).expect("GQA"),
    );
    assert!(matches!(
        result,
        Err(minifield_engine_api::ExecutorError::ResourceLimit(_))
    ));
    assert_eq!(output.as_slice(), &[0.0; 4]);
}

#[test]
fn even_width_convolution_matches_left_padding_and_right_crop() {
    let mut backend = backend();
    let projection = backend
        .upload_f32(
            Shape::new(&[3, 3]).expect("shape"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
        )
        .expect("projection");
    let taps = backend
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[1.0, 10.0])
        .expect("taps");
    let mut output = backend
        .allocate_f32(Shape::new(&[3, 1]).expect("shape"))
        .expect("out");
    backend
        .centered_gated_convolution(
            &mut output,
            &projection,
            &taps,
            &EncoderSegments::single(3).expect("segments"),
            GatedShortConvSpec::new(1, 2).expect("conv"),
        )
        .expect("convolution");
    assert_eq!(output.as_slice(), &[10.0, 21.0, 32.0]);
}

#[test]
fn decoding_averages_endpoint_probabilities_and_keeps_stable_ties() {
    let input = EncoderInput {
        token_ids: vec![1; 6],
        segments: EncoderSegments::single(6).expect("segments"),
        questions: vec![
            PointerQuestion {
                query_index: 0,
                option_indices: vec![1, 2],
                kind: PointerQuestionKind::Choice,
            },
            PointerQuestion {
                query_index: 0,
                option_indices: vec![],
                kind: PointerQuestionKind::Extract {
                    absent_index: 1,
                    source_start: 2,
                    selectable: vec![true, true, false, true],
                    presence_threshold: 0.5,
                },
            },
        ],
    };
    let output = decode_pointer(
        &input,
        vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0, 2.0, 2.0, 100.0, 2.0,
        ],
        vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 2.0, 2.0, 100.0, 2.0,
        ],
    )
    .expect("decode");
    assert!(matches!(
        &output.answers[0],
        PointerAnswer::Choice { index: 0, .. }
    ));
    assert!(matches!(
        &output.answers[1],
        PointerAnswer::Span {
            span: Some([0, 1]),
            ..
        }
    ));
}

#[test]
fn pointer_host_probabilities_preserve_f64_threshold_and_choice_decisions() {
    let input = EncoderInput {
        token_ids: vec![1; 3],
        segments: EncoderSegments::single(3).expect("segments"),
        questions: vec![
            PointerQuestion {
                query_index: 0,
                option_indices: vec![1, 2],
                kind: PointerQuestionKind::Binary { positive_option: 0 },
            },
            PointerQuestion {
                query_index: 0,
                option_indices: vec![],
                kind: PointerQuestionKind::Extract {
                    absent_index: 1,
                    source_start: 2,
                    selectable: vec![true],
                    presence_threshold: 0.5,
                },
            },
            PointerQuestion {
                query_index: 0,
                option_indices: vec![1, 2],
                kind: PointerQuestionKind::Choice,
            },
            PointerQuestion {
                query_index: 0,
                option_indices: vec![1, 2],
                kind: PointerQuestionKind::Ordinal,
            },
        ],
    };
    let delta = 1e-8_f32;
    let logits = vec![
        0.0, -delta, 0.0, 0.0, delta, 0.0, 0.0, -delta, 0.0, 0.0, delta, 0.0,
    ];
    let output = decode_pointer(&input, logits.clone(), logits).expect("decode");
    let expected = 1.0 / (1.0 + f64::from(delta).exp());
    match &output.answers[0] {
        PointerAnswer::Binary { probability, .. } => {
            assert!(*probability < 0.5);
            assert!((*probability - expected).abs() < 1e-15);
        }
        answer => panic!("unexpected binary answer {answer:?}"),
    }
    assert!(matches!(output.answers[1],PointerAnswer::Span{span:None,presence} if presence<0.5));
    assert!(matches!(
        output.answers[2],
        PointerAnswer::Choice { index: 1, .. }
    ));
    assert!(matches!(output.answers[3],PointerAnswer::Ordinal{value,..} if value<0.5));
}

#[test]
fn pointer_presence_retains_tiny_candidates_in_compensated_host_sum() {
    let input = EncoderInput {
        token_ids: vec![1; 10],
        segments: EncoderSegments::single(10).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: vec![],
            kind: PointerQuestionKind::Extract {
                absent_index: 0,
                source_start: 1,
                selectable: vec![true; 9],
                presence_threshold: 0.5,
            },
        }],
    };
    // Primary training probabilities() under CPython 3.12/3.13 sums these
    // weights to 2.0. A naive F64 sum loses the eight tiny candidates and
    // returns 1.9999999999999996, moving presence below the threshold.
    let mut logits = vec![-37.429_947_f32; 10];
    logits[0] = 0.0;
    logits[1] = -3e-16_f32;
    let output = decode_pointer(&input, logits.clone(), logits).expect("decode");
    assert_eq!(
        output.answers[0],
        PointerAnswer::Span {
            span: Some([0, 1]),
            presence: 0.5,
        }
    );
}

#[test]
fn pointer_ordinal_expectation_matches_compensated_primary_host_sum() {
    let input = EncoderInput {
        token_ids: vec![1; 5],
        segments: EncoderSegments::single(5).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: vec![0, 1, 2, 3, 4],
            kind: PointerQuestionKind::Ordinal,
        }],
    };
    let logits = vec![0.0, 0.0, -37.429_947_f32, -37.429_947_f32, -37.429_947_f32];
    let output = decode_pointer(&input, logits.clone(), logits).expect("decode");
    let PointerAnswer::Ordinal {
        value,
        probabilities,
    } = &output.answers[0]
    else {
        panic!("expected ordinal answer");
    };
    // Pinned training's sum(level * probability ...) in CPython 3.12/3.13.
    assert_eq!(*value, f64::from_bits(0x3fe0_0000_0000_0002));
    #[allow(clippy::cast_precision_loss)]
    let naive: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(level, probability)| level as f64 * probability)
        .sum();
    assert_ne!(*value, naive);
}

#[test]
fn pointer_span_adds_f32_logits_in_f64_to_preserve_source_scan_order() {
    let input = EncoderInput {
        token_ids: vec![1; 3],
        segments: EncoderSegments::single(3).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: vec![],
            kind: PointerQuestionKind::Extract {
                absent_index: 0,
                source_start: 1,
                selectable: vec![true, true],
                presence_threshold: 0.5,
            },
        }],
    };
    // F32 addition rounds both candidates to 1e8. Python's host F64 sum
    // distinguishes +1 from +2 and keeps the same earliest start.
    let output =
        decode_pointer(&input, vec![-1e8, 1e8, 1e8], vec![-1e8, 1.0, 2.0]).expect("decode");
    assert!(matches!(
        output.answers[0],
        PointerAnswer::Span {
            span: Some([0, 2]),
            ..
        }
    ));
}

#[test]
fn input_rejects_cross_segment_candidates_before_recording() {
    let input = EncoderInput {
        token_ids: vec![1, 1],
        segments: EncoderSegments::new(vec![1, 2]).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: vec![1],
            kind: PointerQuestionKind::Choice,
        }],
    };
    assert!(
        input
            .validate(
                10,
                EncoderLimits {
                    max_tokens: 2,
                    max_questions: 1
                }
            )
            .is_err()
    );
    assert!(decode_pointer(&input, vec![0.0, 0.0], vec![0.0, 0.0]).is_err());
}

fn tiny_config() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model_type":"lfm2","architectures":["Lfm2BidirectionalForMaskedLM"],"use_cache":false,
        "hidden_size":4,"intermediate_size":4,"num_attention_heads":2,"num_key_value_heads":1,
        "conv_L_cache":3,"vocab_size":8,"num_hidden_layers":2,"layer_types":["conv","full_attention"],
        "rope_theta":10000.0,"tie_word_embeddings":true,"norm_eps":0.00001,"max_position_embeddings":32,
        "minifield_pointer":{"format":"minifield.magicbox-joint-pointer/1","projection_dim":2}
    })).expect("config")
}

fn tiny_weights() -> (
    Rc<RefCell<CpuBackend>>,
    Rc<EncoderTypedWeights<minifield_backend_cpu::CpuBuffer>>,
) {
    tiny_weights_with_config(tiny_config())
}

fn tiny_weights_with_config(
    config: Vec<u8>,
) -> (
    Rc<RefCell<CpuBackend>>,
    Rc<EncoderTypedWeights<minifield_backend_cpu::CpuBuffer>>,
) {
    tiny_weights_on_backend(config, backend())
}

fn tiny_weights_on_backend(
    config: Vec<u8>,
    mut backend: CpuBackend,
) -> (
    Rc<RefCell<CpuBackend>>,
    Rc<EncoderTypedWeights<minifield_backend_cpu::CpuBuffer>>,
) {
    let plan = EncoderWeightPlan::from_config_with_quantization(
        parse_encoder_config(&config).expect("config"),
        Lfm2WeightFormat::Dense,
        &HashMap::new(),
    )
    .expect("plan");
    assert!(
        !plan
            .generic_plan()
            .requirements
            .iter()
            .any(|item| item.role.starts_with("tied_lm_head"))
    );
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for requirement in &plan.generic_plan().requirements {
        let shape = requirement.source_shape;
        let count = usize::try_from(shape.element_count().expect("elements")).expect("count");
        let mut values = vec![0.0_f32; count];
        if requirement.role.contains("norm") {
            values.fill(1.0);
        }
        if requirement.role == "token_embedding" {
            for (index, value) in values.iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                {
                    *value = 0.2 + (index % 4) as f32 * 0.15 + (index / 4) as f32 * 0.03;
                }
            }
        }
        if requirement.role.contains(".ffn.w") || requirement.role.starts_with("pointer.") {
            let columns = usize::try_from(shape.dim(1).expect("columns")).expect("columns");
            for row in 0..usize::try_from(shape.dim(0).expect("rows")).expect("rows") {
                values[row * columns + row] = 1.0;
            }
        }
        let start = payload.len();
        for value in values {
            payload.extend(value.to_le_bytes());
        }
        let dimensions: Vec<u64> = (0..shape.rank())
            .map(|axis| shape.dim(usize::from(axis)).expect("dimension"))
            .collect();
        header.insert(requirement.tensor_name.clone(),serde_json::json!({"dtype":"F32","shape":dimensions,"data_offsets":[start,payload.len()]}));
    }
    let header = serde_json::to_vec(&header).expect("header");
    let mut bytes = u64::try_from(header.len())
        .expect("header length")
        .to_le_bytes()
        .to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    let size = u64::try_from(bytes.len()).expect("size");
    let limits = LoaderLimits {
        max_asset_bytes: size,
        max_header_bytes: size,
        max_source_tensor_bytes: size,
        max_retained_host_bytes: size * 6,
        max_tensor_name_bytes: 1024,
        max_tensors: 100,
        max_rank: 4,
    };
    let config_sha256 = Sha256::digest(&config).into();
    let request = EncoderLoadRequest::new_with_quantization(
        config,
        config_sha256,
        size,
        Sha256::digest(&bytes).into(),
        limits,
        Lfm2WeightFormat::Dense,
        &HashMap::new(),
    )
    .expect("request");
    let mut task = EncoderWeightLoadTask::begin(request).expect("load task");
    let mut provider = MemoryAssetProvider::new(bytes, size);
    for _ in 0..1000 {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => {
                return (
                    Rc::new(RefCell::new(backend)),
                    Rc::new(result.expect("weights")),
                );
            }
        }
    }
    panic!("encoder loader did not complete");
}

fn ready<T: InferenceCompletion>(mut task: T) -> T::Output {
    for _ in 0..1000 {
        match task.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(result) => return result.expect("completion"),
        }
    }
    panic!("encoder completion did not finish");
}

fn tiny_input() -> EncoderInput {
    EncoderInput {
        token_ids: vec![1, 2, 0],
        segments: EncoderSegments::new(vec![1, 1, 0]).expect("segments"),
        questions: vec![PointerQuestion {
            query_index: 0,
            option_indices: vec![0, 1],
            kind: PointerQuestionKind::Choice,
        }],
    }
}

fn reference_hidden_with_eps(token: u32, block_epsilon: f32, final_epsilon: f32) -> [f32; 4] {
    reference_hidden_with_layers(token, block_epsilon, final_epsilon, 2)
}

fn reference_hidden_with_layers(
    token: u32,
    block_epsilon: f32,
    final_epsilon: f32,
    layers: usize,
) -> [f32; 4] {
    let mut hidden = [0.0; 4];
    for (index, value) in hidden.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *value = 0.2 + index as f32 * 0.15 + token as f32 * 0.03;
        }
    }
    let normalize = |input: [f32; 4], epsilon: f32| {
        let inverse = (input.iter().map(|value| value * value).sum::<f32>() / 4.0 + epsilon)
            .sqrt()
            .recip();
        input.map(|value| value * inverse)
    };
    for _ in 0..layers {
        let normalized = normalize(hidden, block_epsilon);
        for (value, normalized) in hidden.iter_mut().zip(normalized) {
            *value += (normalized / (1.0 + (-normalized).exp())) * normalized;
        }
    }
    normalize(hidden, final_epsilon)
}

#[test]
fn encoder_preserves_distinct_block_and_final_normalization_epsilons() {
    let block_epsilon = 0.25_f32;
    let final_epsilon = 0.00001_f32;
    let mut config: serde_json::Value = serde_json::from_slice(&tiny_config()).expect("config");
    config["block_norm_eps"] = serde_json::json!(block_epsilon);
    let (backend, weights) =
        tiny_weights_with_config(serde_json::to_vec(&config).expect("config bytes"));
    let encoder = Lfm2PointerEncoder::new(
        backend,
        weights,
        EncoderLimits {
            max_tokens: 8,
            max_questions: 4,
        },
    )
    .expect("encoder");
    let output = ready(encoder.begin_predict(tiny_input()).expect("task"));
    let query = reference_hidden_with_eps(1, block_epsilon, final_epsilon);
    for (column, token) in [1, 2].into_iter().enumerate() {
        let key = reference_hidden_with_eps(token, block_epsilon, final_epsilon);
        let expected = (query[0] * key[0] + query[1] * key[1]) / 2.0_f32.sqrt();
        assert!((output.start[column] - expected).abs() < 0.000_001);
        assert!((output.end[column] - expected).abs() < 0.000_001);
    }
    let wrong = reference_hidden_with_eps(1, final_epsilon, final_epsilon);
    let wrong_score = (wrong[0] * wrong[0] + wrong[1] * wrong[1]) / 2.0_f32.sqrt();
    assert!((output.start[0] - wrong_score).abs() > 0.001);
    assert_eq!(output.start[2], 0.0);
    assert_eq!(output.end[2], 0.0);
}

#[test]
fn encoder_loader_and_pointer_heads_keep_all_final_ffn_tokens() {
    let mut workspace_bytes = None;
    for layers in [2, 6] {
        let mut config: serde_json::Value = serde_json::from_slice(&tiny_config()).expect("config");
        config["num_hidden_layers"] = serde_json::json!(layers);
        config["layer_types"] = serde_json::json!(
            ["conv", "full_attention"]
                .into_iter()
                .cycle()
                .take(layers)
                .collect::<Vec<_>>()
        );
        let (backend, weights) =
            tiny_weights_with_config(serde_json::to_vec(&config).expect("config bytes"));
        let encoder = Lfm2PointerEncoder::new(
            Rc::clone(&backend),
            weights,
            EncoderLimits {
                max_tokens: 8,
                max_questions: 4,
            },
        )
        .expect("encoder");
        let before = backend
            .borrow()
            .resource_report()
            .total_owned_bytes()
            .expect("resources");
        let mut task = encoder.begin_predict(tiny_input()).expect("task");
        let mut retained = 0;
        let mut output = None;
        for _ in 0..1000 {
            let poll = task.poll_step();
            retained = retained.max(
                backend
                    .borrow()
                    .resource_report()
                    .total_owned_bytes()
                    .expect("resources")
                    - before,
            );
            if let CompletionPoll::Ready(result) = poll {
                output = Some(result.expect("completion"));
                break;
            }
        }
        let output = output.expect("encoder completion did not finish");
        if let Some(expected) = workspace_bytes {
            assert_eq!(retained, expected, "scratch storage grew with layer count");
        } else {
            workspace_bytes = Some(retained);
        }
        let query = reference_hidden_with_layers(1, 0.00001, 0.00001, layers);
        let keys =
            [1, 2].map(|token| reference_hidden_with_layers(token, 0.00001, 0.00001, layers));
        for (column, key) in keys.iter().enumerate() {
            let expected = (query[0] * key[0] + query[1] * key[1]) / 2.0_f32.sqrt();
            assert!(
                (output.start[column] - expected).abs() < 0.000_001,
                "{column}: {} != {expected}",
                output.start[column]
            );
            assert!((output.end[column] - expected).abs() < 0.000_001);
        }
        assert_eq!(output.start[2], 0.0);
        assert_eq!(output.end[2], 0.0);
        assert_eq!(
            backend
                .borrow()
                .resource_report()
                .total_owned_bytes()
                .expect("resources"),
            before
        );
        assert_eq!(
            ready(encoder.begin_predict(tiny_input()).expect("repeat task")),
            output
        );
    }
}

#[test]
fn encoder_task_drop_cancel_limits_and_stale_generation_preserve_ownership() {
    let (backend, weights) = tiny_weights();
    let encoder = Lfm2PointerEncoder::new(
        Rc::clone(&backend),
        weights,
        EncoderLimits {
            max_tokens: 8,
            max_questions: 4,
        },
    )
    .expect("encoder");
    let before = backend
        .borrow()
        .resource_report()
        .total_owned_bytes()
        .expect("resources");
    let mut task = encoder.begin_predict(tiny_input()).expect("task");
    task.cancel().expect("cancel unsubmitted");
    assert!(matches!(task.poll_step(), CompletionPoll::Ready(Err(_))));
    let mut task = encoder.begin_predict(tiny_input()).expect("task");
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    drop(task);
    backend.borrow().poll_retired_fences().expect("retired");
    assert_eq!(
        backend
            .borrow()
            .resource_report()
            .total_owned_bytes()
            .expect("resources"),
        before
    );
    let mut invalid = tiny_input();
    invalid.questions[0].query_index = 2;
    assert!(encoder.begin_predict(invalid).is_err());
    let mut invalid = tiny_input();
    invalid.token_ids[0] = 8;
    assert!(encoder.begin_predict(invalid).is_err());
    backend
        .borrow_mut()
        .advance_generation()
        .expect("generation");
    assert!(encoder.begin_predict(tiny_input()).is_err());
}

#[test]
fn encoder_layer_slices_keep_one_fence_and_preserve_results_and_retirement() {
    let mut config: serde_json::Value = serde_json::from_slice(&tiny_config()).expect("config");
    config["num_hidden_layers"] = serde_json::json!(6);
    config["layer_types"] = serde_json::json!([
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention"
    ]);
    let (backend, weights) = tiny_weights_on_backend(
        serde_json::to_vec(&config).expect("config bytes"),
        backend_with_pending_limit(1),
    );
    let encoder = Lfm2PointerEncoder::new(
        Rc::clone(&backend),
        weights,
        EncoderLimits {
            max_tokens: 8,
            max_questions: 4,
        },
    )
    .expect("encoder");
    let before = backend.borrow().resource_report();
    let mut task = encoder.begin_predict(tiny_input()).expect("task");
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    let first = backend.borrow().resource_report();
    assert_eq!(first.pending_operations, 1);
    assert!(first.scratch_bytes > before.scratch_bytes);
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    let final_slice = backend.borrow().resource_report();
    assert_eq!(final_slice.pending_operations, 1);
    assert!(final_slice.scratch_bytes > first.scratch_bytes);
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    let CompletionPoll::Ready(Ok(output)) = task.poll_step() else {
        panic!("pointer result was not ready after two layer slices and readback");
    };
    let query = reference_hidden_with_layers(1, 0.00001, 0.00001, 6);
    for (column, token) in [1, 2].into_iter().enumerate() {
        let key = reference_hidden_with_layers(token, 0.00001, 0.00001, 6);
        let expected = (query[0] * key[0] + query[1] * key[1]) / 2.0_f32.sqrt();
        assert!((output.start[column] - expected).abs() < 0.000_001);
        assert!((output.end[column] - expected).abs() < 0.000_001);
    }
    assert_eq!(output.start[2], 0.0);
    assert_eq!(output.end[2], 0.0);
    assert!(matches!(
        task.poll_step(),
        CompletionPoll::Ready(Err(minifield_engine_api::ExecutorError::CompletionConsumed))
    ));
    assert_eq!(backend.borrow().resource_report(), before);
    assert_eq!(
        ready(encoder.begin_predict(tiny_input()).expect("repeat task")),
        output
    );
    let mut cancelled = encoder.begin_predict(tiny_input()).expect("cancel task");
    assert!(matches!(cancelled.poll_step(), CompletionPoll::Pending));
    cancelled.cancel().expect("cancel intermediate fence");
    assert_eq!(backend.borrow().resource_report(), before);
    let mut abandoned = encoder.begin_predict(tiny_input()).expect("drop task");
    assert!(matches!(abandoned.poll_step(), CompletionPoll::Pending));
    drop(abandoned);
    backend.borrow().poll_retired_fences().expect("retirement");
    assert_eq!(backend.borrow().resource_report(), before);
    let mut stale = encoder.begin_predict(tiny_input()).expect("stale task");
    assert!(matches!(stale.poll_step(), CompletionPoll::Pending));
    backend
        .borrow_mut()
        .advance_generation()
        .expect("generation");
    assert!(matches!(
        stale.poll_step(),
        CompletionPoll::Ready(Err(minifield_engine_api::ExecutorError::StaleBuffer))
    ));
    assert_eq!(backend.borrow().resource_report(), before);
}

#[test]
fn encoder_admission_rejects_causal_architecture_and_bad_head_metadata() {
    let config: serde_json::Value = serde_json::from_slice(&tiny_config()).expect("json");
    for (key, value) in [
        ("architectures", serde_json::json!(["Lfm2ForCausalLM"])),
        ("use_cache", serde_json::json!(true)),
        (
            "minifield_pointer",
            serde_json::json!({"format":"other","projection_dim":2}),
        ),
    ] {
        let mut modified = config.clone();
        modified[key] = value;
        assert!(parse_encoder_config(&serde_json::to_vec(&modified).expect("json")).is_err());
    }
}
