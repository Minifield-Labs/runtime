#![allow(clippy::expect_used)]

use minifield_engine_api::ExecutorError;
use minifield_executor_core::{
    LayerKind, Lfm2StorageDType, Lfm2WeightPlan, NumericalMode, parse_lfm2_config,
};
use serde_json::{Value, json};

const TINY_CONFIG: &[u8] = include_bytes!("fixtures/numerical-lfm-001/config.json");
const PINNED_CONFIG: &[u8] = include_bytes!("fixtures/pinned-lfm2.5-350m/config.json");

fn parsed(value: &Value) -> Result<minifield_executor_core::Lfm2Config, ExecutorError> {
    parse_lfm2_config(&serde_json::to_vec(value).expect("JSON encode"))
}

#[test]
fn independent_tiny_config_and_pinned_350m_header_parse_to_checked_math() {
    let tiny = parse_lfm2_config(TINY_CONFIG).expect("tiny config");
    assert_eq!(tiny.hidden_size, 16);
    assert_eq!(tiny.raw_intermediate_size, 32);
    assert_eq!(tiny.effective_intermediate_size, 32);
    assert_eq!(tiny.attention_heads, 4);
    assert_eq!(tiny.key_value_heads, 2);
    assert_eq!(tiny.head_dim, 4);
    assert_eq!(tiny.conv_width, 3);
    assert_eq!(tiny.vocab_size, 32);
    assert_eq!(tiny.layers, vec![LayerKind::Conv, LayerKind::FullAttention]);
    assert_eq!(tiny.rope_theta.to_bits(), 1_000_000.0_f32.to_bits());
    assert!(tiny.tie_embedding);
    assert_eq!(tiny.max_position_embeddings, None);
    assert_eq!(
        tiny.numerical_mode,
        NumericalMode::F32ReciprocalSqrtMultiply
    );

    let pinned = parse_lfm2_config(PINNED_CONFIG).expect("pinned config");
    assert_eq!(pinned.hidden_size, 1024);
    assert_eq!(pinned.raw_intermediate_size, 6656);
    assert_eq!(pinned.effective_intermediate_size, 4608);
    assert_eq!(pinned.attention_heads, 16);
    assert_eq!(pinned.key_value_heads, 8);
    assert_eq!(pinned.head_dim, 64);
    assert_eq!(pinned.conv_width, 3);
    assert_eq!(pinned.vocab_size, 65_536);
    assert_eq!(pinned.layers.len(), 16);
    assert_eq!(pinned.max_position_embeddings, Some(128_000));
    assert!(pinned.tie_embedding);
    assert_eq!(pinned.weight_storage_dtype, Lfm2StorageDType::BF16);
    let pinned_plan = Lfm2WeightPlan::from_config(pinned).expect("pinned plan");
    assert_eq!(pinned_plan.physical_tensor_count(), 148);
    assert_eq!(pinned_plan.physical_parameter_count(), Ok(354_483_968));
}

#[test]
fn matching_published_aliases_are_accepted_and_effective_ff_is_exact() {
    let value = json!({
        "model_type": "lfm2", "hidden_size": 1024, "block_dim": 1024, "conv_dim": 1024,
        "intermediate_size": 6656, "block_ff_dim": 6656,
        "block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256, "block_use_swiglu": true, "conv_bias": false,
        "conv_L_cache": 3, "num_attention_heads": 16, "num_heads": 16,
        "num_key_value_heads": 8, "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"], "vocab_size": 65536,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
        "rope_theta": 1_000_000.0, "tie_embedding": true, "tie_word_embeddings": true,
        "norm_eps": 0.00001, "block_norm_eps": 0.00001,
        "max_position_embeddings": 128_000
    });
    let config = parsed(&value).expect("matching aliases");
    assert_eq!(config.effective_intermediate_size, 4608);
    assert_eq!(config.head_dim, 64);
}

#[test]
fn conflicting_aliases_are_rejected_before_model_allocation() {
    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["block_ff_dim"] = json!(6657);
    assert!(matches!(
        parsed(&root),
        Err(ExecutorError::InvalidArgument(_))
    ));

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["num_heads"] = json!(17);
    assert!(matches!(
        parsed(&root),
        Err(ExecutorError::InvalidArgument(_))
    ));

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["rope_theta"] = json!(1234.0);
    assert!(matches!(
        parsed(&root),
        Err(ExecutorError::InvalidArgument(_))
    ));

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["tie_word_embeddings"] = json!(false);
    assert!(matches!(
        parsed(&root),
        Err(ExecutorError::InvalidArgument(_))
    ));

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["dtype"] = json!("float64");
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported(
            "configured LFM2 source storage dtype is unsupported"
        ))
    );
}

#[test]
fn unsupported_or_ambiguous_math_declarations_are_rejected() {
    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["rope_scaling"] = json!({"factor": 2.0});
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported("scaled RoPE is unsupported"))
    );

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["block_use_swiglu"] = json!(false);
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported(
            "only SwiGLU LFM2 blocks are supported"
        ))
    );

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["conv_bias"] = json!(true);
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported(
            "convolution bias is unsupported"
        ))
    );

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["tie_embedding"] = json!(false);
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported(
            "untied embedding and language-model head are unsupported"
        ))
    );

    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root["layer_types"] = json!(["conv", "unsupported"]);
    root["num_hidden_layers"] = json!(2);
    assert_eq!(
        parsed(&root),
        Err(ExecutorError::Unsupported("unsupported LFM2 layer type"))
    );
}

#[test]
fn auto_adjust_requires_its_published_parameters() {
    let mut root: Value = serde_json::from_slice(PINNED_CONFIG).expect("pinned JSON");
    root.as_object_mut()
        .expect("object")
        .remove("block_multiple_of");
    assert!(matches!(
        parsed(&root),
        Err(ExecutorError::InvalidArgument(_))
    ));
}
