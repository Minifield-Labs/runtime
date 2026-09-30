//! INT8 model admission uses an explicit byte contract before backend upload.
#![allow(clippy::unwrap_used)]
use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{ExecutorError, MemoryAssetProvider, ResourceLimits, Shape};
use minifield_executor_core::{
    Lfm2StorageDType, Lfm2WeightFormat, Lfm2WeightPlan, LoadRequest, LoaderLimits, LoaderPoll,
    StorageDType, WeightLayout, WeightLoadTask, WeightPlan, WeightRequirement, parse_lfm2_config,
};
use sha2::{Digest, Sha256};

fn limits() -> LoaderLimits {
    LoaderLimits {
        max_asset_bytes: 4096,
        max_header_bytes: 2048,
        max_source_tensor_bytes: 1024,
        max_retained_host_bytes: 4096,
        max_tensor_name_bytes: 128,
        max_tensors: 100,
        max_rank: 4,
    }
}
#[test]
fn signed_code_admission_rejects_reserved_value_before_upload() {
    let header = serde_json::to_vec(
        &serde_json::json!({"codes":{"dtype":"U8","shape":[1,128],"data_offsets":[0,128]}}),
    )
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    let payload = bytes.len();
    bytes.resize(payload + 128, 0);
    bytes[payload + 53] = 128;
    let config = b"{}".to_vec();
    let request = LoadRequest {
        config_name: "fixture".to_owned(),
        expected_config_sha256: Sha256::digest(&config).into(),
        config_bytes: config,
        declared_asset_bytes: bytes.len() as u64,
        expected_asset_sha256: Sha256::digest(&bytes).into(),
        plan: WeightPlan {
            config_name: "fixture".into(),
            requirements: vec![WeightRequirement {
                role: "codes".into(),
                tensor_name: "codes".into(),
                storage_dtype: StorageDType::U8,
                source_shape: Shape::new(&[1, 128]).unwrap(),
                layout: WeightLayout::SignedInt8Codes,
                tied_to_role: None,
            }],
        },
        limits: limits(),
    };
    let mut provider = MemoryAssetProvider::new(bytes, 4096);
    let mut backend = CpuBackend::new(
        124,
        ResourceLimits {
            max_allocation_bytes: 4096,
            max_total_bytes: 16384,
            max_pending_operations: 16,
        },
    );
    let mut task = WeightLoadTask::begin(request).unwrap();
    loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Err(error)) => {
                assert_eq!(
                    error.cause,
                    ExecutorError::InvalidArgument("INT8 -128 code is reserved")
                );
                break;
            }
            LoaderPoll::Ready(Ok(_)) => panic!("reserved INT8 code was admitted"),
        }
    }
    assert_eq!(backend.resource_report().resident_weight_bytes, 0);
}
#[test]
fn f16_and_int8_role_plans_keep_protected_embedding_and_head_dense() {
    let config=serde_json::to_vec(&serde_json::json!({"model_type":"lfm2","hidden_size":128,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"conv_L_cache":3,"vocab_size":256,"num_hidden_layers":1,"rope_theta":10000.0,"layer_types":["conv"],"dtype":"float16","tie_word_embeddings":true,"norm_eps":0.00001,"block_norm_eps":0.00001})).unwrap();
    let config = parse_lfm2_config(&config).unwrap();
    assert_eq!(config.weight_storage_dtype, Lfm2StorageDType::F16);
    let overrides = std::collections::HashMap::from([(
        "model.layers.0.feed_forward.w1.weight".to_owned(),
        Lfm2WeightFormat::Int8V1,
    )]);
    let plan = Lfm2WeightPlan::from_config_classifier_with_quantization(
        config,
        8,
        Lfm2WeightFormat::MixedV1,
        &overrides,
    )
    .unwrap();
    let requirements = &plan.generic_plan().requirements;
    assert!(
        requirements
            .iter()
            .any(|r| r.role == "token_embedding" && r.storage_dtype == StorageDType::F16)
    );
    assert!(
        requirements
            .iter()
            .any(|r| r.role == "classification_head" && r.storage_dtype == StorageDType::F16)
    );
    let codes = requirements
        .iter()
        .find(|r| r.role == "layer.0.ffn.w1.codes")
        .unwrap();
    assert_eq!(codes.source_shape, Shape::new(&[128, 128]).unwrap());
    assert_eq!(codes.layout, WeightLayout::SignedInt8Codes);
}
