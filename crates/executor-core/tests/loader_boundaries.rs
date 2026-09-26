use std::fmt::Debug;

use minifield_backend_cpu::{CpuBackend, CpuBuffer, CpuCompletion};
use minifield_engine_api::{
    AssetProvider, ByteRange, ExecutorError, MemoryAssetProvider, MemoryAssetRead, ResourceLimits,
};
use minifield_executor_core::{
    Lfm2WeightFormat, LoadRequest, LoaderLimits, LoaderPoll, LoaderStage, ParsedAsset,
    TypedWeights, WeightLayout, WeightLoadTask, WeightPlan, WeightRequirement,
    detect_lfm2_weight_format, parse_safetensors_header,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn required<T, E: Debug>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|error| panic!("synthetic test setup failed: {error:?}"))
}

fn limits(host_bytes: u64) -> LoaderLimits {
    LoaderLimits {
        max_asset_bytes: 4096,
        max_header_bytes: 1024,
        max_source_tensor_bytes: 4096,
        max_retained_host_bytes: host_bytes,
        max_tensor_name_bytes: 64,
        max_tensors: 8,
        max_rank: 4,
    }
}

fn asset(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = required(u64::try_from(header.len())).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn parse(header: &str, payload: &[u8]) -> minifield_engine_api::Result<ParsedAsset> {
    let asset_bytes = required(u64::try_from(8 + header.len() + payload.len()));
    parse_safetensors_header(
        header.as_bytes(),
        required(u64::try_from(header.len())),
        asset_bytes,
        limits(4096),
    )
}

struct TrackingProvider {
    inner: MemoryAssetProvider,
    calls: usize,
}

impl AssetProvider for TrackingProvider {
    type Read = MemoryAssetRead;

    fn read_range(&mut self, range: ByteRange) -> minifield_engine_api::Result<Self::Read> {
        self.calls += 1;
        self.inner.read_range(range)
    }
}

fn drive(
    task: &mut WeightLoadTask<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>,
    provider: &mut TrackingProvider,
    backend: &mut CpuBackend,
) -> Result<TypedWeights<CpuBuffer>, minifield_executor_core::LoaderError> {
    for _ in 0..64 {
        if let LoaderPoll::Ready(result) = task.poll_step(provider, backend) {
            return result;
        }
    }
    panic!("synthetic loader exceeded its bounded step count")
}

#[test]
fn tensor_host_preflight_uses_uploaded_dtype_width_at_exact_boundaries() {
    let config = b"{}";
    for (dtype, elements, uploaded_bytes) in [
        ("U8", 1024, 1024_u64),
        ("F32", 256, 1024),
        ("BF16", 512, 2048),
        ("F16", 512, 2048),
    ] {
        let payload = if dtype == "U8" {
            (0_u8..=u8::MAX).cycle().take(1024).collect::<Vec<_>>()
        } else {
            vec![0; 1024]
        };
        let header = format!(
            "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,1024]}}}}"
        );
        let parsed = required(parse(&header, &payload));
        let bytes = asset(&header, &payload);
        let exact_host_bytes = required(u64::try_from(config.len())) + 1024 + uploaded_bytes;
        for host_bytes in [exact_host_bytes - 1, exact_host_bytes] {
            let mut provider = TrackingProvider {
                inner: MemoryAssetProvider::new(bytes.clone(), 4096),
                calls: 0,
            };
            let mut backend = CpuBackend::new(
                0x10ad,
                ResourceLimits {
                    max_allocation_bytes: 4096,
                    max_total_bytes: 4096,
                    max_pending_operations: 4,
                },
            );
            let tensor = &parsed.tensors[0];
            let mut task = required(WeightLoadTask::begin(LoadRequest {
                config_name: "format".to_owned(),
                config_bytes: config.to_vec(),
                expected_config_sha256: Sha256::digest(config).into(),
                declared_asset_bytes: required(u64::try_from(bytes.len())),
                expected_asset_sha256: Sha256::digest(&bytes).into(),
                plan: WeightPlan {
                    config_name: "format".to_owned(),
                    requirements: vec![WeightRequirement {
                        role: "weight".to_owned(),
                        tensor_name: "weight".to_owned(),
                        storage_dtype: tensor.storage_dtype,
                        source_shape: tensor.source_shape,
                        layout: WeightLayout::Identity,
                        tied_to_role: None,
                    }],
                },
                limits: limits(host_bytes),
            }));
            let result = drive(&mut task, &mut provider, &mut backend);
            if host_bytes == exact_host_bytes {
                let weights = required(result);
                assert_eq!(provider.calls, 3, "{dtype}: prefix, header, and tensor");
                assert_eq!(weights.owned_uploaded_bytes(), uploaded_bytes);
                if dtype == "U8" {
                    assert_eq!(
                        required(weights.buffer_for_role("weight")).as_bytes(),
                        payload
                    );
                }
            } else {
                let Err(error) = result else {
                    panic!("one byte below the staging boundary must fail");
                };
                assert_eq!(error.stage, LoaderStage::TensorDecode, "{dtype}");
                assert!(matches!(error.cause, ExecutorError::ResourceLimit(_)));
                assert_eq!(provider.calls, 2, "{dtype}: reject before the tensor read");
                assert_eq!(backend.resource_report().total_owned_bytes(), Ok(0));
            }
            assert_eq!(task.resource_report().total_loader_bytes(), Ok(0));
        }
    }
}

#[test]
fn empty_tensor_at_shared_range_start_is_independent_of_header_order() {
    let empty = r#""empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}"#;
    let nonempty = r#""weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
    for header in [
        format!("{{{empty},{nonempty}}}"),
        format!("{{{nonempty},{empty}}}"),
    ] {
        let parsed = required(parse(&header, &[0; 4]));
        assert_eq!(parsed.tensors.len(), 2);
        assert_eq!(parsed.asset_bytes - parsed.payload_start, 4);
    }
}

#[test]
fn nonempty_overlapping_ranges_still_reject_in_both_header_orders() {
    let first = r#""first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
    let second = r#""second":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
    for header in [
        format!("{{{first},{second}}}"),
        format!("{{{second},{first}}}"),
    ] {
        assert!(matches!(
            parse(&header, &[0; 4]),
            Err(ExecutorError::InvalidLayout(_))
        ));
    }
}

fn format_asset(format: &Value) -> Vec<u8> {
    asset(
        &required(serde_json::to_string(
            &json!({"__metadata__": {"format": format}}),
        )),
        &[],
    )
}

#[test]
fn known_weight_formats_and_absent_dense_marker_remain_supported() {
    for header in ["{}", r#"{"__metadata__":{}}"#] {
        assert_eq!(
            detect_lfm2_weight_format(&asset(header, &[])),
            Ok(Lfm2WeightFormat::Dense)
        );
    }
    for (marker, format) in [
        ("pt", Lfm2WeightFormat::Dense),
        ("minifield.ternary.v1", Lfm2WeightFormat::TernaryV1),
        ("minifield.nf4.v1", Lfm2WeightFormat::Nf4V1),
        ("minifield.mixed.v1", Lfm2WeightFormat::MixedV1),
    ] {
        assert_eq!(
            detect_lfm2_weight_format(&format_asset(&json!(marker))),
            Ok(format)
        );
    }
}

#[test]
fn explicit_unknown_weight_encodings_and_versions_reject() {
    for marker in [
        "unknown",
        "minifield.ternary.v2",
        "minifield.nf4.v2",
        "minifield.mixed.v2",
    ] {
        assert!(matches!(
            detect_lfm2_weight_format(&format_asset(&json!(marker))),
            Err(ExecutorError::Unsupported(_))
        ));
    }
    for marker in [
        Value::Null,
        json!(false),
        json!(1),
        json!([]),
        json!({"encoding":"ternary","version":2}),
    ] {
        assert!(matches!(
            detect_lfm2_weight_format(&format_asset(&marker)),
            Err(ExecutorError::InvalidArgument(_))
        ));
    }
    for metadata in [Value::Null, json!("minifield.ternary.v2"), json!([])] {
        let header = required(serde_json::to_string(&json!({"__metadata__": metadata})));
        assert!(matches!(
            detect_lfm2_weight_format(&asset(&header, &[])),
            Err(ExecutorError::InvalidArgument(_))
        ));
    }
}
