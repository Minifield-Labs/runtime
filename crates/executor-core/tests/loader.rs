#![allow(clippy::expect_used, clippy::too_many_lines)]

use std::{cell::RefCell, rc::Rc};

use minifield_backend_cpu::{CpuBackend, CpuBuffer, CpuCompletion};
use minifield_engine_api::{
    AllocationClass, AssetProvider, BackendCapabilities, BackendIdentity, BackendLease,
    CompletionPoll, ExecutorError, FenceRetirement, GatedShortConvSpec, GqaSpec,
    InferenceCompletion, InferenceOps, MemoryAssetProvider, MemoryAssetRead, PackedHeadSpec,
    RectCopy2d, ResourceLimits, ResourceReport, Result, RetirementRejection, RotarySpec, Shape,
    TokenId, TokenIds,
};
use minifield_executor_core::{
    Lfm2LoadRequest, Lfm2TypedWeights, Lfm2WeightLoadTask, Lfm2WeightPlan, Lfm2WeightRole,
    LoadRequest, LoaderLimits, LoaderPoll, LoaderStage, StorageDType, WeightLayout, WeightLoadTask,
    WeightPlan, WeightRequirement, parse_lfm2_config, parse_safetensors_header,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CASES: &str = include_str!("fixtures/asset-format-001/cases.json");
const MANIFEST: &str = include_str!("fixtures/asset-format-001/manifest.json");
const TINY_WEIGHTS: &[u8] = include_bytes!("fixtures/numerical-lfm-001-weights.safetensors");
const TINY_CONFIG: &[u8] = include_bytes!("fixtures/numerical-lfm-001-config.json");
const PINNED_CONFIG: &[u8] = include_bytes!("fixtures/pinned-lfm2.5-350m/config.json");

fn limits() -> LoaderLimits {
    LoaderLimits {
        max_asset_bytes: 128 * 1024,
        max_header_bytes: 16 * 1024,
        max_source_tensor_bytes: 64 * 1024,
        max_retained_host_bytes: 128 * 1024,
        max_tensor_name_bytes: 256,
        max_tensors: 128,
        max_rank: 4,
    }
}

fn backend(total: u64) -> CpuBackend {
    CpuBackend::new(
        0x10AD,
        ResourceLimits {
            max_allocation_bytes: total,
            max_total_bytes: total,
            max_pending_operations: 4,
        },
    )
}

fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn prefix_and_header(asset: &[u8]) -> core::result::Result<(u64, &[u8]), ExecutorError> {
    let prefix: [u8; 8] = asset
        .get(..8)
        .ok_or(ExecutorError::OutOfBounds("fixture lacks header prefix"))?
        .try_into()
        .map_err(|_| ExecutorError::InvalidArgument("fixture prefix"))?;
    let length = u64::from_le_bytes(prefix);
    let end = 8_u64
        .checked_add(length)
        .ok_or(ExecutorError::Overflow("fixture header end"))?;
    let end = usize::try_from(end).map_err(|_| ExecutorError::Overflow("fixture header usize"))?;
    Ok((
        length,
        asset
            .get(8..end)
            .ok_or(ExecutorError::OutOfBounds("fixture header"))?,
    ))
}

fn plan_from_parsed(name: &str, parsed: &minifield_executor_core::ParsedAsset) -> WeightPlan {
    WeightPlan {
        config_name: name.to_owned(),
        requirements: parsed
            .tensors
            .iter()
            .map(|tensor| WeightRequirement {
                role: tensor.name.clone(),
                tensor_name: tensor.name.clone(),
                storage_dtype: tensor.storage_dtype,
                source_shape: tensor.source_shape,
                layout: WeightLayout::Identity,
                tied_to_role: None,
            })
            .collect(),
    }
}

fn request(
    config_name: &str,
    config: &[u8],
    asset: &[u8],
    plan: WeightPlan,
    loader_limits: LoaderLimits,
) -> LoadRequest {
    LoadRequest {
        config_name: config_name.to_owned(),
        config_bytes: config.to_vec(),
        expected_config_sha256: hash(config),
        declared_asset_bytes: asset.len() as u64,
        expected_asset_sha256: hash(asset),
        plan,
        limits: loader_limits,
    }
}

fn drive_immediate(
    mut task: WeightLoadTask<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>,
    provider: &mut MemoryAssetProvider,
    runtime: &mut CpuBackend,
) -> core::result::Result<
    minifield_executor_core::TypedWeights<CpuBuffer>,
    minifield_executor_core::LoaderError,
> {
    for _ in 0..256 {
        match task.poll_step(provider, runtime) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => return result,
        }
    }
    panic!("immediate loader exceeded bounded step count")
}

fn drive_lfm_immediate(
    mut task: Lfm2WeightLoadTask<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>,
    provider: &mut MemoryAssetProvider,
    runtime: &mut CpuBackend,
) -> core::result::Result<Lfm2TypedWeights<CpuBuffer>, minifield_executor_core::LoaderError> {
    for _ in 0..256 {
        match task.poll_step(provider, runtime) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => return result,
        }
    }
    panic!("immediate LFM loader exceeded bounded step count")
}

#[test]
fn independent_asset_format_fixture_is_pinned_and_all_34_cases_match_policy() {
    let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest");
    assert_eq!(manifest["cases"], 34);
    assert_eq!(
        manifest["generator_sha256"],
        "ea8316a80135c9d68d538b5ebbdf52e91b66af2e26df5bcd12e816086ecbbc19"
    );
    let document: Value = serde_json::from_str(CASES).expect("cases");
    let cases = document.as_array().expect("case list");
    assert_eq!(cases.len(), 34);
    for case in cases {
        let file = case["file"].as_str().expect("file");
        let bytes = match file {
            "valid_f32.bin" | "header_budget.bin" | "asset_budget.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_f32.bin").to_vec()
            }
            "valid_unpadded_header.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_unpadded_header.bin").to_vec()
            }
            "valid_reverse_header_order.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_reverse_header_order.bin").to_vec()
            }
            "valid_metadata.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_metadata.bin").to_vec()
            }
            "valid_scalar.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_scalar.bin").to_vec()
            }
            "valid_empty.bin" => {
                include_bytes!("fixtures/asset-format-001/valid_empty.bin").to_vec()
            }
            "valid_bf16.bin" => include_bytes!("fixtures/asset-format-001/valid_bf16.bin").to_vec(),
            "short_length_prefix.bin" => {
                include_bytes!("fixtures/asset-format-001/short_length_prefix.bin").to_vec()
            }
            "header_length_overflow.bin" => {
                include_bytes!("fixtures/asset-format-001/header_length_overflow.bin").to_vec()
            }
            "truncated_header.bin" => {
                include_bytes!("fixtures/asset-format-001/truncated_header.bin").to_vec()
            }
            "invalid_utf8.bin" => {
                include_bytes!("fixtures/asset-format-001/invalid_utf8.bin").to_vec()
            }
            "nonobject_header.bin" => {
                include_bytes!("fixtures/asset-format-001/nonobject_header.bin").to_vec()
            }
            "leading_whitespace_header.bin" => {
                include_bytes!("fixtures/asset-format-001/leading_whitespace_header.bin").to_vec()
            }
            "trailing_nonspace_header.bin" => {
                include_bytes!("fixtures/asset-format-001/trailing_nonspace_header.bin").to_vec()
            }
            "duplicate_tensor_name.bin" => {
                include_bytes!("fixtures/asset-format-001/duplicate_tensor_name.bin").to_vec()
            }
            "duplicate_tensor_field.bin" => {
                include_bytes!("fixtures/asset-format-001/duplicate_tensor_field.bin").to_vec()
            }
            "metadata_nonstring.bin" => {
                include_bytes!("fixtures/asset-format-001/metadata_nonstring.bin").to_vec()
            }
            "missing_dtype.bin" => {
                include_bytes!("fixtures/asset-format-001/missing_dtype.bin").to_vec()
            }
            "dimension_boolean.bin" => {
                include_bytes!("fixtures/asset-format-001/dimension_boolean.bin").to_vec()
            }
            "dimension_negative.bin" => {
                include_bytes!("fixtures/asset-format-001/dimension_negative.bin").to_vec()
            }
            "dimension_fractional.bin" => {
                include_bytes!("fixtures/asset-format-001/dimension_fractional.bin").to_vec()
            }
            "shape_multiply_overflow.bin" => {
                include_bytes!("fixtures/asset-format-001/shape_multiply_overflow.bin").to_vec()
            }
            "shape_byte_mismatch.bin" => {
                include_bytes!("fixtures/asset-format-001/shape_byte_mismatch.bin").to_vec()
            }
            "reversed_offsets.bin" => {
                include_bytes!("fixtures/asset-format-001/reversed_offsets.bin").to_vec()
            }
            "offset_outside_asset.bin" => {
                include_bytes!("fixtures/asset-format-001/offset_outside_asset.bin").to_vec()
            }
            "overlap.bin" => include_bytes!("fixtures/asset-format-001/overlap.bin").to_vec(),
            "hole.bin" => include_bytes!("fixtures/asset-format-001/hole.bin").to_vec(),
            "trailing_payload.bin" => {
                include_bytes!("fixtures/asset-format-001/trailing_payload.bin").to_vec()
            }
            "unknown_dtype.bin" => {
                include_bytes!("fixtures/asset-format-001/unknown_dtype.bin").to_vec()
            }
            "unsupported_f64_storage.bin" => {
                include_bytes!("fixtures/asset-format-001/unsupported_f64_storage.bin").to_vec()
            }
            "nonfinite_f32.bin" => {
                include_bytes!("fixtures/asset-format-001/nonfinite_f32.bin").to_vec()
            }
            "nonfinite_bf16.bin" => {
                include_bytes!("fixtures/asset-format-001/nonfinite_bf16.bin").to_vec()
            }
            _ => panic!("unknown fixture file {file}"),
        };
        let expected = case["header_expect"].as_str().expect("expectation");
        let mut configured = limits();
        if let Some(max_header) = case
            .get("limits_override")
            .and_then(|value| value.get("max_header_bytes"))
            .and_then(Value::as_u64)
        {
            configured.max_header_bytes = max_header;
        }
        if let Some(max_asset) = case
            .get("limits_override")
            .and_then(|value| value.get("max_asset_bytes"))
            .and_then(Value::as_u64)
        {
            configured.max_asset_bytes = max_asset;
        }
        let parsed = prefix_and_header(&bytes).and_then(|(length, header)| {
            parse_safetensors_header(header, length, bytes.len() as u64, configured)
        });
        assert_eq!(parsed.is_ok(), expected == "accept", "{}", case["id"]);
    }
}

fn tiny_requirement(
    role: &str,
    name: &str,
    dimensions: &[u64],
    layout: WeightLayout,
    tied: Option<&str>,
) -> WeightRequirement {
    WeightRequirement {
        role: role.to_owned(),
        tensor_name: name.to_owned(),
        storage_dtype: StorageDType::F32,
        source_shape: Shape::new(dimensions).expect("tiny shape"),
        layout,
        tied_to_role: tied.map(str::to_owned),
    }
}

fn tiny_lfm_plan() -> Lfm2WeightPlan {
    Lfm2WeightPlan::from_config(parse_lfm2_config(TINY_CONFIG).expect("tiny checked config"))
        .expect("tiny config-derived plan")
}

fn tiny_plan() -> WeightPlan {
    tiny_lfm_plan().generic_plan().clone()
}

fn tiny_lfm_request(loader_limits: LoaderLimits) -> Lfm2LoadRequest {
    Lfm2LoadRequest::new(
        TINY_CONFIG.to_vec(),
        hash(TINY_CONFIG),
        TINY_WEIGHTS.len() as u64,
        hash(TINY_WEIGHTS),
        loader_limits,
    )
    .expect("checked tiny LFM request")
}

#[test]
fn sealed_tiny_weights_load_as_owned_f32_with_one_tied_embedding_head() {
    assert_eq!(
        hash(TINY_WEIGHTS),
        [
            0x29, 0x73, 0xbe, 0xfc, 0x2b, 0x8c, 0x72, 0x37, 0x5c, 0x4b, 0xff, 0xd4, 0x61, 0x2c,
            0xef, 0x31, 0x0b, 0xef, 0x4e, 0x84, 0xa1, 0xb3, 0xbe, 0x2e, 0x33, 0xfe, 0xab, 0x9b,
            0xa4, 0x31, 0x50, 0x7f,
        ]
    );
    let mut runtime = backend(64 * 1024);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let plan = tiny_lfm_plan();
    assert_eq!(plan.physical_tensor_count(), 21);
    assert_eq!(plan.physical_parameter_count(), Ok(5_512));
    let task = Lfm2WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(
        tiny_lfm_request(limits()),
    )
    .expect("task");
    let weights =
        drive_lfm_immediate(task, &mut provider, &mut runtime).expect("published LFM weights");
    assert_eq!(weights.inner().tensors().len(), 21);
    assert_eq!(weights.inner().owned_uploaded_bytes(), 22_048);
    assert_eq!(runtime.resource_report().resident_weight_bytes, 22_048);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(22_048));
    assert!(core::ptr::eq(
        weights
            .buffer_for(Lfm2WeightRole::TokenEmbedding)
            .expect("embedding"),
        weights
            .buffer_for(Lfm2WeightRole::TiedLmHead)
            .expect("tied head"),
    ));
    let kernel = weights
        .buffer_for(Lfm2WeightRole::Layer {
            index: 0,
            role: minifield_executor_core::Lfm2LayerWeightRole::ConvKernel,
        })
        .expect("kernel");
    let kernel_tensor = weights
        .inner()
        .tensors()
        .iter()
        .find(|tensor| tensor.name == "model.layers.0.conv.conv.weight")
        .expect("kernel tensor");
    assert_eq!(
        kernel_tensor.source_shape,
        Shape::new(&[16, 1, 3]).expect("source shape")
    );
    assert_eq!(
        kernel_tensor.operator_shape,
        Shape::new(&[16, 3]).expect("operator shape")
    );
    assert_eq!(kernel.len(), 48);

    let (header_length, header) = prefix_and_header(TINY_WEIGHTS).expect("tiny header");
    let parsed =
        parse_safetensors_header(header, header_length, TINY_WEIGHTS.len() as u64, limits())
            .expect("tiny parsed");
    for tensor in &parsed.tensors {
        let loaded = weights
            .inner()
            .tensors()
            .iter()
            .find(|loaded| loaded.name == tensor.name)
            .expect("loaded tensor");
        let start = usize::try_from(tensor.bytes.offset).expect("fixture range start");
        let end = usize::try_from(
            tensor
                .bytes
                .offset
                .checked_add(tensor.bytes.len)
                .expect("fixture range end overflow"),
        )
        .expect("fixture range end");
        let expected: Vec<u32> = TINY_WEIGHTS[start..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_bits())
            .collect();
        let actual = runtime
            .read_f32(loaded.buffer())
            .expect("read loaded tensor");
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
            "{}",
            tensor.name
        );
    }
}

#[test]
fn loader_expands_independent_f32_and_bf16_bits_before_f32_compute_upload() {
    let document: Value = serde_json::from_str(CASES).expect("cases");
    for file in ["valid_f32.bin", "valid_bf16.bin"] {
        let asset: Vec<u8> = match file {
            "valid_f32.bin" => include_bytes!("fixtures/asset-format-001/valid_f32.bin").to_vec(),
            "valid_bf16.bin" => include_bytes!("fixtures/asset-format-001/valid_bf16.bin").to_vec(),
            _ => unreachable!(),
        };
        let (length, header) = prefix_and_header(&asset).expect("header");
        let parsed =
            parse_safetensors_header(header, length, asset.len() as u64, limits()).expect("parsed");
        let mut runtime = backend(4096);
        let mut provider = MemoryAssetProvider::new(asset.clone(), 4096);
        let task = WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
            "format",
            b"format-config",
            &asset,
            plan_from_parsed("format", &parsed),
            limits(),
        ))
        .expect("task");
        let weights = drive_immediate(task, &mut provider, &mut runtime).expect("weights");
        let expected_case = document
            .as_array()
            .expect("array")
            .iter()
            .find(|case| case["file"] == file)
            .expect("case");
        let expected = expected_case["decoded_f32_u32"]["weight"]
            .as_array()
            .expect("bits")
            .iter()
            .map(|value| u32::try_from(value.as_u64().expect("bit")).expect("u32 bit pattern"))
            .collect::<Vec<_>>();
        let actual = runtime
            .read_f32(weights.buffer_for_role("weight").expect("weight"))
            .expect("read");
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
            "{file}"
        );
    }
}

#[derive(Debug)]
struct ScriptedRead {
    result: Option<core::result::Result<minifield_engine_api::AssetBytes, ExecutorError>>,
    pending: u8,
}

impl InferenceCompletion for ScriptedRead {
    type Output = minifield_engine_api::AssetBytes;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.pending > 0 {
            self.pending -= 1;
            return CompletionPoll::Pending;
        }
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> minifield_engine_api::Result<()> {
        if self.result.is_none() {
            return Err(ExecutorError::CompletionConsumed);
        }
        self.result = Some(Err(ExecutorError::Cancelled));
        self.pending = 0;
        Ok(())
    }
}

struct ScriptedProvider {
    bytes: Vec<u8>,
    pending: u8,
    calls: usize,
    fail_call: Option<usize>,
    short_call: Option<usize>,
}

impl ScriptedProvider {
    fn new(bytes: &[u8], pending: u8) -> Self {
        Self {
            bytes: bytes.to_vec(),
            pending,
            calls: 0,
            fail_call: None,
            short_call: None,
        }
    }
}

impl AssetProvider for ScriptedProvider {
    type Read = ScriptedRead;

    fn read_range(
        &mut self,
        range: minifield_engine_api::ByteRange,
    ) -> minifield_engine_api::Result<Self::Read> {
        range.validate_within(self.bytes.len() as u64)?;
        self.calls += 1;
        let result = if self.fail_call == Some(self.calls) {
            Err(ExecutorError::BackendFailure("scripted asset read failure"))
        } else {
            let start = usize::try_from(range.offset)
                .map_err(|_| ExecutorError::Overflow("scripted range start exceeds usize"))?;
            let mut end = usize::try_from(range.end()?)
                .map_err(|_| ExecutorError::Overflow("scripted range end exceeds usize"))?;
            if self.short_call == Some(self.calls) && end > start {
                end -= 1;
            }
            Ok(minifield_engine_api::AssetBytes::new(
                self.bytes[start..end].to_vec(),
            ))
        };
        Ok(ScriptedRead {
            result: Some(result),
            pending: self.pending,
        })
    }
}

fn drive_scripted(
    mut task: WeightLoadTask<ScriptedRead, CpuCompletion<()>, CpuBuffer>,
    provider: &mut ScriptedProvider,
    runtime: &mut CpuBackend,
) -> core::result::Result<
    minifield_executor_core::TypedWeights<CpuBuffer>,
    minifield_executor_core::LoaderError,
> {
    for _ in 0..512 {
        match task.poll_step(provider, runtime) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => return result,
        }
    }
    panic!("scripted loader exceeded bounded step count")
}

#[test]
fn deferred_reading_stays_pollable_and_read_failures_or_cancellation_never_publish_weights() {
    let (length, header) = prefix_and_header(TINY_WEIGHTS).expect("header");
    let parsed = parse_safetensors_header(header, length, TINY_WEIGHTS.len() as u64, limits())
        .expect("parsed");
    let config = b"deferred-config";
    let mut runtime = backend(64 * 1024);
    let mut provider = ScriptedProvider::new(TINY_WEIGHTS, 2);
    let mut task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        config,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    assert!(matches!(
        task.poll_step(&mut provider, &mut runtime),
        LoaderPoll::Pending
    ));
    assert_eq!(provider.calls, 1);
    assert_eq!(task.resource_report().pending_requested_bytes, 8);
    assert!(matches!(
        task.poll_step(&mut provider, &mut runtime),
        LoaderPoll::Pending
    ));
    assert!(matches!(
        task.poll_step(&mut provider, &mut runtime),
        LoaderPoll::Pending
    ));
    assert_eq!(
        provider.calls, 1,
        "two deferred polls must not spin or issue another read"
    );
    task.cancel().expect("cancel pending read");
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
    assert_eq!(task.resource_report().total_loader_bytes(), Ok(0));

    let mut runtime = backend(64 * 1024);
    let mut provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    provider.fail_call = Some(4); // after prefix/header and one tensor range; staged weight must drop.
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        config,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    let error = drive_scripted(task, &mut provider, &mut runtime).expect_err("late read failure");
    assert_eq!(error.stage, LoaderStage::TensorRead);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let mut runtime = backend(64 * 1024);
    let mut provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    provider.short_call = Some(1);
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        config,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    let error = drive_scripted(task, &mut provider, &mut runtime).expect_err("short prefix");
    assert_eq!(error.stage, LoaderStage::HeaderPrefix);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let _ = parsed;
}

#[test]
fn identity_upload_and_final_fence_failures_drop_staged_weights_and_preserve_existing_model() {
    let mut runtime = backend(64 * 1024);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let existing = drive_immediate(
        WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
            "lfm2",
            TINY_CONFIG,
            TINY_WEIGHTS,
            tiny_plan(),
            limits(),
        ))
        .expect("existing task"),
        &mut provider,
        &mut runtime,
    )
    .expect("existing model");
    let existing_total = runtime
        .resource_report()
        .total_owned_bytes()
        .expect("existing bytes");
    let embedding_bits = runtime
        .read_f32(
            existing
                .buffer_for_role("token_embedding")
                .expect("embedding"),
        )
        .expect("embedding read")
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();

    let mut wrong_identity = request("lfm2", TINY_CONFIG, TINY_WEIGHTS, tiny_plan(), limits());
    wrong_identity.expected_asset_sha256 = [0_u8; 32];
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let error = drive_immediate(
        WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(wrong_identity)
            .expect("task"),
        &mut provider,
        &mut runtime,
    )
    .expect_err("identity mismatch");
    assert_eq!(error.stage, LoaderStage::Identity);
    assert_eq!(
        runtime.resource_report().total_owned_bytes(),
        Ok(existing_total)
    );
    assert_eq!(
        runtime
            .read_f32(
                existing
                    .buffer_for_role("token_embedding")
                    .expect("embedding")
            )
            .expect("read")
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        embedding_bits
    );

    let mut constrained = backend(2_080);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let error = drive_immediate(
        WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
            "lfm2",
            TINY_CONFIG,
            TINY_WEIGHTS,
            tiny_plan(),
            limits(),
        ))
        .expect("task"),
        &mut provider,
        &mut constrained,
    )
    .expect_err("partial upload failure");
    assert_eq!(error.stage, LoaderStage::Upload);
    assert_eq!(constrained.resource_report().total_owned_bytes(), Ok(0));

    let mut runtime = backend(64 * 1024);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut task = WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    for _ in 0..47 {
        assert!(matches!(
            task.poll_step(&mut provider, &mut runtime),
            LoaderPoll::Pending
        ));
    }
    runtime.request_cancel();
    assert!(matches!(
        task.poll_step(&mut provider, &mut runtime),
        LoaderPoll::Pending
    ));
    let result = task.poll_step(&mut provider, &mut runtime);
    match result {
        LoaderPoll::Ready(Err(error)) => assert_eq!(error.stage, LoaderStage::FinalFence),
        _ => panic!("expected final fence failure"),
    }
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
}

#[test]
fn loader_limits_and_invalid_inventory_fail_before_their_next_read_or_upload() {
    let tiny_limit = LoaderLimits {
        max_asset_bytes: 8,
        ..limits()
    };
    let Err(error) = WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(
        request("lfm2", TINY_CONFIG, TINY_WEIGHTS, tiny_plan(), tiny_limit),
    ) else {
        panic!("declared asset limit unexpectedly accepted");
    };
    assert_eq!(error.stage, LoaderStage::Begin);

    let (length, header) = prefix_and_header(TINY_WEIGHTS).expect("header");
    let _ = parse_safetensors_header(header, length, TINY_WEIGHTS.len() as u64, limits())
        .expect("valid header");
    let config = b"resource-bound-config";
    let mut combined = limits();
    combined.max_retained_host_bytes = config.len() as u64 + length - 1;
    let mut runtime = backend(64 * 1024);
    let mut provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        config,
        TINY_WEIGHTS,
        tiny_plan(),
        combined,
    ))
    .expect("task");
    let error = drive_scripted(task, &mut provider, &mut runtime).expect_err("header host cap");
    assert_eq!(error.stage, LoaderStage::Header);
    assert_eq!(
        provider.calls, 1,
        "header range must not be requested after combined-host preflight"
    );
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let hole = include_bytes!("fixtures/asset-format-001/hole.bin");
    let mut runtime = backend(4096);
    let mut provider = ScriptedProvider::new(hole, 0);
    let hole_plan = WeightPlan {
        config_name: "format".to_owned(),
        requirements: vec![tiny_requirement(
            "weight",
            "weight",
            &[2],
            WeightLayout::Identity,
            None,
        )],
    };
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "format",
        b"format-config",
        hole,
        hole_plan,
        limits(),
    ))
    .expect("task");
    let error = drive_scripted(task, &mut provider, &mut runtime).expect_err("hole inventory");
    assert_eq!(error.stage, LoaderStage::Inventory);
    assert_eq!(
        provider.calls, 2,
        "invalid inventory must stop before any tensor read"
    );
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let mut decode_bound = limits();
    decode_bound.max_retained_host_bytes = TINY_CONFIG.len() as u64 + 4_095;
    let mut runtime = backend(64 * 1024);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let task = WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        decode_bound,
    ))
    .expect("task");
    let error = drive_immediate(task, &mut provider, &mut runtime).expect_err("decode host cap");
    assert_eq!(error.stage, LoaderStage::TensorDecode);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
}

#[test]
#[ignore = "requires MINIFIELD_PINNED_LFM_SAFETENSORS; validates a local, non-vendored 350M asset"]
fn pinned_350m_header_is_structurally_validated_with_exact_content_identity() {
    use std::{fs::File, io::Read as _, path::PathBuf};

    const EXPECTED_ASSET_BYTES: u64 = 708_984_464;
    const EXPECTED_ASSET_SHA256: &str =
        "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e";
    const EXPECTED_HEADER_BYTES: u64 = 16_520;
    const EXPECTED_HEADER_SHA256: &str =
        "f33a14b62b30d6015ca90705fd317186d62507a3a80d30f4fec4038886b580e9";
    const EXPECTED_TENSORS: usize = 148;
    const EXPECTED_ELEMENTS: u64 = 354_483_968;

    let path = PathBuf::from(
        std::env::var("MINIFIELD_PINNED_LFM_SAFETENSORS")
            .expect("set path to the pinned local model.safetensors"),
    );
    let metadata = std::fs::metadata(&path).expect("asset metadata");
    assert_eq!(metadata.len(), EXPECTED_ASSET_BYTES);

    let mut file = File::open(&path).expect("asset");
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).expect("header prefix");
    let header_bytes = u64::from_le_bytes(prefix);
    assert_eq!(header_bytes, EXPECTED_HEADER_BYTES);
    let mut header = vec![0_u8; usize::try_from(header_bytes).expect("header usize")];
    file.read_exact(&mut header).expect("header");
    assert_eq!(
        format!("{:x}", Sha256::digest(&header)),
        EXPECTED_HEADER_SHA256
    );

    let parsed = parse_safetensors_header(
        &header,
        header_bytes,
        metadata.len(),
        LoaderLimits {
            max_asset_bytes: 800 * 1024 * 1024,
            max_header_bytes: 32 * 1024,
            max_source_tensor_bytes: 128 * 1024 * 1024,
            max_retained_host_bytes: 128 * 1024 * 1024,
            max_tensor_name_bytes: 256,
            max_tensors: 256,
            max_rank: 4,
        },
    )
    .expect("pinned asset header is structurally valid");
    assert_eq!(parsed.tensors.len(), EXPECTED_TENSORS);
    assert!(
        parsed
            .tensors
            .iter()
            .all(|tensor| tensor.storage_dtype == StorageDType::BF16)
    );
    assert_eq!(
        parsed
            .tensors
            .iter()
            .map(|tensor| tensor.source_shape.element_count().expect("element count"))
            .sum::<u64>(),
        EXPECTED_ELEMENTS
    );
    assert!(
        parsed
            .tensors
            .iter()
            .any(|tensor| tensor.name == "model.embed_tokens.weight")
    );
    assert!(
        parsed
            .tensors
            .iter()
            .any(|tensor| tensor.name == "model.layers.0.conv.conv.weight")
    );

    let plan = Lfm2WeightPlan::from_config(
        parse_lfm2_config(PINNED_CONFIG).expect("checked pinned config"),
    )
    .expect("pinned config plan");
    assert_eq!(plan.physical_tensor_count(), EXPECTED_TENSORS);
    assert_eq!(plan.physical_parameter_count(), Ok(EXPECTED_ELEMENTS));
    let bindings = plan
        .validate_parsed_asset(
            &parsed,
            LoaderLimits {
                max_asset_bytes: 800 * 1024 * 1024,
                max_header_bytes: 32 * 1024,
                max_source_tensor_bytes: 128 * 1024 * 1024,
                max_retained_host_bytes: 128 * 1024 * 1024,
                max_tensor_name_bytes: 256,
                max_tensors: 256,
                max_rank: 4,
            },
        )
        .expect("config-derived exact pinned inventory");
    assert_eq!(bindings.len(), EXPECTED_TENSORS + 1);

    let mut hash_file = File::open(path).expect("asset for hash");
    let mut digest = Sha256::new();
    let mut chunk = [0_u8; 16 * 1024];
    loop {
        let count = hash_file.read(&mut chunk).expect("asset hash read");
        if count == 0 {
            break;
        }
        digest.update(&chunk[..count]);
    }
    assert_eq!(format!("{:x}", digest.finalize()), EXPECTED_ASSET_SHA256);
}

#[test]
fn config_derived_lfm_inventory_rejects_name_shape_effective_ff_and_separate_head_before_upload() {
    let (header_bytes, header) = prefix_and_header(TINY_WEIGHTS).expect("tiny header");
    let parsed =
        parse_safetensors_header(header, header_bytes, TINY_WEIGHTS.len() as u64, limits())
            .expect("tiny parsed");
    let plan = tiny_lfm_plan();
    assert_eq!(
        plan.validate_parsed_asset(&parsed, limits())
            .expect("original exact inventory")
            .len(),
        22
    );

    let mut renamed = parsed.clone();
    renamed.tensors[0].name = "model.unexpected.weight".to_owned();
    assert_eq!(
        plan.validate_parsed_asset(&renamed, limits()),
        Err(ExecutorError::UnexpectedTensor)
    );

    let mut reshaped = parsed.clone();
    let tensor = reshaped
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "model.layers.0.feed_forward.w1.weight")
        .expect("w1");
    tensor.source_shape = Shape::new(&[16, 32]).expect("same source element count");
    assert_eq!(
        plan.validate_parsed_asset(&reshaped, limits()),
        Err(ExecutorError::InvalidShape(
            "tensor does not match required shape or dtype"
        ))
    );

    let mut changed_config: Value = serde_json::from_slice(TINY_CONFIG).expect("tiny config JSON");
    changed_config["intermediate_size"] = Value::from(31_u64);
    let changed_config = serde_json::to_vec(&changed_config).expect("changed config bytes");
    let changed_plan = Lfm2WeightPlan::from_config(
        parse_lfm2_config(&changed_config).expect("checked changed config"),
    )
    .expect("changed plan");
    assert_eq!(
        changed_plan.validate_parsed_asset(&parsed, limits()),
        Err(ExecutorError::InvalidShape(
            "tensor does not match required shape or dtype"
        ))
    );

    let mut separate_head = parsed.clone();
    let mut extra = separate_head.tensors[0].clone();
    extra.name = "lm_head.weight".to_owned();
    // Model a separately stored, nonempty final head appended after the original payload.
    assert_ne!(extra.bytes.len, 0);
    extra.bytes.offset = separate_head.asset_bytes;
    separate_head.asset_bytes = separate_head
        .asset_bytes
        .checked_add(extra.bytes.len)
        .expect("separate head asset length");
    separate_head.tensors.push(extra);
    assert_eq!(
        plan.validate_parsed_asset(&separate_head, limits()),
        Err(ExecutorError::UnexpectedTensor)
    );
    let runtime = backend(64 * 1024);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
}

#[test]
fn loader_preflights_declared_prefix_and_tensor_host_bytes_before_provider_reads() {
    let (header_bytes, header) = prefix_and_header(TINY_WEIGHTS).expect("tiny header");
    let parsed =
        parse_safetensors_header(header, header_bytes, TINY_WEIGHTS.len() as u64, limits())
            .expect("tiny parsed");

    let generic_config = b"{}";
    let mut prefix_limited = limits();
    prefix_limited.max_retained_host_bytes = 9;
    let mut prefix_provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    let mut runtime = backend(64 * 1024);
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "format",
        generic_config,
        TINY_WEIGHTS,
        plan_from_parsed("format", &parsed),
        prefix_limited,
    ))
    .expect("generic diagnostic task");
    let error =
        drive_scripted(task, &mut prefix_provider, &mut runtime).expect_err("prefix host cap");
    assert_eq!(error.stage, LoaderStage::HeaderPrefix);
    assert_eq!(prefix_provider.calls, 0);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let mut declared_short_provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    let mut runtime = backend(64 * 1024);
    let mut short_request = request(
        "format",
        generic_config,
        TINY_WEIGHTS,
        plan_from_parsed("format", &parsed),
        limits(),
    );
    short_request.declared_asset_bytes = 7;
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(short_request)
        .expect("short declared task");
    let error = drive_scripted(task, &mut declared_short_provider, &mut runtime)
        .expect_err("declared asset shorter than prefix");
    assert_eq!(error.stage, LoaderStage::HeaderPrefix);
    assert_eq!(declared_short_provider.calls, 0);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));

    let mut tensor_limited = limits();
    // The tiny header fits, but the first 2,048-byte source plus its decoded F32 array does not.
    tensor_limited.max_retained_host_bytes = 3_000;
    let mut tensor_provider = ScriptedProvider::new(TINY_WEIGHTS, 0);
    let mut runtime = backend(64 * 1024);
    let task = WeightLoadTask::<ScriptedRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        tensor_limited,
    ))
    .expect("tensor bounded task");
    let error =
        drive_scripted(task, &mut tensor_provider, &mut runtime).expect_err("tensor host cap");
    assert_eq!(error.stage, LoaderStage::TensorDecode);
    assert_eq!(
        tensor_provider.calls, 2,
        "only prefix and header reads may occur before tensor host preflight"
    );
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
}

#[test]
fn loader_rejects_colliding_backend_identity_after_first_upload_without_mixing_weights() {
    let mut first = backend(64 * 1024);
    let mut second = backend(64 * 1024);
    assert_eq!(
        first.identity(),
        second.identity(),
        "public diagnostic IDs collide"
    );
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut task = WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    for _ in 0..6 {
        assert!(matches!(
            task.poll_step(&mut provider, &mut first),
            LoaderPoll::Pending
        ));
    }
    assert!(
        first.resource_report().resident_weight_bytes > 0,
        "one physical weight has been uploaded before the backend switch"
    );
    match task.poll_step(&mut provider, &mut second) {
        LoaderPoll::Ready(Err(error)) => {
            assert_eq!(error.stage, LoaderStage::Backend);
            assert_eq!(error.cause, ExecutorError::WrongBackend);
        }
        _ => panic!("cross-instance backend switch unexpectedly continued"),
    }
    assert_eq!(first.resource_report().total_owned_bytes(), Ok(0));
    assert_eq!(second.resource_report().total_owned_bytes(), Ok(0));
}

#[derive(Debug, Default)]
struct DeferredFenceState {
    cancel_calls: u32,
    dropped_unready: u32,
}

#[derive(Debug)]
struct DeferredFence {
    pending: u8,
    fail_cancel: bool,
    state: Rc<RefCell<DeferredFenceState>>,
}

impl Drop for DeferredFence {
    fn drop(&mut self) {
        if self.pending != 0 {
            self.state.borrow_mut().dropped_unready += 1;
        }
    }
}

impl InferenceCompletion for DeferredFence {
    type Output = ();

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.pending != 0 {
            self.pending -= 1;
            CompletionPoll::Pending
        } else {
            CompletionPoll::Ready(Ok(()))
        }
    }

    fn cancel(&mut self) -> Result<()> {
        self.state.borrow_mut().cancel_calls += 1;
        if self.fail_cancel {
            Err(ExecutorError::BackendFailure(
                "queued fence cancellation is not yet confirmed",
            ))
        } else {
            self.pending = 0;
            Ok(())
        }
    }
}

#[derive(Debug)]
struct RetiredDeferredFence {
    fence: DeferredFence,
    _retained: Vec<CpuBuffer>,
}

#[derive(Debug, Default)]
struct DeferredFenceRetirement {
    retired: RefCell<Vec<RetiredDeferredFence>>,
}

impl FenceRetirement<DeferredFence, CpuBuffer> for DeferredFenceRetirement {
    fn retire(
        &self,
        fence: DeferredFence,
        retained: Vec<CpuBuffer>,
    ) -> core::result::Result<(), RetirementRejection<DeferredFence, CpuBuffer>> {
        self.retired.borrow_mut().push(RetiredDeferredFence {
            fence,
            _retained: retained,
        });
        Ok(())
    }

    fn quarantine_rejected(&self, rejected: RetirementRejection<DeferredFence, CpuBuffer>) {
        let (_, fence, retained) = rejected.into_parts();
        self.retired.borrow_mut().push(RetiredDeferredFence {
            fence,
            _retained: retained,
        });
    }

    fn poll_retired(&self) {
        let mut retired = core::mem::take(&mut *self.retired.borrow_mut());
        let mut pending = Vec::new();
        for mut entry in retired.drain(..) {
            match entry.fence.poll_step() {
                CompletionPoll::Pending => pending.push(entry),
                CompletionPoll::Ready(_) => drop(entry),
            }
        }
        self.retired.borrow_mut().extend(pending);
    }
}

/// Test-only queued adapter: all uploads are retained by the loader until this delayed fence
/// resolves, so no loader test relies only on a deferred asset provider.
struct DeferredFenceBackend {
    cpu: CpuBackend,
    pending_fence_polls: u8,
    fail_fence_cancel: bool,
    state: Rc<RefCell<DeferredFenceState>>,
    retirement: Rc<DeferredFenceRetirement>,
}

impl DeferredFenceBackend {
    fn new(total: u64, pending_fence_polls: u8, fail_fence_cancel: bool) -> Self {
        Self {
            cpu: backend(total),
            pending_fence_polls,
            fail_fence_cancel,
            state: Rc::new(RefCell::new(DeferredFenceState::default())),
            retirement: Rc::new(DeferredFenceRetirement::default()),
        }
    }
}

impl InferenceOps for DeferredFenceBackend {
    type Buffer = CpuBuffer;
    type Fence = DeferredFence;
    type Readback = CpuCompletion<Vec<f32>>;
    type FenceRetirement = DeferredFenceRetirement;

    fn identity(&self) -> BackendIdentity {
        self.cpu.identity()
    }
    fn lease(&self) -> BackendLease {
        self.cpu.lease()
    }
    fn fence_retirement(&self) -> Rc<Self::FenceRetirement> {
        Rc::clone(&self.retirement)
    }
    fn poll_retired_fences(&self) -> Result<()> {
        self.retirement.poll_retired();
        Ok(())
    }
    fn capabilities(&self) -> BackendCapabilities {
        self.cpu.capabilities()
    }
    fn resource_report(&self) -> ResourceReport {
        self.cpu.resource_report()
    }
    fn advance_generation(&mut self) -> Result<()> {
        self.cpu.advance_generation()
    }
    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        self.cpu.allocate_f32_classified(shape, class)
    }
    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        self.cpu.upload_f32_classified(shape, values, class)
    }
    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        self.cpu.upload_u8_classified(shape, bytes, class)
    }
    fn fence(&self) -> Result<DeferredFence> {
        Ok(DeferredFence {
            pending: self.pending_fence_polls,
            fail_cancel: self.fail_fence_cancel,
            state: Rc::clone(&self.state),
        })
    }
    fn read_f32_async(&self, buffer: &CpuBuffer) -> Result<CpuCompletion<Vec<f32>>> {
        self.cpu.read_f32_async(buffer)
    }
    fn copy(&self, output: &mut CpuBuffer, input: &CpuBuffer) -> Result<()> {
        self.cpu.copy(output, input)
    }
    fn copy_rect_2d(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        self.cpu.copy_rect_2d(output, input, rectangle)
    }
    fn gather_rows(
        &self,
        output: &mut CpuBuffer,
        table: &CpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.cpu.gather_rows(
            output,
            table,
            match ids {
                TokenIds::Host(ids) => TokenIds::Host(ids),
                TokenIds::Device(buffer) => TokenIds::Device(buffer),
            },
        )
    }
    fn gather_columns(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        columns: &[TokenId],
    ) -> Result<()> {
        self.cpu.gather_columns(output, input, columns)
    }
    fn argmax(&self, output: &mut CpuBuffer, input: &CpuBuffer) -> Result<()> {
        self.cpu.argmax(output, input)
    }
    fn add(&self, output: &mut CpuBuffer, left: &CpuBuffer, right: &CpuBuffer) -> Result<()> {
        self.cpu.add(output, left, right)
    }
    fn multiply(&self, output: &mut CpuBuffer, left: &CpuBuffer, right: &CpuBuffer) -> Result<()> {
        self.cpu.multiply(output, left, right)
    }
    fn linear(&self, output: &mut CpuBuffer, input: &CpuBuffer, weight: &CpuBuffer) -> Result<()> {
        self.cpu.linear(output, input, weight)
    }
    fn packed_linear(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
    ) -> Result<()> {
        self.cpu.packed_linear(output, input, codes, scales)
    }
    fn packed_gather_rows(
        &self,
        output: &mut CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.cpu.packed_gather_rows(
            output,
            codes,
            scales,
            match ids {
                TokenIds::Host(ids) => TokenIds::Host(ids),
                TokenIds::Device(buffer) => TokenIds::Device(buffer),
            },
        )
    }
    fn row_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu.row_rms_norm(output, input, weight, epsilon)
    }
    fn head_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu
            .head_rms_norm(output, input, weight, heads, epsilon)
    }
    fn split_half_rotary(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        self.cpu.split_half_rotary(output, input, positions, spec)
    }
    fn causal_gqa(
        &self,
        output: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        value: &CpuBuffer,
        key_cache: &mut CpuBuffer,
        value_cache: &mut CpuBuffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()> {
        self.cpu.causal_gqa(
            output,
            query,
            key,
            value,
            key_cache,
            value_cache,
            cache_len,
            spec,
        )
    }
    fn gated_short_convolution(
        &self,
        output: &mut CpuBuffer,
        projection: &CpuBuffer,
        kernel: &CpuBuffer,
        history: &mut CpuBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.cpu
            .gated_short_convolution(output, projection, kernel, history, spec)
    }
    fn swiglu(&self, output: &mut CpuBuffer, gate: &CpuBuffer, up: &CpuBuffer) -> Result<()> {
        self.cpu.swiglu(output, gate, up)
    }
    #[allow(clippy::too_many_arguments)]
    fn packed_linear_pair(
        &self,
        out_a: &mut CpuBuffer,
        out_b: &mut CpuBuffer,
        input: &CpuBuffer,
        codes_a: &CpuBuffer,
        scales_a: &CpuBuffer,
        codes_b: &CpuBuffer,
        scales_b: &CpuBuffer,
    ) -> Result<()> {
        self.cpu
            .packed_linear_pair(out_a, out_b, input, codes_a, scales_a, codes_b, scales_b)
    }
    fn packed_swiglu_linear(
        &self,
        output: &mut CpuBuffer,
        gate: &CpuBuffer,
        up: &CpuBuffer,
        codes: &CpuBuffer,
        scales: &CpuBuffer,
    ) -> Result<()> {
        self.cpu
            .packed_swiglu_linear(output, gate, up, codes, scales)
    }
    fn add_row_rms_norm(
        &self,
        sum: &mut CpuBuffer,
        normed: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
        weight: &CpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu
            .add_row_rms_norm(sum, normed, left, right, weight, epsilon)
    }
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        &self,
        query_out: &mut CpuBuffer,
        key_out: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        query_weight: &CpuBuffer,
        key_weight: &CpuBuffer,
        positions: &[u64],
        rope: RotarySpec,
        key_value_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu.qk_norm_rope(
            query_out,
            key_out,
            query,
            key,
            query_weight,
            key_weight,
            positions,
            rope,
            key_value_heads,
            epsilon,
        )
    }
}

#[test]
fn deferred_fence_withholds_lfm_publication_and_unconfirmed_cancel_drains_staged_weights() {
    let mut success_backend = DeferredFenceBackend::new(64 * 1024, 1, false);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut success = WeightLoadTask::<MemoryAssetRead, DeferredFence, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("success task");
    for _ in 0..47 {
        assert!(matches!(
            success.poll_step(&mut provider, &mut success_backend),
            LoaderPoll::Pending
        ));
    }
    assert!(matches!(
        success.poll_step(&mut provider, &mut success_backend),
        LoaderPoll::Pending
    ));
    assert!(matches!(
        success.poll_step(&mut provider, &mut success_backend),
        LoaderPoll::Pending
    ));
    assert!(
        success_backend.resource_report().resident_weight_bytes > 0,
        "fence-pending staged weights remain backend-owned"
    );
    assert!(matches!(
        success.poll_step(&mut provider, &mut success_backend),
        LoaderPoll::Ready(Ok(_))
    ));

    let mut uncertain_backend = DeferredFenceBackend::new(64 * 1024, 1, true);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut cancelled = WeightLoadTask::<MemoryAssetRead, DeferredFence, CpuBuffer>::begin(
        request("lfm2", TINY_CONFIG, TINY_WEIGHTS, tiny_plan(), limits()),
    )
    .expect("cancel task");
    for _ in 0..47 {
        assert!(matches!(
            cancelled.poll_step(&mut provider, &mut uncertain_backend),
            LoaderPoll::Pending
        ));
    }
    assert!(matches!(
        cancelled.poll_step(&mut provider, &mut uncertain_backend),
        LoaderPoll::Pending
    ));
    let error = cancelled.cancel().expect_err("unconfirmed queued cancel");
    assert_eq!(error.stage, LoaderStage::Cancelled);
    assert!(
        uncertain_backend.resource_report().resident_weight_bytes > 0,
        "staged weights cannot be dropped before the fence confirms completion"
    );
    assert!(matches!(
        cancelled.poll_step(&mut provider, &mut uncertain_backend),
        LoaderPoll::Pending
    ));
    match cancelled.poll_step(&mut provider, &mut uncertain_backend) {
        LoaderPoll::Ready(Err(error)) => {
            assert_eq!(error.stage, LoaderStage::Cancelled);
            assert_eq!(error.cause, ExecutorError::Cancelled);
        }
        _ => panic!("drained cancellation did not finish"),
    }
    assert_eq!(
        uncertain_backend.resource_report().total_owned_bytes(),
        Ok(0),
        "draining releases known-complete staged CPU buffers"
    );
}

#[test]
fn late_backend_and_generation_rejection_drain_original_fence_before_releasing_staged_weights() {
    let mut owner = DeferredFenceBackend::new(64 * 1024, 2, true);
    let mut foreign = DeferredFenceBackend::new(64 * 1024, 2, true);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut task = WeightLoadTask::<MemoryAssetRead, DeferredFence, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    for _ in 0..47 {
        assert!(matches!(
            task.poll_step(&mut provider, &mut owner),
            LoaderPoll::Pending
        ));
    }
    assert!(matches!(
        task.poll_step(&mut provider, &mut owner),
        LoaderPoll::Pending
    ));
    let staged_bytes = owner
        .resource_report()
        .total_owned_bytes()
        .expect("staged bytes");
    assert_eq!(staged_bytes, 22_048);

    assert!(matches!(
        task.poll_step(&mut provider, &mut foreign),
        LoaderPoll::Pending
    ));
    assert_eq!(owner.state.borrow().cancel_calls, 1);
    assert_eq!(
        owner.resource_report().total_owned_bytes(),
        Ok(staged_bytes),
        "late foreign backend rejection retains every staged weight"
    );
    assert!(matches!(
        task.poll_step(&mut provider, &mut owner),
        LoaderPoll::Pending
    ));
    assert!(matches!(
        task.poll_step(&mut provider, &mut owner),
        LoaderPoll::Pending
    ));
    match task.poll_step(&mut provider, &mut owner) {
        LoaderPoll::Ready(Err(error)) => {
            assert_eq!(error.stage, LoaderStage::Backend);
            assert_eq!(error.cause, ExecutorError::WrongBackend);
        }
        _ => panic!("foreign backend rejection did not drain to its original error"),
    }
    assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));

    let mut generation_owner = DeferredFenceBackend::new(64 * 1024, 2, true);
    let mut generation_provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut generation_task = WeightLoadTask::<MemoryAssetRead, DeferredFence, CpuBuffer>::begin(
        request("lfm2", TINY_CONFIG, TINY_WEIGHTS, tiny_plan(), limits()),
    )
    .expect("generation task");
    for _ in 0..47 {
        assert!(matches!(
            generation_task.poll_step(&mut generation_provider, &mut generation_owner),
            LoaderPoll::Pending
        ));
    }
    assert!(matches!(
        generation_task.poll_step(&mut generation_provider, &mut generation_owner),
        LoaderPoll::Pending
    ));
    let staged_bytes = generation_owner
        .resource_report()
        .total_owned_bytes()
        .expect("generation staged bytes");
    generation_owner
        .advance_generation()
        .expect("generation change");
    assert!(matches!(
        generation_task.poll_step(&mut generation_provider, &mut generation_owner),
        LoaderPoll::Pending
    ));
    assert_eq!(generation_owner.state.borrow().cancel_calls, 1);
    assert_eq!(
        generation_owner.resource_report().total_owned_bytes(),
        Ok(staged_bytes),
        "late generation rejection retains every staged weight"
    );
    assert!(matches!(
        generation_task.poll_step(&mut generation_provider, &mut generation_owner),
        LoaderPoll::Pending
    ));
    assert!(matches!(
        generation_task.poll_step(&mut generation_provider, &mut generation_owner),
        LoaderPoll::Pending
    ));
    match generation_task.poll_step(&mut generation_provider, &mut generation_owner) {
        LoaderPoll::Ready(Err(error)) => {
            assert_eq!(error.stage, LoaderStage::Backend);
            assert_eq!(error.cause, ExecutorError::StaleBuffer);
        }
        _ => panic!("generation rejection did not drain to its original error"),
    }
    assert_eq!(
        generation_owner.resource_report().total_owned_bytes(),
        Ok(0)
    );
}

#[test]
fn dropping_after_unconfirmed_cancel_transfers_fence_and_staged_weights_to_backend_retirement() {
    let mut runtime = DeferredFenceBackend::new(64 * 1024, 2, true);
    let state = Rc::clone(&runtime.state);
    let mut provider = MemoryAssetProvider::new(TINY_WEIGHTS.to_vec(), 64 * 1024);
    let mut task = WeightLoadTask::<MemoryAssetRead, DeferredFence, CpuBuffer>::begin(request(
        "lfm2",
        TINY_CONFIG,
        TINY_WEIGHTS,
        tiny_plan(),
        limits(),
    ))
    .expect("task");
    for _ in 0..47 {
        assert!(matches!(
            task.poll_step(&mut provider, &mut runtime),
            LoaderPoll::Pending
        ));
    }
    assert!(matches!(
        task.poll_step(&mut provider, &mut runtime),
        LoaderPoll::Pending
    ));
    let staged_bytes = runtime
        .resource_report()
        .total_owned_bytes()
        .expect("staged bytes");
    assert_eq!(staged_bytes, 22_048);
    assert!(task.cancel().is_err());
    assert_eq!(state.borrow().cancel_calls, 1);

    drop(task);
    assert_eq!(
        runtime.resource_report().total_owned_bytes(),
        Ok(staged_bytes),
        "task drop transfers unresolved work to backend retirement"
    );
    assert_eq!(
        state.borrow().dropped_unready,
        0,
        "retired fence is not dropped before a terminal poll"
    );
    runtime.poll_retired_fences().expect("first retired poll");
    assert_eq!(
        runtime.resource_report().total_owned_bytes(),
        Ok(staged_bytes)
    );
    runtime.poll_retired_fences().expect("second retired poll");
    assert_eq!(
        runtime.resource_report().total_owned_bytes(),
        Ok(staged_bytes)
    );
    runtime
        .poll_retired_fences()
        .expect("terminal retired poll");
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
    assert_eq!(state.borrow().dropped_unready, 0);
}

#[test]
fn present_nonstring_lfm_dtype_rejects_before_asset_read_or_weight_publication() {
    let malformed = [
        Value::Null,
        Value::Bool(false),
        json!(123),
        json!({"nested": "value"}),
        json!([]),
    ];
    for dtype in malformed {
        let mut config: Value = serde_json::from_slice(TINY_CONFIG).expect("tiny config");
        config["dtype"] = dtype;
        let config_bytes = serde_json::to_vec(&config).expect("malformed config bytes");
        assert_eq!(
            Lfm2LoadRequest::new(
                config_bytes.clone(),
                hash(&config_bytes),
                TINY_WEIGHTS.len() as u64,
                hash(TINY_WEIGHTS),
                limits(),
            ),
            Err(ExecutorError::InvalidArgument(
                "LFM2 config dtype must be a supported string"
            ))
        );
        let runtime = backend(64 * 1024);
        assert_eq!(
            runtime.resource_report().total_owned_bytes(),
            Ok(0),
            "config rejection precedes asset provider reads and all uploads"
        );
    }
}

#[test]
fn lfm_load_request_rejects_config_digest_before_provider_reads_or_uploads() {
    let request = Lfm2LoadRequest::new(
        TINY_CONFIG.to_vec(),
        [0_u8; 32],
        TINY_WEIGHTS.len() as u64,
        hash(TINY_WEIGHTS),
        limits(),
    )
    .expect("config parses before identity check");
    let Err(error) =
        Lfm2WeightLoadTask::<MemoryAssetRead, CpuCompletion<()>, CpuBuffer>::begin(request)
    else {
        panic!("wrong config digest unexpectedly accepted");
    };
    assert_eq!(error.stage, LoaderStage::Identity);
    let runtime = backend(64 * 1024);
    assert_eq!(runtime.resource_report().total_owned_bytes(), Ok(0));
}
