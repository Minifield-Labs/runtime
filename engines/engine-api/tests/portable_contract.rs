#![allow(clippy::expect_used)]

use minifield_engine_api::{
    AssetLimits, AssetManifest, AssetProvider, BackendCapabilities, BackendIdentity, BackendKind,
    BufferAccess, BufferDescriptor, ByteRange, CompletionPoll, DType, DTypeSet, ExecutorError,
    InferenceCompletion, MemoryAssetProvider, OperationKind, OperationSet, PrecisionPolicy,
    ReadyCompletion, ResourceLimits, ResourceReport, Shape, TensorLayout, TensorRecord,
    TensorRequirement, validate_asset_manifest,
};

fn limits() -> AssetLimits {
    AssetLimits {
        max_asset_bytes: 128,
        max_tensor_bytes: 32,
        max_tensors: 4,
    }
}

#[test]
fn shape_layout_and_offset_bounds_are_checked_without_wraparound() {
    let overflow = Shape::new(&[u64::MAX, 2]).expect("shape");
    assert_eq!(
        overflow.element_count(),
        Err(ExecutorError::Overflow("shape element count overflows u64"))
    );

    let shape = Shape::new(&[2, 3]).expect("shape");
    let layout = TensorLayout::strided(DType::F32, shape, &[4, 1], 4).expect("layout");
    assert_eq!(layout.element_byte_offset(&[1, 2]), Ok(28));
    assert_eq!(
        layout.validate_within(31),
        Err(ExecutorError::OutOfBounds("layout exceeds storage bytes"))
    );
    assert_eq!(layout.validate_within(32), Ok(()));
    assert_eq!(
        layout.element_byte_offset(&[2, 0]),
        Err(ExecutorError::OutOfBounds(
            "tensor coordinate exceeds dimension"
        ))
    );
    assert_eq!(
        TensorLayout::strided(DType::F32, shape, &[3], 0),
        Err(ExecutorError::InvalidLayout(
            "stride count differs from rank"
        ))
    );
}

#[test]
fn capability_identity_and_resource_contracts_reject_wrong_combinations() {
    let identity = BackendIdentity {
        kind: BackendKind::Cpu,
        ordinal: 0,
        owner: 11,
        generation: 3,
    };
    let layout =
        TensorLayout::contiguous(DType::F32, Shape::new(&[2]).expect("shape")).expect("layout");
    let descriptor = BufferDescriptor {
        backend: identity,
        allocation: 7,
        layout,
        access: BufferAccess::ReadWrite,
    };
    let other_owner = BackendIdentity {
        owner: 12,
        ..identity
    };
    assert_eq!(
        descriptor.validate_for(other_owner),
        Err(ExecutorError::WrongBackend)
    );
    let stale = BackendIdentity {
        generation: 4,
        ..identity
    };
    assert_eq!(
        descriptor.validate_for(stale),
        Err(ExecutorError::StaleBuffer)
    );

    let capabilities = BackendCapabilities {
        dtypes: DTypeSet::only(DType::F32),
        operations: OperationSet::empty().with(OperationKind::Copy),
        precision: PrecisionPolicy {
            weights: DType::F32,
            activations: DType::F32,
            cache: DType::F32,
            accumulation: DType::F32,
        },
        max_rank: 2,
        max_elements: 8,
        max_allocation_bytes: 32,
        supports_nonblocking_completion: true,
    };
    assert!(
        capabilities
            .validate(DType::F32, OperationKind::Copy, 1, 2, 8)
            .is_ok()
    );
    assert_eq!(
        capabilities.validate(DType::BF16, OperationKind::Copy, 1, 2, 4),
        Err(ExecutorError::InvalidDType(
            "dtype is unsupported by backend"
        ))
    );
    assert_eq!(
        capabilities.validate(DType::F32, OperationKind::Linear, 1, 2, 8),
        Err(ExecutorError::Unsupported(
            "operation is unsupported by backend"
        ))
    );

    let report = ResourceReport {
        scratch_bytes: 8,
        pending_operation_bytes: 12,
        pending_operations: 1,
        ..ResourceReport::default()
    };
    assert_eq!(report.total_owned_bytes(), Ok(20));
    assert_eq!(
        report.validate(ResourceLimits {
            max_allocation_bytes: 32,
            max_total_bytes: 19,
            max_pending_operations: 1,
        }),
        Err(ExecutorError::ResourceLimit(
            "owned resource bytes exceed configured limit"
        ))
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn asset_manifest_enforces_exact_tensor_names_shapes_ranges_and_ties() {
    let shape = Shape::new(&[2, 2]).expect("shape");
    let manifest = AssetManifest {
        config_name: "lfm-config".to_owned(),
        asset_bytes: 32,
        tensors: vec![TensorRecord {
            name: "embedding".to_owned(),
            dtype: DType::F32,
            shape,
            bytes: ByteRange { offset: 8, len: 16 },
        }],
    };
    let requirements = vec![
        TensorRequirement {
            role: "embedding".to_owned(),
            tensor_name: "embedding".to_owned(),
            dtype: DType::F32,
            shape,
            tied_to_role: None,
        },
        TensorRequirement {
            role: "head".to_owned(),
            tensor_name: "embedding".to_owned(),
            dtype: DType::F32,
            shape,
            tied_to_role: Some("embedding".to_owned()),
        },
    ];
    let bindings =
        validate_asset_manifest(&manifest, "lfm-config", &requirements, limits()).expect("bind");
    assert_eq!(bindings.len(), 2);
    assert_eq!(bindings[0].tensor_index, bindings[1].tensor_index);

    let invalid_tie = vec![
        requirements[0].clone(),
        TensorRequirement {
            role: "head".to_owned(),
            tensor_name: "other".to_owned(),
            dtype: DType::F32,
            shape,
            tied_to_role: Some("embedding".to_owned()),
        },
    ];
    assert_eq!(
        validate_asset_manifest(&manifest, "lfm-config", &invalid_tie, limits()),
        Err(ExecutorError::InvalidTie)
    );

    let duplicate = AssetManifest {
        tensors: vec![manifest.tensors[0].clone(), manifest.tensors[0].clone()],
        ..manifest.clone()
    };
    assert_eq!(
        validate_asset_manifest(&duplicate, "lfm-config", &requirements, limits()),
        Err(ExecutorError::DuplicateName)
    );

    let undeclared = AssetManifest {
        tensors: vec![
            manifest.tensors[0].clone(),
            TensorRecord {
                name: "adapter.weight".to_owned(),
                dtype: DType::F32,
                shape: Shape::new(&[1]).expect("shape"),
                bytes: ByteRange { offset: 24, len: 4 },
            },
        ],
        ..manifest.clone()
    };
    assert_eq!(
        validate_asset_manifest(&undeclared, "lfm-config", &requirements, limits()),
        Err(ExecutorError::UnexpectedTensor)
    );

    let two_requirements = vec![
        requirements[0].clone(),
        TensorRequirement {
            role: "norm".to_owned(),
            tensor_name: "norm".to_owned(),
            dtype: DType::F32,
            shape: Shape::new(&[2]).expect("shape"),
            tied_to_role: None,
        },
    ];
    let overlapping = AssetManifest {
        tensors: vec![
            manifest.tensors[0].clone(),
            TensorRecord {
                name: "norm".to_owned(),
                dtype: DType::F32,
                shape: Shape::new(&[2]).expect("shape"),
                bytes: ByteRange { offset: 20, len: 8 },
            },
        ],
        ..manifest.clone()
    };
    assert_eq!(
        validate_asset_manifest(&overlapping, "lfm-config", &two_requirements, limits()),
        Err(ExecutorError::InvalidLayout(
            "physical tensor byte ranges overlap"
        ))
    );

    let self_tie = vec![TensorRequirement {
        role: "embedding".to_owned(),
        tensor_name: "embedding".to_owned(),
        dtype: DType::F32,
        shape,
        tied_to_role: Some("embedding".to_owned()),
    }];
    assert_eq!(
        validate_asset_manifest(&manifest, "lfm-config", &self_tie, limits()),
        Err(ExecutorError::InvalidTie)
    );
}

#[test]
fn checked_layout_constructor_rejects_end_overflow() {
    assert_eq!(
        TensorLayout::strided(
            DType::F32,
            Shape::new(&[1]).expect("shape"),
            &[1],
            u64::MAX - 3,
        ),
        Err(ExecutorError::Overflow("layout end overflows u64"))
    );
}

#[test]
fn memory_asset_reads_and_completions_are_portable_and_consumed_once() {
    let mut provider = MemoryAssetProvider::new(vec![1, 2, 3, 4], 3);
    let mut read = provider
        .read_range(ByteRange { offset: 1, len: 2 })
        .expect("read");
    match read.poll_step() {
        CompletionPoll::Ready(Ok(bytes)) => assert_eq!(bytes.as_slice(), &[2, 3]),
        other => panic!("unexpected poll state: {other:?}"),
    }
    assert_eq!(
        read.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed))
    );
    match provider.read_range(ByteRange { offset: 3, len: 2 }) {
        Err(ExecutorError::OutOfBounds("byte range exceeds source bytes")) => {}
        Err(other) => panic!("unexpected out-of-range read error: {other:?}"),
        Ok(_) => panic!("out-of-range read unexpectedly succeeded"),
    }

    let mut completion = ReadyCompletion::new(Ok(9_u32));
    completion.cancel().expect("cancel");
    assert_eq!(
        completion.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::Cancelled))
    );
}

#[test]
fn token_mask_rejects_short_masks_and_bad_indexes_without_panicking() {
    let ids = [1_u32, 2];
    let short_mask = [true];
    let chunk = minifield_engine_api::TokenChunk::masked(&ids, &short_mask);
    assert_eq!(
        chunk.is_valid(1),
        Err(ExecutorError::InvalidArgument(
            "token validity mask length differs from token IDs"
        ))
    );
    assert_eq!(
        chunk.is_valid(2),
        Err(ExecutorError::OutOfBounds(
            "token index exceeds physical chunk"
        ))
    );
}
