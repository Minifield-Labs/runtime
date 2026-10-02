#![allow(clippy::expect_used)]

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, MemoryAssetProvider, ResourceLimits, TokenChunk,
    TokenExecutor,
};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2WeightLoadTask,
    LoaderLimits, LoaderPoll,
};
use sha2::{Digest, Sha256};

const CONFIG: &[u8] = include_bytes!("fixtures/numerical-lfm-001-config.json");
const WEIGHTS: &[u8] = include_bytes!("fixtures/numerical-lfm-001-weights.safetensors");
const CLASSES: [usize; 3] = [2, 5, 7];

#[test]
fn automatic_cache_keeps_a_tail_and_counts_only_the_required_passes() {
    let (backend, weights) = load(classifier_weights(), Some(3));
    let mut classifier = Lfm2Classifier::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 32,
        },
    )
    .expect("classifier");
    let original: Vec<_> = (1..=20).collect();
    let mut changed = original.clone();
    changed[0] = 2;
    for (ids, passes, positions, rebuilds, reused, fallback) in [
        (&original[..], 2, 20, 1, 19, false),
        (&original[..], 1, 1, 0, 19, false),
        (&original[..19], 2, 19, 1, 18, false),
        (&original[..19], 1, 1, 0, 18, false),
        (&original[..4], 1, 4, 0, 0, true),
        (&changed[..], 1, 20, 0, 0, true),
    ] {
        let expected = ready(
            classifier
                .classify(TokenChunk::all(ids))
                .expect("reference"),
        );
        let before = classifier.inference_work();
        let mut task = classifier
            .classify_cached(TokenChunk::all(ids))
            .expect("cached");
        let actual = loop {
            match task.poll_step() {
                CompletionPoll::Pending => {}
                CompletionPoll::Ready(result) => break result.expect("cached logits"),
            }
        };
        let stats = task.cache_stats();
        assert_eq!(stats.rebuilds, rebuilds);
        assert_eq!(stats.reused_tokens, reused);
        assert_eq!(stats.fallback_used, fallback);
        drop(task);
        let work = classifier.inference_work().since(before);
        assert_eq!(work.forward_passes, passes);
        assert_eq!(work.token_positions_processed, positions);
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{actual:?} != {expected:?}");
        }
    }
}

#[test]
fn cancelled_rebuild_and_invalid_inputs_preserve_the_published_cache() {
    let (backend, weights) = load(classifier_weights(), Some(3));
    let mut classifier = Lfm2Classifier::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 32,
        },
    )
    .expect("classifier");
    let ids: Vec<_> = (1..=20).collect();
    let expected = ready(
        classifier
            .classify_cached(TokenChunk::all(&ids))
            .expect("cached"),
    );
    let mut rebuild = classifier
        .classify_cached(TokenChunk::all(&ids[..19]))
        .expect("rebuild");
    rebuild.cancel().expect("cancel before submission");
    assert!(matches!(rebuild.poll_step(), CompletionPoll::Ready(Err(_))));
    assert_eq!(rebuild.cache_stats().rebuilds, 0);
    drop(rebuild);
    let before = classifier.inference_work();
    for invalid in [&[][..], &[32][..], &[1; 33][..]] {
        assert!(
            classifier
                .classify_cached(TokenChunk::all(invalid))
                .is_err()
        );
    }
    assert_eq!(classifier.inference_work(), before);
    let actual = ready(
        classifier
            .classify_cached(TokenChunk::all(&ids))
            .expect("reuse original"),
    );
    assert_eq!(actual, expected);
    let work = classifier.inference_work().since(before);
    assert_eq!(work.forward_passes, 1);
    assert_eq!(work.token_positions_processed, 1);
}

#[test]
fn work_counts_cached_and_full_passes_with_the_actual_head_width() {
    let (backend, weights) = load(classifier_weights(), Some(3));
    let mut classifier = Lfm2Classifier::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 32,
        },
    )
    .expect("classifier");
    let initial = classifier.inference_work();
    assert_eq!(initial, minifield_executor_core::InferenceWork::default());
    let base = ready(
        classifier
            .prefill_base(TokenChunk::all(&[1, 2, 3]))
            .expect("prefill"),
    );
    let warm = classifier.inference_work().since(initial);
    assert_eq!(warm.forward_passes, 1);
    assert_eq!(warm.token_positions_processed, 3);
    // H=16, I=32, one conv + one attention block; final FFN executes one row.
    assert_eq!(warm.estimated_flops, 23_712);
    let before = classifier.inference_work();
    ready(
        classifier
            .classify_tail(&base, TokenChunk::all(&[4, 5]))
            .expect("tail"),
    );
    let tail = classifier.inference_work().since(before);
    assert_eq!(tail.forward_passes, 1);
    assert_eq!(tail.token_positions_processed, 2);
    assert_eq!(tail.estimated_flops, 17_248);
    let before = classifier.inference_work();
    ready(
        classifier
            .classify(TokenChunk::all(&[1, 2, 3]))
            .expect("full"),
    );
    assert_eq!(
        classifier.inference_work().since(before).estimated_flops,
        23_808
    );
    let before = classifier.inference_work();
    assert!(classifier.classify(TokenChunk::all(&[])).is_err());
    assert_eq!(classifier.inference_work(), before);
}

fn classifier_weights() -> Vec<u8> {
    classifier_weights_for(&CLASSES)
}

fn classifier_weights_for(rows: &[usize]) -> Vec<u8> {
    let header_len = usize::try_from(u64::from_le_bytes(WEIGHTS[..8].try_into().expect("length")))
        .expect("header length");
    let mut header: serde_json::Value =
        serde_json::from_slice(&WEIGHTS[8..8 + header_len]).expect("header");
    let mut payload = WEIGHTS[8 + header_len..].to_vec();
    let embedding = &header["model.embed_tokens.weight"];
    let offset =
        usize::try_from(embedding["data_offsets"][0].as_u64().expect("offset")).expect("offset");
    let start = payload.len();
    for row in rows {
        let bytes = payload[offset + row * 16 * 4..offset + (row + 1) * 16 * 4].to_vec();
        payload.extend_from_slice(&bytes);
    }
    header["classification_head.weight"] = serde_json::json!({"dtype":"F32","shape":[rows.len(),16],"data_offsets":[start,payload.len()]});
    let mut encoded = serde_json::to_vec(&header).expect("header bytes");
    while encoded.len() % 8 != 0 {
        encoded.push(b' ');
    }
    let mut result = u64::try_from(encoded.len())
        .expect("length")
        .to_le_bytes()
        .to_vec();
    result.extend(encoded);
    result.extend(payload);
    result
}

fn load(
    bytes: Vec<u8>,
    classes: Option<u32>,
) -> (
    CpuBackend,
    minifield_executor_core::Lfm2TypedWeights<minifield_backend_cpu::CpuBuffer>,
) {
    let size = u64::try_from(bytes.len()).expect("size");
    let limits = LoaderLimits {
        max_asset_bytes: size,
        max_header_bytes: 1 << 20,
        max_source_tensor_bytes: size,
        max_retained_host_bytes: size * 6,
        max_tensor_name_bytes: 1024,
        max_tensors: 1024,
        max_rank: 4,
    };
    let request = Lfm2LoadRequest::discover(
        CONFIG.to_vec(),
        Sha256::digest(CONFIG).into(),
        &bytes,
        Sha256::digest(&bytes).into(),
        limits,
        classes,
    )
    .expect("request");
    let mut backend = CpuBackend::new(
        1,
        ResourceLimits {
            max_allocation_bytes: 1 << 24,
            max_total_bytes: 1 << 28,
            max_pending_operations: 512,
        },
    );
    let mut task = Lfm2WeightLoadTask::begin(request).expect("task");
    let mut provider = MemoryAssetProvider::new(bytes, size);
    loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => return (backend, result.expect("weights")),
        }
    }
}

fn ready<T: InferenceCompletion>(mut task: T) -> T::Output {
    for _ in 0..10_000 {
        match task.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(result) => return result.expect("completion"),
        }
    }
    panic!("completion timeout");
}

#[test]
fn classifier_matches_selected_dense_logits_and_resets_state() {
    let limits = Lfm2ExecutionLimits {
        max_logical_tokens: 64,
    };
    let (backend, weights) = load(WEIGHTS.to_vec(), None);
    let mut lm = Lfm2Executor::new(backend, weights, limits).expect("lm");
    let (backend, weights) = load(classifier_weights(), Some(3));
    let mut classifier = Lfm2Classifier::new(backend, weights, limits).expect("classifier");
    for ids in [&[1, 10, 9, 3][..], &[1, 8][..], &[1, 10, 9, 3][..]] {
        let prefix = ready(lm.prefill(TokenChunk::all(ids)).expect("prefill"));
        let expected = ready(lm.next_logits(&prefix).expect("logits"));
        let actual = ready(classifier.classify(TokenChunk::all(ids)).expect("classify"));
        assert_eq!(actual.len(), 3);
        for (index, row) in CLASSES.iter().enumerate() {
            assert!((actual[index] - expected[*row]).abs() < 1e-5);
        }
    }
    assert!(classifier.classify(TokenChunk::all(&[])).is_err());
    assert!(classifier.classify(TokenChunk::all(&[32])).is_err());
    assert!(classifier.classify(TokenChunk::all(&[1; 65])).is_err());
}

#[test]
fn classifier_and_language_model_heads_cannot_be_confused() {
    let limits = Lfm2ExecutionLimits {
        max_logical_tokens: 64,
    };
    let (backend, weights) = load(classifier_weights(), Some(3));
    assert!(Lfm2Executor::new(backend, weights, limits).is_err());
    let (backend, weights) = load(WEIGHTS.to_vec(), None);
    assert!(Lfm2Classifier::new(backend, weights, limits).is_err());
}

#[test]
fn cached_classifier_tail_accepts_33_classes_with_a_32_token_vocabulary() {
    let rows: Vec<_> = (0..33).map(|index| index % 32).collect();
    let (backend, weights) = load(classifier_weights_for(&rows), Some(33));
    let mut classifier = Lfm2Classifier::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: 64,
        },
    )
    .expect("classifier");
    let expected = ready(
        classifier
            .classify(TokenChunk::all(&[1, 5, 7, 9]))
            .expect("full classification"),
    );
    let base = ready(
        classifier
            .prefill_base(TokenChunk::all(&[1, 5]))
            .expect("base"),
    );
    let actual = ready(
        classifier
            .classify_tail(&base, TokenChunk::all(&[7, 9]))
            .expect("cached classification"),
    );
    assert_eq!(actual.len(), 33);
    assert_eq!(actual, expected);
    assert_eq!(
        classifier.lut2_mode(),
        minifield_executor_core::Lfm2Lut2Mode::Auto
    );
    assert_eq!(
        classifier
            .inspect_backend(CpuBackend::identity)
            .expect("inspect")
            .owner,
        1
    );
}
