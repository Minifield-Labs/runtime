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
    let request = match classes {
        Some(classes) => Lfm2LoadRequest::new_classifier(
            CONFIG.to_vec(),
            Sha256::digest(CONFIG).into(),
            size,
            Sha256::digest(&bytes).into(),
            limits,
            classes,
        ),
        None => Lfm2LoadRequest::new(
            CONFIG.to_vec(),
            Sha256::digest(CONFIG).into(),
            size,
            Sha256::digest(&bytes).into(),
            limits,
        ),
    }
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
