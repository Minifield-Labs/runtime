#![allow(clippy::expect_used)]

use std::{
    cell::{Cell, RefCell},
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

use minifield_backend_cpu::{CpuBackend, CpuBuffer, CpuCompletion};
use minifield_engine_api::{
    BackendCapabilities, BackendIdentity, BackendLease, CompletionPoll, ExecutorError,
    FenceRetirement, GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps,
    MemoryAssetProvider, MemoryAssetRead, PackedHeadSpec, RectCopy2d, ResourceLimits,
    ResourceReport, Result, RetirementRejection, RotarySpec, Shape, TokenChoiceExecutor,
    TokenChunk, TokenExecutor, TokenId, TokenIds,
};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2WeightLoadTask, LoaderLimits,
    LoaderPoll,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn loader_limits(bytes: usize) -> LoaderLimits {
    let bytes = u64::try_from(bytes).expect("asset size");
    LoaderLimits {
        max_asset_bytes: bytes,
        max_header_bytes: 1 << 20,
        max_source_tensor_bytes: bytes,
        max_retained_host_bytes: bytes.checked_mul(2).expect("host limit"),
        max_tensor_name_bytes: 1024,
        max_tensors: 1024,
        max_rank: 4,
    }
}

fn backend() -> CpuBackend {
    CpuBackend::new(
        0xE0_2C,
        ResourceLimits {
            max_allocation_bytes: 128 * 1024 * 1024,
            max_total_bytes: 512 * 1024 * 1024,
            max_pending_operations: 256,
        },
    )
}

fn load_with_backend<B: InferenceOps<Buffer = CpuBuffer>>(
    config: &[u8],
    weights: Vec<u8>,
    mut backend: B,
    limits: Lfm2ExecutionLimits,
) -> Lfm2Executor<B> {
    let request = Lfm2LoadRequest::new(
        config.to_vec(),
        digest(config),
        u64::try_from(weights.len()).expect("asset bytes"),
        digest(&weights),
        loader_limits(weights.len()),
    )
    .expect("checked request");
    let mut task: Lfm2WeightLoadTask<MemoryAssetRead, B::Fence, CpuBuffer> =
        Lfm2WeightLoadTask::begin(request).expect("load task");
    let mut provider = MemoryAssetProvider::new(weights, 128 * 1024 * 1024);
    let weights = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(Ok(weights)) => break weights,
            LoaderPoll::Ready(Err(error)) => panic!("loader error: {error:?}"),
        }
    };
    Lfm2Executor::new(backend, weights, limits).expect("executor")
}

fn load(config: &[u8], weights: Vec<u8>) -> Lfm2Executor<CpuBackend> {
    load_with_backend(
        config,
        weights,
        backend(),
        Lfm2ExecutionLimits {
            max_logical_tokens: 64,
        },
    )
}

fn ready<T, C: InferenceCompletion<Output = T>>(completion: &mut C) -> T {
    for _ in 0..10_000 {
        match completion.poll_step() {
            CompletionPoll::Pending => {}
            CompletionPoll::Ready(Ok(value)) => return value,
            CompletionPoll::Ready(Err(error)) => panic!("completion error: {error:?}"),
        }
    }
    panic!("completion did not become ready")
}

fn external_path(variable: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(
        || panic!("{variable} must name the external immutable fixture directory"),
        PathBuf::from,
    )
}

fn read_row(path: &Path, tensor: &str, row: usize) -> Vec<f32> {
    let bytes = fs::read(path).expect("expected safetensors");
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[..8].try_into().expect("header length"),
    ))
    .expect("header usize");
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).expect("header JSON");
    let record = &header[tensor];
    assert_eq!(record["dtype"], "F32");
    let shape = record["shape"].as_array().expect("shape");
    let columns = usize::try_from(shape[1].as_u64().expect("columns")).expect("columns usize");
    let offsets = record["data_offsets"].as_array().expect("offsets");
    let begin = usize::try_from(offsets[0].as_u64().expect("begin")).expect("begin usize");
    let end = usize::try_from(offsets[1].as_u64().expect("end")).expect("end usize");
    let payload = &bytes[8 + header_len + begin..8 + header_len + end];
    payload[row * columns * 4..(row + 1) * columns * 4]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-6_f32 + 3.0e-5_f32 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected}"
        );
    }
}

/// Runs only against the external, immutable genuinely-trained tiny artifact and independent
/// Torch/JAX reference. The source tree deliberately stores no trained weights or private oracle.
#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT and MINIFIELD_TRAINED_TINY_STATE_ROOT"]
fn trained_tiny_base8_next_logits_match_independent_reference() {
    let artifact = external_path("MINIFIELD_TRAINED_TINY_ARTIFACT");
    let state = external_path("MINIFIELD_TRAINED_TINY_STATE_ROOT");
    let config = fs::read(artifact.join("model/config.json")).expect("trained config");
    let weights = fs::read(artifact.join("model/model.safetensors")).expect("trained weights");
    let mut executor = load(&config, weights);
    let mut prefill = executor
        .prefill(TokenChunk::all(&[1, 3, 7, 11, 6, 14, 2, 5]))
        .expect("prefill");
    let prefix = ready(&mut prefill);
    let mut logits = executor.next_logits(&prefix).expect("logits");
    let actual = ready(&mut logits);
    let expected = read_row(&state.join("expected.safetensors"), "base8.logits", 7);
    assert_close(&actual, &expected);
}

struct ExpectedFile {
    bytes: Vec<u8>,
    payload_start: usize,
    header: Value,
}
impl ExpectedFile {
    fn open(path: &Path) -> Self {
        let bytes = fs::read(path).expect("expected safetensors");
        let header_len = usize::try_from(u64::from_le_bytes(
            bytes[..8].try_into().expect("header length"),
        ))
        .expect("header usize");
        let header = serde_json::from_slice(&bytes[8..8 + header_len]).expect("header JSON");
        Self {
            bytes,
            payload_start: 8 + header_len,
            header,
        }
    }
    fn row(&self, tensor: &str, row: usize) -> Vec<f32> {
        let record = &self.header[tensor];
        assert_eq!(record["dtype"], "F32");
        let columns = usize::try_from(
            record["shape"]
                .as_array()
                .expect("shape")
                .last()
                .expect("last dimension")
                .as_u64()
                .expect("columns"),
        )
        .expect("columns usize");
        let offsets = record["data_offsets"].as_array().expect("offsets");
        let begin = usize::try_from(offsets[0].as_u64().expect("begin")).expect("begin usize");
        let end = usize::try_from(offsets[1].as_u64().expect("end")).expect("end usize");
        self.bytes[self.payload_start + begin + row * columns * 4..self.payload_start + end]
            .chunks_exact(4)
            .take(columns)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
            .collect()
    }
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ID array")
        .iter()
        .map(|value| u32::try_from(value.as_u64().expect("ID")).expect("ID u32"))
        .collect()
}

fn prefix_for<B: InferenceOps>(
    executor: &mut Lfm2Executor<B>,
    ids: &[u32],
) -> minifield_executor_core::Lfm2Prefix<B> {
    let mut task = executor.prefill(TokenChunk::all(ids)).expect("prefill");
    ready(&mut task)
}

fn logits_for<B: InferenceOps>(
    executor: &mut Lfm2Executor<B>,
    prefix: &minifield_executor_core::Lfm2Prefix<B>,
) -> Vec<f32> {
    let mut task = executor.next_logits(prefix).expect("next logits");
    ready(&mut task)
}

#[allow(clippy::cast_possible_truncation)]
fn assert_score(actual: f32, expected: f64) {
    // CandidateScore's transport field is deliberately F32 after F64 normalization.
    let expected = expected as f32;
    let tolerance = 1.0e-5_f32 + 3.0e-5_f32 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[derive(Debug, Default)]
struct DeferredState {
    fence_polls: Cell<u8>,
    readback_polls: Cell<u8>,
    fail_fence: Cell<bool>,
    fail_cancel: Cell<bool>,
    fail_after_n_copies: Cell<Option<u32>>,
    dropped_pending_fences: Cell<u32>,
    dropped_pending_readbacks: Cell<u32>,
}

impl DeferredState {
    /// Counts down an armed copy failure: after the configured number of copy
    /// calls succeed, the next one reports a backend failure.
    fn inject_copy_failure(&self) -> bool {
        match self.fail_after_n_copies.get() {
            Some(0) => true,
            Some(remaining) => {
                self.fail_after_n_copies.set(Some(remaining - 1));
                false
            }
            None => false,
        }
    }
}

#[derive(Debug)]
struct DeferredFence {
    pending: u8,
    fail_cancel: bool,
    state: Rc<DeferredState>,
}

impl Drop for DeferredFence {
    fn drop(&mut self) {
        if self.pending != 0 {
            self.state
                .dropped_pending_fences
                .set(self.state.dropped_pending_fences.get().saturating_add(1));
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
        if self.fail_cancel {
            Err(ExecutorError::BackendFailure(
                "deferred fence cancellation was not confirmed",
            ))
        } else {
            self.pending = 0;
            Ok(())
        }
    }
}

#[derive(Debug)]
struct DeferredReadback {
    inner: CpuCompletion<Vec<f32>>,
    pending: u8,
    state: Rc<DeferredState>,
}

impl Drop for DeferredReadback {
    fn drop(&mut self) {
        if self.pending != 0 {
            self.state
                .dropped_pending_readbacks
                .set(self.state.dropped_pending_readbacks.get().saturating_add(1));
        }
    }
}

impl InferenceCompletion for DeferredReadback {
    type Output = Vec<f32>;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.pending != 0 {
            self.pending -= 1;
            CompletionPoll::Pending
        } else {
            self.inner.poll_step()
        }
    }

    fn cancel(&mut self) -> Result<()> {
        self.pending = 0;
        self.inner.cancel()
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

    fn has_unresolved(&self) -> bool {
        !self.retired.borrow().is_empty()
    }
}

/// Test-only queued adapter. Its numerical work delegates to the owned scalar CPU backend, while
/// its independent fence/readback completions make shared executor lifetime behavior observable.
struct DeferredBackend {
    cpu: CpuBackend,
    state: Rc<DeferredState>,
    retirement: Rc<DeferredFenceRetirement>,
}

impl DeferredBackend {
    fn new(state: Rc<DeferredState>) -> Self {
        Self {
            cpu: backend(),
            state,
            retirement: Rc::new(DeferredFenceRetirement::default()),
        }
    }
}

#[allow(clippy::too_many_lines)]
impl InferenceOps for DeferredBackend {
    type Buffer = CpuBuffer;
    type Fence = DeferredFence;
    type Readback = DeferredReadback;
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
        class: minifield_engine_api::AllocationClass,
    ) -> Result<Self::Buffer> {
        self.cpu.allocate_f32_classified(shape, class)
    }

    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: minifield_engine_api::AllocationClass,
    ) -> Result<Self::Buffer> {
        self.cpu.upload_f32_classified(shape, values, class)
    }

    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: minifield_engine_api::AllocationClass,
    ) -> Result<Self::Buffer> {
        self.cpu.upload_u8_classified(shape, bytes, class)
    }

    fn fence(&self) -> Result<Self::Fence> {
        if self.state.fail_fence.get() {
            return Err(ExecutorError::BackendFailure(
                "injected failure while submitting deferred fence",
            ));
        }
        Ok(DeferredFence {
            pending: self.state.fence_polls.get(),
            fail_cancel: self.state.fail_cancel.get(),
            state: Rc::clone(&self.state),
        })
    }

    fn read_f32_async(&self, buffer: &Self::Buffer) -> Result<Self::Readback> {
        Ok(DeferredReadback {
            inner: self.cpu.read_f32_async(buffer)?,
            pending: self.state.readback_polls.get(),
            state: Rc::clone(&self.state),
        })
    }

    fn copy(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        if self.state.inject_copy_failure() {
            return Err(ExecutorError::BackendFailure(
                "injected failure while recording a deferred copy",
            ));
        }
        self.cpu.copy(output, input)
    }

    fn copy_rect_2d(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        if self.state.inject_copy_failure() {
            return Err(ExecutorError::BackendFailure(
                "injected failure while recording a deferred copy",
            ));
        }
        self.cpu.copy_rect_2d(output, input, rectangle)
    }

    fn gather_rows(
        &self,
        output: &mut Self::Buffer,
        table: &Self::Buffer,
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
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        columns: &[TokenId],
    ) -> Result<()> {
        self.cpu.gather_columns(output, input, columns)
    }

    fn argmax(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        self.cpu.argmax(output, input)
    }

    fn add(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        self.cpu.add(output, left, right)
    }

    fn multiply(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        self.cpu.multiply(output, left, right)
    }

    fn linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
    ) -> Result<()> {
        self.cpu.linear(output, input, weight)
    }

    fn packed_linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        self.cpu.packed_linear(output, input, codes, scales)
    }

    fn packed_gather_rows(
        &self,
        output: &mut Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
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
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu.row_rms_norm(output, input, weight, epsilon)
    }

    fn head_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu
            .head_rms_norm(output, input, weight, heads, epsilon)
    }

    fn split_half_rotary(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        self.cpu.split_half_rotary(output, input, positions, spec)
    }

    fn causal_gqa(
        &self,
        output: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
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
        output: &mut Self::Buffer,
        projection: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.cpu
            .gated_short_convolution(output, projection, kernel, history, spec)
    }

    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()> {
        self.cpu.swiglu(output, gate, up)
    }

    fn packed_linear_pair(
        &self,
        out_a: &mut Self::Buffer,
        out_b: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        self.cpu
            .packed_linear_pair(out_a, out_b, input, codes_a, scales_a, codes_b, scales_b)
    }

    fn packed_swiglu_linear(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        self.cpu
            .packed_swiglu_linear(output, gate, up, codes, scales)
    }

    fn packed_swiglu_pair(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        self.cpu
            .packed_swiglu_pair(output, input, codes_a, scales_a, codes_b, scales_b)
    }

    fn add_row_rms_norm(
        &self,
        sum: &mut Self::Buffer,
        normed: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()> {
        self.cpu
            .add_row_rms_norm(sum, normed, left, right, weight, epsilon)
    }

    fn qk_norm_rope(
        &self,
        query_out: &mut Self::Buffer,
        key_out: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        query_weight: &Self::Buffer,
        key_weight: &Self::Buffer,
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
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT and MINIFIELD_TRAINED_TINY_STATE_ROOT"]
#[allow(clippy::too_many_lines)]
fn trained_tiny_complete_state_fixture_matches_prefix_cache_forks_masks_and_scores() {
    let artifact = external_path("MINIFIELD_TRAINED_TINY_ARTIFACT");
    let state_root = external_path("MINIFIELD_TRAINED_TINY_STATE_ROOT");
    let cases: Value =
        serde_json::from_slice(&fs::read(state_root.join("cases.json")).expect("cases"))
            .expect("case JSON");
    let expected = ExpectedFile::open(&state_root.join("expected.safetensors"));
    let config = fs::read(artifact.join("model/config.json")).expect("trained config");
    let weights = fs::read(artifact.join("model/model.safetensors")).expect("trained weights");
    let mut executor = load(&config, weights);

    for (name, sequence) in cases["sequences"].as_object().expect("sequences") {
        let sequence_ids = ids(&sequence["ids"]);
        let mut prefix = prefix_for(&mut executor, &[]);
        for (row, token) in sequence_ids.iter().copied().enumerate() {
            let mut append = executor
                .append_known(&prefix, TokenChunk::all(&[token]))
                .expect("append");
            prefix = ready(&mut append);
            let actual = logits_for(&mut executor, &prefix);
            assert_close(
                &actual,
                &expected.row(sequence["logits_tensor"].as_str().expect("tensor"), row),
            );
        }
        assert_eq!(
            prefix.logical_length(),
            u64::try_from(sequence_ids.len()).expect("length u64"),
            "{name}"
        );
        assert_eq!(prefix.token_history(), sequence_ids, "{name}");
    }

    let base = ids(&cases["sequences"]["base8"]["ids"]);
    let base_expected = expected.row("base8.logits", 7);
    for partition in cases["append_partitions"].as_array().expect("partitions") {
        let mut prefix = prefix_for(&mut executor, &[]);
        let mut offset = 0_usize;
        for width in ids(partition) {
            let end = offset + usize::try_from(width).expect("partition width usize");
            let mut append = executor
                .append_known(&prefix, TokenChunk::all(&base[offset..end]))
                .expect("partition append");
            prefix = ready(&mut append);
            offset = end;
        }
        assert_eq!(offset, base.len());
        assert_close(&logits_for(&mut executor, &prefix), &base_expected);
    }

    let parent_ids = ids(&cases["sequences"]["prefix4"]["ids"]);
    let parent = prefix_for(&mut executor, &parent_ids);
    let mut a_fork = executor.fork(&parent).expect("fork A");
    let a = ready(&mut a_fork);
    let mut b_fork = executor.fork(&parent).expect("fork B");
    let b = ready(&mut b_fork);
    let mut append_a = executor
        .append_known(&a, TokenChunk::all(&[9, 7, 12]))
        .expect("append A");
    let a = ready(&mut append_a);
    let mut append_b = executor
        .append_known(&b, TokenChunk::all(&[10, 4]))
        .expect("append B");
    let b = ready(&mut append_b);
    assert_close(
        &logits_for(&mut executor, &parent),
        &expected.row("prefix4.logits", 3),
    );
    assert_close(
        &logits_for(&mut executor, &a),
        &expected.row("parent4_branch_a.logits", 6),
    );
    assert_close(
        &logits_for(&mut executor, &b),
        &expected.row("parent4_branch_b.logits", 5),
    );
    let mut append_a2 = executor
        .append_known(&a, TokenChunk::all(&[6]))
        .expect("append A again");
    let a2 = ready(&mut append_a2);
    assert_close(
        &logits_for(&mut executor, &a2),
        &expected.row("parent4_branch_a_append.logits", 7),
    );
    assert_close(
        &logits_for(&mut executor, &b),
        &expected.row("parent4_branch_b.logits", 5),
    );

    let physical = ids(&cases["mask_cases"][0]["physical_ids"]);
    let valid: Vec<bool> = cases["mask_cases"][0]["valid"]
        .as_array()
        .expect("mask")
        .iter()
        .map(|value| value.as_bool().expect("bool"))
        .collect();
    let masked = prefix_for(&mut executor, &[]);
    let mut masked_task = executor
        .append_known(&masked, TokenChunk::masked(&physical, &valid))
        .expect("masked append");
    let masked = ready(&mut masked_task);
    assert_close(
        &logits_for(&mut executor, &masked),
        &expected.row("prefix4.logits", 3),
    );
    let mut all_masked = executor
        .append_known(&masked, TokenChunk::masked(&[99, 98], &[false, false]))
        .expect("all masked append");
    let all_masked = ready(&mut all_masked);
    assert_eq!(all_masked.token_history(), masked.token_history());
    assert_close(
        &logits_for(&mut executor, &all_masked),
        &expected.row("prefix4.logits", 3),
    );

    for score_case in cases["candidate_scores"].as_array().expect("score cases") {
        let prefix = prefix_for(&mut executor, &ids(&score_case["prefix_ids"]));
        let candidate_values: Vec<Vec<u32>> = score_case["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(|candidate| ids(&candidate["ids"]))
            .collect();
        let candidate_refs: Vec<&[u32]> = candidate_values.iter().map(Vec::as_slice).collect();
        let mut score_task = executor
            .score_candidates(&prefix, &candidate_refs)
            .expect("score");
        let scores = ready(&mut score_task);
        for (actual, candidate) in scores
            .iter()
            .zip(score_case["candidates"].as_array().expect("candidates"))
        {
            assert_eq!(
                actual.token_count,
                usize::try_from(candidate["token_count"].as_u64().expect("count"))
                    .expect("count usize")
            );
            assert_score(
                actual.log_probability,
                candidate["sum_log_probability"].as_f64().expect("score"),
            );
        }
        let reference_name = score_case["prefix_reference"].as_str().expect("reference");
        let reference = &cases["sequences"][reference_name];
        let row = usize::try_from(reference["next_logits_row"].as_u64().expect("row"))
            .expect("row usize");
        assert_close(
            &logits_for(&mut executor, &prefix),
            &expected.row(reference["logits_tensor"].as_str().expect("tensor"), row),
        );
    }
}

#[test]
#[ignore = "requires MINIFIELD_MODEL_STATE_ROOT external independent numerical fixture"]
fn sealed_random_tiny_base8_remains_an_independent_cpu_oracle() {
    let state_root = external_path("MINIFIELD_MODEL_STATE_ROOT");
    let config = include_bytes!("fixtures/numerical-lfm-001-config.json").to_vec();
    let weights = include_bytes!("fixtures/numerical-lfm-001-weights.safetensors").to_vec();
    let mut executor = load(&config, weights);
    let prefix = prefix_for(&mut executor, &[1, 3, 7, 11, 6, 14, 2, 5]);
    let expected = ExpectedFile::open(&state_root.join("expected.safetensors"));
    assert_close(
        &logits_for(&mut executor, &prefix),
        &expected.row("full_logits", 7),
    );
}

fn trained_inputs() -> (Vec<u8>, Vec<u8>) {
    let artifact = external_path("MINIFIELD_TRAINED_TINY_ARTIFACT");
    (
        fs::read(artifact.join("model/config.json")).expect("trained config"),
        fs::read(artifact.join("model/model.safetensors")).expect("trained weights"),
    )
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT"]
fn trained_tiny_capacity_masks_invalid_ids_and_foreign_instance_are_rejected_prepublication() {
    let (config, weights) = trained_inputs();
    let mut limited = load_with_backend(
        &config,
        weights,
        backend(),
        Lfm2ExecutionLimits {
            max_logical_tokens: 1,
        },
    );
    assert!(matches!(
        limited.prefill(TokenChunk::all(&[1, 3])),
        Err(ExecutorError::OutOfBounds(_))
    ));
    let prefix = prefix_for(&mut limited, &[1]);
    let before = logits_for(&mut limited, &prefix);
    assert!(matches!(
        limited.append_known(&prefix, TokenChunk::all(&[3])),
        Err(ExecutorError::OutOfBounds(_))
    ));
    assert!(matches!(
        limited.score_candidates(&prefix, &[&[3]]),
        Err(ExecutorError::OutOfBounds(_))
    ));
    let mut empty_scores = limited
        .score_candidates(&prefix, &[&[]])
        .expect("empty candidate at capacity");
    let empty_scores = ready(&mut empty_scores);
    assert_eq!(empty_scores.len(), 1);
    assert_eq!(empty_scores[0].token_count, 0);
    assert!(empty_scores[0].log_probability.abs() <= f32::EPSILON);

    assert!(matches!(
        limited.append_known(&prefix, TokenChunk::all(&[3, u32::MAX])),
        Err(ExecutorError::OutOfBounds(_))
    ));
    assert!(matches!(
        limited.append_known(&prefix, TokenChunk::masked(&[3], &[])),
        Err(ExecutorError::InvalidArgument(_))
    ));
    let mut all_masked = limited
        .append_known(&prefix, TokenChunk::masked(&[u32::MAX], &[false]))
        .expect("invalid masked token is not gathered");
    let all_masked = ready(&mut all_masked);
    assert_eq!(all_masked.token_history(), prefix.token_history());
    assert_close(&logits_for(&mut limited, &all_masked), &before);
    assert_close(&logits_for(&mut limited, &prefix), &before);

    let (config, weights_a) = trained_inputs();
    let weights_b =
        fs::read(external_path("MINIFIELD_TRAINED_TINY_ARTIFACT").join("model/model.safetensors"))
            .expect("second trained weights");
    let mut first = load(&config, weights_a);
    let mut second = load(&config, weights_b);
    let first_prefix = prefix_for(&mut first, &[1]);
    assert!(matches!(
        second.append_known(&first_prefix, TokenChunk::all(&[3])),
        Err(ExecutorError::WrongBackend)
    ));
    assert!(matches!(
        second.next_logits(&first_prefix),
        Err(ExecutorError::WrongBackend)
    ));
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT"]
#[allow(clippy::too_many_lines)]
fn deferred_model_tasks_retain_sources_reject_stale_readback_and_quarantine_fence_failure() {
    let (config, weights) = trained_inputs();
    let state = Rc::new(DeferredState::default());
    let mut executor = load_with_backend(
        &config,
        weights,
        DeferredBackend::new(Rc::clone(&state)),
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    );

    let prefix = prefix_for(&mut executor, &[1, 3]);
    let sibling = prefix.clone();
    state.fence_polls.set(3);
    state.fail_cancel.set(true);
    let mut pending = executor
        .append_known(&prefix, TokenChunk::all(&[7]))
        .expect("append task");
    assert!(matches!(pending.poll_step(), CompletionPoll::Pending));
    assert!(matches!(pending.poll_step(), CompletionPoll::Pending));
    drop(prefix);
    drop(sibling);
    assert!(matches!(
        pending.cancel(),
        Err(ExecutorError::BackendFailure(_))
    ));
    assert!(matches!(
        pending.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed))
    ));
    assert!(
        executor
            .resource_report()
            .expect("live deferred executor")
            .cache_bytes
            > 0,
        "the unresolved fence retains staged cache buffers"
    );
    drop(pending);
    assert_eq!(
        state.dropped_pending_fences.get(),
        0,
        "the task transferred the fence to retirement instead of dropping it"
    );
    for _ in 0..4 {
        executor.poll_retired().expect("retirement poll");
    }
    let retired = executor.resource_report().expect("report after retirement");
    assert_eq!(retired.pending_operation_bytes, 0);
    assert_eq!(retired.cache_bytes, 0);

    state.fail_cancel.set(false);
    state.fence_polls.set(3);
    let parent = prefix_for(&mut executor, &[1, 3]);
    let sibling = parent.clone();
    let mut dropped = executor
        .append_known(&parent, TokenChunk::all(&[7]))
        .expect("drop task");
    assert!(matches!(dropped.poll_step(), CompletionPoll::Pending));
    assert!(matches!(dropped.poll_step(), CompletionPoll::Pending));
    drop(parent);
    drop(sibling);
    drop(dropped);
    assert!(
        executor
            .resource_report()
            .expect("report before retired drop")
            .cache_bytes
            > 0
    );
    for _ in 0..4 {
        executor.poll_retired().expect("drop retirement poll");
    }
    assert_eq!(
        executor
            .resource_report()
            .expect("report after dropped task retirement")
            .cache_bytes,
        0
    );
    assert_eq!(state.dropped_pending_fences.get(), 0);

    state.fence_polls.set(0);
    state.readback_polls.set(0);
    let readback_prefix = prefix_for(&mut executor, &[1]);
    state.readback_polls.set(3);
    let mut dropped_readback = executor
        .next_logits(&readback_prefix)
        .expect("pending readback");
    assert!(matches!(
        dropped_readback.poll_step(),
        CompletionPoll::Pending
    ));
    drop(dropped_readback);
    assert_eq!(
        state.dropped_pending_readbacks.get(),
        1,
        "dropping a pending readback does not publish a host result"
    );
    state.readback_polls.set(2);
    let prefix = prefix_for(&mut executor, &[1]);
    let mut logits = executor.next_logits(&prefix).expect("deferred readback");
    assert!(matches!(logits.poll_step(), CompletionPoll::Pending));
    executor
        .advance_backend_generation()
        .expect("invalidate backend generation");
    assert!(matches!(logits.poll_step(), CompletionPoll::Pending));
    assert!(matches!(logits.poll_step(), CompletionPoll::Pending));
    assert!(matches!(
        logits.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::StaleBuffer))
    ));
    drop(logits);
    assert!(matches!(
        executor.next_logits(&prefix),
        Err(ExecutorError::StaleBuffer)
    ));
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT"]
fn dropping_executor_handle_during_deferred_prefill_keeps_task_context_alive() {
    let (config, weights) = trained_inputs();
    let state = Rc::new(DeferredState::default());
    state.fence_polls.set(2);
    let mut executor = load_with_backend(
        &config,
        weights,
        DeferredBackend::new(Rc::clone(&state)),
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    );
    let mut task = executor.prefill(TokenChunk::all(&[1])).expect("prefill");
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    drop(executor);
    let prefix = ready(&mut task);
    assert_eq!(prefix.logical_length(), 1);
    assert_eq!(state.dropped_pending_fences.get(), 0);
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT"]
fn deferred_fence_submission_failure_after_model_ops_quarantines_executor_without_publication() {
    let (config, weights) = trained_inputs();
    let state = Rc::new(DeferredState::default());
    let mut executor = load_with_backend(
        &config,
        weights,
        DeferredBackend::new(Rc::clone(&state)),
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    );
    state.fail_fence.set(true);
    let mut task = executor
        .prefill(TokenChunk::all(&[1]))
        .expect("prefill task is constructed before backend work");
    assert!(matches!(task.poll_step(), CompletionPoll::Pending));
    assert!(matches!(
        task.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::BackendFailure(_)))
    ));
    assert!(matches!(
        executor.resource_report(),
        Err(ExecutorError::BackendFailure(_))
    ));
    let mut rejected = executor
        .prefill(TokenChunk::all(&[3]))
        .expect("token validation does not publish work");
    assert!(matches!(
        rejected.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::BackendFailure(_)))
    ));
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_ARTIFACT"]
fn deferred_branch_copy_failure_quarantines_recorded_work_and_retains_base() {
    let (config, weights) = trained_inputs();
    let state = Rc::new(DeferredState::default());
    let mut executor = load_with_backend(
        &config,
        weights,
        DeferredBackend::new(Rc::clone(&state)),
        Lfm2ExecutionLimits {
            max_logical_tokens: 8,
        },
    );
    let mut base_task = executor
        .prefill_choice_base(TokenChunk::all(&[1, 3]))
        .expect("choice base prefill");
    let base = ready(&mut base_task);
    // One branch copy is recorded before the injected failure, so the failing
    // call leaves destination buffers referenced by work with no completion
    // boundary.
    state.fail_after_n_copies.set(Some(1));
    assert!(matches!(
        executor.append_choice_logits(&base, TokenChunk::all(&[7]), &[2, 5]),
        Err(ExecutorError::BackendFailure(_))
    ));
    assert!(
        matches!(
            executor.resource_report(),
            Err(ExecutorError::BackendFailure(_))
        ),
        "recorded branch copies without a fence must quarantine the executor"
    );
    drop(base);
}
