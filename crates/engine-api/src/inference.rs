//! Finite backend-neutral inference operations.

use std::rc::Rc;

use crate::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendLease, ExecutorError,
    FenceRetirement, GatedShortConvSpec, GqaSpec, InferenceCompletion, PackedHeadSpec, RectCopy2d,
    ResourceReport, Result, RotarySpec, Shape, TokenId,
};

/// Row selector source for gather operations.
///
/// `Host` carries caller-held token ids. `Device` points at a backend-resident
/// f32 `[T]` buffer of exact integer row indices, typically produced by
/// [`InferenceOps::argmax`], so a sampled token can feed an embedding gather
/// without a host roundtrip. A non-finite device id produces a non-finite
/// output row on deferred backends; backends that validate operands eagerly
/// may reject it at call time instead.
#[derive(Clone, Copy)]
pub enum TokenIds<'a, B: InferenceOps + ?Sized> {
    Host(&'a [u32]),
    Device(&'a B::Buffer),
}

/// Finite, backend-neutral inference operation contract.
///
/// Implementors own buffer storage and expose pollable fence/readback completion types.
/// A device backend may enqueue kernel methods; it retains every queued source, destination, and
/// staging buffer until the next submitted fence or readback completion releases it. Shared model
/// code must poll those completions and must not require a backend-wide blocking read or sync.
/// This intentionally names only model-execution primitives; it is not a tensor graph or general
/// expression system.
pub trait InferenceOps {
    type Buffer;
    type Fence: InferenceCompletion<Output = ()>;
    type Readback: InferenceCompletion<Output = Vec<f32>>;
    type FenceRetirement: FenceRetirement<Self::Fence, Self::Buffer> + 'static;

    fn identity(&self) -> BackendIdentity;

    /// Return this backend's non-forgeable actual-instance lease and current generation.
    fn lease(&self) -> BackendLease;
    /// Return the backend-owned retirement queue for unresolved submission fences.
    fn fence_retirement(&self) -> Rc<Self::FenceRetirement>;
    /// Advance backend-owned abandoned fence retirement without a blocking synchronization.
    fn poll_retired_fences(&self) -> Result<()>;
    fn capabilities(&self) -> BackendCapabilities;
    fn resource_report(&self) -> ResourceReport;

    /// Highest total backend-accounted storage since construction, including
    /// completion results and physical pools. Driver memory isn't included.
    fn peak_accounted_bytes(&self) -> Result<u64> {
        Err(ExecutorError::Unsupported(
            "backend has no peak resource counter",
        ))
    }
    fn advance_generation(&mut self) -> Result<()>;

    /// Allocate an f32 buffer with an explicit lifetime/accounting class.
    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Upload f32 values into an explicitly classified backend-owned buffer.
    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Upload raw bytes into an explicitly classified backend-owned buffer.
    /// Used for opaque packed payloads such as ternary code streams; the bytes
    /// are not interpreted as scalars by the buffer contract.
    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<Self::Buffer>;

    /// Allocate transient work storage. Model/prefix loaders must use the classified form.
    fn allocate_f32(&mut self, shape: Shape) -> Result<Self::Buffer> {
        self.allocate_f32_classified(shape, AllocationClass::Scratch)
    }

    /// Allocate f32 storage whose initial contents are unspecified. Callers
    /// must fully overwrite the buffer before any consumer reads it. The
    /// default forwards to the zero-initializing classified allocation;
    /// backends that pay a per-allocation zeroing cost can override it.
    fn allocate_f32_uninit(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        self.allocate_f32_classified(shape, class)
    }

    /// Upload transient work storage. Model/prefix loaders must use the classified form.
    fn upload_f32(&mut self, shape: Shape, values: &[f32]) -> Result<Self::Buffer> {
        self.upload_f32_classified(shape, values, AllocationClass::Scratch)
    }

    /// Submit a completion boundary without forcing a device-wide synchronous wait.
    fn fence(&self) -> Result<Self::Fence>;

    /// Submit host-visible f32 readback. The completion owns output until polling is ready;
    /// a device backend retains source storage and its staging buffer until that point.
    fn read_f32_async(&self, buffer: &Self::Buffer) -> Result<Self::Readback>;

    fn copy(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()>;

    /// Copy one checked rectangular range between distinct packed rank-two buffers.
    /// Overlap on the same physical allocation is unsupported until a separately qualified
    /// view/alias contract exists.
    fn copy_rect_2d(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        rectangle: RectCopy2d,
    ) -> Result<()>;

    /// Row-wise argmax over f32 `[T, V]` logits: `output[t]` is the index of
    /// the first strict maximum in row `t`, written as an exact f32 integer,
    /// or NaN when the row contains any non-finite element. `output` is f32
    /// `[T]`. `V` must be at most `1 << 24` so indices stay exactly
    /// representable. The NaN marker lets a tiny readback double as the
    /// finiteness check for the whole logits row.
    fn argmax(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()>;

    /// `argmax` with a per-element candidate mask: only positions whose bit
    /// is set in `mask` may win. `mask` is one bit per element, LSB-first
    /// inside each u64 word (`mask[i / 64]` bit `i % 64`); its length must be
    /// `ceil(width / 64)`. Non-finite values at allowed positions still
    /// poison the row to NaN; masked-out values are skipped entirely. A row
    /// with no allowed candidate produces NaN. The default implementation
    /// reports `Unsupported` rather than silently ignoring the constraint.
    fn argmax_masked(
        &self,
        _output: &mut Self::Buffer,
        _input: &Self::Buffer,
        _mask: &[u64],
    ) -> Result<()> {
        Err(ExecutorError::Unsupported(
            "masked argmax is not implemented for this backend",
        ))
    }

    fn gather_rows(
        &self,
        output: &mut Self::Buffer,
        table: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()>;

    /// Column-wise gather over a contiguous f32 `[rows, width]` input:
    /// `output[r, k] = input[r, columns[k]]`. `output` is f32
    /// `[rows, columns.len()]`. Caller order and duplicate column ids are
    /// preserved, so `columns` acts as a typed selector list. An empty
    /// `columns` is a valid no-op producing `[rows, 0]` output. Out-of-range
    /// ids are rejected. Unlike [`InferenceOps::gather_rows`], ids always
    /// arrive as host `TokenId`s; callers needing only a few columns can use
    /// this to avoid a full-width host readback.
    fn gather_columns(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        columns: &[TokenId],
    ) -> Result<()>;

    /// Gather packed ternary rows and dequantize them into an f32 `[ids, K]` output.
    ///
    /// `codes` is a U8 `[rows, K/4]` buffer in `minifield.ternary.v1` layout: each
    /// group of 128 weights occupies 32 consecutive bytes, and weight `j` of a
    /// group sits at byte `j / 4`, bits `2 * (j % 4)`. `scales` is the f32
    /// `[rows, K/128]` group-scale stream decoded from the packed file's FP16
    /// scales. Decoded weight `w = (code - 1) * scale`.
    fn packed_gather_rows(
        &self,
        output: &mut Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()>;

    /// Packed ternary linear: `output[t, r] = sum_k input[t, k] * w[r, k]` where
    /// `w` is the `minifield.ternary.v1` dequantization of `codes`/`scales` as
    /// documented on [`InferenceOps::packed_gather_rows`]. `input` is f32
    /// `[T, K]`, `codes` is U8 `[R, K/4]`, `scales` is f32 `[R, K/128]`, and
    /// `output` is f32 `[T, R]`. Group scales apply inside each 128-weight
    /// group, matching the reference dequantized matvec.
    fn packed_linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()>;

    /// Paired packed ternary linear over one shared input:
    /// `out_a[t, r] = sum_k input[t, k] * wa[r, k]` and
    /// `out_b[t, r] = sum_k input[t, k] * wb[r, k]`. Both weight sets share the
    /// `minifield.ternary.v1` layout and must have identical `[R, K]` shapes, so
    /// `out_a` and `out_b` have identical `[T, R]` shapes. Semantically equal to
    /// two `packed_linear` calls; backends may issue them as one dispatch.
    #[allow(clippy::too_many_arguments)]
    fn packed_linear_pair(
        &self,
        out_a: &mut Self::Buffer,
        out_b: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()>;

    /// Packed ternary linear over an on-the-fly `SiLU(gate) * up` activation:
    /// `output[t, r] = sum_k (silu(gate[t,k]) * up[t,k]) * w[r, k]` where `w` is
    /// the `minifield.ternary.v1` dequantization of `codes`/`scales`. `gate` and
    /// `up` are f32 `[T, K]`, `codes` is U8 `[R, K/4]`, `scales` is f32
    /// `[R, K/128]`, and `output` is f32 `[T, R]`. Semantically equal to a
    /// `swiglu` into scratch followed by `packed_linear`.
    fn packed_swiglu_linear(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()>;

    /// Paired packed linear over one shared input with a fused `SwiGLU`
    /// epilogue: `output[t, r] = silu(a[t, r]) * b[t, r]` where `a` and `b`
    /// are the `packed_linear` results of `input` against `codes_a`/`scales_a`
    /// and `codes_b`/`scales_b` respectively. Both weight sets share the
    /// `minifield.ternary.v1` layout and must have identical `[R, K]` shapes,
    /// and `output` is f32 `[T, R]`. Semantically equal to
    /// `packed_linear_pair` followed by `swiglu`; backends may fuse the
    /// activation into the projection epilogue.
    #[allow(clippy::too_many_arguments)]
    fn packed_swiglu_pair(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()>;

    /// Whether this backend consumes ternary code streams repacked into the
    /// two-weight LUT2 layout. When true, loaders may call
    /// `repack_ternary_lut2` once per packed stream at load and route the
    /// resulting buffers to the `*_lut2` ops. False keeps everything on the
    /// raw `minifield.ternary.v1` decode.
    fn supports_ternary_lut2(&self) -> bool {
        false
    }

    /// Rearrange one `minifield.ternary.v1` code stream into the backend's
    /// two-weight LUT2 layout: each pair of two-bit codes becomes a nibble
    /// indexing an activation pair table. Lossless, same byte count, done
    /// once per tensor at load. The result is only meaningful to the
    /// `*_lut2` ops; it is not interchangeable with raw ternary codes.
    fn repack_ternary_lut2(&mut self, codes: &Self::Buffer) -> Result<Self::Buffer> {
        let _ = codes;
        Err(ExecutorError::Unsupported(
            "ternary lut2 repack is unsupported by backend",
        ))
    }

    /// `packed_linear` over a LUT2-repacked ternary code stream. Callers must
    /// keep the raw `codes` stream for paths that decode it directly. The
    /// default rejects the layout; supporting backends override.
    fn packed_linear_lut2(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        let _ = (output, input, codes, scales);
        Err(ExecutorError::Unsupported(
            "packed lut2 linear is unsupported by backend",
        ))
    }

    /// `packed_swiglu_pair` over LUT2-repacked ternary code streams: the
    /// shared activation table is built once and consumed by both streams.
    /// Same call contract as `packed_swiglu_pair` with LUT2 codes in place of
    /// raw ternary codes.
    #[allow(clippy::too_many_arguments)]
    fn packed_swiglu_pair_lut2(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        let _ = (output, input, codes_a, scales_a, codes_b, scales_b);
        Err(ExecutorError::Unsupported(
            "packed lut2 swiglu pair is unsupported by backend",
        ))
    }

    /// Fused residual add plus row RMS norm: `sum = left + right` and
    /// `normed` is the row RMS norm of `sum` scaled by `weight`, with `sum`
    /// and `normed` as distinct `[T, C]` outputs. `left`, `right` are
    /// `[T, C]` inputs and `weight` is `[C]`. Semantically equal to `add`
    /// followed by `row_rms_norm` over the sum.
    fn add_row_rms_norm(
        &self,
        sum: &mut Self::Buffer,
        normed: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()>;

    /// Fused per-head RMS norm plus split-half rotary for query and key rows:
    /// `query_out` is `rope(head_rms_norm(query, query_weight))` and `key_out`
    /// is `rope(head_rms_norm(key, key_weight))`, evaluated per `[token, head]`
    /// row with `positions` per token. `rope` carries the query head geometry
    /// and theta; `key_value_heads` carries the key head geometry. Semantically
    /// equal to `head_rms_norm` then `split_half_rotary` on each tensor.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<()>;
    fn add(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()>;
    fn multiply(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()>;
    fn linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
    ) -> Result<()>;
    fn row_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()>;

    /// Apply RMS normalization independently to every [token, head] row.
    fn head_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()>;

    /// Split-half rotary embedding for packed `[tokens, heads * head_dim]` values.
    fn split_half_rotary(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()>;

    /// Append packed K/V rows to the supplied caches and calculate causal GQA output.
    /// The cache length changes only after all shapes and input bounds have been accepted.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<()>;

    /// Apply B*V gated short convolution and update a [width-1, hidden] rolling
    /// U history. `projection` is the fused `[tokens, 3 * hidden]` in-projection
    /// output: token `t`'s B, C, and V rows sit at offsets `t * 3h + {0, h, 2h}`.
    fn gated_short_convolution(
        &self,
        output: &mut Self::Buffer,
        projection: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()>;

    /// Apply SiLU(gate) * up elementwise over equal contiguous layouts.
    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()>;
}
