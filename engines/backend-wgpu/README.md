# backend-wgpu

`InferenceOps` implementation over wgpu 30 for the portable executor. F32 plus
the `minifield.ternary.v1` and `minifield.nf4.v1` packed paths; it runs the
full LFM2.5 decode loop on device, including the greedy sample.

## What it does

- One shared `CommandEncoder` per backend. Portable op calls record compute
  dispatches, buffer copies, and deferred staging maps; `fence()` and
  `read_f32_async()` are the only submission boundaries.
- Completion is nonblocking. `poll_step` pumps `device.poll(PollType::Poll)`
  and observes `on_submitted_work_done` serial confirmations and deferred
  `map_buffer_on_submit` callbacks. There is no `PollType::Wait` and no
  blocking map receive.
- Readback uses pooled `MAP_READ` staging buffers. A readback dropped or
  cancelled before its map resolves parks the staging buffer until the
  callback lands, so a buffer is never recycled under an in-flight map.
- Kernel parameters travel through a shared 256-byte-slot uniform ring with
  dynamic offsets. No push constants, no native immediates, no subgroups, no
  f16. A ring wrap mid-batch forces a submission because `write_buffer`
  ordering only exists across submission boundaries.
- Buffers come from a size-classed pool (powers of two under 1 MiB, sixteenth
  steps above). A dropped buffer is quarantined until the submission serial
  that could still reference it is confirmed complete, then recycled.

## Kernels

WGSL lives in `src/kernels.rs`: fill, binary add/multiply, 2D rect copy,
row gather, lane-grouped shared-reduction GEMV (m == 1) with vec4 activation
loads and adaptive lanes-per-row, 16x16 tiled GEMM (m > 1), row/head RMS norm,
host-table split-half RoPE, per-token causal GQA with recorded cache appends,
gated short convolution (single-dispatch `conv_step` for decode; staged
history assembly for multi-token prefill), SwiGLU, packed ternary GEMV /
pair / SwiGLU / gather, the same four packed ops for `minifield.nf4.v1`
plus small-batch variants that decode each codes word once per 8-row tile.
A load-time repack kernel can rewrite ternary FFN code streams into the
backend-private LUT2 pair-nibble layout (see docs/ffn-prefill-experiments.md
round 5 for the encoding); `packed_linear_lut2` and the paired
`packed_swiglu_pair_lut2` consume those streams through the 32x64
grouped-lookup tile while raw codes stay resident for short-row paths.
NF4 prefill with at least 32 rows uses `nf4_prefill.wgsl`: a 32x32 output
tile shares decoded weights and activations, with a 2x2 register fragment
per invocation. Linear, paired linear, and fused SwiGLU share the tiled body.
The attention path includes
a batched causal GQA covering a block of prefill tokens in one dispatch,
fused add-RMS-norm, fused QK-norm+RoPE, and a two-stage argmax for wide rows. `argmax_masked` adds a read-only u32
candidate bitset binding (LSB-first over `ceil(width / 64)` u64 words on the
host) gated by a params flag: masked-out elements are skipped before the
finiteness check, so their NaN or infinity cannot poison the row, and a fully
masked row yields NaN. The unmasked path rebinds the logits buffer at the
mask slot and clears the flag, so both paths share one pipeline. Grids
flatten workgroup IDs so n up to 65_536 and beyond fits WebGPU's 65_535
workgroups-per-dimension limit.

## Diagnostics

`MINIFIELD_WGPU_STATS` (any non-empty value) enables host-side timing and
per-kernel dispatch counts, printed when the device drops. `tests/kernel_bench.rs`
is a developer microbenchmark at real LFM2.5-230M shapes:

    cargo test --release -p minifield-backend-wgpu --test kernel_bench -- --nocapture

## Device limits

The device requests the adapter's maximum `max_storage_buffer_binding_size`
and `max_buffer_size`; WebGPU defaults (128 MiB / 256 MiB) cannot hold a
268 MB f32 lm_head. Everything else stays at the portable default limits.

## Tests

`tests/parity.rs` compares every op against `crates/backend-cpu` on synthetic
tensors and skips cleanly when no adapter exists:

    cargo test -p minifield-backend-wgpu

## Deferred

Subgroup-intrinsic reductions, f16 compute, wasm32 hosting, and wider decode
fusion (fusing the row-norm epilogue into consuming GEMVs; multi-token decode
batches behind one submission) are follow-up tasks. The adapter exposes
`SUBGROUP` and `TIMESTAMP_QUERY`, but kernels stay portable for now.
