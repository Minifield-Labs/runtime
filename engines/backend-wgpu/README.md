# backend-wgpu

`InferenceOps` implementation over wgpu 30 for the portable executor. F32 only;
it is the GPU foundation for the ternary-execution plan, not the ternary path
itself.

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
row gather, shared-memory-reduction GEMV (m == 1) with a vec4 fast path,
16x16 tiled GEMM (m > 1), row/head RMS norm, host-table split-half RoPE,
per-token causal GQA with recorded cache appends, two-pass gated short
convolution with rolling history, and SwiGLU. Grids flatten workgroup IDs so
n up to 65_536 and beyond fits WebGPU's 65_535 workgroups-per-dimension limit.

## Device limits

The device requests the adapter's maximum `max_storage_buffer_binding_size`
and `max_buffer_size`; WebGPU defaults (128 MiB / 256 MiB) cannot hold a
268 MB f32 lm_head. Everything else stays at the portable default limits.

## Tests

`tests/parity.rs` compares every op against `crates/backend-cpu` on synthetic
tensors and skips cleanly when no adapter exists:

    cargo test -p minifield-backend-wgpu

## Deferred

Ternary weight decode, fused ternary kernels, wasm32 hosting, and subgroup or
f16 fast paths are separate follow-up tasks.
