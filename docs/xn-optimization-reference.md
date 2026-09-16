# xn: potential runtime optimization reference

Reviewed September 14, 2026. Status: source research and proposed comparisons. No dependency was added, source was copied into runtime, or upstream build/benchmark was run.

[gradium-ai/xn](https://github.com/gradium-ai/xn) is a Rust inference framework with CPU quantization kernels and optional Metal, CUDA, Vulkan, and WebGPU backends. Its workspace includes `xn-core` and speech-model code in `xn-moshi`; a separate CUDA FlashAttention crate is excluded from the default workspace. These are useful implementation references for our Rust runtime.

Pin this review to commit [`91c700563e12dc1463a13903fe7b11472da02153`](https://github.com/gradium-ai/xn/tree/91c700563e12dc1463a13903fe7b11472da02153), whose workspace version is `0.2.4`. The latest commit inspected gates WASM relaxed-SIMD instructions by target feature. Recheck changes and compatibility before using a newer revision.

Our target remains **batch 1, one resident model, and one reusable KV allocation**. Multiple CPU threads or GPU commands can accelerate that single sequence. The [experiment index](../experiments/README.md) owns order and status; the [runtime procedure](procedure.md) owns acceptance.

## Code worth studying

All source links below point to the reviewed commit.

| Reference | What to try in our runtime |
| --- | --- |
| [Q8 weight repacking](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/quantized/repack.rs) | Interleave 4 output columns in runs of 8 values so a kernel can stream adjacent weights. Its reversible layout preserves the original Q8 blocks and supports an ARM/NEON path. Compare layout, activation-quantization cost, and matrix-vector execution separately. Count load-time repacking, temporary copies, and retained memory. |
| [WASM SIMD quantized kernels](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/quantized/simd128.rs) | Study packed Q4/Q8 dot products, vector unpacking, and scale application against our scalar reference. Build a portable SIMD variant and qualify optional relaxed SIMD separately. Compare numerical error as well as time. |
| [CPU single-query attention](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/cpu_backend.rs#L1251) and [dispatch conditions](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/ops.rs#L275) | Online softmax accumulates the weighted result without materializing a score vector for every head. It accepts a view of the populated cache with a larger backing capacity. Adapt head mapping for GQA, preserve mask/position semantics, and compare against composed attention. |
| [Persistent CPU thread pool](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/threadpool.rs) | Reuse workers across tiny operations and partition output work. Compare 1, 2, and 4 kernel threads at batch 1 where supported. Include idle CPU use, sustained power/thermal behavior, and host responsiveness when choosing spin/park behavior. |
| [GPU command recording, buffer pooling, and profiling](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/webgpu_backend/mod.rs) | Record several operations before submission, cache pipelines, and reuse buffers after their GPU consumers finish. Track dispatches, submissions, copies, readbacks, pool retention, and host wait time. Set a physical memory cap for our pool. |
| [Preallocated KV cache and active views](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/models/kv_cache.rs#L3) | Append into existing storage and expose only the active prefix. Add our own logical truncation, prefix identity/invalidation, cancellation handling, and bounded retention. This supports the planned prefix-plus-current-tail allocation. |

## Compatibility limits

**The fused decode path needs GQA adaptation.** Its public API requires equal query and KV head counts. Falcon-E has 16 query heads and 2 KV heads. Map groups of query heads to the compact cache directly; expanding KV heads would increase traffic and temporary storage. The inspected CPU kernel supports head dimensions up to 256, which includes our candidates' 128.

The generic [Llama attention implementation](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/models/llama.rs#L175) concatenates growing KV tensors, expands KV heads, and computes composed attention. It doesn't demonstrate the preallocated cache and fused decode path wired into a compatible Falcon decoder. The rotating cache elsewhere in the repository changes context retention and needs model-specific sliding-window semantics.

The inspected [WebGPU device setup](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/src/webgpu_backend/mod.rs#L214) requires native push constants. Its execution path also uses blocking polling and synchronous readback. Browser use needs compatible parameter buffers, asynchronous completion, and our Worker lifecycle. The GPU path targets F32 and falls back through the host for other dtypes. These source patterns don't establish a browser-ready backend or 16-bit KV storage.

The [WASM build configuration](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/.cargo/config.toml) still enables `+relaxed-simd`. The new source guard allows a build without those instructions, but producing that artifact requires explicit build settings and target-browser validation. The native thread pool also needs separate browser threading support; a native measurement doesn't establish WASM worker behavior.

The quantized references use GGML formats. Q8 repacking requires positive matrix dimensions with output width divisible by 4 and input width divisible by 32. Those formats and their activation quantization don't establish Falcon's packed ternary compatibility. Preserve exact model/export semantics and keep any Q4/Q8 candidate comparison explicit.

## Comparisons mapped to the experiment protocols

1. **[Decoder baseline and profiling](../experiments/0001-decoder-baseline.md):** add per-operation timing and memory attribution once the exact decoder works. On the target device, identify whether weight reads, attention, dispatch, allocation, or readback dominates. Use that evidence to select an xn technique.
2. **[Prefix reuse](../experiments/0002-prefix-session-reuse.md) and [bounded attention](../experiments/0003-bounded-attention.md):** compare cache appends/active views and GQA-aware single-query attention against the numerical reference at the same context lengths. Measure both active bytes and allocated capacity. Keep prefill correctness and causal masking covered separately.
3. **[Packed execution](../experiments/0004-packed-weight-execution.md):** compare scalar and SIMD matrix-vector kernels on exact exported shapes. Trial repacking only for compatible formats, with identical effective weights and activation semantics. Include setup costs, scratch allocations, and complete-action latency.
4. **Scheduling and GPU overhead:** compare supported thread counts under [0004](../experiments/0004-packed-weight-execution.md), and command submissions/buffer reuse under [0003](../experiments/0003-bounded-attention.md). Keep batch 1 fixed. Retain a change only when device gains survive repeated product requests and the existing numerical, behavioral, memory, and responsiveness gates.

Use the upstream [attention benchmark](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/examples/sdpa_decode_bench.rs) and [correctness cases](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/tests/tensor_tests.rs#L1069) as harness references. The benchmark reports the minimum average from repeated rounds; our device reports still need p50/p95, sample counts, and sustained runs.

The [WASI benchmark](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/xn-core/examples/wasm_benchmarks.rs) rotates 24 Q8 weight matrices (about 102 MiB total), a useful pattern for exposing weight-streaming cost beyond a hot matrix. It uses 125 activation rows and Wasmtime. Add `m = 1` decode shapes, realistic values, and actual-browser measurements before drawing batch-1 browser conclusions; select a working set appropriate to the measured device.

The inspected [CI workflow](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/.github/workflows/rust-ci.yml) defines default-feature checks/tests across native operating systems. It doesn't establish optional GPU/backend or browser qualification. This review inspected source and test definitions only.

## Adoption boundary

Keep xn as a source reference and potential comparison engine until a measured bottleneck justifies a focused trial. Isolate any future dependency or borrowed kernel under `engines/`, pin its revision, and preserve provenance. The project declares MIT/Apache-2.0 licensing in [Cargo.toml](https://github.com/gradium-ai/xn/blob/91c700563e12dc1463a13903fe7b11472da02153/Cargo.toml); retain applicable license files and upstream notices, including the repacking code's GGML attribution, if code is reused.

No framework migration, new model format, or performance gain is accepted by adding this reference.
