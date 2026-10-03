# WebGPU backend

`InferenceOps` over wgpu 30, with F32 arithmetic, canonical ternary/NF4/signed-INT8 packed weights, and optional ternary LUT2 repacks. The same backend runs natively and through browser WebGPU.

## Layout

Operation families live in `dense`, `packed`, `normalization`, `attention_convolution`, and `sampling`. `dispatch` implements the portable trait. `device` owns allocation pools, submission, and polling; `completion` owns fence/readback lifetimes.

WGSL bodies live in `src/shaders/`. `kernels.rs` registers them. Research shaders are under `src/shaders/experimental/` and require `experimental-kernels`. The explicit experimental API carries a typed kernel choice and validates the representation it consumes.

## Execution

Operations record into a shared command encoder. Fences and async readbacks submit batches. `poll_step` observes device progress and map callbacks without a blocking wait. A host must pump completions; browser callers yield to the event loop.

Buffers return to a pending queue when their final logical owner drops. Submission serials quarantine them until completion, then pools can reuse them. Pending/cancelled readback maps retain staging until the callback resolves.

Uniform parameters use a bounded host/device ring. Ring wrap submits recorded work before overwriting slots. Device limits, caller allocation limits, and total accounted bytes bound admission.

## Policies and accounting

Construct with `WgpuOptions` for diagnostics or experimental NF4 staging precision. Default staging is F32. F16 modes need the experimental feature and adapter SHADER_F16 support. Core code doesn't parse environment variables.

`dispatch_counts`, `stats`, `adapter_info`, `resource_report`, and `peak_accounted_bytes` provide inspectable evidence. Physical size classes, padding, pools, staging, retained result buffers, and fixed uniforms remain charged while owned. Logical weights/caches are classified separately; extra physical storage is reported as scratch. GPU driver/pipeline allocations and process RSS aren't measured by this report.

INT8 uses its own decode headers with the shared F32 GEMM template at every row count. Its group-128 scales and signed byte layout are explicit, and experimental F16 staging doesn't change this arithmetic path.

Raw packed APIs accept canonical codes only. LUT2/PN4 buffers have separate layout tags. Small-row paths retain canonical weights, while qualified shapes can use the explicitly admitted repack. The executor's duplicate-weight budget controls repack admission.

Ternary row gather assigns each invocation one packed word and writes its 16 decoded F32 values. The aligned word shares one ID validation and group-scale load; invalid device IDs fill all 16 outputs with the existing invalid-ID value.

## Verification

```sh
scripts/check.sh gpu
cargo test --release --locked -p minifield-backend-wgpu --test kernel_bench -- --ignored --nocapture
```

Run commands from the repository root. The required GPU tier runs serial CPU/GPU parity, low-bit conformance, and allocation/layout regressions. Missing adapters fail that tier. Portable workspace tests may skip only a genuinely unavailable adapter.

[FFN experiments](../../docs/ffn-prefill-experiments.md) preserve historical kernel comparisons. [Bundle qualification](../../tools/qualification/README.md) records actual assets, modes, dispatch counts, numerical parity, and matched timings. A native pass doesn't establish browser support.
