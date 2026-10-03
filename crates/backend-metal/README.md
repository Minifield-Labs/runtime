# Native Metal backend

`MetalBackend` implements `InferenceOps` and `EncoderOps` directly through Apple's Metal framework. It owns its device, queue, buffers, MSL pipelines and completion tracking. It has no WebGPU dependency.

```rust,ignore
use minifield_backend_metal::MetalBackend;
use minifield_engine_api::ResourceLimits;

let backend = MetalBackend::new(1, ResourceLimits {
    max_allocation_bytes: 2 << 30,
    max_total_bytes: 8 << 30,
    max_pending_operations: 64,
})?;
```

Construction requires macOS and an actual Metal device. Other targets compile a construction rejection, preserving portable workspace builds.

## Execution and ownership

Operations record adjacent dispatches through one serial compute encoder in a retained-reference `MTLCommandBuffer`. Copies and submission end the encoder; a later dispatch starts a new one. `fence()` and `read_f32_async()` submit. Their `poll_step()` reads command status and returns pending or terminal completion without blocking.

Each batch keeps `Rc` ownership of every referenced allocation. Submitted batches remain in a backend-owned queue until completion, even if callers drop buffers or completion handles. Abandoned model tasks transfer their fence and buffers into `MetalFenceRetirement`; foreign-instance rejections return the full payload intact.

Submitted work can't be cancelled. `cancel()` reports `Unsupported`, allowing the executor to retain the real unresolved fence. Readback copies into private shared staging before submission, so later writes can't change that snapshot.

Allocations aren't pooled yet. Dropping an allocation releases its class charge only after its last queued or submitted owner releases it. New shared storage is zeroed before encoding; uploads initialize it before GPU access. CPU reads only touch private staging after its copy command completes.

## Math and stored encodings

Dense arithmetic, activations, caches and accumulation use F32. Model loaders handle dense FP16 storage. Packed matrices execute directly from canonical U8 code streams and decoded F32 group-128 scales:

- Ternary: 4 two-bit codes per byte, LSB first, `(code - 1) * scale`.
- NF4: 2 four-bit codebook indices per byte, low nibble first.
- INT8: signed two's-complement bytes multiplied by group scales.

The validated model loader rejects reserved codes and malformed artifacts. Backend methods check ownership, generation, dtype, shape products, output geometry and alias policy before encoding. Fused QK admission also validates the complete key rotary descriptor before allocating or recording work.

The baseline gives each linear output a sequential reduction. Attention gives each query/head a disjoint score row and output row. Causal KV cache append and history update use ordered encoders; encoder attention and centered convolution isolate contiguous segments.

This branch implements one cooperative canonical packed family with an 8×32×32 tile and 256 complete-group threads. It stages F32 input and decoded/scaled weights, while each output retains ascending-K scalar accumulation.

Single and pair entry points cover the existing input/epilogue fusion modes and independently formatted pairs. Unsupported shapes or observed pipeline limits retain scalar dispatch.

RMS normalization uses one complete 256-thread group per row for widths from 256 through `u32::MAX - 255` when the compiled pipeline supports 32-lane SIMD groups and its thread/memory limits fit. Each SIMD group sums its F32 partials, then one thread combines the 8 group sums. Widths outside that range and incompatible pipeline limits retain the scalar kernel; this changes the addition order for admitted rows.

Metal fast math is disabled. Rotary parameters reproduce the portable contract's explicit F64-to-F32 frequency/trig boundaries on the host, then normalization and rotation execute on Metal. Canonical ternary remains the native path; LUT2 repacking isn't advertised.

## Counters and resource reports

`device_info()` reports the selected native device name and registry ID. `dispatch_counts()` returns the actual MSL entry points recorded, including `dense_linear`, `packed_linear`, `packed_pair`, `rms_norm_simd`, `attention` and `centered_conv`. The candidate adds `packed_linear_tile8` and `packed_pair_tile8` without renaming scalar counters; every packed operation still records one dispatch.

`resource_report()` counts physical classified storage, including allocations retained by pending work. Pending bytes include readback staging and reserved capacity for the temporary byte vector and returned F32 vector. Those host vectors each obey the per-allocation cap, while their combined reservation obeys the total cap. `peak_accounted_bytes()` preserves the highest total. Driver memory, pipelines and general process allocations need separate measurements.

## Source map

| File | Responsibility |
| --- | --- |
| `lib.rs` | Device/batch ownership, accounting, admission and public construction |
| `bridge.rs` | Audited Objective-C calls and synchronized shared-storage access |
| `operations.rs` | Finite inference operations and checked dispatch geometry |
| `packed.rs` | Canonical format inference, fixed tile selection and complete groups |
| `kernels.rs` | Finite typed native registry and stable counter names |
| `encoder.rs` | Complete-sequence attention and centered convolution |
| `completion.rs` | Pollable completions, readback snapshots and fence retirement |
| `kernels.metal` | Independent MSL kernels with parameter layouts |
| `shaders/packed_tile8.metal` | Candidate cooperative packed single/pair bodies |

Unsafe code is denied throughout the crate and allowed only in `bridge.rs`. The workspace policy remains unchanged.

## Checks

```sh
cargo +1.89.0 test --locked -p minifield-backend-metal --lib
cargo +1.89.0 clippy --locked -p minifield-backend-metal --all-targets -- -D warnings
cargo +1.89.0 check --locked -p minifield-backend-metal --target wasm32-unknown-unknown
MINIFIELD_REQUIRE_GPU=1 cargo +1.89.0 test --locked -p minifield-backend-metal --lib --test parity --test packed_tile8 -- --ignored --nocapture --test-threads=1
```

Hardware tests are explicitly ignored by portable CI. The hardware command requires actual native Metal construction and shader compilation, and fails if either is unavailable.

The hardware gate includes `--lib`, `parity` and `packed_tile8`: 1 capability unit, 14 parity cases and 4 tile8 cases. The parity cases include dependent dispatches across copy/submission boundaries and RMS widths 64, 255, 256, 257, 1024 and 1025, with zero rows and varying gains. The private unit retains two JSON records of actual compiled tile8 limits and requires their admission.

Both reference-device pipelines report execution width 32 and a 1,024-thread limit; their static allocations are 5,248 and 9,472 bytes against the device's 32,768-byte capacity.
