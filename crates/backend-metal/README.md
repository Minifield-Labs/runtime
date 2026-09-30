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

Operations record into a retained-reference `MTLCommandBuffer`. `fence()` and `read_f32_async()` submit. Their `poll_step()` reads command status and returns pending or terminal completion without blocking.

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

Metal fast math is disabled. Rotary parameters reproduce the portable contract's explicit F64-to-F32 frequency/trig boundaries on the host, then normalization and rotation execute on Metal. Canonical ternary remains the native path; LUT2 repacking isn't advertised.

## Counters and resource reports

`device_info()` reports the selected native device name and registry ID. `dispatch_counts()` returns the actual MSL entry points recorded, including `dense_linear`, `packed_linear`, `packed_pair`, `attention` and `centered_conv`.

`resource_report()` counts physical classified storage, including allocations retained by pending work. Pending bytes include readback staging and reserved capacity for the temporary byte vector and returned F32 vector. Those host vectors each obey the per-allocation cap, while their combined reservation obeys the total cap. `peak_accounted_bytes()` preserves the highest total. Driver memory, pipelines and general process allocations need separate measurements.

## Source map

| File | Responsibility |
| --- | --- |
| `lib.rs` | Device/batch ownership, accounting, admission and public construction |
| `bridge.rs` | Audited Objective-C calls and synchronized shared-storage access |
| `operations.rs` | Finite inference operations and checked dispatch geometry |
| `encoder.rs` | Complete-sequence attention and centered convolution |
| `completion.rs` | Pollable completions, readback snapshots and fence retirement |
| `kernels.metal` | Independent MSL kernels with parameter layouts |

Unsafe code is denied throughout the crate and allowed only in `bridge.rs`. The workspace policy remains unchanged.

## Checks

```sh
cargo +1.89.0 test --locked -p minifield-backend-metal --lib
cargo +1.89.0 clippy --locked -p minifield-backend-metal --all-targets -- -D warnings
cargo +1.89.0 check --locked -p minifield-backend-metal --target wasm32-unknown-unknown
MINIFIELD_REQUIRE_GPU=1 cargo +1.89.0 test --locked -p minifield-backend-metal --test parity -- --ignored --test-threads=1
```

Hardware tests are explicitly ignored by portable CI. The hardware command requires actual native Metal construction and shader compilation, and fails if either is unavailable. See [the foundation experiment](../../docs/research/runtime-native-metal-foundation-2026-09-29.md) for recorded evidence.
