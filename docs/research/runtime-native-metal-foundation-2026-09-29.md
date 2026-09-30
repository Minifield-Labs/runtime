# Native Metal foundation, September 29, 2026

## Hypothesis

A dedicated Metal implementation is necessary to evaluate native Metal independently from WGPU-through-Metal. Both paths can share the finite executor contracts while retaining their own buffers, queues, kernels and completion ownership.

This chunk establishes that separate implementation and a correctness baseline. Performance tuning starts after the frozen campaign admits actual models on both paths.

## Evidence and trace

The existing GPU crate constructs a WGPU device and records WGSL pipelines. Its macOS adapter uses Metal underneath, but it remains the WGPU implementation.

The finite `InferenceOps` contract already carries instance leases, generation checks, resource classes, pollable completion and abandoned-fence retirement. `EncoderOps` adds complete-sequence attention and centered convolution without changing causal cache operations.

A native backend can implement both contracts directly. Apple's normal command buffers retain resource references by default, and submitted work exposes status observation. The new backend also retains its own allocation ownership until the corresponding command buffer reaches a terminal state. [Apple's retained-reference documentation](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences), [command-buffer documentation](https://developer.apple.com/documentation/metal/mtlcommandbuffer).

Shared-memory visibility requires completion before the CPU consumes GPU writes. Readback therefore records a private snapshot copy, submits it and only reads staging after completion. Subsequent writes target the original allocation and can't change that snapshot. [Apple's resource synchronization guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Mem-Obj/Mem-Obj.html).

Review found 2 numerical boundaries to preserve. The encoder attention mask must guard value reads as well as score reads, since `0 * infinity` creates NaN. Rotary frequency and trig calculations have explicit portable F64-to-F32 rounding points, so native dispatch stages the same parameter table before GPU rotation.

## Fix

Added `crates/backend-metal` with independent native `MTLDevice`, `MTLCommandQueue`, retained-reference command buffers, shared buffers and MSL pipelines. The crate has no WGPU dependency or CPU inference fallback.

The backend supports dense linear/normalization/rotary, elementwise and selector operations, canonical ternary/NF4/INT8 projections, causal attention and convolution, and segment-isolated encoder attention and centered convolution. Mixed projection pairs decode each stream using its own format parameter.

The baseline keeps reductions explicit: one thread per linear output, one per normalization row, and one per attention query/head. The MSL file names dimensions and documents every parameter layout. Metal fast math is disabled.

Each batch retains every referenced allocation. Submission serials tag independent command-buffer ownership; the pending queue releases allocations only after observing a terminal status. Completed storage isn't pooled in this foundation.

`MetalFenceRetirement` preserves abandoned task buffers and returns ownership intact when admission sees a foreign backend instance. Submitted cancellation reports `Unsupported`, keeping a real unresolved fence available for retirement.

Physical allocations retain their original resource-class charge through pending work. Readback accounts private staging and reserves host result capacity. A high-water counter records peak backend-accounted bytes; device name/registry ID and actual MSL dispatch counts are available to the evaluation host.

Unsafe code is isolated in `bridge.rs`, covering checked Objective-C dispatch and synchronized shared-storage access. The rest of the crate denies unsafe code. Workspace lints remain unchanged.

## Test

| Check | Result |
| --- | --- |
| `cargo test --offline -p minifield-backend-metal --lib` | 7 passed after review fixes |
| `cargo clippy --offline -p minifield-backend-metal --all-targets -- -D warnings` | Passed |
| `cargo check --offline -p minifield-backend-metal --target wasm32-unknown-unknown` | Passed |
| `cargo clippy --offline -p minifield-backend-metal --target wasm32-unknown-unknown --lib -- -D warnings` | Passed |
| `MINIFIELD_REQUIRE_GPU=1 cargo +1.89.0 test --locked --offline -p minifield-backend-metal --test parity -- --ignored --test-threads=1` | 12 passed, 0 failed, 0 skipped, 0.71 s after review fixes |

The hardware gate exercised native framework shader compilation and dispatch. CPU parity covered dense/fused operations, all 3 packed formats, causal and bidirectional GQA, both convolution modes and rotary normalization.

Lifecycle tests covered duplicate diagnostic owner IDs on distinct actual instances, stale generations, readback snapshots across later writes, early buffer drops and ownership-preserving retirement rejection. A separate infinity fixture verified that inactive encoder rows never contaminate valid output.

The initial 8-case hardware output is preserved outside Git at `experiments/2026-09-29-runtime-gpu-hill-climb/runs/native-metal-foundation-parity.log`. The reviewed 12-case output is at `experiments/2026-09-29-runtime-gpu-hill-climb/notes/native-metal-reviewed-gate.log`. Actual-model qualification is recorded separately by the campaign host with model/precision hashes and selected device identity.

The installed Xcode command-line `metal` executable reported a missing Metal Toolchain component. No component was downloaded. Runtime framework compilation succeeded for every registered shader during the native hardware gate.

## Review fixes

Independent review found 3 contract boundaries that the first hardware fixtures missed. QK normalization built the key rotary descriptor after recording query work; readback applied the per-allocation cap to a 2-vector aggregate; row gather compared a selector against a rounded F32 table size.

QK now checks matching head dimensions and constructs the key rotary descriptor before scratch allocation or command recording. Portable admission tests cover odd and even mismatches; hardware fixtures assert that rejection preserves output sentinels, dispatch counts and resource counters.

Readback now admits its temporary byte vector and F32 result as 2 independent allocations under the shared total cap. The portable accounting regression reserves 4 separate 16-byte buffers under `max_allocation_bytes=16` and `max_total_bytes=64`, then verifies failed per-buffer and aggregate admissions preserve counters. The hardware fixture runs an actual 16-byte snapshot under those limits.

Dense and packed gather now guard the exact unsigned conversion range, convert the selector and compare integer row bounds. The hardware fixture selects row `16,777,216` from a `16,777,217`-row table, checks an invalid nearby id, and rejects `2^32` before conversion. The packed fixture uses approximately 576 MiB of native storage.

Upload byte staging now preflights resource admission and uses fallible checked reservation. Additional convolution fixtures cover a sequence shorter than its rolling history, even-width centered padding/crop and segment isolation.

The expanded hardware suite passed all 12 cases in an isolated GPU window: 0 failures, 0 skips, 0.71 s. The repaired large-row selector, tight readback allocation budget and side-effect-free QK rejection all ran on the actual native Metal implementation.

## Resolution

The native implementation now passes its finite-operation and ownership gates independently from WGPU. It can enter actual-model qualification as `native_metal`, with arithmetic precision, artifact size and output parity reported separately for each precision.

This foundation establishes correctness evidence. It makes no speedup claim. The frozen campaign must still verify each supplied model/precision cell before native performance candidates can be promoted.

The first likely tuning targets are parallel linear reductions, reusable completed storage and bounded attention score workspaces. Each needs its own hypothesis, matched same-precision output check and spaced repeated timing runs.
