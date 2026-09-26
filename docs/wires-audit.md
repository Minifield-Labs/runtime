> Historical transfer audit. The Q4/MFQ8 compatibility probe remains in `tools/quant-reference`. The JavaScript adapter and lifecycle helpers described below were retired in the 2026-09-25 restructure.

# Wires transfer: runtime audit

Reviewed September 11, 2026. Source hashes and destination mappings are recorded in wires-provenance.json. This audit covers the extracted components, with the source GPU experiment retained as design evidence.

## Reused implementation

| Source in unrvl-embdb | Destination | Decision |
| --- | --- | --- |
| src/quant.rs, src/error.rs | tools/quant-reference/src/q4.rs, error.rs | Extract scalar packed-INT4 operations and shape checks; preserve source tests and MIT notice |
| experiments/cubecl-q4-probe/src/w8.rs | tools/quant-reference/src/q8.rs | Extract row-local INT8 CPU math and checked reading; remove fixed encoder assumptions |
| web/unrvl-embdb.js | src/inference/wasm-memory.mjs, src/context/token-input.mjs | Preserve fresh memory views and copied results; add pre-coercion input and finite-output validation |
| experiments/cubecl-q4-probe/src/wasm.rs and w8_runner.rs | src/inference/resident-model.mjs | Adapt resident ownership into a bounded FIFO lifecycle |
| Optimization notes and experience report | src/inference/sequence-buckets.mjs and procedure additions | Bound shape specialization and record device-specific measurements |

The helper modules and scalar crate are independently testable. A browser Worker host, a decoder, GPU kernels, and device-loss fallback still require implementation.

## Findings and repairs

### Model replacement raced with pending inference

The source encode/trace functions take the global model out of its slot, await GPU work, then unconditionally put it back. A load or clear during that await can be overwritten by the older model. A second concurrent inference also sees an empty slot.

The adapted ResidentModel queues load, run, unload, and disposal on one bounded FIFO. The slot stays owned throughout a request, and replacement waits for completion. Failed jobs don't poison the queue; same-key loads reuse the current resource.

Tests suspend a running request, queue replacement and clear, and verify disposal order and the final empty state. Cancellation waits for operation settlement before cleanup. Adapter completion must include the last GPU consumer, even after an abort.

### Typed-array coercion hid bad inputs

The source converts token IDs with Uint32Array.from and masks with Uint8Array.from before validation. Fractional IDs truncate and out-of-range values wrap before Rust can inspect them.

The new input helper checks raw values, token budgets, mask lengths, and binary masks before copying them into typed arrays. It also rejects an all-masked input.

### WASM results need fresh views and finite checks

The source correctly reacquires views after potential linear-memory growth and copies results before freeing their buffers. This pattern was retained.

The extracted helper adds alignment/range checks and rejects non-finite output. Its tests grow a real WebAssembly.Memory in Node, then confirm the copied result remains independent of later memory writes. This exercises the memory API; it isn't a browser GPU benchmark.

### Generalizing the W8 parser required stronger public boundaries

The source's private matrix constructor computes ceil division before checking a zero group size. Its fixed-shape reader guards group=64, so that constructor order wasn't an exposed file-loading path there.

The extracted public constructor validates nonzero dimensions before division. MFQ8 parsing checks overflow, exact lengths, padding, normal positive scales, and an explicit element budget before allocation. Matrix calls validate lengths and finite inputs/results.

Q4 retains the source representation, including continuous nibbles across odd-width row boundaries. Added tests cover that boundary and distinct row-local scales. Q4 math remains a low-level scalar reference; callers must apply their finite-output checks.

## What needs a separate implementation

The full CubeCL graph targets embedding inference: fixed BERT dimensions, bidirectional attention, pooling, and an embedding head. Product decoders add autoregressive execution, KV-cache lifetime, and different position/normalization semantics.

The prerelease CubeCL/WGPU dependencies, generated shaders, bundled finance checkpoint, and model-specific ABI weren't imported. The source ownership lessons can guide the selected backend without tying Minifield to that encoder.

W8 is a useful numeric reference. A 22.7M-parameter W8 encoder fitting roughly 24 MB doesn't imply that an 8-bit 1B model fits a 500 MB delivery budget.

## Corrections to the experience report

The inspected W8 runner has 12 logical launches per layer plus 5 outside the layer loop, giving 77 for 6 layers. The report's approximate 65 and the older notes' 5 + 10 × layers formula don't match that source body. Count actual dispatches in each future benchmark.

The reported browser warm median is based on 3 runs on one M1 Max setup. The parity fixture covers 7 embeddings and 3 query rankings. Preserve those results as historical evidence with those limits.

The roughly 79 KB bundle-overhead statement compares compressed bundled WASM against the raw model. A distribution comparison should measure both packaging choices with the same transport compression and cache conditions.

The saturated-GELU NaN, shader-literal issue, asynchronous validation, and queued-handle lifetime failures are source-reported findings. They weren't reproduced on fresh browser hardware in this transfer.

## Runtime rules carried forward

1. Keep a CPU oracle for each accelerated operation. Compare intermediate tensors and product decisions.
2. Initialize adapters and read back results asynchronously in the browser.
3. Keep immutable weights resident. Retain activation buffers until their final submitted consumer completes.
4. Isolate in-flight request memory, or serialize requests as this initial owner does.
5. Cache initialization and pipelines under artifact, engine, numerical settings, and bounded shape keys.
6. Measure prompt prefill and token-by-token decode separately. Account for KV cache, readback, and transient loading copies.
7. Use a Worker for blocking host work and a measured CPU fallback when its compatibility/performance justify it.
8. Validate shader failures and finite output explicitly. Inspect device loss and memory pressure.
9. Keep padding masks and positions consistent with training. Bucket selection alone doesn't implement padding semantics.
10. Compare cold and warm distributions, actual transfer bytes, memory peaks, and complete product tasks on target browsers.

## Verification

From the runtime root:

```sh
npm test
cargo test --manifest-path tools/quant-reference/Cargo.toml --offline
cargo clippy --manifest-path tools/quant-reference/Cargo.toml --offline --all-targets -- -D warnings
cargo build --manifest-path tools/quant-reference/Cargo.toml --offline --example matrix_probe
cargo build --manifest-path tools/quant-reference/Cargo.toml --offline --target wasm32-unknown-unknown
python scripts/check_contracts.py
```

Rust tests cover imported Q4 behavior, signed bytes, partial groups, malformed/truncated files, budgets, scales, and call shapes. JavaScript tests cover ownership, failed operations, cancellation, backpressure, memory growth, inputs, and bucket limits.

Cross-language tests run from training against the explicitly supplied matrix_probe executable. No source repository is required by either consumer. Full browser model execution and device benchmarks remain to be done.
