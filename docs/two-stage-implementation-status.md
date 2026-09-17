# Custom Rust inference implementation status

This draft implements an inference-only runtime owned by this repository. The target is an embeddable Rust library and a thin plaintext input/output executable. Caller policy, tool execution, UI, networking, and training stay outside the executor.

## Implemented foundation

- Backend-neutral shape, dtype, allocation, resource, completion, tensor, token, and asset contracts.
- Owned scalar FP32 CPU kernels, including the arithmetic, linear, normalization, convolution, rotary-position, and attention operations required by the planned model.
- Config parsing and model-derived weight inventory, duplicate-aware bounded tensor loading, asset checksums, tied weights, and typed model roles.
- Backend instance leases, read-bound checks, pending-fence retirement on abandonment, and checked retirement admission that preserves rejected fence/buffer ownership.
- Separate pure Rust schema/argument trace crate with strict JSON, numeric and pattern handling, union and finite-container semantics.

## Validation and limits

The prior CPU/operator foundation passed independent operator and rounding fixtures. The accepted loader correction passed 46 ordinary owned tests, with one external-asset test ignored in the ordinary run, strict Clippy, and a wasm32 compile check. An explicitly enabled local asset test verified the pinned model header and inventory: 148 physical tensors and 354,483,968 parameters. Independent review and debug/release counterexamples verified asynchronous retirement, same-instance stale-generation cleanup, strict dtype admission, and rejection of foreign-instance fences/buffers without losing their ownership. Full model execution is now being implemented.

The separate protocol package passed 45 tests, formatting, all-target strict Clippy, and documentation checks. Its opt-in local corpus bridge matched all 4,134 rows for fields currently exposed. Routing candidate records, forced operations, globally interleaved operation order, and probe global operation indices are still missing from that bridge.

Bulk generated protocol traces, pattern oracles, and binary64 oracle dumps are excluded from Git. Small default regressions remain in the standalone suite; complete oracle qualification uses an explicitly supplied local artifact directory, documented in engines/decoding-protocol/README.md. Historical test counts above describe the original full local fixture run.

The committed 24 KiB numerical weight fixture is synthetic random test data for a two-layer, 16-hidden model. It is not a pretrained or trained model checkpoint. Actual model assets and private corpus inputs stay outside Git.

## Remaining

- Full model execution assembled from loaded typed weights.
- Prefix/cache ownership, append, branch, cancellation, and complete candidate scoring.
- Tokenizer, text generation loop, and plaintext executable.
- Owned CUDA and Metal backends, quantized kernels, artifact integration, and cross-backend qualification.
- Browser delivery testing; wasm32 compilation alone does not establish browser correctness or performance.

Historical JavaScript and excluded engine prototypes are not the new Rust executor. No end-to-end custom runtime or backend performance claim is made at this checkpoint.

This snapshot was taken from the existing implementation checkout. Remote main has newer legacy runtime changes; integration remains required before merge.

## Checkpoint practice

The coordinator owns Git operations for this repository because runtime and protocol agents edit separate areas concurrently. Commit coherent checkpoints, include actual test outcomes, and push progress at roughly 30-minute intervals while work is active. Never commit raw experiment outputs, customer data, or production weights.

Snapshot recorded: 2026-09-17T01:18:39.589211+00:00
