# Custom Rust inference implementation status

This draft implements an inference-only runtime owned by this repository. The target is an embeddable Rust library and a thin plaintext input/output executable. Caller policy, tool execution, UI, networking, and training stay outside the executor.

## Implemented foundation

- Backend-neutral shape, dtype, allocation, resource, completion, tensor, token, and asset contracts.
- Owned scalar FP32 CPU kernels, including the arithmetic, linear, normalization, convolution, rotary-position, and attention operations required by the planned model.
- Config parsing and model-derived weight inventory, duplicate-aware bounded tensor loading, asset checksums, tied weights, and typed model roles.
- Backend instance leases, read-bound checks, pending-fence retirement on abandonment, and checked retirement admission that preserves rejected fence/buffer ownership.
- Separate pure Rust schema/argument trace crate with strict JSON, numeric and pattern handling, union and finite-container semantics.

## Validation and limits

The prior CPU/operator foundation passed independent operator and rounding fixtures. The accepted loader correction passed 46 ordinary owned tests, with one external-asset test ignored in the ordinary run, strict Clippy, and a wasm32 compile check. An explicitly enabled local asset test verified the pinned model header and inventory: 148 physical tensors and 354,483,968 parameters. Independent review and debug/release counterexamples verified asynchronous retirement, same-instance stale-generation cleanup, strict dtype admission, and rejection of foreign-instance fences/buffers without losing their ownership. Full FP32 model execution is implemented, including convolution and attention cache state, prefix forks, append operations, cooperative cancellation, and complete candidate scoring. Independent public-API review reproduced and corrected extreme-logit normalization and stale score publication. All six review probes now pass. A separate Rust consumer matches all six frozen trained-tiny plaintext cases exactly, with source unchanged during verification.

The separate protocol package now exposes public route framing and complete true/false candidates, forced array transitions, interleaved operation order, and global probe indices. The frozen corpus bridge matched all 4,134 rows with no remaining unqualified trace fields; debug-label naming differences are reported separately. After the coordinator corrected empty-description admission, all 43 default package tests passed with nine external-oracle tests ignored by default. The preceding full external oracle run passed all nine suites, and formatting, strict Clippy, rustdoc and wasm32 compilation passed. These remain teacher-driven traces; live model decoding and caller policy integration are separate work.

Bulk generated protocol traces, pattern oracles, and binary64 oracle dumps are excluded from Git. Small default regressions remain in the standalone suite; complete oracle qualification uses an explicitly supplied local artifact directory, documented in engines/decoding-protocol/README.md. Historical test counts above describe the original full local fixture run.

The committed 24 KiB numerical weight fixture is synthetic random test data for a two-layer, 16-hidden model. It is not a pretrained or trained model checkpoint. Actual model assets and private corpus inputs stay outside Git.

The owned tokenizer passed ten compact tests and two external oracle suites. Coverage includes 1,145 exact encode/BOS/decode cases, streaming and byte handling, all 1,112,064 Unicode scalar classifications, strict malformed-asset checks, and native/wasm32 compilation. Tokenizer assets remain caller supplied.

The portable Rust generation library and native `minifield-infer` binary are implemented. The binary reads bounded UTF-8 stdin, preserves prompt whitespace, applies only the explicit BOS choice, and writes generated plaintext without an envelope. Twelve compact tests pass, and all six hash-verified trained-tiny cases match both library token IDs/text and actual binary stdin/stdout bytes. Independent coordinator checks cover Unicode output, malformed UTF-8 and duplicate options. Strict Clippy, formatting, docs and the generation library wasm32 compile check pass. Actual browser execution is the next separate gate.

## Remaining

- Quantized model execution and useful-model qualification beyond the tiny diagnostic.
- Owned CUDA and Metal backends, quantized kernels, artifact integration, and cross-backend qualification.
- Browser delivery testing; wasm32 compilation alone does not establish browser correctness or performance.

Historical JavaScript and excluded engine prototypes are not the new Rust executor. The trained tiny diagnostic establishes CPU numerical integration only. Actual asynchronous device ownership, low-precision execution, useful model quality, and backend performance remain unqualified.

This snapshot was taken from the existing implementation checkout. Remote main has newer legacy runtime changes; integration remains required before merge.

## Checkpoint practice

The coordinator owns Git operations for this repository because runtime and protocol agents edit separate areas concurrently. Commit coherent checkpoints, include actual test outcomes, and push progress at roughly 30-minute intervals while work is active. Never commit raw experiment outputs, customer data, or production weights.

Snapshot recorded: 2026-09-17T01:18:39.589211+00:00
