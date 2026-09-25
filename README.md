# Minifield Runtime

Minifield Runtime is an inference-only Rust component for executing delivered models locally. It is intended to embed in native applications and browser/WASM callers, with a thin executable accepting plaintext input and returning plaintext output. Product policy, tool execution, UI, network services, jobs, training, and artifact distribution belong to callers.

The active Rust workspace owns its execution and kernels. It has no third-party tensor or inference framework. CPU/WASM is the portable baseline; CUDA and Metal are planned low-level backends.
This independent repository includes audited scalar Q4/Q8 matrix references,
serialized model ownership, WASM memory checks, token validation, bounded
sequence buckets, and a native mistral.rs adapter for complete LFM2 training
model exports. Browser Worker hosting and product integration remain to be
implemented.

## Current implementation

- engines/engine-api: backend-neutral tensor, resource, completion, asset, token, and finite inference-operation contracts.
- crates/backend-cpu: owned scalar FP32 storage and arithmetic, packed-ternary and packed-NF4, fused decode, linear, normalization, convolution, rotary-position, and attention kernels.
- engines/backend-wgpu: wgpu 30 backend implementing the same finite operation contract (F32 plus `minifield.ternary.v1` and `minifield.nf4.v1` packed kernels) with batched command recording and nonblocking completions.
- crates/executor-core: checked configuration and bounded typed weight loading, full LFM2 execution, prefix caches, append/fork operations, and complete candidate scoring.
- crates/text-tokenizer: owned bounded tokenizer asset parsing, byte-level BPE, Unicode pretokenization, and incremental UTF-8 decoding.
- crates/text-generation: backend-neutral bounded greedy plaintext generation and typed binary-criterion choice scoring over a caller-provided tokenizer and token executor; choice scoring evaluates each criterion serially and reads back only its true/false logits.
- crates/infer-cli: native bounded local-bundle loader and plaintext stdin/stdout executable with explicit BOS and capacity options.
- engines/decoding-protocol: separate pure Rust schema/argument framing and teacher-trace component for caller integration.

The model executor, tokenizer, bounded greedy generation loop, and plaintext binary now run the trained tiny diagnostic model through owned Rust APIs. Quantized execution and device backends remain implementation work. Read [implementation status](docs/two-stage-implementation-status.md) for exact validation and remaining gaps.

Historical JavaScript helpers and excluded engine prototypes are not part of the new Rust execution core. The scalar CPU implementation provides a correctness baseline; it makes no throughput claim.

## Local checks

Use Rust 1.89 with the wasm32 target installed:

    cargo +1.89.0 fmt --all --check
    cargo +1.89.0 test --workspace --locked
    cargo +1.89.0 clippy --workspace --all-targets --locked -- -D warnings
    cargo +1.89.0 check --workspace --target wasm32-unknown-unknown --locked

Opt-in checks requiring external model assets or private corpus inputs are documented in their test sources and are separate from the standalone synthetic suite. A wasm32 compile check is not browser execution qualification.

## Boundaries

Backend-neutral Rust owns model loading and execution policy within a caller-supplied resource budget. Backends own storage, finite kernels, and device completion. Opaque ownership and generation checks prevent foreign or stale tensors from being reused.

Only synthetic test fixtures live with source. Actual model weights, private corpus records, training output, and run logs remain outside Git. The tiny numerical loader fixture contains generated random values and is retained solely for standalone tests.

This repository must work as a standalone clone and cannot import sibling repositories by filesystem path.
