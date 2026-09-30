# Minifield Runtime

Local LFM2 inference in Rust, with CPU and WebGPU backends. The runtime loads dense or packed weights, tokenizes input, runs generation or classification, and supports reusable prefixes and constrained decoding.

This is an early implementation with tested numerical paths and explicit limits. Hardware support and model quality require evidence for the actual bundle and device. Trained weights and a finished product integration aren't included.

## Start here

The repository pins Rust 1.89.0, including rustfmt, Clippy, and the WASM target. Portable checks also use Python 3.11+, uv, and Node.js 22+.

```sh
scripts/check.sh quick
scripts/check.sh ci
```

`ci` runs portable tests, lints, contracts, JavaScript, and converter checks. GPU and browser qualification require actual hardware. See [the development procedure](docs/procedure.md).

Run a local language-model bundle on the CPU:

```sh
printf 'Hello' | cargo run --release --locked -p minifield-infer -- --model-dir /absolute/model --bos true --max-output-tokens 12 --max-context-tokens 64
```

The directory must contain `config.json`, `model.safetensors`, and `tokenizer/tokenizer.json`. The CLI consumes the prompt exactly as supplied. Callers own chat templates and product policy.

Use [bundle qualification](tools/qualification/README.md) for GPU classification and matched quantization benchmarks, [the browser harness](web/README.md) for WASM execution, and [offline converters](tools/converters/README.md) for packaging or explicitly requested quantization.

## Repository map

| Path | Responsibility |
| --- | --- |
| `crates/engine-api` | Finite operation, tensor, ownership, completion, and resource contracts |
| `crates/backend-cpu` | Scalar reference implementation and packed arithmetic |
| `crates/backend-wgpu` | GPU storage, dispatch, shaders, completion, and diagnostics |
| `crates/kernels-simd` | Isolated portable SIMD kernel work |
| `crates/executor-core` | Bounded loader, LFM2 execution, classification, and prefix state |
| `crates/text-tokenizer` | Bounded BPE assets, tokenization, and incremental decoding |
| `crates/text-generation` | Bounded generation and candidate/choice scoring |
| `crates/json-grammar` | Byte-level decoding constraints and token masks |
| `crates/decoding-protocol` | Schema validation, framing, and deterministic teacher traces |
| `crates/infer-cli` | Native plaintext host |
| `web` | WASM bindings and browser development harness |
| `tools/converters` | Independent Python conversion package |
| `tools/qualification` | Native bundle correctness and timing reports |
| `tools/quant-reference` | Separate MFQ8/Q4 compatibility reference |
| `contracts`, `examples` | Pinned contracts and compact synthetic fixtures |
| `docs/research` | Historical plans and research proposals |

## Supported execution

The active model is LFM2 with the configuration subset validated by the loader. Dense F32/BF16/F16 assets execute as F32. Packed `minifield.ternary.v1`, `minifield.nf4.v1`, and signed `minifield.int8.v1` matrices use group-128 scales; mixed formats resolve per weight role. CPU and WebGPU implement the same `InferenceOps` contract.

The model file specifies weight representation. The runtime chooses compatible kernels from the backend, tensor shape, and explicit memory policy. Backend-private repacks stay in memory. Experimental kernels require a Cargo feature and typed selection. See [architecture](docs/architecture.md).

Native Metal and browser WebGPU are exercised on a reference Apple device. Other GPU families need their own qualification. Dedicated CUDA and Metal backends, general model imports, and mobile/browser compatibility matrices remain future work.

## Boundaries

Runtime owns inference and its evidence. Applications own authorization, tool execution, UI, and product sessions. Training owns optimizer state, QAT, evaluation datasets, and release metadata. Converters operate offline; Rust builds never invoke Python.

This is a standalone repository with no sibling source imports. Weights, customer data, generated reports, and logs stay outside Git. See [contributing](CONTRIBUTING.md), [security reporting](SECURITY.md), and [third-party notices](THIRD_PARTY.md).
