# Minifield Runtime

Run a delivered product specialist and connect its supported actions to the host application.

This independent repository includes audited scalar Q4/Q8 matrix references, serialized model ownership, WASM memory checks, token validation, and bounded sequence buckets. Full decoder loading, GPU execution, Worker hosting, and product integration remain to be implemented.

Read the [detailed procedure and success criteria](docs/procedure.md) for implementation order, required artifacts, validation, failure handling, and the first milestone.

The [ordered experiment plan](experiments/README.md) contains individual protocols for the decoder baseline, prefix reuse, memory, packed execution, action syntax, and conditional KV compression/speculative decoding.

The [Wires transfer audit](docs/wires-audit.md) records source provenance, repaired lifecycle/input issues, benchmark limits, and the remaining browser work.

## Ownership

- src/inference/: model loading, tokenizer/template application, and decoding interface.
- src/context/: authorized observations, product policy, history, and context limits.
- src/tools/: schema validation, tool dispatch, results, and error handling.
- src/sessions/: the model/action loop, cancellation, budgets, and undo boundaries.
- engines/: backend-specific implementations and compatibility tests.
- products/: host integrations and supported-action mappings.
- configs/: reviewed configuration templates.

The current foundation uses portable JavaScript modules and a dependency-free Rust reference crate. The deployed decoder/backend will be selected during the deployment proof. Python supports development-time contract checks.

Platform owns the product UI/API, job history, artifact registry, and release controls from the first version. Runtime consumes registered bundles and returns validation evidence tied to bundle, engine, product, and device versions. Core inference remains Rust with a thin browser integration layer.

## Input contract

Load self-contained model bundles matching the pinned schema in contracts/model-bundle/. Check checksums, supported engine/format, product version, and memory/context constraints before starting a session. Refuse unknown versions and fixtures in a real loader.

The host app enforces permissions and confirmation rules. Keep private training metadata and judgments outside model-visible context. Logging and any cloud fallback need an explicit data-flow decision consistent with local AI processing.

See [bundle rules](contracts/model-bundle/v0.1.0/README.md). Model packing belongs to the training repository; this repository owns evidence that the delivered artifact runs correctly on the target device.

## Local checks

Use Python 3.11 or newer for the contract checks:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/check_contracts.py
```

The check validates schemas, pinned snapshots, example hashes, and handoff consistency. It also checks that malformed records are rejected. It doesn't run a teacher, a trainer, or model inference.

Run the audited JavaScript and scalar kernel tests with Node 22+ and Rust 1.85+:

```sh
npm test
cargo test --manifest-path engines/quant-reference/Cargo.toml --offline
cargo clippy --manifest-path engines/quant-reference/Cargo.toml --offline --all-targets -- -D warnings
```

See the [reference crate](engines/quant-reference/README.md) for WASM compilation and cross-language matrix checks. The tiny MFQ8 fixture is a kernel contract, separate from complete model bundles.

## First implementation

Choose one reference device and engine. Load an intact candidate, round-trip a tool call, and measure downloaded bytes, cold load, peak memory, and complete-task latency. Integrate a trained product bundle after this path works.
