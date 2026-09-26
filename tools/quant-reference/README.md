# Quantized CPU references

This dependency-free Rust crate extracts the reusable Q4 matrix operations from Wires and adapts its W8 matrix/parser into a bounded MFQ8 reader. See [the audit](../../docs/wires-audit.md) for source provenance, changes, and limits.

Q4 retains continuous packed nibbles, row-local scales, batched matrix multiplication, and embedding-row lookup. Q8 reads the [matrix contract](../../contracts/quant-matrix/v0.1.0/README.md), validates dimensions and input/output lengths, and computes a scalar FP32 reference.

```sh
cargo test --manifest-path tools/quant-reference/Cargo.toml --offline
cargo clippy --manifest-path tools/quant-reference/Cargo.toml --offline --all-targets -- -D warnings
cargo build --manifest-path tools/quant-reference/Cargo.toml --offline --example matrix_probe
cargo build --manifest-path tools/quant-reference/Cargo.toml --offline --target wasm32-unknown-unknown
```

Run these commands from the runtime repository root. The WASM target must already be installed. The crate cross-compiles to a Rust WASM library; a browser ABI and complete decoder remain to be implemented.

matrix_probe takes a matrix file and a whitespace-separated FP32 input file, then prints output values as JSON. Its element budget is fixed at 16,777,216 for diagnostics. Production loaders need budgets derived from the target device.

The MIT license from the source is preserved in LICENSE. This crate does not import Wires, download weights, or require CubeCL.
