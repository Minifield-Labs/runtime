# Portable engine contracts

This dependency-free crate defines finite inference operations, tensor descriptors, backend identity/generation, capabilities, resource limits, completion/retirement, bounded assets, and token execution interfaces.

Modules separate these contracts without changing their public reexports. Backend implementations own allocation and operations; executors own model equations. Keep filesystem access, GPU APIs, environment parsing, and model-specific policy outside this crate.

Run `cargo test -p minifield-engine-api --locked` from the repository root. The portable contract tests exercise bounded descriptors, operation capabilities, ownership, and asset boundaries.
