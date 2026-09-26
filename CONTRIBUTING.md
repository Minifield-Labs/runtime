# Contributing

Read [architecture](docs/architecture.md) and [the development procedure](docs/procedure.md) before changing a format or dispatch rule. Use the pinned Rust toolchain and locked dependencies. Python converter tooling is an independent package under `tools/converters`.

Keep a change focused on one observable behavior or one structural boundary. Use conventional commit subjects. A pull request should explain the concrete problem, the resulting behavior, the checks that ran, and any external assets or hardware those checks required.

Run `scripts/check.sh quick` while developing and `scripts/check.sh ci` before review. Kernel or dispatch changes also require `scripts/check.sh gpu` on a real adapter. Browser changes require the actual browser qualification described in [web/README.md](web/README.md). Compilation alone doesn't establish browser support.

A new encoding needs independent byte fixtures, malformed-input rejection, dequantized reference comparisons, and actual bundle qualification. A new kernel needs boundary-shape tests and dispatch evidence showing it ran. Keep quality loss from quantization separate from implementation parity and performance.

Keep models, customer data, generated reports, and run logs outside Git. Commit only compact synthetic contract fixtures or their deterministic generators. Preserve notices when adapting source. Avoid imports from sibling repositories and dependencies on a parent workspace.
