# Tests

The initial contract checks live in `scripts/check_contracts.py`. Add behavioral tests here as implementation begins, using small synthetic fixtures.

Model quality, application conformance, and device performance require separate evaluations. Passing schema checks doesn't establish any of them.

Run npm test for the JavaScript lifecycle, cancellation, memory-growth, token-input, and bucket checks. The scalar Rust tests live in engines/quant-reference/ and include source tests plus extracted-format and boundary regressions. See docs/wires-audit.md for commands and limits.
