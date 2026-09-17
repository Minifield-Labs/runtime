# Draft-5 decoding protocol

This isolated crate implements pure-Rust strict JSON admission, binary64 and
raw-integer checks, RFC 8785 SafeJSON, schema normalization and original
instance validation, trace planning, ECMAScript-pattern admission, and
independently tokenized segments. It owns no model, tokenizer implementation,
tool execution, or product effects.

The draft artifact is fixtures/decoding-protocol.draft-5.json with SHA-256
e78b6ecfa07a71cf21203ae697aec7b92a19768bbc02e320be58ad1452150fde.

## Default checks

cargo test -p minifield-decoding-protocol --locked uses compact source and
fixture checks for SafeJSON, numeric boundaries, schema assertions, trace
ownership, finite domains, and representative ECMAScript semantics.

## Opt-in full oracle qualification

Large independently generated JCS, V8-pattern, and exact trace oracles stay
outside Git. Preserve their original bundle directories under an absolute root,
set MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT to that root, then run:

cargo test -p minifield-decoding-protocol --locked -- --ignored

Each ignored test requires the root, rejects relative or missing paths, pins the
bundle manifest.json SHA-256, and verifies every manifest-listed payload hash
before reading expected values. A bad root or modified oracle fails the test.
