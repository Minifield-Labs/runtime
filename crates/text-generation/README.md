# `minifield-text-generation`

This crate provides portable, bounded greedy plaintext generation over a caller-provided
`TokenExecutor` and `Tokenizer`. It has no filesystem, network, prompt-template, tool, routing,
or model-loading policy.

`generate` checks the complete 65,536-logit head, rejects non-finite logits, chooses the lowest
ID on an exact maximum tie, and stops only for an explicit caller stop ID. It checks context and
output bounds before every candidate, validates candidate decoding before model append, and only
publishes the prefix and text after append completes. Callers choose BOS insertion, special-token
rendering, and cancellation handling through `GenerationRequest`.

Run the compact checks with:

```text
cargo +1.89.0 test -p minifield-text-generation --locked
cargo +1.89.0 clippy -p minifield-text-generation --all-targets --locked -- -D warnings
cargo +1.89.0 check -p minifield-text-generation --target wasm32-unknown-unknown --locked
```

The crate compiles for WASM but does not supply a browser executor or browser smoke harness.