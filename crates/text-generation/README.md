# `minifield-text-generation`

This crate provides portable, bounded greedy plaintext generation over a caller-provided
`TokenExecutor` and `Tokenizer`. It has no filesystem, network, prompt-template, tool, routing,
or model-loading policy.

`generate` checks the complete 65,536-logit head, rejects non-finite logits, chooses the lowest
ID on an exact maximum tie, and stops only for an explicit caller stop ID. It checks context and
output bounds before every candidate, validates candidate decoding before model append, and only
publishes the prefix and text after append completes. Callers choose BOS insertion, special-token
rendering, and cancellation handling through `GenerationRequest`.

`choose` performs typed binary-criterion choice scoring over a `TokenChoiceExecutor`. A shared
`base_prompt` is prefilled once in a single multi-token pass with `prefill_choice_base`, then each
named criterion's `tail` is evaluated serially: `append_choice_logits` branches the immutable base,
appends the short tail, and reads back only its true/false selector logits (no generated tokens, no
published branch prefix, no full-vocabulary readback). `prepare_choice` tokenizes the base and
criteria up front, requiring every tail and selector to be a compositional continuation, and
`finish_choice` softmaxes each criterion's true-minus-false evidence so async hosts can pump the
completions themselves.

Run the compact checks with:

```text
cargo +1.89.0 test -p minifield-text-generation --locked
cargo +1.89.0 clippy -p minifield-text-generation --all-targets --locked -- -D warnings
cargo +1.89.0 check -p minifield-text-generation --target wasm32-unknown-unknown --locked
```

The crate compiles for WASM but does not supply a browser executor or browser smoke harness.