# llama.cpp oracle capture (T2/T3)

Captures reference logits and greedy token streams for LFM2.5-230M so the Rust
executor can be checked position-by-position against an independent
implementation of the same math. This is the correctness anchor from
`docs/ternary-format-v1.md`: architecture validation on dense weights, kept
separate from any ternary kernel work.

## What it produces

All artifacts land under `tmp/oracle/` (gitignored, ~1 GB):

- `LFM2.5-230M-BF16.gguf`: the published BF16 GGUF, sha256-pinned in the script.
- `out/pNN.logits.f32`: raw f32 logits `[n_tokens][65536]` per prompt position.
- `out/pNN.tokens.txt`: greedy generations from `llama-completion` (temp 0).
- `manifest.json`: model/build/tool versions, per-prompt token ids, emitted
  ids, hashes, and anomalies. `prompts/pNN` is a JSON object keyed by prompt id.

## Running it

```sh
scripts/oracle/run-oracle.sh
```

The script downloads the GGUF, clones and builds llama.cpp `b11046` CPU-only
(`-DGGML_METAL=OFF -DGGML_NATIVE=OFF`), builds `dump_logits` (a small C tool
that reads `llama_get_logits_ith` rows as f32), runs six prompts with f32 KV
(`-ctk f32 -ctv f32`), and writes the manifest.

Note: in `b11046`, `llama-cli` is a chat REPL that applies the chat template.
The batch interface used here is `llama-completion`. BOS is auto-prepended by
llama.cpp (`vocab_add_bos=1`), so the manifest's prompt ids already include it.

## Comparing against the Rust executor

```sh
MINIFIELD_LFM25_BUNDLE_DIR=/path/to/bundle \
MINIFIELD_LFM25_ORACLE_DIR=tmp/oracle \
cargo test -p minifield-executor-core --release --test oracle_lfm25 -- --ignored --nocapture
```

The bundle dir must contain `config.json` and `model.safetensors` (the BF16
safetensors, not the GGUF). The test teacher-forces each oracle prompt,
compares `next_logits` at every position, then checks the generated chain
under matched history. Disagreements are only tolerated at near-ties (top-2
gap ≤ 0.05); a confident wrong answer fails.
