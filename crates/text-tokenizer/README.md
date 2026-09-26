# `minifield-text-tokenizer`

This crate implements the pinned LFM byte-level BPE tokenizer in Rust. It is a
backend-neutral library: callers supply tokenizer JSON bytes and text, while
this crate performs no filesystem, network, model, generation, prompt, or tool
execution work.

`Tokenizer::from_json_bytes` admits only the declared profile: no normalizer,
the ordered Unicode-aware Split expression followed by `ByteLevel`, BPE with no
dropout, unknown-token, byte-fallback, or merge-ignore mode, and the BOS-only
template. It validates vocabulary IDs, merge definitions, added-token IDs and
content, then encodes ordinary text and exact added-token substrings. BOS ID 1
is emitted only through `EncodeOptions { add_special_tokens: true }`; no EOS is
inserted.

The model head has 65,536 scores. IDs 64,402 through 65,535 intentionally have
no tokenizer mapping. `decode` and `token_bytes` return `UnmappedToken` for
those IDs rather than silently dropping them. `StreamingDecoder` buffers an
incomplete UTF-8 scalar across token chunks and returns an error if final bytes
are invalid. `skip_special_tokens` skips only definitions marked `special`; the
six non-special added tokens remain visible.

The default resource limits reject oversized assets, input, pretokenized
pieces, output ID sequences, BPE rank-lookups, and added-token matching. The
simple merger charges each adjacent rank lookup across the entire `encode`
call against `max_merge_steps`. Added-token matching charges every attempted
trie byte edge, including failed edges, against `max_added_token_steps` across
the same call. Both work limits default to 16,777,216 steps and return their
specific work-limit error before doing work beyond the budget.

Run the compact default suite with:

```text
cargo +1.89.0 test -p minifield-text-tokenizer --locked
cargo +1.89.0 clippy -p minifield-text-tokenizer --all-targets --locked -- -D warnings
```

The full generated oracle is deliberately outside Git. Run the ignored,
hash-verified suites only with explicit artifact roots:

```text
MINIFIELD_TEXT_TOKENIZER_FIXTURE_ROOT=/absolute/.../fixtures/text-tokenizer-001 \
MINIFIELD_TEXT_TOKENIZER_ASSET=/absolute/.../tokenizer.json \
MINIFIELD_TEXT_TOKENIZER_UNICODE_PROFILE_ROOT=/absolute/.../fixtures/tokenizer-unicode-profile-001 \
cargo +1.89.0 test -p minifield-text-tokenizer --locked -- --ignored
```

The tests verify the fixture and asset SHA-256 values before reading expected
records. A missing, wrong, or altered configured path fails; no fallback source
or retokenization substitutes for the oracle.
