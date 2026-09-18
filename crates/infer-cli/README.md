# `minifield-infer`

`minifield-infer` is the native I/O adapter for a local LFM2 bundle. It reads one strict UTF-8
prompt from stdin through EOF and writes only generated plaintext to stdout. Diagnostics use
stderr and failures exit nonzero. It does not add a chat template or execute product actions.

The required model directory contains `config.json`, `model.safetensors`, and
`tokenizer/tokenizer.json`. BOS handling and token/context limits are explicit:

```text
printf 'Hello' | cargo +1.89.0 run -p minifield-infer --locked -- \
  --model-dir /absolute/model --bos true --max-output-tokens 12 --max-context-tokens 64
```

The executable verifies model/config bytes during the existing bounded LFM2 load. Its caller
limits bound prompt reads, asset reads, loader staging, backend allocations, and generated tokens.
EOS ID 7 is the only implicit artifact stop; other special IDs are ordinary generated IDs unless
the caller changes the generic library stop list.

Run compact checks with:

```text
cargo +1.89.0 test -p minifield-infer --locked
cargo +1.89.0 clippy -p minifield-infer --all-targets --locked -- -D warnings
```

The full trained-tiny oracle remains external. It is hash-verified and runs both the library path
and the compiled binary only when these variables name the immutable sources:

```text
MINIFIELD_TRAINED_TINY_TEXT_FIXTURE_ROOT=/absolute/fixtures/trained-tiny-text-001 \
MINIFIELD_TRAINED_TINY_ARTIFACT=/absolute/trained-tiny-artifact \
cargo +1.89.0 test -p minifield-infer --test trained_tiny --locked -- --ignored
```