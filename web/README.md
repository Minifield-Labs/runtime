# Browser demo: packed ternary LFM2 on WebGPU

Runs the `minifield.ternary.v1` packed LFM2.5-230M bundle entirely in a browser
tab: model fetch, weight upload, greedy decode, and token streaming, with all
inference in the Rust `minifield-backend-wgpu` engine compiled to wasm32.

This is a developer harness, not a product surface. The naive ternary export
produces incoherent text; the demo exists to exercise the end-to-end path and
show decode throughput.

## Build

```sh
cargo build --target wasm32-unknown-unknown -p minifield-web-demo --release
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/minifield_web_demo.wasm
```

`web/pkg/` is gitignored build output. The `wasm-bindgen` CLI version must
match the crate version pinned in `web/Cargo.toml`.

## Run

Serve the repository root so both `web/` and the local bundle under
`tmp/models/` resolve:

```sh
python3 -m http.server 8642
```

Open `http://localhost:8642/web/index.html` in Chrome or Edge (WebGPU
required). The page fetches `config.json`, the packed safetensors, and
`tokenizer.json` from `tmp/models/lfm2.5-230m/` relative to the server root.

## How it works

- `WgpuBackend::new_async` creates the adapter/device through the browser's
  WebGPU promise path; the blocking constructor would starve on wasm32.
- `load()` builds the packed executor from fetched bytes via the standard
  `Lfm2WeightLoadTask`/`MemoryAssetProvider` path.
- `generate()` mirrors `text-generation`'s greedy loop but pumps each
  completion cooperatively: a pending poll yields one macrotask so
  `map_async`/`on_submitted_work_done` can resolve. The page provides
  `__minifieldYield` (a `MessageChannel` post, unclamped); `setTimeout(0)` is
  the fallback.
- Each sampled id streams back through an `on_token` callback; the page
  renders `id | decoded fragment` chips and reports tokens/second.
- `generate_json()` runs the same loop under a decode constraint:
  `crates/json-grammar`'s byte-level acceptor yields an allowed-token bitset
  per step, and `prefill_masked`/`append_argmax_masked` gate the on-device
  argmax so only grammar-continuable tokens can win. The page's Tool call
  button uses `AssistantCallEnforcer` with the comma-separated names input:
  output is exactly the serialized assistant body
  `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
  where `<name>` is one of the registered names, matching the
  lfm2-chatml-tool-json training serializer. The crate also exposes
  `JsonEnforcer` for general JSON-shaped output and `ToolCallEnforcer` for
  the simpler `{"<name>":true|false}` shape.
- `choose(names, prompts)` runs `text-generation`'s typed choice scoring.
  Each criterion is an independent prompt scored serially: one multi-token
  prefill pass, then a gather of its true/false selector logits, so only 2
  logits per criterion cross to the host. It resolves to the standard answer
  `{ type: "choice", choice, confidence, probabilities }`, where
  `probabilities` maps each criterion name to its relative score. The page's
  "Structured choice" section drives it with a state textarea and one
  `name: description` criterion per line.
- The prompt is wrapped in the lfm2-chatml-tool-json template
  (`<|im_start|>` turns plus an `Available tools:` system block). The system
  block is fixed per tool-name set, so `warm_tools` prefills it once at load
  and `generate_json` appends only the short user/assistant tail onto the
  cached KV prefix; the tail's pending sample is verified against the
  grammar mask, with a full masked prefill as the fallback.
