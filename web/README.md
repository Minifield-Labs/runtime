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
