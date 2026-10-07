# Browser WebGPU harness

The browser harness loads LFM2 assets, performs generation/classification in Rust/WASM, and streams results through thin JavaScript bindings. It requires WebGPU. Model quality depends on the supplied weights.

## Build and run

```sh
npm ci
cargo install wasm-bindgen-cli --version 0.2.127 --locked
npm run build:web
python3 -m http.server 8642 --bind 127.0.0.1
```

The CLI version must match `web/Cargo.toml`. `web/pkg/` is ignored generated output.
Deploy `minifield_web_demo.js` and `minifield_web_demo_bg.wasm` together. The JavaScript
includes telemetry and has no external module imports. For blob URL imports in a
Worker, pass WASM bytes to `init({ module_or_path: bytes })`. TypeScript definitions
and `assets.json` are optional build metadata.

Put a local bundle at `models/demo/`, or open `http://localhost:8642/web/index.html?bundle=../models/your-model`. The bundle contains `config.json`, `model.safetensors`, and `tokenizer/tokenizer.json`. Serve only a directory whose contents you're willing to expose locally.

The page is a development harness. Product UI, application authorization, and tool execution belong to the host application.

## Actual browser qualification

```sh
scripts/check.sh browser --bundle /absolute/bundle --prompts /absolute/prompts.json --classes 8 --expected /absolute/native-result.json --out /absolute/browser-result.json
```

This command builds the WASM package, then launches installed Chrome with an isolated temporary profile. Use `--chrome /absolute/executable` or `CHROME_BIN` to select the browser. Node.js 22+ and the matching wasm-bindgen CLI are required. No browser download or existing user profile is used.

Prompts are a nonempty JSON array of nonempty strings. `--expected` accepts a native qualification result, native classifier JSON result, or hash-bound expected-logit fixture. The weight, config, tokenizer, and prompt hashes and class count must match. Omitting expected evidence checks full/cached parity and recovery only.

The check runs actual WebGPU inference, full and cached classification, 64 concurrent scheduler yields, empty-input rejection, and successful inference afterward. It requires finite outputs, matching argmax, and the declared tolerances. It records browser/adapter identity, WASM/asset hashes, source revision/dirty status, timing samples, and comparison results. A deadline or absent adapter fails the command.

Compilation uses `scripts/check.sh wasm`. JavaScript helper checks use `npm test`. Neither command establishes browser inference support.

The runner uses official [Chrome headless](https://developer.chrome.com/docs/automation-and-testing/headless) and [DevTools Runtime](https://chromedevtools.github.io/devtools-protocol/tot/Runtime/) interfaces.

## Binding behavior

Generation and classifier calls report a content-free terminal record. Configure origin/environment or
disable delivery with `configure_telemetry`; flush before terminating a Worker with
`flush_telemetry`. See [runtime telemetry](../docs/runtime-telemetry.md) for the complete record
and deployment settings. Build with `npm run build:web` and distribute the paired
JavaScript and WASM files from `web/pkg/`.

- Async backend creation and cooperative completion polling keep the browser event loop active. `__minifieldYield` uses a queued MessageChannel scheduler; every concurrent waiter resolves.
- `load` and `generate` run bounded greedy language-model inference with incremental token callbacks.
- `generate_json` applies grammar masks to on-device argmax. Grammar validity and registered tool names don't establish application authorization or complete JSON Schema validation.
- `choose` prefills shared context once and scores criterion tails serially, reading only selector logits.
- `load_classifier` loads an explicit dense classifier head `[classes, hidden]`. Dense F32/BF16 and packed ternary/NF4 backbones share the same API.
- `classify` starts fresh state. `classify_cached` reuses a shared prefix when possible, with independent class width and input vocabulary. Callers own prompts, action masks, and application state.
- `load_pointer_encoder(config, weights, tokenizer, maxTokens?)` loads the bidirectional model, with a joint-token limit (default 2048) and a 32-question limit. Larger limits cost GPU memory and time. `tokenize(text, bos)` returns IDs and UTF-8 byte offsets into the original text, with `[0,0]` for an inserted BOS; offsets map back through the tokenizer's normalizer. `load_tokenizer(bytes)` returns the same `tokenize` without a model or GPU, so hosts can lay out requests first; `predict(JSON)` accepts `token_ids`, optional `segment_ids`, and explicit pointer questions, then returns scores and decoded answers. Hosts own wording, joint layouts, and conversion from source-relative token spans to source characters; pointer calls don't emit telemetry.

## Native bundle evidence

Use [tools/qualification](../tools/qualification/README.md) for repeated native measurements. The native example emits one structured JSON result with asset hashes, token IDs, logits, timings, dispatch counts, policy, and adapter details.

```sh
cargo run --release --locked -p minifield-web-demo --example classify -- /absolute/bundle /absolute/prompts.json --classes 8 --lut2 auto --prefix true
```

The default tokenizer path is `tokenizer/tokenizer.json`. Legacy bundles require an explicit `--tokenizer` path. Native results complement the browser check.
