# Runtime inference telemetry

The browser bindings and native CLI report one record when a high-level inference returns.
`classify`, `classify_cached`, and `choose` use `single_step`. `generate` and `generate_json`
use `autoregressive`. Model loading and `warm_tools` don't emit inference records. Rejected
input after model load emits a failed inference with zero predictions.

The default receiver is `https://telemetry.minifieldlabs.com/insert/jsonline`, the deployed
VictoriaLogs JSON-line endpoint. VictoriaLogs assigns receipt time; records contain elapsed
durations and no start/end timestamps. This schema is `minifield.runtime-inference/1`.

## Deployment configuration

Browser configuration applies to the WASM module instance. Set it in the same Worker that
loads the runtime, before calling inference:

```js
import init, { configure_telemetry, flush_telemetry } from "./pkg/minifield_web_demo.js";
await init();
configure_telemetry({
  integrationId: "example-command",
  productId: "example-product",
  applicationId: "example-app",
  applicationVersion: "1.2.0",
  environment: "production",
});

// Deployment opt-out:
configure_telemetry({ enabled: false });

// Optional alternative receiver:
configure_telemetry({ enabled: true, endpoint: "https://metrics.example.com/insert/jsonline" });

// Before explicitly terminating the inference Worker:
await flush_telemetry();
```

Origin defaults to the executing page/Worker's HTTP(S) origin, with path, query, and fragment
removed. `websiteOrigin` can explicitly name the embedding application origin. Localhost
defaults to `development`; other origins default to `production`. Preview/test deployments
should set their environment explicitly. Integration ID defaults to `minifield-web`.
Deployment labels accept up to 128 ASCII letters, digits, or `._:/@+-`.

The CLI supports `MINIFIELD_TELEMETRY=0` (also `false` or `off`) to disable delivery.
Optional settings are `MINIFIELD_TELEMETRY_ENDPOINT`, `MINIFIELD_ENVIRONMENT`,
`MINIFIELD_INTEGRATION_ID`, `MINIFIELD_PRODUCT_ID`, `MINIFIELD_APPLICATION_ID`, and
`MINIFIELD_APPLICATION_VERSION`. Defaults are the receiver above, `production`, and
`minifield-infer`. Native library embedders supply their own origin/platform fields and
delivery through `run_with_reporter`; `run_with_io` performs no network I/O.

Endpoints require HTTPS; loopback HTTP is accepted for local tests. Configuration changes
discard pending browser batches and abort active requests. Browser batching waits up to 1
second, keeps at most 64 queued records / 48 KiB, and permits 1 request in flight.
Overflow records are dropped. Requests use write-only `no-cors` delivery, omit credentials
and referrer, use the required `follow` redirect mode, and time out after 5 seconds. The native binary makes one
bounded request after producing stdout, waiting at most 5 seconds before exit. Delivery failure never changes inference
results. Delivery is best effort; these are operational statistics, not billing records.

## Record fields

All listed fields are present. `null` means unavailable or inapplicable; zero is a measured
zero. IDs use `inf_` plus a UUIDv7 encoded according to the
[TypeID specification](https://github.com/jetify-com/typeid).

| Field | Meaning |
| --- | --- |
| `schema_version` | `minifield.runtime-inference/1` |
| `inference_id` | Unique UUIDv7 TypeID for this terminal inference |
| `mode` | `single_step` or `autoregressive` |
| `status` | `completed` or `failed` for the current host APIs |
| `error_code` | Bounded stage code on failure, never error text |
| `runtime.version` | Runtime package version |
| `runtime.build_id` | SHA256 of Rust/WGSL/Metal/browser module sources, Cargo files, target, and build profile |
| `runtime.git_commit` | Build-time source revision, nullable for source archives |
| `runtime.target` | `native` or `wasm` |
| `model.id` | `sha256:` followed by the bundle fingerprint |
| `model.name` | Human-readable model name, currently unavailable from the raw-assets load API |
| `model.revision` | Bundle fingerprint |
| `model.bundle_sha256` | Domain-separated SHA256 of config, weights, and tokenizer SHA256 digests, in that order |
| `model.architecture` | `lfm2` |
| `model.parameter_count` | Logical dense-equivalent parameters, tied weights counted once |
| `model.weight_formats` | Sorted inventory, including dense formats, packed formats, and scale formats |
| `origin.integration_id` | SDK/deployment integration label |
| `origin.product_id` | Optional product label |
| `origin.website_origin` | Scheme, host, and port only; nullable for native |
| `origin.application_id` | Optional native/browser application label |
| `origin.application_version` | Optional application release |
| `origin.environment` | `production`, `preview`, `development`, or `test` |
| `hardware.kind` | `cpu` for the CLI or a software adapter, `gpu` for a GPU adapter |
| `hardware.vendor` | Coarse vendor from the actual adapter when exposed, otherwise null |
| `hardware.architecture` | Native CPU architecture (`arm`, `x64`, etc.); null when unavailable |
| `platform.host` | `native` or `browser` |
| `platform.os` | `{family, major_version}` or null; version currently null |
| `platform.browser` | `{family, major_version}` or null; full user-agent string excluded |
| `execution.backend` | `cpu` or `webgpu` for the current hosts |
| `execution.compute_precisions` | `["fp32"]` for current production kernels |
| `execution.elapsed_ms` | Tokenization through terminal inference result; excludes loading and delivery |
| `execution.tokenization_ms` | Elapsed through tokenization, when available |
| `execution.time_to_first_token_ms` | Time to first emitted token; null for single-step or no token |
| `execution.tokens` | `{input, output}` logical token counts; classifier/choice output is zero |
| `execution.cache` | `{token_positions_reused, rebuilds}` for prefix branching within this call |
| `execution.prefill` | Phase record described below |
| `execution.decode` | Phase record for autoregression, null for single-step |
| `execution.estimated_flops` | Decimal string for total estimated arithmetic, preserving integer precision |
| `execution.flops_estimator_version` | `lfm2-dense-equivalent/1` |
| `execution.flops_estimate_coverage` | `complete` for completed calls, `partial` for failures |
| `execution.fallback_used` | A cached attempt required full-prompt recomputation |
| `single_step` | `{predictions_produced, alternatives_evaluated}` or null |
| `autoregressive` | `{decoding_mode, constraint, max_output_tokens, stop_reason}` or null |

Each phase contains `elapsed_ms`, `forward_passes`, `token_positions_processed`, and
`estimated_flops`. Prefill covers input preparation after tokenization and all prompt-side
passes, including cache creation and discarded work before fallback. Decode begins after
the first usable prefix. A failed prefill leaves decode measurements unavailable. Single-step
criterion scoring puts all passes in prefill.

`predictions_produced` is 1 for successful classification/choice and 0 on failure.
`alternatives_evaluated` is the classifier width or number of criteria. Choice input tokens
count the base once plus all criterion tails. Cache reuse counts each branched prefix,
including prefixes constructed within the same inference. Decode KV-cache reads are already
accounted for by the FLOP estimator and aren't added to the prompt-cache reuse counter.

Autoregression currently uses `greedy` decoding. Constraint is `none` or `tool_call`.
The requested maximum is retained; stop reason distinguishes `end_token`, `output_limit`,
`context_limit`, `constraint_complete`, and `error`. Browser generation appends each emitted
token except the final one; native generation currently appends every emitted token. Counters
record those actual passes instead of deriving work from output-token count.

GPU architecture stays null because the current wgpu adapter interface doesn't expose the
browser's architecture field. It isn't inferred from the browser's CPU architecture or from
a device marketing name. Unknown browser vendors also stay null. No machine ID is created.

## FLOP estimator

The estimate counts dense-equivalent matrix products, causal QK/AV products, and convolution,
using 2 FLOPs per multiply-add. It excludes normalization, nonlinearities, dequantization,
sampling, and data movement. `complete` means every recorded forward pass was included in
this estimator's defined scope; it doesn't claim a hardware instruction count or energy use.

The executor counts actual rows, cached context lengths, final-layer FFN row reduction, and
the actual classifier/vocabulary head width. Mixed precision doesn't change the equivalent
arithmetic count. Warmup outside an inference is excluded; newly built caches and fallback
recomputation inside the inference are included. Parameter count comes from an equivalent
dense inventory so packed codes and scales don't inflate it.

The bundle hash is `SHA256("minifield.runtime-bundle/1\0" || config_digest || weights_digest || tokenizer_digest)`,
where the separator ends with a single zero byte and each digest is 32 raw bytes. It describes
the exact loaded assets, independent of filenames or download URLs.

## Data and verification

The record builder receives counts and timings. Prompts, generated text, token IDs, logits,
selected classes, tool names, full URLs, error messages, machine identifiers, and user/session
identifiers are excluded. Browser transport sends no cookies and no referrer. No storage or
session identifier is used for telemetry.

`cargo test -p minifield-runtime-telemetry -p minifield-infer -p minifield-executor-core`
checks TypeID conformance, phase totals, mixed formats, logical parameter counts, full/cached
work, native completion boundaries, rejected input, and zero-output work. `npm test` checks
batch bounds, configuration, opt-out, origin stripping, and failed delivery.

The browser harness checks actual WebGPU classification and delivery to a loopback receiver.
With `--generation-bundle`, it also checks generation, choice, constrained fallback, and
failed generation. Its synthetic LM must have a 65,536-token head, a contiguous tokenizer
covering ASCII and single-token `true`/`false`, and weights that greedily select `a`.
All qualification records use environment `test` and a local receiver.
