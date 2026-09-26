# Bundle qualification

Run a real classifier bundle through the native wgpu host and save the evidence. Profiles contain explicit paths and settings. They don't contain a developer's private model directory.

```sh
scripts/check.sh qualify /path/to/profile.json
scripts/check.sh bench /path/to/profile.json
python3 tools/qualification/runner.py bench /path/to/profile.json \
  --output runs/qualification/result.json
```

The check script builds the locked Rust 1.89 release example first. The direct Python command uses the already built `target/release/examples/classify`, or the profile's explicit `command` array. It never invokes a shell.

## Profile

Paths resolve against the profile's directory. `tokenizer` resolves against `bundle`, with `tokenizer/tokenizer.json` as the default. Use `"tokenizer": "tokenizer.json"` explicitly for a legacy root tokenizer.

```json
{
  "schema_version": 1,
  "name": "classifier-device-check",
  "bundle": "../models/classifier",
  "prompts": "prompts.json",
  "classes": 8,
  "context": 512,
  "repetitions": 3,
  "warmups": 1,
  "timeout_seconds": 120,
  "max_lut2_bytes": 67108864,
  "lut2_mode": "auto",
  "cached": false,
  "allow_lut2_fallback": false,
  "comparison_kind": "implementation_parity",
  "expected_logits": "expected.json",
  "tolerances": {"absolute": 0.0001, "relative": 0.0001}
}
```

`prompts.json` is a nonempty array of nonempty strings. A cached run also needs a nonempty common token prefix, with at least 1 token left in every tail. Context and class count are explicit admission limits.

Unknown profile fields, nonfinite numbers, duplicate fields, and invalid types reject. `qualify` requires `expected_logits` or `reference_run`. Benchmarks require at least 2 repetitions.

## Reference evidence

An expected-logit file binds the logits to their inputs:

```json
{
  "schema_version": 1,
  "classes": 8,
  "context": 512,
  "artifacts": {
    "weights_sha256": "<actual SHA-256>",
    "config_sha256": "<actual SHA-256>",
    "tokenizer_sha256": "<actual SHA-256>",
    "prompts_sha256": "<actual SHA-256>"
  },
  "logits": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
}
```

Replace the example values with independently recorded evidence. The matrix must contain 1 row per prompt and exactly `classes` finite values per row. `reference_run` can point to a previous successful runner result instead.

`implementation_parity` requires identical weight, config, tokenizer, and prompt hashes, matching argmax, and logits within tolerance. `quantization_quality` permits different weight hashes while keeping the other inputs matched. Its output reports measured logit drift and argmax agreement against that reference. Neither comparison establishes task accuracy without labeled product evidence.

Parity tolerance uses `abs(actual - reference) <= absolute + relative * abs(reference)`. Quality comparisons require explicit `quality_thresholds`, for example `{"max_absolute_delta": 0.1, "minimum_argmax_agreement": 0.95}`. Those values belong to the profile and must be justified by its application. There are no built-in quality scores or model-specific thresholds.

## Matched benchmarks

`bench` runs raw, down-only LUT2, and automatic LUT2 on the same weight file, through both full and cached inference. It interleaves the 6 variants and reverses their order on alternating repetitions.

Every process warms its loaded model before measuring. Initialization, warmup, cached-base prefill, individual inference timings, and total process wall time are stored separately. The median is the median of each repetition's mean prompt inference time; the raw samples remain in the result.

Kernel parity compares these variants against raw/full with identical artifacts. Comparisons between different quantized weight files belong to `quantization_quality`.

LUT2 dispatch expectations follow the submitted token rows. Full inference submits each whole prompt; cached inference submits the common base and each tail separately. The runtime uses fused FFN tiles at 96 or more rows with compatible packed gate/up/down weights, and always runs the final layer's FFN at 1 row. A run with no eligible LUT2 tile reports `effective_lut2_mode: "not_applicable"` and `lut2_fallback: false`. Eligible tiles still require repacked codes and the expected down/pair dispatch counts, including warmup passes. Results expose `submitted_token_rows` and `expected_lut2_dispatch_counts` alongside actual counts.

Set `reference_run` and `max_relative_slowdown` to enforce a latency regression bound, such as `1.10` for a maximum ratio of 1.10. That reference must contain the same artifacts and each matching variant. Check the saved adapter and platform before interpreting a comparison across machines.

## Recorded evidence and failures

Each JSON result contains artifact and profile hashes, Git revision, dirty-worktree status, a source fingerprint before and after the run, executable hashes, platform, adapter identity, requested and effective LUT2 modes, skipped roles, recorded kernel dispatch counts, raw logits, token IDs, timings, and accounted backend memory.

Memory is the backend's accounted bytes after the run, including physical GPU pools, staging, and fixed uniforms. It doesn't measure process RSS, driver allocations, or peak usage. Timing includes host submission and completion observation.

Missing assets, empty inputs, nonfinite or incomplete output, mismatched hashes/logits, unexpected LUT2 or software-adapter fallback, missing dispatch evidence, process errors, and polling/process deadlines fail the run with a nonzero exit. Partial evidence never receives a passing status.

The native example accepts explicit arguments and retains strict compatibility parsing for `MINI_FFN_LUT2`, `CLASSIFY_PREFIX`, and `MINIFIELD_WGPU_STATS`. `CLASSIFY_PREFIX=0` means false; misspelled LUT2 modes reject. Explicit arguments take precedence.

## Portable checks

```sh
scripts/check.sh quick
scripts/check.sh ci
scripts/check.sh gpu
scripts/check.sh wasm
scripts/check.sh browser --bundle /path/to/bundle --prompts /path/to/prompts.json
uv run --locked --project tools/qualification python -m unittest discover -s tools/qualification/tests -v
```

`quick` checks formatting and the CPU, SIMD, executor, grammar, tokenizer, and harness tests. `ci` runs locked workspace tests and Clippy, contract checks, the independent quantization reference, active JavaScript tests, and locked converter tests, lint, formatting, and Rust compatibility checks. Missing prerequisites fail. Install Rust 1.89, Node 22 or newer, and uv; Python dependencies resolve from each tool's committed lockfile.

`gpu` sets `MINIFIELD_REQUIRE_GPU=1`, enables the experimental-kernels feature, and runs the GPU library, parity, and low-bit suites serially. A missing adapter must fail that tier.

`wasm` checks compilation. `browser` builds the release WASM package and forwards arguments to `scripts/check_browser.mjs`; it requires an installed Chrome and a wasm-bindgen CLI matching Cargo.lock. The CI workflow runs portable checks and WASM compilation; actual GPU and browser execution remain hardware release gates.

Harness unit tests use a tiny fake executable to exercise validation, result transport, deadlines, comparisons, and error reporting. They make no model or kernel claim.
