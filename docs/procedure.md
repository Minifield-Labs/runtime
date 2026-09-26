# Development and release procedure

Ship a reproducible runtime whose admitted formats, numerical behavior, and tested platforms are explicit. Each change needs checks proportional to the behavior it can affect.

## Development loop

1. Read the relevant crate/tool documentation and identify the observable change.
2. Preserve artifact and public API compatibility unless an explicit version change is intended.
3. Add a regression for a real failure or a new boundary. Mechanical moves should preserve behavior and public signatures.
4. Run `scripts/check.sh quick`, then affected hardware/format checks.
5. Run `scripts/check.sh ci` before review. Record commands, results, skips, and hardware requirements accurately.
6. Review the final diff, documentation, and tracked files. Use conventional commits.

The pinned toolchain and lockfiles support reproducibility. Core libraries don't read environment configuration or import sibling repositories. Host programs pass typed settings.

## Required checks

| Change | Required evidence |
| --- | --- |
| Every code change | Formatting, relevant tests, `quick` |
| Merge candidate | `ci`: workspace tests/Clippy, contract pins, JS, reference tools, converters, cross-language checks |
| Shader, GPU ownership, dispatch | `gpu` on a real adapter, including representation and shape boundaries |
| Loading/execution or visible output | Actual bundle qualification against matched expected results |
| Browser bindings/host | `wasm`, actual browser execution, JavaScript tests |
| Performance claim | Matched repeated benchmark, correctness pass, artifact/device identity, raw samples |
| Quantization algorithm | Independent byte fixtures, malformed inputs, same-weight parity, separate quality evaluation |

`gpu` requires an adapter and fails when none is available. Portable workspace tests may skip adapter-dependent cases; that output doesn't establish GPU qualification. `wasm` checks compilation only.

Hosted CI covers portable checks and WASM compilation. Hardware release evidence comes from a real supported host. Never label a skipped GPU test or a WASM build as a browser pass.

## Four measurements

**Encoding conformance:** known bytes, layout order, scale rounding, codebook values, reserved codes, dimensions, malformed headers. Use hand-defined fixtures and independently decoded values.

**Implementation parity:** CPU/GPU, raw/repacked, and full/cached execution on the same decoded weights. Require finite values, declared absolute/relative tolerances, and stable argmax. Exercise dispatch boundaries, awkward dimensions, limits, cancellation, and recovery.

**Quantization quality:** compare the quantized artifact with a dense/training reference on frozen evaluation data. Report task accuracy, rejection and false rejection where relevant, plus logit/argmax diagnostics. Runtime parity can't establish model quality.

**Performance:** use the same assets, prompts, context, precision, adapter, and power conditions. Separate initialization, warmup, cached-base prefill, steady inference, and wall time. Record sample counts and dispersion before interpreting small differences.

## Bundle qualification

Use an explicit profile as described in [tools/qualification](../tools/qualification/README.md). Paths resolve relative to the profile, so private directories never enter source.

```sh
scripts/check.sh qualify /absolute/profile.json
scripts/check.sh bench /absolute/profile.json
```

The runner records hashes, source identity, adapter, requested/effective policy, dispatches, timings, resource accounting, tokens, and logits. Benchmarks interleave raw/down/auto across full/cached execution, with warmups and repeated runs. Incomplete/nonfinite output, unexpected fallback, incompatible references, and declared gate failures reject.

Historical [FFN measurements](ffn-prefill-experiments.md) explain LUT2 and useful shapes. They belong to their recorded revision/device. Use a fresh matched baseline when accepting a change.

## Browser gate

Build with the locked `wasm-bindgen` version and run [actual browser qualification](../web/README.md). Execute WebGPU inference, compare full/cached results, exercise rejected input and recovery, and save browser/adapter identities.

A browser-specific failure blocks browser support. Native Metal parity doesn't cover JavaScript scheduling, WASM memory, asynchronous device creation, or browser callbacks.

## Publication checklist

- A fresh clone runs portable checks without sibling repositories, private models, or global Python packages.
- README commands, supported-format claims, active links, and crate paths match the tree.
- Contracts and locks are pinned. Attribution and third-party notices are retained.
- Only compact synthetic fixtures are tracked. Weights, customer prompts, credentials, generated reports, and personal paths stay outside Git.
- Hardware claims name the tested device/browser and evidence. Gaps are explicit.
- Experimental kernels are opt-in and can't silently replace defaults.
- Resource and input limits have boundary tests and useful errors.
- Archive qualification output with the release revision and asset hashes outside Git.

A product release also needs application authorization, privacy, session behavior, and task-quality gates. This repository doesn't implement or certify those application behaviors.
