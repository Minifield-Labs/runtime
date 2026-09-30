# Freeze the yardstick before climbing

Date: September 29, 2026. Scope: the campaign controller and native host contract.
This records the harness work before any performance change earns promotion.

## Hypothesis

The useful score is sustained completed predictions/s from the models we'll
deploy. A kernel timer helps explain a result, but the promotion gate needs to
include recording, submission, completion, readback, and task decoding.

The classifier and pointer encoder have different workloads. FP16, INT8, NF4,
ternary, and the supplied mixed QAT artifact also carry different weights.
Each admitted model/precision/backend gets its own immutable reference and its
own performance score. WGPU on Metal and a dedicated native Metal implementation
need separate builds, identities, and evidence.

We expect pairing to reduce interference from changing system conditions.
Independent fresh processes, fixed work, and a 15-second gap between measured
runs should make the score harder to push around with a lucky sample.

## Fix

Added a standard-library Python controller under
[tools/hillclimb](../../tools/hillclimb/README.md). Its
[native protocol](../../tools/hillclimb/protocol.md) describes actual complete
predictions and the information that must accompany them.

A campaign freezes the criteria, inputs, same-precision oracle, model/config/
tokenizer/input hashes, Python controller, and complete native evaluation-host
crate. Running it requires the campaign digest printed during freeze. Replacing
the lock and recomputing its internal hash won't satisfy that external pin.

The environment is also explicit. Reviewed build/host allowlists are the entire
child environment; telemetry is forced off. Ambient PROFILE, RUSTFLAGS, Cargo
profile overrides, compiler wrappers, injected libraries, and GPU-selection
overrides reject. Resolved tool binaries, hashes, `rustc -vV`, and global/ancestor
Cargo configs enter the record and must stay stable across both builds and the
screen/confirmation pair.

The controller builds champion and candidate using exact arguments:

```text
cargo +1.89.0 build --release --locked --package minifield-evaluation
  --bin minifield-eval --no-default-features --target-dir PER_BUILD_BACKEND_DIR
  [--features BACKEND_FEATURES]
```

Each build/backend gets a separate target directory. The build command, source
revision, dirty state, source fingerprint, and executable digest enter the
evidence. Source fingerprints include modified and untracked files, plus
Git-ignored source files that could still enter a build. Generated target,
dependency, and cache directories are excluded explicitly.

Both builds first run correctness against the original same-precision oracle.
Every measured cycle is checked again. The controller requires complete counts,
finite outputs, frozen numerical tolerances, and identical discrete predictions.
Reference widths and pointer question contracts are validated before building.

Pointer outputs retain the start/end score matrices and the continuous decoded
values. Their discrete decisions retain option indices, rounded ordinal levels,
binary decisions, and spans or absence. That lets us tolerate declared numeric
drift while still rejecting a changed product decision.

The host reports hashes of the exact bytes it loaded. Those must match the
oracle. It also reports its actual backend/API/device/driver, measured-loop
dispatch counts, and peak backend-accounted memory. Requested WGPU/Metal cannot
quietly become CPU or another API. Required kernel counts must be present.

## Frozen criteria

The reference manifest in the tool README specifies this initial promotion
profile:

| Criterion | Value |
| --- | --- |
| Independent runs per build/cell | 8 |
| Paired ABBA blocks | 4 |
| Gap after measured process ends | At least 15 seconds |
| Minimum confirmed rate improvement | 1% |
| Protected slowdown allowance | 2% |
| Confidence interval | 95% paired-block percentile bootstrap |
| Bootstrap resamples / seed | 10,000 / 47 |
| Numeric absolute / relative tolerances | Explicit per campaign; example 0.0001 / 0.0001 |
| Memory budget | Explicit peak backend-accounted bytes |

These thresholds are reviewed campaign choices. A/A measurements and actual
bundle parity need to justify the final deployment settings before freeze.
Changing a criterion starts a new campaign; it can't repair a losing candidate
inside an existing campaign.

The score for 1 process is completed predictions divided by total measured
prediction-loop seconds. The cell score is the arithmetic mean of independent
process rates. Initialization, warmup, per-prediction latency, and process wall
time remain separate fields.

ABBA means champion, candidate, candidate, champion in fresh processes.
Statistics resample whole ABBA blocks, preserving their paired structure.
Each resample recomputes the ratio of candidate and champion rate totals,
matching the declared arithmetic-mean score. Averaging block ratios can hide a
regression when blocks have different absolute rates, so those ratios remain
diagnostics only.
There is no score averaged across different models, precisions, backends, or
full/cached modes.

Exploratory 4-run campaigns have 2 blocks. They remain inconclusive and can't
qualify a candidate for promotion, even when their point estimate looks good.
The enforced minimum is 4 blocks for a promotion decision. More evidence can be
declared before a campaign starts.

Every protected cell must satisfy its slowdown bound. At least 1 cell must clear
the improvement bound. A passing screen produces `eligible_for_confirmation`.
A separate fresh confirmation must bind the exact same source and executable
identities and pass the same gates before the controller reports `promoted`.

Confirmation pins the screening-result digest externally. It reconstructs the
prior decision from every locked cell and raw sample, rather than trusting saved
status labels or statistics. At least 1 same screened cell must improve again;
a win that moves from classifier to encoder can't substitute for it.

The manifest predeclares candidate and finalist attempt limits. A persistent
flocked hash-chain ledger reserves each run before execution. Failed attempts
consume the budget and retain evidence. Review a multiplicity policy before
freeze: the attempt bound limits repeated testing, while the percentile interval
continues to describe its own per-cell measurement scope.

The native host's pending-completion loop sleeps 1ms. This policy is part of its
frozen source. Throughput includes that polling behavior, completion, readback,
and decoding. A kernel candidate can't change the measurement policy to earn
a gain.

## Test

The portable suite executes tiny synthetic native hosts through the real
transport, validation, timing-order, and decision paths. An injected monotonic
clock verifies all 15-second gaps without making the tests wait minutes.

48 tests passed on September 29, 2026:

```sh
PYTHONPATH=tools/hillclimb python3 -m unittest discover \
  -s tools/hillclimb/tests -v
```

The suite checks ABBA order, separate release targets, externally pinned lock
identity, changed weights/oracles/native hosts, actual loaded hashes, incomplete
outputs, changed decisions, nonfinite values, backend substitution, missing
dispatches, memory budgets, impossible reported timings, and process deadlines.
It also changes source and executable files after the final host returns to
prove that a late mutation can't receive a passing result.

The source identity test creates a real temporary Git repository. It verifies
that tracked edits, untracked modules, and ignored source modules change the
fingerprint, while generated target files don't. Another test proves that a
2-block exploratory result cannot enter confirmation.

Additional tests reject rehashed altered status/statistics, deleted cells,
incomplete prior raw samples, changed derived rates, unpinned screens, and a
different winning cell at confirmation. Environment tests cover PROFILE,
RUSTFLAGS, Cargo profile/wrapper knobs, changes during execution, and telemetry
forced to 0 in actual test hosts. Global and ancestor Cargo config changes,
compiler identity changes, and exhausted persistent budgets also reject.
Identical config bytes at distinct external ancestor paths are also rejected;
checkout-local configs retain their relative path identity.

Final provenance regressions reject unchanged Cargo configs that point at a
mutable external compiler or wrapper. Active configs are parsed as standard
TOML and can't select a compiler through `[build]` or Rust/Cargo/compiler `[env]`
overrides. HOME/Cargo/Rustup lookup directories must be absolute. Legacy config
filename priority preserves both recorded hashes, and malformed active TOML
rejects before building. External config `include` directives also reject,
including an unchanged directive whose included file changes between attempts.

Ruff lint and formatting passed for the controller package. Its dependency-free
uv lock was generated offline. The portable tests make no GPU or model throughput
claim; those require actual locked campaigns and device execution.

## Resolution

The controller now has a concrete acceptance decision and preserved rejection
evidence. It refuses stale or changed references, changed harness code, missing
work, changed predictions, unintended execution paths, and unsupported hardware.
Failures retain process stdout/stderr, requests, and a failing result.

Actual deployment freeze still depends on the native host, both model execution
paths, protected-precision packing, and real bundle references passing their
own checks. That work belongs in subsequent stacked changes. The classifier,
encoder, and dedicated Metal cells must each supply actual execution evidence;
an unavailable backend remains a failure rather than inheriting another score.

After those prerequisites pass, run A/A, review the thresholds once, and freeze
the deployment campaign. Kernel changes can then climb against the same yardstick.
Keep a fixed original correctness reference while updating the performance
champion, so tolerated numeric drift can't accumulate unnoticed.
