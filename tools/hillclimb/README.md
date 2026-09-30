# Frozen runtime campaigns

Compare complete predictions from 2 identified runtime builds against the same
precision-specific correctness oracle. Each model, stored precision, arithmetic
mode, backend, and full/cached mode gets an independent score. Results never
average CPU with GPU or NF4 with ternary.

The controller uses Python 3.11+ and the standard library. Model files, frozen
campaigns, reference outputs, and generated evidence stay outside Git.

## Workflow

1. Prepare the real deployment bundles and frozen inputs. Keep embeddings and
   output heads at their admitted higher precision. Record packing provenance
   and sizes independently of runtime timing.
2. Record 1 complete prediction per input using an identified starting runtime
   at each stored precision. Bind that reference to the exact model, config,
   tokenizer, and input hashes. An independently implemented mathematical oracle
   is useful additional evidence; it has a separate identity and purpose.
3. Review the campaign's tolerances, repetitions, dispatch requirements, memory
   budget, and improvement thresholds. Freeze it only after all references exist.
4. Commit the baseline and candidate source. Run the same locked campaign for
   each change. Never edit the evaluation code, criteria, inputs, or references
   during that campaign.
5. Confirm an eligible finalist with a fresh process sequence. The confirmation
   binds the exact source and executable identities from screening.

Run from the repository root:

```sh
PYTHONPATH=tools/hillclimb python3 -m minifield_hillclimb freeze \
  /private/experiment/campaign.json \
  --output /private/experiment/campaign.lock.json

PYTHONPATH=tools/hillclimb python3 -m minifield_hillclimb run \
  /private/experiment/campaign.lock.json \
  --expected-campaign-sha256 SHA_PRINTED_BY_FREEZE \
  --champion /absolute/baseline-checkout \
  --candidate /absolute/candidate-checkout \
  --output /private/experiment/screen-001

PYTHONPATH=tools/hillclimb python3 -m minifield_hillclimb run \
  /private/experiment/campaign.lock.json \
  --expected-campaign-sha256 SHA_PRINTED_BY_FREEZE \
  --champion /absolute/baseline-checkout \
  --candidate /absolute/candidate-checkout \
  --confirmation-of /private/experiment/screen-001/result.json \
  --expected-screen-sha256 RESULT_SHA_PRINTED_BY_SCREEN \
  --output /private/experiment/confirmation-001
```

Keep the printed campaign digest in the experiment's reviewed record. The run
command requires that external digest, so a rehashed replacement lock can't
silently relax a gate. A new harness or new criterion needs a separately reviewed
campaign. Existing output directories and locks are never overwritten.

## Campaign manifest

Paths resolve relative to the manifest. The tokenizer path resolves relative to
the bundle, including a legacy root-level `tokenizer.json` when explicitly set.
The lock resolves paths and pins their content hashes.

```json
{
  "schema_version": 1,
  "name": "deployment-runtime-v1",
  "build": {
    "cargo": "cargo",
    "rust_toolchain": "1.89.0",
    "package": "minifield-evaluation",
    "binary": "minifield-eval",
    "host_source": "/absolute/reviewed-checkout/crates/evaluation-host",
    "host_relative_path": "crates/evaluation-host",
    "features_by_backend": {
      "cpu_reference": [],
      "wgpu_metal": ["wgpu"],
      "native_metal": ["metal"]
    }
  },
  "environment": {
    "build": {"PATH": "/reviewed/rustup/bin:/usr/bin:/bin", "HOME": "/reviewed/home"},
    "host": {"PATH": "/reviewed/rustup/bin:/usr/bin:/bin", "HOME": "/reviewed/home"}
  },
  "attempt_budget": {
    "candidate_limit": 12,
    "finalist_limit": 3,
    "ledger": "attempts.json"
  },
  "criteria": {
    "independent_runs": 8,
    "cooldown_seconds": 15,
    "minimum_improvement": 0.01,
    "maximum_slowdown": 0.02,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.0001,
    "bootstrap_resamples": 10000,
    "confidence": 0.95,
    "seed": 47,
    "minimum_paired_blocks": 4,
    "resource_limit_bytes": 4294967296,
    "build_deadline_seconds": 1800
  },
  "cells": [
    {
      "id": "classifier-nf4-wgpu-full",
      "model": "classifier",
      "precision": "nf4",
      "arithmetic": "f32",
      "backend": "wgpu_metal",
      "bundle": "bundles/classifier-nf4",
      "tokenizer": "tokenizer/tokenizer.json",
      "inputs": "classifier-inputs.json",
      "reference": "references/classifier-nf4.json",
      "task": "classifier",
      "classes": 8,
      "context": 512,
      "mode": "full",
      "warmups": 2,
      "measured_cycles": 20,
      "deadline_seconds": 180,
      "lut2_mode": "off",
      "max_lut2_bytes": 67108864,
      "dispatch_requirements": {"gemm": 1}
    }
  ]
}
```

The values above demonstrate the schema. Actual references, class counts, context,
dispatch names, tolerances, and memory budget must describe the chosen bundles.
There are no built-in accuracy or model-specific tolerance claims.

Storage precision is `fp16`, `int8`, `nf4`, `ternary`, or `mixed_qat`. Arithmetic
is independently pinned to `f32` in this version. Each cell records its actual
weight hash; a precision label alone doesn't establish packing conformance.
The packing manifest and independent decoder checks provide that evidence.

Build commands use exact arguments, pinned Rust 1.89.0, `--release --locked`, and
separate target directories for each build/backend. All builds complete before
measurement. The actual host response must identify the requested implementation
and API. An unavailable dedicated Metal backend fails its own cell.

The complete native evaluation-host crate is part of the frozen harness.
`host_source` identifies the reviewed host at freeze time; `host_relative_path`
locates it in both measured checkouts. Its content has to match the frozen digest.
The Python controller is independently pinned. Source changes to either harness
require a new reviewed campaign before measuring kernels.

The manifest also pins a reviewed allowlist of environment variables for builds
and hosts. Include every present setting from the allowed keys, including PATH,
HOME, locale, temporary directories, and build-only SDK/Cargo/Rustup locations.
Freeze and run reject missing or changed settings. Child processes receive only
these explicit values; `MINIFIELD_TELEMETRY=0` is forced in both phases.
Ambient PROFILE, RUSTFLAGS, Cargo profile overrides, compiler wrappers, injected
libraries, and GPU-selection overrides reject. They can't silently bias a run.

Builds permit PATH, HOME, CARGO_HOME, RUSTUP_HOME, TMPDIR, TMP, TEMP, LANG, LC_ALL,
SDKROOT, MACOSX_DEPLOYMENT_TARGET, and SOURCE_DATE_EPOCH. Hosts permit PATH, HOME,
TMPDIR, TMP, TEMP, LANG, and LC_ALL. Review their actual values before freezing.
The paths above demonstrate the schema; use the reviewed host's real values.
HOME and any admitted CARGO_HOME or RUSTUP_HOME must be absolute paths. This keeps
controller lookup and subprocess lookup on the same directories.

Each run records resolved cargo/rustup entry points, the actual selected cargo
and rustc binaries and hashes, cargo version, and `rustc -vV`. It fingerprints
global Cargo config files and every source-directory ancestor's `.cargo/config`
and `config.toml`, including absent files. Both builds must have matching active
configs and tools. Changes during execution or between screening and
confirmation reject.

Active Cargo configs are parsed as standard TOML before building. Compiler
selection through `build.rustc`, `build.rustc-wrapper`, or
`build.rustc-workspace-wrapper` rejects. Cargo `[env]` Rust/Cargo/compiler
overrides, plus HOME/PATH replacement, also reject. Config `include` directives
reject because they load settings outside the recorded file set. The supported
path uses the probed toolchain directly. When both config filenames exist, Cargo
selects the legacy filename; both paths and hashes remain recorded. See the
[Cargo configuration reference](https://doc.rust-lang.org/cargo/reference/config.html).

## Measurement and decisions

Every cell first checks both builds against the frozen same-precision oracle.
Every later measured prediction is checked again. Finite logits, exact complete
predictions, fixed numerical tolerances, case IDs, counts, memory, and required
dispatches all have to pass. An unexpectedly substituted backend rejects.

An ABBA block runs champion, candidate, candidate, champion in 4 fresh processes.
There are at least 4 independent runs per build, arranged in complete blocks.
Each measured process receives the same warmup count and fixed prediction cycles.
The next measured process starts at least 15 seconds after the previous process
ends. A monotonic clock measures that gap, including between different cells.

For each process, sustained predictions/s is completed predictions divided by
the sum of measured case durations. The reported score is the arithmetic mean of
these independently measured rates for that cell. Initialization, warmup, and
process wall time are retained separately. Prediction latencies and every output
remain in the raw samples.

Paired statistics compare the candidate/champion rate ratio within each entire
ABBA block. A deterministic percentile bootstrap resamples whole blocks. It
never treats successive dispatches within a process as independent measurements.
Fewer blocks than the frozen minimum make the result inconclusive.

Confirmation requires at least 4 independent ABBA blocks, which is 8 runs per
build. A 4-run exploratory campaign provides 2 blocks and stays inconclusive
even when its point estimate looks faster. It can't enter confirmation or earn
promotion. Use the 8-run manifest above for a promotion campaign.

The lower interval bound has to satisfy the protected slowdown allowance for
every cell. At least 1 cell must clear the minimum improvement bound. No cell can
hide another's regression through an aggregate score. Passing screening reports
`eligible_for_confirmation`; a fresh matched confirmation of the identical
finalist reports `promoted` only when the same gates pass again.

Confirmation requires the externally retained screening-result SHA-256 printed
by the run command. Before confirming, the controller validates every locked cell
and every raw correctness/measurement sample, recomputes rates and statistics,
and rejects altered summaries or incomplete evidence. At least 1 cell that
improved during screening must improve again. A different winning cell can't
substitute for it.

The bootstrap interval summarizes these observed blocks. Run an A/A campaign
before selecting thresholds and use the same frozen criteria for each candidate.
The attempt budget declares a maximum number of screens and confirmations before
freeze. A persistent locked hash-chain ledger reserves each attempt before work
starts; failed executions consume attempts. Exhaustion rejects before building.
Preserve the ledger and all referenced run directories. A new reviewed campaign
needs a new ledger. The chain verifies retained records. Keep ledger backups and
digests beside the reviewed experiment evidence.

The bounded attempt budget doesn't establish a family-wide false-promotion rate.
Review the confidence threshold and multiplicity policy with the workload before
freeze, and record the planned total candidate/finalist count. The per-cell
percentile intervals retain their stated scope.

## Retained evidence

Each new output directory contains the lock, source revisions and fingerprints,
exact build arguments, executable hashes, build/host stdout and stderr, requests,
tool/environment/config identities, attempt reservation, backend identity,
raw outputs, dispatch counts, timings, and decisions. Timeouts
kill the whole process group and preserve partial stdout/stderr. A failed process
or invalid response leaves a failing result, never a partial pass.

The controller verifies frozen artifact/reference/harness hashes before every
host execution and after it, source fingerprints and executable hashes on both
sides of execution, and both sources again at the end. The host also reports
hashes of the exact assets it loaded, which must match the frozen oracle.
Source fingerprints cover tracked, modified, untracked, and Git-ignored source
files. Generated `target`, `.venv`, Python caches, and node_modules trees are
excluded. Evidence directories must stay outside both measured checkouts.
Keep measured checkouts untouched and suspend unrelated GPU work during a run.

Memory gates use **peak backend-accounted bytes**. They don't measure driver
allocations or process RSS. Capture those separately when a change targets them.
Packed artifact size belongs beside each cell's deployment evidence.

## Portable tests

```sh
PYTHONPATH=tools/hillclimb python3 -m unittest discover \
  -s tools/hillclimb/tests -v
```

Tests execute tiny synthetic hosts and the real response-validation/decision
path. An injected clock exercises the full cooldown without 15-second sleeps.
They cover frozen identities, backend fallback, missing dispatches, wrong and
incomplete predictions, nonfinite outputs, memory/timing gates, deadlines,
ABBA ordering, separate targets, and fresh confirmation. They establish harness
behavior; actual bundle/device results provide runtime evidence.
