# Measure completed predictions through one frozen host

## Hypothesis

Kernel microbenchmarks can miss upload, task construction, completion polling,
readback, and decoded decisions. A hill-climb needs complete predictions from
the actual deployment artifacts, with initialization and warmup recorded
separately and each backend identified explicitly.

## Fix

Added `minifield-eval`, a small native host with separate modules for request
admission, consumed-asset hashing, model preparation, backend selection,
completion, prediction, measurement, and reporting. The Python controller owns
build pairing, process order, cooling intervals, and promotion decisions.

The request pins task, backend, F32 arithmetic, exact bundle/tokenizer/inputs,
context, full/cached mode, warmup count, measured cycles, deadline, and repack
policy. Unknown fields reject. Each backend constructs its own implementation;
unavailable native Metal or WGPU selections fail before measurement.

The host loads the bundle once and resolves inputs before timing. Cached
classification constructs an immutable common-prefix base outside the measured
phase. Every measured tail completes from that same base. Pointer inference
uses complete bidirectional execution and all question heads.

Every prediction includes task construction, recorded operations, submission,
polling, readback, and decoded decisions. Pending completion sleeps for a fixed
1 ms. Deadlines check recording, polling, loading, and final decoding; the
controller also enforces a process deadline. A failed or late prediction never
produces a completed success record.

Output preserves every raw F32 logit exactly as an F64 JSON value. Pointer
probabilities and expectations retain the decoder's F64 values. Stable discrete
decisions are separate fields. Ordinal admission checks the rounded integer
level, preserving legitimate expectation rounding slightly beyond an endpoint.

Dispatch counts are measured-phase deltas after warmup. A checked GPU-only aggregate counts their total while retaining every raw name. Freezing that aggregate permits legitimate kernel replacements without pinning the campaign to a particular algorithm. Changed backend identity, raw-name collisions, or aggregate overflow reject. The resource snapshot
is taken while the model and any cached base remain owned. The high-water count
includes initialization and warmup, with its accounting scope stated explicitly.
It doesn't claim driver memory or process RSS.

## Test

Synthetic host tests exercise real CPU loader/executor paths, full/cached
classification, pointer execution, actual asset hashes, exact fixed work,
complete continuous/discrete output serialization, unsupported backends,
nonfinite output, resource snapshots, and deadline failures.

Independent review found an ordinal endpoint case where compensated expectation
is `10.000000000000002` for 11 options. Its correct rounded decision is `10`.
Rejecting the continuous value before rounding excluded a valid result; the
endpoint regression now preserves the value and checks the integer decision.

Actual pre-freeze qualification completed 14 cells on each GPU implementation:
5 classifier artifacts in full/cached mode and 4 encoder precisions. All typed
decisions match the independent references. Classifier cells and FP16/INT8/NF4
encoder cells pass the independent numerical guard. MagicBox ternary retains a
strict numerical failure, scoped in the [reference investigation](runtime-independent-model-references-2026-09-29.md).

The independent [decoder proof](runtime-pointer-host-summation-2026-09-29.md)
checks 91 F64 values and 44 decisions across supported Python versions and Rust
debug/release. These checks qualify mathematical and host boundaries; their
single-cycle correctness timings don't establish a performance gain.

One controller fixture used a 0.5-second Python startup budget while parallel
checks compiled Rust. The process correctly timed out before printing, producing
an empty retained log. Test-only budgets now allow 3 seconds for that fixture
and 1 second for the direct process fixture, against their deliberate 20-second
stalls. Benchmark deadlines, controller behavior, and scoring criteria are
unchanged.

## Resolution

The host is ready to become part of the frozen benchmark driver after portable,
hardware, and actual-bundle checks. Record starting outputs separately per
backend and full/cached mode, prove unchanged-build A/A stability, then bind
source, executable, input, reference, and criterion identities before candidates.

The immutable original reference stays fixed after champion promotions. Each
candidate must match exact typed decisions and the original numerical guard,
then earn a sustained-throughput gain and repeat it in fresh confirmation.
This host carries no performance claim of its own.
