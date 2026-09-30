# Pointer host summation correction

## Hypothesis

Promoting pointer decoding to FP64 still leaves a threshold error when ordinary
addition loses small positive candidates. With F32 logits
`[0, -3e-16, -37.429947 repeated 8 times]` in both heads, naive accumulation gives
presence `0.4999999999999999`; the pinned training decoder gives `0.5` and returns
a span at the default threshold.

The primary source is training revision
`96fd486d1601bb910b390652bb21d53463e8237a`.
[`field_decode.probabilities`](https://github.com/Minifield-Labs/minifield-training/blob/96fd486d1601bb910b390652bb21d53463e8237a/src/minifield_training/evaluation/field_decode.py)
normalizes Python floats with `math.exp` and Python `sum`.
[`pointer.decode`](https://github.com/Minifield-Labs/minifield-training/blob/96fd486d1601bb910b390652bb21d53463e8237a/src/minifield_training/evaluation/pointer.py)
uses the same builtin for ordinal expectations. Both are ordinary Python float
streams. The pinned project requires Python `>=3.12,<3.14`.

CPython's float summation uses Neumaier compensation in both
[3.12.11](https://github.com/python/cpython/blob/v3.12.11/Python/bltinmodule.c#L2464)
and [3.13.7](https://github.com/python/cpython/blob/v3.13.7/Python/bltinmodule.c#L2520).
The source doesn't call `math.fsum` or a NumPy reduction for these operations.

## Fix

The Rust typed decoder now uses one private compensated FP64 accumulator for
softmax denominators and ordinal expectations. It processes terms in candidate
order, tracks the rounding residue of each addition, and adds the accumulated
residue once at the end. The magnitude branch compares values directly because
both streams contain finite nonnegative terms and their admitted bounds prevent
overflow.

The policy is fixed in runtime code. It preserves each scalar subtraction and
addition, uses no fused operation, and doesn't depend on a future Python builtin
implementation. Network arithmetic and raw logits remain F32. The existing
separate-head softmax averaging, FP64 span addition, threshold, scan-order ties,
integer ordinal rounding, and JSON fields retain their defined behavior.

## Test

Both new executor regressions failed before the change. The extraction fixture
returned absent, and an ordinal fixture returned `0.5000000000000003` instead
of the primary source's `0.5000000000000002`. Both now pass in debug and release.
A host integration test runs the actual typed decoder and verifies that the
corrected presence, ordinal expectation, raw logits, and discrete decisions
survive the existing JSON path.

Independent validation executes the archived primary Python functions directly,
without importing training packages. It decodes stored raw logits from all 12
FP16/INT8/NF4/ternary cases plus 4 synthetic boundaries. Python 3.12.11 and 3.13.7
produce the same 91 continuous FP64 bit patterns and 44 discrete decisions;
the Rust decoder matches those outputs in debug and release. This validation executes no model and
changes no stored network outputs or artifacts.

The outside-Git evidence is in the workspace experiment's
`runs/decoder-compensated-sum/`: complete pinned source files and hashes, a primary
function harness, Rust debug/release decoder outputs, and `parity-report.json`.
Customer-derived logits remain in that experiment; committed regressions use
small synthetic vectors.

## Resolution

Pointer host summation now has an explicit numerical contract matching the
supported CPython versions on the qualified host. The scalar accumulation policy
is independent of interpreter version. Bit-identical decoding across every
platform still requires matching the FP64 exponential implementation; exact
threshold ties can expose a different platform's `exp` rounding. The pinned
training package excludes Python 3.11, whose plain float summation can produce a
different boundary decision.

This is a correctness prerequisite for the frozen eval. It carries no kernel
speed claim and doesn't alter the outstanding ternary network parity diagnosis.
