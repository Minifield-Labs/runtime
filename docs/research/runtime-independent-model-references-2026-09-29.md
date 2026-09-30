# Independent references for the production-model hill-climb

## Hypothesis

A correct GPU optimization preserves the outputs of the exact stored model.
The supplied classifier and pointer encoder give the evaluation real matrix
shapes, operator mixes, vocabularies, and output heads. Each stored precision
needs an independent reference; a separately quantized model changes the
question being tested.

The baseline runtime alone is an insufficient mathematical reference for a new
loader or encoder. A shared implementation mistake could survive a comparison
between 2 runtime revisions.

## Fix

Built an independent NumPy reference in the dated experiment. It reads the
prepared safetensors bytes and implements signed INT8, ternary, and NF4 decoding
from their format definitions. It imports no converter quantizer, training
model, or runtime executor.

The reference executes F32 RMS normalization, split-half RoPE, gated causal or
centered convolution, grouped-query attention with the correct visibility,
SwiGLU residual blocks, final normalization, and the task's output projections.
Stored F16 and BF16 values expand to F32. That matches the runtime's current
arithmetic boundary.

Pointer decoding uses the average of start/end option softmaxes. Extraction
compares the absent marker with selectable source tokens and finds a contiguous
half-open source span. Ties retain the caller's first option or the earliest
shortest valid span. Reference output includes raw start/end scores and the
continuous probabilities/presence values as well as discrete predictions.

The pinned training decoder promotes network scores to FP64 host values before
softmax, ordinal expectation, presence thresholds, and span-score addition.
The reference now follows that boundary. Rust uses compensated FP64 summation
for softmax denominators and ordinal expectations, matching the inspected
CPython 3.12/3.13 policy for finite nonnegative terms in candidate order.

Frozen classifier inputs contain 3 existing production board prompts and 2 held
out board variations. The pinned full and pruned tokenizers each produce
345–347 tokens. Frozen pointer inputs use the training joint layout: a BOS-led
question, BOS-led options, then a BOS-led source. The 3 public requests contain
111, 220, and 308 tokens, with 2, 4, and 4 questions.

The Rust tokenizer produced all token IDs. Every query, option, and source run
starts with BOS ID 1. The inputs contain no private labels or expected outputs.
Question markers and selection masks remain explicit host inputs.

## Test

Hand-calculated tests independently establish signed INT8 bytes, ternary pair
order, NF4 nibble order, scale application, RMS normalization, split-half
rotation, stable option ties, extraction boundaries around a masked gap, FP64
span addition, and decisions near 0.5. All 9 tests pass under Python
3.12.11/NumPy 2.2.6 and Python 3.13.7/NumPy 2.5.3.

The reference completed every frozen case for all 9 prepared artifacts:
4 classifier precisions, 4 pointer-encoder precisions, and the supplied mixed
QAT classifier. Every output is finite. Each reference file binds its expected
values to the complete weights, config, tokenizer, and input hashes. It also
records NumPy version, reference source hash, inspected mathematical source
pins, and resolved-token provenance.

The initial artifact provenance was corrected before freezing the campaign.
Every stored tensor hash matched before and after the metadata correction,
which preserves the reference's mathematical inputs. That correction leaves
expected values unchanged, and the reference asset hashes name the final files.

A separate decoder correction replayed the 12 saved pointer cases from their
raw scores. All 18,672 raw F32 logits and model artifacts stayed unchanged;
80 continuous decoder values changed by at most `6.369013638707344e-8`, and all
40 discrete decisions stayed identical. The proof preserves raw-prefix hashes,
network-function AST hashes, both reference-source versions, and the separate
decoder execution environment.

Executing the archived primary training decoder functions gives 0 bit
mismatches against Rust debug and release across 16 cases, 91 continuous FP64
values, and 44 decisions in both supported Python versions. The cases include
`[0, -3e-16, -37.429947 repeated 8 times]`: compensated summation produces
presence `0.5`, while naive FP64 accumulation produces `0.4999999999999999`.
This verifies the decoder boundary for these inputs and execution environments.

Actual Apple M1 Max qualification completed 14 cells on each GPU backend:
full/cached classification for 5 artifacts, plus full inference for 4 encoder
precisions. Backend identity and dispatch counts establish independent native
Metal and WGPU-through-Metal execution. These are correctness runs; their
timings aren't sustained-performance measurements.

| Backend | Independent typed decisions | Independent numerical gate |
| --- | --- | --- |
| WGPU-through-Metal | Match in all 14 cells | Pass in 13 cells; ternary encoder fails |
| Native Metal | Match in all 14 cells | Pass in 13 cells; ternary encoder fails |

The numerical gate remains `abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)`.
Ternary encoder maximum absolute drift reaches `0.00322223` on WGPU and
`0.00483179` on native Metal across the 3 requests. CPU also fails the strict
independent comparison for that artifact. All 10 typed pointer decisions match
on each backend. The independent numerical failures remain in the record.

A temporary CPU trace isolates the 111-token ternary request. Replaying each
operation from identical captured inputs makes all 181 tested normalization,
projection, centered convolution, head normalization, RoPE, attention, and
SwiGLU operations pass the original numerical gate. Accumulated F32 rounding is
supported for that trace. It doesn't establish the cause of every GPU
discrepancy or qualify the 220/308-token requests' individual primitives.

## Resolution

The hill-climb has an independent mathematical reference for every evaluated
model and precision. Keep its numerical pass/fail results and exact typed
decisions alongside the backend qualification record.

Before candidate development, run A/A calibration with the same starting
runtime and exact artifacts on each backend and execution mode. Freeze the
starting runtime's continuous outputs separately for every model, precision,
backend, and full/cached mode. Bind these references to source, executable,
artifact, and input hashes. A/A stability is required before campaign freeze;
the qualification runs above don't complete that gate.

Candidates must pass the original `1e-4` absolute and `1e-4` relative guard
against their immutable starting-runtime reference, and match the common
independent typed decisions exactly. The initial anchor stays fixed after
promotion, so repeated small changes can't accumulate unchecked drift.
Replacing an anchor or changing criteria requires a new campaign. The strict
independent ternary failure remains visible under this policy.

These references define same-artifact implementation parity checks. They aren't
a model quality score, and no candidate can earn a performance gain by changing
its precision, tokenization, question layout, or discrete predictions.
Runtime parity and platform qualification are separate evidence from the
NumPy computation reported here.

## Reproduction and pins

The `2026-09-29-runtime-gpu-hill-climb` experiment stores
`scripts/numpy_reference.py`, its hand-calculated tests, frozen inputs, token
IDs, and `references/{bundle-id}.json` outside Git.
`prepared-artifacts-all.json` names each bundle and its final weight hash.
The reference script consumes those assets through NumPy and safetensors byte
ranges; it doesn't depend on a sibling source import.

The decoder proof lives in `runs/reference-decoder-f64-correction.{json,md}`,
`runs/reference-decoder-f64-correction-sources/`, and
`runs/decoder-compensated-sum/parity-report.json`. Hardware qualification lives
in `runs/pre-freeze-{wgpu,native}-001/`. The conditioning investigation lives in
`notes/ternary-conditioning-trace.md`, `runs/ternary-layer-trace/`, and
`runs/ternary-rope-runtime-policy-diagnostic/`.

The inspected reference code is pinned to
`96fd486d1601bb910b390652bb21d53463e8237a` for the backbone/pointer equations
and `736fe888a6754d0f80657879e080ac02dbfc154d` for segment-aware encoder
boundaries. Those pins describe the equations inspected. The supplied
checkpoints' actual training Git revisions are unknown, and their source
identities are bound to the supplied file hashes and recorded run/source IDs.
