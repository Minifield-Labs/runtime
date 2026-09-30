# Native evaluation protocol, version 1

The controller launches a newly built native executable:

```text
minifield-eval --request /absolute/request.json
```

The host writes exactly 1 JSON response to stdout. Diagnostics go to stderr.
Unknown fields, duplicate fields, malformed values, unimplemented tasks, and
unavailable backends reject with a nonzero exit. A host must complete all GPU work
and read back the outputs before it counts a prediction.

## Request

```json
{
  "schema_version": 1,
  "backend": "wgpu_metal",
  "arithmetic": "f32",
  "task": "classifier",
  "bundle": "/absolute/bundle",
  "tokenizer": "/absolute/bundle/tokenizer/tokenizer.json",
  "inputs": "/absolute/inputs.json",
  "classes": 8,
  "context": 512,
  "mode": "full",
  "phase": "measure",
  "warmups": 2,
  "measured_cycles": 20,
  "deadline_seconds": 180,
  "lut2_mode": "off",
  "max_lut2_bytes": 67108864
}
```

`backend` is `cpu_reference`, `wgpu_metal`, or `native_metal`. WGPU/Metal and a
dedicated native Metal implementation remain separate identities. The host
checks the actual selected API and rejects a fallback.

`task` is `classifier` or `pointer`; `classes` is required only for classification.
`mode` is `full` or `cached`. Cached mode must use the model's admitted common
prefix behavior and reject incompatible inputs. Its initialization timing
includes preparing the reusable base. Each completed prediction executes the
same immutable input with valid cache ownership and branch state.

`phase` is `correctness` or `measure`. Correctness requests have 0 warmups and
1 measured cycle. Measured requests use the frozen positive warmup and cycle
counts. Warmup outputs don't enter measured cycles, but must not corrupt state.
The controller also enforces the process deadline.

## Inputs

```json
{
  "schema_version": 1,
  "task": "classifier",
  "cases": [
    {"id": "case-1", "text": "Product input"},
    {"id": "case-2", "token_ids": [1, 12, 7]}
  ]
}
```

Each classifier case provides exactly 1 of `text` and `token_ids`. Pointer cases
provide `id`, `token_ids`, `questions`, and optional `segments`. Omitted segments
are all 1; 0 masks a token. All candidates and selected source positions must
share the query token's active segment. Input IDs are distinct.

```json
{
  "id": "pointer-case",
  "token_ids": [1, 12, 7, 4],
  "segments": [1, 1, 1, 1],
  "questions": [
    {"query_index": 0, "option_indices": [1, 2], "kind": {"type": "choice"}},
    {
      "query_index": 0,
      "option_indices": [],
      "kind": {
        "type": "extract",
        "absent_index": 1,
        "source_start": 2,
        "selectable": [true, true],
        "presence_threshold": 0.5
      }
    }
  ]
}
```

Question kinds are `choice`, `ordinal`, `binary`, and `extract`. Choice and ordinal
have distinct, nonempty option indices. Binary has exactly 2 options and a
`positive_option` index of 0 or 1. Extract has the fields shown above. Its absent
marker cannot be selectable, and its presence threshold is finite in [0,1].

## Response

```json
{
  "schema_version": 1,
  "artifacts": {
    "weights_sha256": "digest of weights actually loaded",
    "config_sha256": "digest of config actually loaded",
    "tokenizer_sha256": "digest of tokenizer actually loaded",
    "inputs_sha256": "digest of inputs actually loaded"
  },
  "backend": {
    "implementation": "wgpu_metal",
    "api": "metal",
    "device": "actual adapter identity",
    "driver": "actual driver identity"
  },
  "initialization_seconds": 0.5,
  "warmup_seconds": 0.2,
  "cases": [
    {
      "id": "case-1",
      "completed_predictions": 2,
      "elapsed_seconds": 0.21,
      "latencies_seconds": [0.1, 0.1],
      "outputs": [[0.1, 0.9], [0.1, 0.9]],
      "predictions": [1, 1]
    }
  ],
  "dispatch_counts": {"actual_kernel_name": 12},
  "resources": {
    "accounted_bytes": 1000000,
    "peak_accounted_bytes": 1200000
  }
}
```

The example case is a 2-class, 2-cycle response. Actual widths and counts must
match the request and oracle. Cases match the input IDs and order exactly.

`api` is `cpu` or `metal`; implementation must equal the requested backend.
All backend identity fields must remain identical across matched processes.
The CPU host reports its implementation/device identity rather than a GPU name.

Each case includes 1 finite output vector, complete prediction, and positive
latency per measured cycle. `completed_predictions` equals the requested cycles.
Classification predictions use stable argmax with the smallest index on ties.
Pointer decisions are an ordered array with 1 object per question:

| Question | Exact decision | Continuous payload appended to output |
| --- | --- | --- |
| Choice | `{"type":"choice","index":0}` | Option probabilities |
| Ordinal | `{"type":"ordinal","level":1}` | Expected value, then option probabilities |
| Binary | `{"type":"binary","value":true}` | Positive probability, then option probabilities |
| Extract | `{"type":"span","start":0,"end":2}` or `{"type":"absent"}` | Presence probability |

Ordinal decisions round the nonnegative expected value to the nearest level,
with a .5 tie rounding upward. Binary decisions use probability >= .5. Span
indices form a half-open source-relative token range. Each pointer output vector
starts with flattened start scores `[questions,tokens]`, then flattened end
scores, then the continuous payloads in question order. Continuous values use
frozen numerical tolerances; discrete decisions must match exactly.

The response's artifact digests come from the bytes the host actually loaded.
They must exactly match the immutable reference, including the input JSON hash.

`elapsed_seconds` spans only the complete prediction loop for the case, including
host submission and completion/readback. It includes loop overhead. The sum of
per-prediction latencies cannot exceed that elapsed time. It excludes initialization
and warmup, which have separate durations. Each measured cycle must execute real
inference; a host cannot substitute stored outputs or skip cache branches.

Dispatch counts cover the measured prediction loops only, subtracting the
post-warmup counter snapshot. GPU cells declare explicit minimum counts for their
intended path. A kernel count of 0 never proves that path executed.

The native host's completion loop sleeps 1ms for each pending poll. This policy
is frozen with the host source. Timings include completion, readback, decoding,
and that polling policy; kernel campaigns can't change it to earn a speed gain.

Peak memory means the maximum **backend-accounted bytes** observed during that
process. Ending accounted bytes cannot exceed it. Driver allocation, process RSS,
and stored artifact size are separate measurements.

## Immutable reference

```json
{
  "schema_version": 1,
  "task": "classifier",
  "artifacts": {
    "weights_sha256": "actual model.safetensors digest",
    "config_sha256": "actual config.json digest",
    "tokenizer_sha256": "actual tokenizer digest",
    "inputs_sha256": "actual inputs JSON digest"
  },
  "cases": [
    {"id": "case-1", "output": [0.1, 0.9], "prediction": 1}
  ]
}
```

The reference contains 1 output and prediction per input. Optional `provenance`
records the starting source/executable and capture procedure. Record those
identities when creating deployment references. Hash the exact reference file
when freezing; never regenerate it from an optimized candidate.

The controller compares every scalar using the frozen inequality:

```text
abs(actual - oracle) <= absolute_tolerance + relative_tolerance * abs(oracle)
```

Every complete prediction must equal its same-precision oracle. Each precision
gets its own reference. Quantization quality across different representations
is a separate evaluation and never changes this implementation-parity gate.
