# Evaluation host

`minifield-eval` loads an actual runtime bundle once, prepares frozen inputs, warms the selected backend, and reports completed predictions. The qualification controller owns candidate/champion pairing and the `15` second gap between fresh measured processes.

Build the CPU reference host:

```sh
cargo build --locked --release -p minifield-evaluation
```

Build the WGPU host and evaluate on a hardware Metal adapter:

```sh
cargo build --locked --release -p minifield-evaluation --features wgpu
./target/release/minifield-eval --request /absolute/request.json
```

Build the dedicated native Metal host on macOS:

```sh
cargo build --locked --release -p minifield-evaluation --features metal
```

Backend selection is explicit. `cpu_reference` runs the Rust CPU implementation. `wgpu_metal` requires the `wgpu` feature and a hardware Metal adapter. `native_metal` requires the `metal` feature on macOS and constructs the independent native backend. The host never substitutes WGPU or CPU for a missing backend or device.

## Request contract

The input is strict JSON with `schema_version: 1`. Unknown fields reject.

```json
{
  "schema_version": 1,
  "backend": "wgpu_metal",
  "arithmetic": "f32",
  "task": "classifier",
  "bundle": "/absolute/bundle",
  "tokenizer": "/absolute/bundle/tokenizer/tokenizer.json",
  "inputs": "/absolute/classifier-inputs.json",
  "classes": 3,
  "context": 512,
  "mode": "full",
  "warmups": 2,
  "measured_cycles": 6,
  "phase": "measure",
  "deadline_seconds": 120,
  "lut2_mode": "auto",
  "max_lut2_bytes": 67108864
}
```

The bundle contains `config.json` and `model.safetensors`. Arithmetic is fixed to F32 activation and accumulation; stored weights may use an admitted F16, INT8, NF4, ternary, or mixed representation. Each representation needs its own matched reference.

`phase` is `correctness` or `measure`. Correctness requires `measured_cycles: 1`. Both phases use the same model, completion, readback, and decision path. Configured warmups precede the recorded cycles in either phase. The controller normally sets correctness warmups to `0`.

Classifier mode is `full` or `cached`. Cached mode finds the longest shared token prefix while leaving at least `1` tail token per case. It prepares this base once outside measured work; every cycle completes a fresh tail prediction from that immutable base. Pointer tasks require `mode: full`, omit `classes`, and use a bidirectional encoder bundle with the four pointer heads.

`lut2_mode` is `raw`, `off`, `down`, or `auto`. It selects admitted lossless representations for the causal classifier. Pointer inference consumes its canonical stored operands. The model file and same-precision reference stay fixed when changing this runtime option.

## Frozen inputs

Classifier cases supply exactly one of `text` or `token_ids`. Text tokenization happens before warmup and measurement. Frozen IDs bypass tokenization but retain the tokenizer asset hash in the evidence.

```json
{
  "schema_version": 1,
  "task": "classifier",
  "cases": [
    {"id": "request-a", "token_ids": [1, 31, 42]},
    {"id": "request-b", "token_ids": [1, 31, 57]}
  ]
}
```

Pointer cases supply frozen token IDs and question roles. Segment `0` marks padding. Each nonzero segment label occupies one contiguous request; rotary positions restart inside each segment. Omitting `segments` makes every token active in a single segment.

```json
{
  "schema_version": 1,
  "task": "pointer",
  "cases": [{
    "id": "request-a",
    "token_ids": [1, 20, 21, 30, 31],
    "segments": [1, 1, 1, 1, 1],
    "questions": [
      {"query_index": 0, "option_indices": [1, 2], "kind": {"type": "choice"}},
      {"query_index": 0, "option_indices": [], "kind": {
        "type": "extract", "absent_index": 1, "source_start": 3,
        "selectable": [true, true], "presence_threshold": 0.5
      }}
    ]
  }]
}
```

The other question kinds are `{"type":"ordinal"}` and `{"type":"binary","positive_option":1}`. Allowed candidates must be in the query's active segment. Question counts are bounded to `64`; token capacity is bounded by the caller, model config, and encoder's `8192` token limit.

To freeze texts with the actual runtime tokenizer:

```sh
./target/release/minifield-eval --tokenize /absolute/tokenizer.json /absolute/texts.json
```

`texts.json` is an array of strings. Stdout is the corresponding array of token-ID arrays, with automatic BOS insertion disabled.

## Timing and completion policy

Initialization includes reading and hashing the consumed assets, admission, upload, model construction, and cached-base preparation. Warmup duration is separate. Every recorded cycle then measures one complete prediction, including host task construction, recording, submission, polling, readback, and decoded decisions.

The completion loop checks its deadline before and after every poll and sleeps `1` ms after `Pending`. This fixed policy contributes to end-to-end latency. CPU recording may finish inside a single poll; a late result still fails. Deadline checks also cover loading and final host decoding. A failure exits without a success record.

Measured cycles run back to back on fixed inputs. No sleep or optional output length changes the work inside a cycle. The outer controller spaces independent processes and pairs revisions. Warmup dispatches are excluded from measured dispatch counts.

## Output and evidence

Stdout contains one JSON object. Its fields are `schema_version`, `backend`, `artifacts`, `initialization_seconds`, `warmup_seconds`, `cases`, `dispatch_counts`, and `resources`.

Backend evidence records `implementation`, `api`, `device`, and `driver`. WGPU preserves driver-reported strings, including empty values, inside an explicit versioned identity. Native Metal records its registry ID and Objective-C binding version, with unavailable driver versions stated explicitly. Artifact evidence hashes the exact consumed weights, config, tokenizer, and inputs.

Each case records `completed_predictions`, `elapsed_seconds`, per-cycle `latencies_seconds`, continuous `outputs`, and discrete `predictions`. Classifier outputs are logits and predictions are stable first-maximum class indices. Pointer outputs concatenate start logits `[Q,T]`, end logits `[Q,T]`, then continuous answer values in question order:

- Choice: option probabilities.
- Ordinal: continuous expected level, then option probabilities.
- Binary: positive-option probability, then option probabilities.
- Extraction: presence probability.

Discrete pointer decisions are `choice/index`, `ordinal/level` (an integer), `binary/value` (a boolean), `span/start/end` (half-open source-relative tokens), or `absent`. The controller checks continuous tolerance and exact discrete decisions separately. Backend logits retain their F32 values, promoted exactly to FP64 for JSON reporting. Pointer probability, presence, expected-level, and span-score calculations use FP64 host arithmetic, matching the pinned training decoder; they aren't rounded to F32 before a threshold or tie decision.

The decoder applies Neumaier compensation to softmax denominators and ordinal expectations in candidate order. This fixes a Python 3.12/3.13 numerical policy explicitly: tiny positive terms survive accumulation, so threshold decisions match the pinned source. Integer ordinal levels round half away from zero. Admission checks the rounded level; the continuous expectation retains any endpoint rounding residue. The host preserves these FP64 continuous values through JSON reporting.

Dispatch counts are the difference between snapshots immediately after warmup and after the measured loops. Decreasing or disappearing counters and changed backend identity reject. GPU responses also include `gpu_dispatches`, the checked sum of measured raw-kernel deltas. This stable aggregate permits a frozen dispatch requirement while candidates introduce new kernel names. Raw names/counts remain recorded; CPU responses omit the aggregate. Resources report current and peak backend-accounted bytes. The peak spans initialization, warmup, and measured work; it isn't driver memory or process RSS. CPU internal mathematical vectors aren't included in this backend counter.

## Checks

```sh
cargo test --locked -p minifield-evaluation
cargo clippy --locked -p minifield-evaluation --all-targets --features wgpu -- -D warnings
```

Tests load compact synthetic classifier and pointer assets through the real CPU loader/executors, compare full/cached results, verify consumed hashes, exercise fixed work and continuous/discrete serialization, reject nonfinite values, check deadlines across CPU recording, and prove unsupported backend selections fail. Hardware measurements require the controller's actual-bundle qualification.
