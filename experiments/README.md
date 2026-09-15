# Runtime experiments

This directory owns the ordered runtime experiment plan. All experiments are planned and have no recorded runs. The [runtime procedure](../docs/procedure.md) owns product acceptance and release; the [Wafer reading](../docs/wafer-inference-experiments.md) and [xn reference](../docs/xn-optimization-reference.md) supply research context.

## Order

| ID | Experiment | Start when | Status |
| --- | --- | --- | --- |
| 0001 | [Complete decoder and device baseline](0001-decoder-baseline.md) | A candidate bundle, numerical reference, and reference device are selected | Planned |
| 0002 | [Prefix and session reuse](0002-prefix-session-reuse.md) | 0001 provides a correct incremental decoder and comparable timings | Planned |
| 0003 | [Bounded prefill and decode allocations](0003-bounded-attention.md) | 0001 exposes attention allocations and positions/masks can be checked | Planned |
| 0004 | [Packed-weight execution](0004-packed-weight-execution.md) | 0001 provides exact export semantics and operation profiles | Planned |
| 0005 | [Constrained action generation](0005-constrained-action-generation.md) | The baseline action format, parser, and retry policy are fixed | Planned |
| 0006 | [KV-cache compression](0006-kv-cache-compression.md) | KV storage or traffic blocks a useful context length after earlier work | Planned, conditional |
| 0007 | [Speculative decoding, including DSpark](0007-speculative-decoding.md) | Decode still accounts for enough action latency to justify extra work and memory | Planned, conditional |

Start with 0001, then follow the order above. Each comparison uses a named control configuration. Earlier experiments can be rejected or deferred without blocking later independent work. Record the reason for any change in order; skip conditional experiments when their trigger is absent.

## Scope and ownership

Keep batch 1, one active sequence, and a resident target model. Prefill chunks, multiple kernel threads, and verification of several draft tokens all operate within that sequence. General request batching and parallel candidate search require a separate product decision.

Core inference stays in Rust with thin browser integration. Runtime owns inference, context, tools, sessions, and device evidence. Training supplies versioned bundles and numerical references; platform persists canonical jobs, artifacts, and product validation results. Consume versioned artifacts without importing sibling source trees.

These files contain reusable protocols and concise decision summaries. Product-specific inputs, adapters, private fixtures, and development evidence belong in named, dated folders under the parent workspace's `experiments/`. Keep generated datasets, weights, customer content, and raw logs outside Git. Deployed runs consume registered immutable inputs and persistent artifacts, independently of local workspace paths.

## Shared comparison rules

Before a run, fill a [run record](run-record-template.md) with the exact bundle digest, runtime revision, device/host, product contract, inputs, control, treatment, and settings. Select numerical tolerances, resource limits, behavioral margins, and a minimum useful improvement before inspecting results. Deployment limits come from procedure section 1; an unset limit means the candidate can't be accepted yet.

Change one factor at a time. Freeze tokenizer/template, observations, decoding, stopping, tool behavior, and retry policy except for the factor explicitly under test. After isolated comparisons, measure the chosen combination against both its immediate control and the original baseline. Use deterministic synthetic cases for parity and separate representative product cases for usefulness.

Proposed screening: 30 warm observations per configuration, 10 cold launches per named cache state, and a 10-minute repeated-task run. Alternate control/treatment order and retain failures. Report sample counts, p50/p95, denominators, and uncertainty. These are screening settings; repeat noisy comparisons and increase sampling before release qualification.

Name cache states: missing downloaded assets, cached files with a fresh process, resident model, and resident reusable prefix. Keep input/output lengths, power mode, background load, and thermal conditions visible. Native and browser results get separate records.

Measure download, verification, load/compile, tokenization, prefill, first token, action validation, host tool execution, first completed action, and complete-request latency. Time to a validated action and time to its completed host effect are separate metrics. For clarification/rejection responses, report completion latency separately from action cases.

Account for resident weights, staging copies, KV payload, allocated capacity, scratch, runtime, and host app memory. Count shared allocations once. Record CPU/GPU copies and measurement blind spots. Include cancellation, release, repeated sessions, app responsiveness, and sustained behavior.

## Decisions and evidence

Each file defines its experiment-specific criteria. Shared requirements are preserved task success, scope rejection, clarification, permission handling, false rejection, and zero observed forbidden effects. Predeclare behavioral margins and check per-family regressions. The procedure's acceptance gates still apply to the exact delivered bundle and host.

Record a decision of accepted, rejected, deferred, or inconclusive with evidence IDs and the compared versions. A baseline can reveal resource failures that motivate optimization; record those failures and keep product qualification pending. An optimization is accepted only when it meets its declared objective, numerical/behavioral requirements, and applicable resource budgets.

Update the experiment's result section and this index together. Keep raw evidence in the configured artifact location and retain concise, non-sensitive conclusions here. A microbenchmark improvement alone doesn't establish a product improvement.
