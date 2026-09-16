# 0001: Complete decoder and device baseline

Status: planned. Owner: runtime inference and host integration, with training supplying the immutable bundle and numerical reference.

## Description and hypothesis

Build the first complete Rust decoder and measure one real host action. The current candidate is Falcon-E; record the exact bundle and any explicitly selected alternative. Include tokenizer/template handling, positions, incremental KV updates, sampling, stopping, cancellation, and resource ownership.

Hypothesis: the delivered model can reproduce the training/export reference within declared tolerances and execute repeated batch-1 requests through the target host. Stage timings and memory accounting will expose the next useful optimization.

## Prerequisites

- Select the reference device, browser/native host, engine, and limits from [procedure section 1](../docs/procedure.md#1-define-the-reference-deployment).
- Register a real model bundle and matching numerical outputs. Contract fixtures alone can't establish model execution.
- Fix public context, supported host tools, permissions, and synthetic/product evaluation cases. Keep expected outcomes outside model-visible input.
- Fill the [run record](run-record-template.md) under the [shared comparison rules](README.md#shared-comparison-rules).

## Comparison and method

1. Compare tokenizer IDs, serialized input, layer intermediates, final logits, and incremental outputs against the reference on fixed small contexts. Exercise cold and repeated execution.
2. Load once and serialize requests through an explicit lifecycle. Implement the intended browser Worker path when qualifying a browser target; preserve asynchronous GPU completion before buffer reuse/release.
3. Complete an authorized host action and check its actual effect. Exercise rejection, clarification, tool errors, cancellation before/during execution, and a subsequent request.
4. Profile supported 512/2,048/4,096/8,192-token synthetic contexts with 32/128-token diagnostic continuations. Record unsupported lengths and failures. Use natural stopping on product requests; diagnostic lengths aren't product limits.
5. Run the shared warm/cold/sustained screen. Record the intact-model baseline first, then repeat for the trained bundle when available. Give each bundle its own baseline.

## Measurements

Record all shared stage timings and memory categories, including action-ready versus host-completed time. Add per-operation timing, weight conversions, dispatches, submissions, transfers/readbacks, cache growth, allocation counts, and cancellation latency. Document profiling overhead and memory visibility limits.

Record numerical errors, token/action differences, task success, correct scope rejection, false rejection, and forbidden effects. Preserve per-family results and exact configurations.

## Completion and stop criteria

Complete the baseline when supported workloads run repeatedly, numerical checks pass, a real action round trip works, lifecycle/error cases behave correctly, and another engineer can reproduce the report. Persist device validation through platform's product path for product acceptance.

Resource failures remain explicit findings. Use them to select an optimization or a smaller supported workload; keep affected deployment qualification pending. Stop on numerical mismatch, stale/corrupted state, forbidden effects, or unsafe resource growth and resolve correctness before collecting comparative performance claims.

## Deliverables and result

Deliver the exact runnable configuration, numerical report, stage/memory profile, product results, failure list, and ranked bottlenecks. Add immutable evidence IDs and the decision here after execution.

Result: not run. No baseline measurements or qualification decision recorded.

## References

- [Runtime procedure](../docs/procedure.md) and [Wires foundation audit](../docs/wires-audit.md).
- [Wafer reading and cache arithmetic](../docs/wafer-inference-experiments.md).
- [xn profiling and benchmark limits](../docs/xn-optimization-reference.md).
