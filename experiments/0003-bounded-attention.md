# 0003: Bounded prefill and decode allocations

Status: planned. Owner: runtime inference and the selected engine.

## Description and hypothesis

Process long prompts with bounded chunks and attention scratch, then decode from compact grouped-query KV storage. Preserve absolute positions and causal masks throughout.

Hypothesis: tiled attention, controlled prefill chunks, and reusable buffers reduce peak memory and interruption delay while preserving numerical behavior and useful task latency.

## Prerequisites

- [0001](0001-decoder-baseline.md) supplies an attention reference, allocation profile, and exact tensor layouts.
- Fix the prefix policy from [0002](0002-prefix-session-reuse.md), or explicitly keep reuse disabled in both arms.
- Record supported lengths, scratch/capacity limits, responsiveness and cancellation budgets in the [run record](run-record-template.md).

## Comparison and method

1. Compare supported prefill chunks of 256, 512, and 1,024 tokens against the backend default with the same total prompt. Keep attention algorithm and cache state fixed for this comparison.
2. Compare reference prefill with tiled prefill, then compare reference decode with fused single-query decode as a separate change. Check boundaries just below, at, and above chunk sizes, plus partial final chunks, nonzero prefix offsets, and cache capacity exceeding active length.
3. Preserve the configured number of KV heads. Map query groups onto compact KV storage directly and verify output against the reference. Avoid retaining expanded head copies or a full prompt-by-prompt attention score tensor in the optimized path.
4. Separately trial reusable scratch and pooled buffers with capped physical retention. Release/recycle only after GPU completion. Measure cancellation during prefill and decode, cleanup, and the next request.
5. Where profiles show submission overhead, compare command recording/submission strategies independently. Validate the actual browser/native APIs and include pool retention and host responsiveness in the [shared screen](README.md#shared-comparison-rules).

## Measurements

Record prefill, first-token, action-ready, and complete-action latency; peak/retained memory; scratch scaling with context; allocation count; compilation keys; dispatches/submissions; and readbacks. Measure cancellation delay and host responsiveness under sustained requests.

The [xn reference](../docs/xn-optimization-reference.md) includes an online-softmax decode kernel, active cache views, and buffer pooling. Its fused API assumes equal query/KV head counts and its WebGPU path requires native features. Adapt and test these boundaries before using its patterns in our runtime.

## Acceptance and stop criteria

Accept a configuration when required context lengths fit the declared memory limit, numerical/product checks pass, and cancellation/responsiveness meet their budgets. Require the declared memory or latency gain; record any tradeoff against first-action latency before adopting it.

Reject position/mask errors, expanded KV retention that breaks the budget, premature buffer reuse, leaks, or unsupported browser operations. Defer extra fusion when it doesn't improve a measured bottleneck. Keep isolated results before comparing the combined configuration.

## Deliverables and result

Deliver a context/chunk comparison, parity cases, allocation-lifetime design, device traces, and selected limits with evidence IDs.

Result: not run. No chunk size, fused path, or pooling policy accepted.

## References

- [FlashAttention](https://arxiv.org/abs/2205.14135).
- [MLX LM prefill settings](https://github.com/ml-explore/mlx-lm#long-prompts-and-generations).
- [Wafer attention memory analysis](../docs/wafer-inference-experiments.md#falcon-and-minicpm5-cache-arithmetic).
