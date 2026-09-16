# 0002: Prefix and session reuse

Status: planned. Owner: runtime context and sessions; coordinate serializer changes with training.

## Description and hypothesis

Retain valid KV state for a stable product prefix or the longest identical conversation prefix. At batch 1, keep one reusable allocation and append the current request's tail.

Hypothesis: repeated app instructions, tool schemas, and history avoid enough prefill work to reduce first-action latency within the retained-memory budget, while preserving uncached decisions.

## Prerequisites

- [0001](0001-decoder-baseline.md) supplies correct incremental decoding and reproducible timings.
- Freeze tokenizer/template, position/mask semantics, product contract, and authorization scope.
- Define prefix identity, physical capacity, idle retention/release policy, and the [run record](run-record-template.md) gates.

## Comparison and method

1. Compare 3 configurations: resident model with no cross-request KV reuse; stable product-prefix reuse; longest identical token-prefix reuse across turns. Hold the model and decoding policy fixed.
2. Retain the valid prefix, append the new tail, and reset logical length to the reusable boundary for independent requests. Overwrite the old tail. For continuing sessions, retain the longer valid history within budget. For a fully cached prompt, preserve the required final logits or recompute the boundary token before sampling.
3. Exercise exact hits, partial hits, and misses. Change an early token, tool schema, permission scope, observation, template, and bundle. Recompute from the first changed token or invalidate the cache as required.
4. Check logout, context edits, cancellation, overflow, idle eviction, and repeated sessions. Key reuse by exact token prefix, bundle, template/tokenizer, engine/numerical settings, product contract, and authorization scope. The host still checks permissions when executing tools.
5. Run the [shared screen](README.md#shared-comparison-rules) on requests with several reuse lengths and the same resulting context. Verify logits and decisions against uncached execution under the fixed policy.

## Measurements

Record reused/new tokens, hit/miss cause, prefill time, first-action and complete-request latency, and retained/peak memory. Distinguish active length from allocated capacity and report the high-water mark after truncation and idle release.

The [existing cache arithmetic](../docs/wafer-inference-experiments.md#falcon-and-minicpm5-cache-arithmetic) gives 48 MiB for a 2,048-token 16-bit prefix on either current candidate. Adding a 1,024-token tail gives 72 MiB in one allocation; a separate prefix snapshot would raise payload to 120 MiB. Include actual dtype, copies, alignment, and scratch in measured totals.

## Acceptance and stop criteria

Accept when repeated-request latency clears the predeclared improvement threshold and uncertainty check, cache-hit/miss decisions agree with the reference, and retention/cancellation stay within budget. New tokens still read the prefix during attention; measure decode separately.

Reject any stale context, cross-scope reuse, or corrupted tail behavior. Defer if reuse is rare or its latency gain doesn't justify retention. Keep the no-reuse path as the control and fallback.

## Deliverables and result

Deliver the cache identity/invalidation policy, bounded lifecycle design, parity cases, memory/latency comparison, and chosen reuse policy with evidence IDs.

Result: not run. No reuse policy accepted.

## References

- [xn preallocated caches and compatibility limits](../docs/xn-optimization-reference.md).
- [MLX LM prompt caching](https://github.com/ml-explore/mlx-lm#long-prompts-and-generations).
- [Inside vLLM reading priorities](../docs/wafer-inference-experiments.md#additional-resource-inside-vllm).
