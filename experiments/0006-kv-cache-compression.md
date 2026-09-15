# 0006: KV-cache compression

Status: planned, conditional. Owner: runtime attention, with training/evaluation checking behavior.

## Description and hypothesis

Reduce the precision of retained keys and values when cache memory or traffic blocks a useful context length. Keep model weights unchanged and account for compression metadata and processing cost.

Hypothesis: an 8-bit cache can meet a named device/context requirement with acceptable task behavior and latency. Consider 4-bit only after the 8-bit comparison establishes a reason to continue.

## Prerequisites and trigger

- [0001](0001-decoder-baseline.md) supplies a correct reference; [0002](0002-prefix-session-reuse.md) and [0003](0003-bounded-attention.md) establish bounded cache retention and attention behavior.
- Review [0004](0004-packed-weight-execution.md) so weight conversions and copies aren't being misidentified as KV cost.
- Record the context length and measured KV memory/traffic limit that this experiment should resolve. Defer if the desired workload already fits and KV isn't a meaningful cost.
- Establish a validated FP16/BF16 cache control. If the backend currently stores FP32, validate conversion to 16-bit as its own step first.
- Predeclare quantization parameters, task margins, context/latency limits, and [run gates](run-record-template.md).

## Comparison and method

1. Compare 16-bit and 8-bit cache storage with the same bundle, prompts, prefix policy, decoding, and attention layout. Name the scale/grouping scheme separately for keys and values.
2. Count scales, alignment, conversion buffers, and any recent high-precision region. Keep original KV head counts and record actual allocated capacity.
3. Exercise short and long sessions, prefix hits/misses, appends, truncation, cache eviction, chunk boundaries, and cancellation. Include exact entity IDs, numeric arguments, retrieval from earlier turns, and corrected user instructions.
4. Compare numerical outputs and product outcomes against the 16-bit control. Evaluate drift over long sessions and rare action families, alongside average task success.
5. If 8-bit leaves a justified gap, compare a specified 4-bit method against both controls. Keep that decision and its tolerances explicit. Repeat the [shared screen](README.md#shared-comparison-rules) on the exact host.

## Measurements

Record KV payload, metadata, active/allocated bytes, scratch, full-process peak memory, cache quantization/dequantization time, prefill/decode latency, and first-action/complete-request latency. Distinguish storage reductions from observed reductions in memory traffic.

Report numerical error, task success, numeric/entity mistakes, rejection, false rejection, and permission behavior by context length. Use the [existing payload table](../docs/wafer-inference-experiments.md#falcon-and-minicpm5-cache-arithmetic) only as an estimate; record measured allocation totals.

## Acceptance and stop criteria

Accept only if the cache resolves the named device/context limit and meets predeclared numerical, behavioral, latency, and memory gates. Record the exact quantization scheme and supported context range.

Reject unacceptable long-session drift, argument corruption, forbidden effects, or conversion costs that erase the useful gain. Stop at 8-bit when it meets the requirement and further compression lacks a measured purpose. Preserve the 16-bit control for regression and fallback.

## Deliverables and result

Deliver the cache format/precision policy, full memory accounting, context-length comparison, long-session evaluations, and decision with evidence IDs.

Result: not run. Trigger has not been established; no reduced-precision cache accepted.

## References

- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) motivates different treatment of key and value quantization. Its results require validation on our exact models and workloads.
- [Wafer cache arithmetic](../docs/wafer-inference-experiments.md#falcon-and-minicpm5-cache-arithmetic).
