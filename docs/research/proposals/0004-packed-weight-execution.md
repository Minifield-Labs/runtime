# 0004: Packed-weight execution

Status: planned. Owner: runtime engines and training export.

## Description and hypothesis

Measure how shipped weights become resident tensors and how kernels consume them. Optimize matrix-vector execution without silently changing the model's effective weights or activation quantization.

Hypothesis: consuming packed weights with suitable layouts and SIMD kernels reduces weight traffic and conversion overhead enough to improve batch-1 action latency within download and memory limits.

## Prerequisites

- [0001](0001-decoder-baseline.md) supplies exact export semantics, numerical reference outputs, and operation profiles.
- Hold prefix, attention, and allocation policies fixed after reviewing [0002](0002-prefix-session-reuse.md) and [0003](0003-bounded-attention.md).
- Inventory shapes, packing, scale granularity, excluded tensors, activation quantization, accumulation precision, and rounding. Record the [shared gates](README.md#shared-comparison-rules) in a [run record](run-record-template.md).

## Comparison and method

1. List each tensor's download format, resident format, scales, conversions, staging copies, and load-time cost. Include embeddings and output projections in the whole-model accounting.
2. Compare compatible scalar/reference and optimized kernels on identical effective weights and activations. For Falcon, reproduce the packed ternary semantics; retain explicit tests for edge values, scale handling, and unsupported shapes.
3. Trial SIMD and layout changes separately. Use single-row decode shapes and representative prefill shapes. Include weight working sets large enough to expose streaming cost on the measured device, alongside hot-cache diagnostics.
4. Trial repacking only for compatible formats. Count repack time, temporary copies, alignment, and retained memory. GGML Q4/Q8 layouts from xn require their own format checks before use.
5. If CPU scheduling dominates, compare 1, 2, and 4 kernel threads where supported while keeping request batch 1. Record spin/park behavior, idle CPU use, sustained thermal/power behavior, and app contention.
6. Integrate the selected kernel and repeat complete product requests. Pin any native comparison engine and verify exact model conversion support before comparing it. Record native and browser results separately.

## Measurements

Record per-operation time and numerical error, decode time per token, prefill time, first-action/complete-request latency, download, load/repack time, resident/peak memory, and transfers. Label estimated byte traffic separately from hardware measurements. Report unsupported shapes and fallback frequency.

Compare a dense Q4 candidate in a separate model/format run with product success and memory/latency results. That comparison has different effective weights and needs its own baseline and behavioral qualification.

## Acceptance and stop criteria

Accept when numerical checks pass and the integrated candidate clears the declared device-level improvement threshold within all applicable budgets. Product behavior, stopping, rejection, and permission handling must remain within the predeclared gates.

Reject silent format/activation changes, output divergence beyond tolerance, or load/memory regressions that exceed budget. Stop tuning a microkernel when end-to-end measurements show no useful effect. Preserve the compatible reference path for regression checks.

## Deliverables and result

Deliver the tensor-format inventory, conversion provenance, scalar/SIMD comparisons, supported-shape table, integrated device report, and chosen dispatch policy with evidence IDs.

Result: not run. No optimized packed kernel or new format accepted.

## References

- [xn quantized layouts, WASM features, and benchmark limits](../../xn-optimization-reference.md).
- [onebitllms](https://github.com/tiiuae/onebitllms) and [BitNet deployment code](https://github.com/microsoft/BitNet), subject to exact Falcon compatibility checks.
- [Existing scalar reference crate](../../../tools/quant-reference/README.md).
