# 0007: Speculative decoding, including DSpark

Status: planned, conditional. Owner: runtime decoding and engines; training owns any matched draft-model training/export.

## Description and hypothesis

Propose several tokens cheaply and have the target verify them together. Keep one active sequence at batch 1, with bounded draft state and target verification buffers.

Hypothesis: enough proposed tokens are accepted to repay drafting, verification, and state-management costs, reducing first-action latency within download and memory budgets. Quantized small models need their own evidence because lower target weight traffic can change that tradeoff.

## Prerequisites and trigger

- [0001](0001-decoder-baseline.md) provides a correct target decoder and stage timings. Review experiments 0002–0005 and fix their selected settings for both arms. Experiment 0006 can remain deferred.
- Measure decode's share of latency and required output lengths. Use the expected saving to decide whether a trial could meet the declared minimum useful improvement.
- Support multi-token verification, target/draft state rollback, and exact stop handling. Set bounds on extra weights, cached features/KV, and scratch before integrating a drafter.
- For a learned draft, record matching target architecture, tokenizer, target feature interfaces, and training/export provenance. Recheck acceptance after product fine-tuning, pruning, or target quantization.
- Record the control, treatments, numerical/behavioral requirements, and resource/improvement gates in the [run record](run-record-template.md) before execution.

## Comparison and method

1. Keep ordinary greedy target decoding as the control. Compare prompt-lookup proposals first when product traces contain useful repetition. Compare a compatible learned drafter separately; a failed lookup trial doesn't establish that learned drafting will fail.
2. Sweep a small set of supported proposal lengths, starting with 2 and 4 and adding the model's native block size where compatible. Keep the target bundle and runtime settings fixed and include all draft work in timing.
3. Verify proposals against the target, emit only verified tokens, and discard rejected state. Test rejection at each position, all accepted tokens, EOS/stop strings within a block, output limits, cancellation, and continued sessions.
4. With constrained generation enabled, apply the same grammar state and token-mask semantics at each verified position. Roll back tentative grammar and cache state together. The host executes only validated final actions.
5. Run short product actions and longer responses separately under the [shared screen](README.md#shared-comparison-rules). An optional native LFM reproduction can establish a reference implementation; qualify our exact quantized specialist and browser path independently.
6. Start with fixed verification windows. Evaluate confidence-based window selection separately if measured verification waste justifies it. Sampling beyond greedy requires its own acceptance algorithm and distribution checks before use.

## Measurements

Record proposed/accepted tokens, accepted length per cycle, rejection positions, draft time, verification time, rollback cost, output length, and first-action/complete-request latency. Measure cold download/load, target-plus-draft resident memory, feature/KV retention, verification scratch, and sustained responsiveness.

Estimate potential end-to-end savings with `new time / old time = (1 - f) + f / s`, where `f` is the baseline fraction spent decoding and `s` is the measured decode speedup including draft overhead. Separately account for new cold-load costs. Use actual task timing for the final decision.

## Acceptance and stop criteria

Accept only when the exact target's greedy token sequence, stopping, validated actions, and product behavior match the control, and first-action latency clears the predeclared improvement gate within all resource limits. Diagnose numerical differences in verification kernels before claiming output parity.

Reject stale speculative state, unverified actions, or budget overruns. Defer if output is too short, acceptance too low, or draft/verification cost leaves no useful task-level gain. Keep ordinary decoding available. Any draft-training cost must be recorded alongside the deployment benefit.

## DSpark evidence and limits

Liquid's [August 2026 DSpark report](https://www.liquid.ai/blog/lfm2.5-dspark) reports a mean 2.54× decode speedup for LFM2.5-1.2B on M4 Max at batch 1, greedy decoding, and FP16 with experimental Metal kernels. This supports testing speculation at small model sizes; our product performance remains unmeasured.

The [published LFM draft](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF) is target-specific and shares embeddings/output-head weights with that target. Its files are roughly 594 MB in FP16 and 175 MB in Q4, before runtime state. Falcon/MiniCPM need compatible draft design and training. Pin actual revisions and formats at execution time.

## Deliverables and result

Deliver the proposer/verification design, rollback and parity cases, accepted-length distributions, memory/cost report, product comparison, and adopt/defer decision with evidence IDs.

Result: not run. Trigger has not been established; no speculative path or draft model accepted.

## References

- [DSpark paper](https://arxiv.org/abs/2607.05147).
- [Inside vLLM reading priorities](../../wafer-inference-experiments.md#additional-resource-inside-vllm).
