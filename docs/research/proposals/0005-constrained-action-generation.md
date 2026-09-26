# 0005: Constrained action generation

Status: planned. Owner: runtime tools and sessions; training supplies the matching output format.

The [2026-09-15 research and incremental plan](../../constrained-decoding-research.md) supplies library evidence, compiler boundaries, repository-specific serializer findings, and proposed measurement gates. It doesn't establish an accepted implementation or measured performance.

## Description and hypothesis

Constrain generated tokens to a supported response/action grammar and measure the complete request, including parsing and repair cycles. Keep valid paths for actions, clarification, scope rejection, permission limits, and ordinary responses.

Hypothesis: preventing malformed output reduces repair cycles enough to improve successful task completion and latency after grammar compilation and masking costs are included.

## Compiler boundary and increments

Compile versioned public function names and argument schemas together with the exact assistant response/stop format. Pair each name with its own argument schema. Keep a zero-call response available. Tool descriptions and observations remain public context; execution-time permissions remain with the host.

Authored trajectories supply training and evaluation cases. Their order, sampled parameter grids, fixture IDs and simulated results mustn't restrict the runtime grammar. The model can compose supported calls across successive actual observations, within explicit application rules and session budgets.

1. Freeze the decoding profile, serializer fixtures, unsupported-keyword report and stop behavior.
2. Compare pinned LLGuidance and current XGrammar using the exact tokenizer, native and WASM execution, valid/invalid sequences and independent schema validation. These first 2 increments can proceed before a complete decoder exists.
3. Once 0001 works, compare validation/repair alone, response-format masking, and function-schema masking. Hold model, response format, call-count policy and repair policy fixed.
4. Package and cache the chosen grammar implementation, with version checks and cold-load/memory evidence.
5. Evaluate unseen compositions in a real test host or simulator. Then test a name-first serializer with matching training as a separate change.
6. Trial token skipping and dynamic constraints only when profiles justify them. Qualify the selected combination through the deployed product path.

The current Falcon training serializer alphabetizes JSON keys, placing arguments before the function name. Preserve this for the initial comparison; a name-first format needs a new serializer version and matching training. Current authored replay requires exact expected call sequences, so it can't establish success for alternative valid trajectories. See the research note for inspected source locations.

## Prerequisites

- [0001](0001-decoder-baseline.md) provides a working action path and representative requests.
- Freeze the exact tokenizer, response envelope, parser, validator, retry budget, and decoding policy. Record the control runtime, including accepted earlier optimizations.
- Define supported grammar features, response branches, and failure behavior. Register the [run record](run-record-template.md) and behavioral margins before comparing.

## Comparison and method

1. Compare current generation plus parsing/repair with grammar-constrained generation using the same model and requests. Hold the retry policy fixed so changed repair frequency remains observable.
2. Include valid actions with enums, numbers, IDs, escaped strings, nested arguments, and relevant boundary values. Include ambiguous, unrelated, mixed-scope, and permission-limited requests with valid response paths.
3. Measure grammar compilation separately for cold and reused grammars. Key reuse by grammar, tokenizer, response format, and engine versions. Test version changes and invalid schemas.
4. Measure per-token mask construction/application and any GPU readback. Test tokenization edge cases, EOS, output limits, empty allowed-token sets, cancellation, and malformed partial output.
5. Execute validated calls through the actual host authorization path and verify effects. The grammar controls syntax; the application enforces permission and argument constraints. Require truthful handling of failures and partial completion.
6. Repeat the [shared screen](README.md#shared-comparison-rules). Keep changes to model training or response verbosity in separate comparisons so the grammar's effect remains attributable.

## Measurements

Record malformed calls, wrong syntactically valid calls, retries, generated tokens, grammar compile time/size, cache hits, mask time, transfers, first-action latency, and complete-request latency. Include complete task success, false rejection, correct rejection, clarification, and forbidden effects by request family. Report repaired and regressed tasks as paired outcomes. Hold out workflow combinations before augmentation, and score their actual final state separately from authored replay.

Token/logit equality with unconstrained generation isn't the hypothesis here because masking deliberately changes allowed outputs. Verify mask correctness and evaluate behavior against the fixed product expectations.

## Acceptance and stop criteria

Accept when the predeclared task-level objective improves beyond measurement uncertainty after all grammar overhead, and every required response branch and behavioral/resource gate passes. A reduction in malformed JSON alone is insufficient.

Reject forced actions on requests requiring clarification/rejection, invalid token masks, excessive host/GPU synchronization, or forbidden effects. Defer dependency adoption when compilation, memory, or runtime support exceeds budget. Keep a defined, validated failure path when grammar generation can't continue.

## Deliverables and result

Deliver the versioned grammar specification, tokenizer/branch coverage cases, compilation/masking profile, product comparison, and dependency/format decision with evidence IDs.

Result: not run. No grammar engine or constrained decoding policy accepted.

## References

- [XGrammar browser SDK](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/javascript_api.html) and [tool-call structures](https://xgrammar.mlc.ai/docs/latest/structural_tag/tool_calling_and_reasoning.html). Verify the pinned version and exact model format during implementation.
- [Inside vLLM reading priorities](../../wafer-inference-experiments.md#additional-resource-inside-vllm).
- [Runtime tool and acceptance procedure](../../procedure.md).
