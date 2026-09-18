# Engineering documentation

Keep repository-specific design decisions, interface changes, and operational instructions here. Company strategy and cross-repository architecture live in the parent workspace's docs/ directory; they aren't required to run this repository's checks.

Record accepted decisions with numbered files such as `0001-rollout-transport.md`. Keep implementation details beside the module they describe when that makes ownership clearer.

- [Runtime procedure](procedure.md): implementation order and delivery gates.
- [Runtime experiments](../experiments/README.md): ordered protocols, hypotheses, comparisons, criteria, status, and a shared run-record template.
- [Constrained decoding research](constrained-decoding-research.md): function-derived grammars, Rust/WASM options, quality and performance evidence, and an incremental plan that preserves new action sequences.
- [Wires audit](wires-audit.md): reused foundations and their verification limits.
- [Wafer inference reading](wafer-inference-experiments.md): research sources, Falcon/MiniCPM cache estimates, and links to the experiment protocols.
- [xn optimization reference](xn-optimization-reference.md): pinned Rust kernel, cache, and scheduling references, with batch-1 trials and browser/model compatibility limits.
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm), Aleksa Gordić: engine internals and inference techniques, with [batch-1 reading priorities](wafer-inference-experiments.md#additional-resource-inside-vllm).
