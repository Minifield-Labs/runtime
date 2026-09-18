# Constrained decoding for product actions

Status: researched proposal, 2026-09-15. No grammar engine has been integrated or benchmarked in this repository. [Experiment 0005](../experiments/0005-constrained-action-generation.md) owns execution and acceptance.

## Recommendation

Ship schema-constrained decoding with the model runtime. Compile the action language from versioned public tool definitions and the model's response format. Let the model compose calls across successive observations, within the application's capabilities and session limits.

The expected gain is fewer malformed calls and repair turns. Near-zero added generation latency is plausible with a good implementation; preserved task quality needs measurement on our model and product. A valid call can still select the wrong object, misunderstand the request, or arrive after permission has changed.

Start with an LLGuidance integration spike because the core is Rust. Compare it with current XGrammar, which has an explicit browser SDK. Adopt one after native and browser measurements. Building our own general grammar engine would absorb effort already spent by both projects.

## 1. What enforcement actually does

An autoregressive model computes scores for the next token. Before sampling, an incremental grammar matcher identifies tokens that can extend the current output into a valid response. Invalid tokens receive a score of negative infinity, then sampling chooses among the remaining tokens. LLGuidance documents this algorithm and the opportunity to overlap mask computation with model execution. [Implementation explanation](https://guidance-ai.github.io/llguidance/llg-go-brrr).

```text
Current output prefix
    -> compute allowed-token mask on CPU
    -> combine with next-token scores from model
    -> sample an allowed token
    -> advance matcher and model state
    -> repeat until a valid terminal state

Completed assistant response
    -> strict parsing and full argument validation
    -> current host permission, confirmation and state checks
    -> execute supported call
    -> return actual result to model
```

A token can contain part of a name, several punctuation characters, or partial UTF-8 bytes. Validity depends on its entire byte sequence and the current parser state. An `@` is valid inside a JSON string; a token containing a closing quote and the next field can cross several grammar boundaries. A character blacklist or independently tokenized list of function names won't handle this correctly.

Apply the mask before top-k/top-p truncation and sample from the remaining distribution. Otherwise, an early sampling filter can discard every valid continuation. Keep the original prefix when selecting the token; discarded candidates must never enter the KV cache.

A successful, completed generation can satisfy the supported grammar. Cancellation, token limits and engine failures can still leave an incomplete prefix. Such output has no executable meaning. Stop tokens must only be available where the selected response protocol permits completion.

## 2. Compile functions and response syntax

The compiler inputs should be:

1. The product's callable names and explicit argument schemas.
2. A versioned assistant response format, including non-action responses and terminal tokens.
3. The exact tokenizer and its token-to-byte mapping.
4. Explicit runtime limits and the host's currently available tool set.

WebMCP exposes a tool's name, description and JSON Schema `inputSchema`. MCP has the analogous input contract. Use those definitions or registered immutable snapshots as the source. A bare JavaScript function signature doesn't contain enough information to recover enums, units, required fields or valid values. [WebMCP specification](https://webmachinelearning.github.io/webmcp/), [MCP tool contract](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).

| Information | Runtime treatment |
| --- | --- |
| Callable name | Exact allowed alternatives |
| Input object, fields, required keys, enums, types | Compile into each function's argument grammar |
| Numeric bounds, string patterns, richer schema features | Enforce during decoding when the selected compiler supports them correctly; validate the full contract before dispatch |
| Assistant response and end-of-turn format | Compile alongside calls; preserve a zero-call response |
| Tool descriptions, units and intended use | Keep in public context and training; they guide meaning |
| Ordered authored trajectories | Training examples and evaluation evidence |
| Sampled parameter grids, fixture IDs and simulated results | Development data; never infer the production action language from them |
| Permissions and changing product state | Check through the host at execution time |

Compile a union of complete call schemas, with each name tied to its own argument schema. A name enum next to an unrelated union of argument objects would allow the arguments for one function to be paired with another function.

For the workspace's synthetic `list_objects` and `modify` tools, the compiler can derive the dimension enum and the declared 1–200 numeric range. The training sampler's 0.5 increment doesn't become a runtime restriction: the tool schema doesn't declare that increment. Likewise, fixture object IDs don't become a production enum.

### Preserve new trajectories

At each decision, the model may request a supported call or produce a non-action response. After a call executes, it receives the actual result and decides again. This allows new combinations, repetitions and recovery paths within the product's rules.

For example, a user could request changes to 2 dimensions. The model could compose a lookup and 2 `modify` calls even if that exact conversation never appeared in training. The grammar permits the sequence; learning and evaluation determine whether the model chooses it correctly.

Keep any required preconditions explicit in the host contract. A transaction rule or required confirmation is legitimate enforcement. Inferring a mandatory call order from the order in an example would accidentally exclude valid behavior.

Adding a function to the catalog makes its syntax available immediately. Competent use of a new function, or an unseen composition of known functions, still needs evidence.

## 3. What the research establishes

### Fast masking is practical

LLGuidance's authors report roughly 50 microseconds of CPU work per token for a 128k vocabulary, averaged over their JSONSchemaBench workload. They also report a latency tail, including masks over 1 millisecond. These are their measurements, with their grammars and hardware. [LLGuidance project](https://github.com/guidance-ai/llguidance).

XGrammar's original work precomputes reusable token checks and overlaps grammar work with GPU inference. The January 2026 XGrammar 2 paper adds dynamic dispatch, just-in-time compilation and caching across grammars. Its SGLang function-calling experiments report latency within 6% of unconstrained generation, including small models. Its end-to-end comparison omitted LLGuidance because that integration failed, so it doesn't establish a universal winner. [XGrammar paper](https://arxiv.org/abs/2411.15100), [XGrammar 2 paper, sections 4.1–4.4](https://arxiv.org/html/2601.04426v1).

Our inference target is batch 1 on customer hardware. Published serving measurements justify trying the techniques; our CPU scheduling, WebGPU transfers, browser startup and thermal behavior determine the actual cost.

### Valid output and good decisions need separate measurements

Grammar-Aligned Decoding demonstrates that ordinary token masking changes the distribution over complete responses. A locally legal early choice can lead toward an unlikely continuation. Its proposed alignment method uses additional sampling; the paper supplies a quality warning, rather than a reason to add expensive search to our first runtime. [Grammar-Aligned Decoding](https://arxiv.org/html/2405.21047v2).

NVIDIA's May 2026 Bash study combined constrained generation with syntax-check retries. Across 13 models and 299 tasks, mean success rose from 62.5% to 75.2%. Across 3,887 model/task pairs, 181 previously successful cases regressed. The domain and retry treatment differ from our JSON tool calls, but the paired evaluation is exactly the pattern we should copy. [NVIDIA study](https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/).

JSONSchemaBench measures efficiency, schema coverage and task quality separately. It finds both acceptance of invalid instances and rejection of valid instances in tested engines. Its results are version-specific, so use its methodology and test corpus without treating historical rankings as a current dependency decision. [JSONSchemaBench](https://arxiv.org/html/2501.10868).

### Schema support needs a declared boundary

LLGuidance documents fixed property order, limits on schema combinations and duplicate-key handling, and a lenient option that ignores unsupported constraints. Keep lenient conversion disabled. The current llama.cpp grammar documentation explicitly warns that unsupported features can be skipped silently and lists restrictions on numeric bounds and other keywords. Successful compilation alone can't establish full schema enforcement. [LLGuidance schema support](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md), [llama.cpp grammar limitations](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

Define a Minifield decoding profile and report, for every schema keyword, whether it is enforced during generation, checked before execution, or unsupported. Unsupported contracts should fail packaging unless a deliberate validator-only treatment is declared. Preserve the original schema for the execution validator.

Object key order and whitespace can have a canonical generation representation without removing semantic JSON values. Preserve optional-field omission, explicit nulls, legitimate additional properties and numeric meaning. Test those distinctions instead of silently adopting a library's defaults.

## 4. How to keep it fast

These are proposed integration choices, subject to device measurements.

### Compile once and reuse

Normalize and check function contracts during release preparation. Package the source contract, response specification and compiler identity. Reuse immutable tokenizer/grammar data while giving each active generation its own matcher state. XGrammar explicitly separates shared compiled objects from per-request matchers. [Engine integration](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html).

Precompute supported artifacts at packaging time where practical. Otherwise compile during bundle initialization and keep the result resident. Measure compressed download size, deserialization and peak memory against load-time compilation before choosing between them.

XGrammar supports versioned serialization of compiled grammars and checks tokenizer metadata on loading. This makes ahead-of-time packaging plausible. Verify that the chosen native and browser bindings expose the needed format; serialization support in a Python API doesn't prove browser compatibility. [Serialization documentation](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/serialization.html).

Cache identity must cover the tool contract, assistant format, tokenizer files and byte-decoding behavior, vocabulary size and special tokens, compiler/version/options, serialization ABI and runtime capability profile. Hash the exact generation representation as well as semantic schema identity, since property order can affect decoding.

### Keep masking beside the sampler

Run parser work inside the inference Worker. With GPU inference, start mask computation once the preceding token is known, alongside the next model forward pass. Upload a packed mask and apply it within, or directly beside, the GPU sampling kernel.

For an illustrative 131,072-token vocabulary, a 1-bit mask is 16 KiB; FP32 scores occupy 512 KiB. Avoid adding a full-score GPU-to-CPU readback solely for grammar checks. The actual vocabulary may be padded beyond the tokenizer's size, which the integration must handle. [XGrammar vocabulary setup](https://xgrammar.mlc.ai/docs/latest/defining_structures/json_generation.html).

Mask transfer, synchronization and application still cost time. CPU-only execution has no GPU work behind which to hide parser work. Measure both paths, including p95 latency and browser responsiveness.

Masking ordinarily retains the model's forward computation. It prevents invalid choices and may save repair turns; it doesn't automatically make transformer layers cheaper.

### Keep common cases small

Reuse per-function structures and compose the currently available tool set. Avoid eagerly compiling every possible permission subset. Refresh availability between assistant decisions, and let the host reject stale calls if state changes during generation.

Keep small static enums in the grammar. Large lists of live object IDs can dominate compilation and may omit legitimate targets. Treat dynamic ID constraints as a later experiment based on an authoritative, complete observation.

### Measure token skipping separately

Forced strings can sometimes be inserted in a block, reducing serial generation steps. They still need to pass through the model to update positions and the KV cache before later predictions.

A forced byte prefix can end inside a token the model would otherwise choose. LLGuidance's fast-forward documentation explains how naive forcing produces non-canonical tokenization and can damage subsequent output. Start with ordinary masked decoding; only add token skipping through a tested tokenizer-aware interface. [Fast-forward token handling](https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md).

## 5. How to protect decision quality

- Preserve a zero-call response for clarification, rejection, permission explanations and normal completion. Never require a tool call for every user turn. XGrammar's model-aware structures distinguish automatic, required and forced tool selection. [Tool-call structure documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/tool_calling_and_reasoning.html).
- Match the model's trained response format. A final-answer JSON schema applied to an incompatible tool-call protocol can exclude the call's opening tokens. Keep any model-specific reasoning behavior consistent with its training and evaluate changes separately.
- Retain descriptions and public context. The mask identifies legal forms; the model still needs the request, object references, units and tool meanings.
- Ground user-visible success messages in completed host results. Syntactically valid response text can still claim that a pending or failed action succeeded.
- Start with ordinary supervised fine-tuning on the same serialization used at inference. Keep conventional loss on the correct targets. Any mask-aware training objective is a separate experiment; inference masking alone supplies no reason to change the loss.
- Track newly broken tasks alongside repaired ones, including wrong-but-valid calls. Compare clarification, scope rejection, false rejection and permission behavior by request family.
- During synthetic evaluation, optionally measure how often the unmasked preferred token is blocked and how much probability lies outside the grammar. Frequent intervention can expose format mismatch. These diagnostics aren't calibrated confidence scores and shouldn't trigger automatic substitution of a different action.
- If the matcher cannot continue, stop the partial response. Permit only the declared bounded recovery path after discarding that partial output. Never execute a repaired fragment or silently disable enforcement for a call.

## 6. Findings specific to this workspace

The runtime has lifecycle, memory and token-input foundations, plus planned experiment 0005. A complete decoder, tool dispatch and session loop remain to be implemented. The baseline experiment names Falcon-E as the current candidate. This research supplies design and acceptance work, without claiming an implemented constrained runtime.

The current training serializer declares `falcon-chatml-tool-json-v2` and `falcon-chatml-tool-json-v3`. Assistant bodies contain `content` and `tool_calls`; each call contains `id`, `name` and an argument object. The OpenAI-compatible dataset interchange uses a different nested call representation and JSON-encoded argument strings. Compile for the serialized model target, then adapt to the host call interface.

Training's `json_io.canonical()` sorts keys. Consequently, a serialized call has this order:

```json
{"arguments":{"object":"obj-1","property":"width","value":80},"id":"call-1","name":"modify"}
```

That is compatible with a grammar containing a union of complete tool variants, but the matcher must retain relevant alternatives while generating arguments. It also means the model produces arguments before explicitly naming the tool. Preserve this order for the first same-model comparison. Later, test a new assistant serializer that places `name` before `arguments`, with matching training and a versioned bundle. Leave generic canonical artifact hashing intact.

The current `prediction_rollout.compare()` evaluates authored replay: it supplies a fixture result only after an exact expected name/argument sequence matches. That correctly prevents fixture leakage, but it can't establish success on alternative valid trajectories. Keep replay for regression testing and add a product simulator or real test host that returns results for any supported call and judges final state.

Source locations inspected on 2026-09-15:

- Runtime: `src/inference/README.md`, `docs/procedure.md`, `experiments/0001-decoder-baseline.md`, `experiments/0005-constrained-action-generation.md`, `contracts/model-bundle/v0.1.0/model-bundle.schema.json`.
- Training repository: `src/minifield_training/sft/data.py`, `src/minifield_training/sft/prediction_rollout.py`, `src/minifield_training/json_io.py`.
- Data-generation repository: `examples/resize.yaml`, `src/openai.rs`, `README.md`.

These are cross-repository observations. Implementations must consume versioned artifacts and installed interfaces.

## 7. Incremental implementation plan

### Increment 0: Freeze the contract and response format

**Owner:** runtime, with training supplying serialization fixtures and backend supplying immutable tool versions.

Define the decoding profile, complete call-schema union, zero-call response, stop conditions and keyword support report. Keep the existing serializer for the baseline. Detect conflicting definitions for the same callable name/version. Define parser limits and strict duplicate-key/non-finite-number rejection.

**Deliverable:** a small synthetic contract corpus, versioned response specification and compiler input manifest. No inference engine is needed for this increment.

**Gate:** every intended response type is representable, unsupported schema behavior is explicit, and no trajectory ordering or sampled fixture values enter the compiled contract.

### Increment 1: Compare grammar libraries without model inference

**Owner:** runtime inference.

Build a narrow adapter with compile, create matcher, fill mask, accept token, terminal-state and reset operations. Trial pinned LLGuidance first. Its Rust crate exposes a `wasm` feature; check the actual feature combination and Worker behavior. Trial current XGrammar through its supported browser SDK as the comparison. Chromium's use of LLGuidance alone doesn't establish our standalone WASM path. [LLGuidance Cargo features](https://github.com/guidance-ai/llguidance/blob/main/parser/Cargo.toml), [XGrammar browser SDK](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/javascript_api.html).

Replay the exact tokenizer's valid and invalid token sequences. Include overlapping function names, arguments before names, escaped strings, Unicode, numbers, nested structures, omitted/null fields, invalid EOS, special/padded token IDs, empty tool sets and exhausted budgets. Use an independent full-schema validator to check completed outputs. Include positive cases to catch accidental restrictions.

**Deliverable:** correctness results, native/WASM build evidence, cold compile/load cost, memory, mask p50/p95/max, and library choice. Use small, representative and stress tool catalogs; label synthetic sizes.

**Gate:** the supported profile passes conformance, required targets build and execute, and costs fit provisional device budgets. A native benchmark can't qualify the browser implementation.

### Increment 2: Integrate minimal masking into the decoder baseline

**Owner:** runtime inference, tools and sessions. **Dependency:** experiment 0001's working decoder and host action path.

Add masking to the sampler with grammar work overlapping GPU execution where possible. Initially use ordinary token-by-token generation, a fixed available tool set and the current response format. Freeze whether the tested bundle supports 0/1 or multiple calls per response; all comparison arms must share that policy. Execute dependent actions through successive real observations.

Compare 3 arms on identical contexts and settings: current generation with validation/repair, response-format-only masking, and response format plus function schemas. Keep repairs bounded and identical in policy across arms.

**Deliverable:** complete-task and valid-action latency, syntax failures, retries, wrong valid calls, response-branch behavior and memory. Measure masking itself and added transfers separately.

**Gate:** correct mask application, no partial dispatch, preserved response branches and no new forbidden effects. Keep the dependency experimental until behavioral and performance comparisons pass.

### Increment 3: Package and cache the selected implementation

**Owner:** runtime for compilation/matching; training/export for assembly; backend/platform for registered artifacts and releases.

Ship the public contract, response specification, capability report and grammar identity with the model. Add serialized compiled assets only if their total delivery/startup tradeoff wins. Verify cache invalidation on tokenizer, tool, format and engine changes. Bound memory and compilation work. Test unload, cancellation, corrupt assets and resumed sessions.

Use a new model-bundle contract version if new required asset roles or metadata are introduced. Keep the consumed v0.1.0 snapshot immutable. Runtime supplies its compiler through a versioned package or worker interface; sibling source imports remain prohibited.

**Deliverable:** reproducible bundle build and compatibility checks tied to the exact runtime/device evidence.

**Gate:** cached and fresh loads produce equivalent accepted behavior, incompatible artifacts fail clearly, and cold-start/download/memory budgets pass.

### Increment 4: Prove composition and tune the model format

**Owner:** training/evaluation, product environment and runtime.

Hold out complete workflow families and combinations of known tools before wording augmentation. Include new tool orders, repeated calls, novel valid argument combinations, changed observations, missing information and recovery after tool failures. A valid alternative sequence receives results from the simulator/test host and is judged by outcome.

First evaluate the same model with and without constraints. Then run a separate matched training comparison for a name-first assistant serializer. Test any content placement or generated call-ID changes independently. Each target format gets its own grammar, checkpoint and version identity.

**Deliverable:** paired repaired/regressed-task report and evidence for unseen composition, with clarification and false rejection measured separately.

**Gate:** task success meets predeclared non-inferiority margins by family and composition performs within its declared target. Syntax improvements alone don't pass.

### Increment 5: Add optimizations only when profiles justify them

**Owner:** runtime, with training involved for output-format changes.

Trial tokenizer-safe forced-span processing, per-function cache reuse for changing availability, and small authoritative dynamic enums one at a time. Keep a path to ask for missing information when the valid action set is empty. Refresh dynamic observations and permissions before execution.

Consider compact action encodings or vocabulary/head specialization only if generated syntax remains a material bottleneck. These require fresh model/format compatibility evidence. Speculative decoding stays in experiment 0007 and must preserve matcher state on acceptance and rollback.

**Gate:** each change improves its stated objective against the immediately preceding accepted configuration and the original baseline, with quality and resource gates intact.

### Increment 6: Qualify through the deployed product path

**Owner:** runtime and platform, with backend persisting validation evidence.

Select an exact registered bundle in the product, run device validation, inspect the result and retrieve it after reconnecting. Record model, grammar, tool contract, runtime and host versions together. Keep promotion and rollback tied to those versions.

**Gate:** the actual delivered browser/native path passes complete-request, authorization, failure, lifecycle and device tests. No release claim follows from a tokenizer microbenchmark alone.

## 8. Proposed measurement gates

These are planning defaults, not measured results. Freeze device/resource limits and behavioral margins before looking at comparison outcomes.

| Area | Proposed gate |
| --- | --- |
| Contract conformance | All committed positive/negative cases pass for the declared profile; zero silently ignored keywords |
| Execution | Zero observed dispatches of incomplete/schema-invalid calls or forbidden effects; investigate every failure |
| Task quality | Paired evaluation with a proposed 1 percentage point non-inferiority margin, checked with uncertainty and by critical family; an underpowered result stays inconclusive |
| Response behavior | Separate targets for correct clarification, scope rejection and false rejection; no aggregate score can hide forced-action regressions |
| Warm speed | Proposed maximum 5% added p50 and p95 latency to a valid action on already-valid cases; separately require the complete-request reliability/latency objective to improve |
| Cold delivery | Report download bytes, initialization/compile/load time and peak memory against the deployment budget |
| Novel composition | Hold out workflow combinations and score actual final state; report separately from authored replay |
| Sustained use | Follow the existing 30 warm observations, 10 cold launches and 10-minute repeated-task screen; expand samples for behavioral qualification |

Count every request, including timeouts and failures, when reporting success and latency. A faster average over only surviving successes can reward a broken treatment. Report time to a validated action, time to its actual effect and full request completion separately.

## First concrete work item

Implement increments 0 and 1 as a small runtime change: the contract profile, serializer fixtures and tokenizer-only LLGuidance/XGrammar comparison. They can proceed while the decoder baseline is built. Keep the existing experiment order for model/device measurements; revisit it explicitly when an independent measurement becomes possible.

The first product outcome to prove is simple: a valid call reaches the host with negligible extra delay, an ambiguous request can ask a question, and a new valid sequence of known actions remains available to the model.
