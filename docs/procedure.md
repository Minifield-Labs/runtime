# Model runtime, product integration, and release procedure

Status: implementation plan. The runtime boundaries, bundle contract, fixtures, and contract checks exist. No inference engine or BuildUp integration is implemented yet.

This repository owns loading the delivered artifact, assembling public context, executing supported product actions, managing sessions, and measuring device behavior. Training supplies packed model bundles; the product environment supplies evaluation semantics.

All thresholds must be agreed for the actual reference device and product scope. A successful desktop inference demonstration doesn't establish browser support, acceptable mobile performance, or furniture-plan correctness.

## 1. Define the reference deployment

Record the intended operating system, browser/native host, CPU/GPU architecture, available acceleration, RAM, storage, context, and supported offline behavior. Name a reproducible reference device.

Choose explicit limits for:

- Total downloadable model, tokenizer, adapter, and required runtime assets.
- Peak process memory, including weights, KV cache, temporary buffers, and host application state.
- Cold-load time, time to first visible action, and complete-request latency.
- Maximum context, generated tokens, tool calls, concurrent sessions, and retries.
- Sustained performance during repeated realistic tasks.
- Version support, updates, interruption handling, and failure messaging.

A provisional 500 MB or 1 GB model budget can guide experiments, but the selected limit must be recorded before accepting a candidate. Measure total download and memory separately.

**Output:** configs/ deployment specification and an experiment note explaining the chosen first engine.

**Advance when:** the reference device and limits are explicit, the product can expose authorized tools/context, and the backend supports a candidate format worth testing.

## 2. Build the bundle loader

Accept a bundle directory or resolved artifact identifier. Parse the manifest using the pinned contract and reject unknown schema versions. Real sessions must reject purpose=contract_fixture.

1. Resolve all files inside the artifact root. Reject traversal, unexpected external paths, and unapproved remote fetches.
2. Check each asset's SHA-256 and byte count before loading.
3. Require weights, tokenizer assets, model template, and public product contract.
4. Check product identity/version, engine compatibility, weight format, quantization, adapter behavior, and context limits.
5. Check the asset set fits the storage budget. Use measured or conservative resource estimates before allocating expensive buffers.
6. Keep bundle metadata immutable during a session. Publish updates through a separate version switch.
7. Surface actionable failures for missing assets, incompatible devices, corrupt downloads, and unsupported products.

Store tokenizer/template and engine revisions in diagnostics. Follow the exact tested adapter path; loading a floating-point adapter beside a packed base and merging/repacking it are separate configurations.

**Output:** loader, compatibility validation, asset-integrity tests, and clear error behavior.

**Advance when:** a valid fixture passes structural checks, malformed/corrupt fixtures fail, and an actual candidate bundle loads on the reference device. The fixture itself contains no model and can't satisfy the final condition.

## 3. Implement one inference backend

Keep backend-specific code and dependencies under engines/<backend>/. Define a small interface for load, generate, cancel, release, and resource/performance reporting.

1. Load the exact tokenizer, template, model, and adapters declared in the bundle.
2. Reproduce the tested decoding policy, stop conditions, and context limits.
3. Return generated token IDs and completion metadata when the backend supports diagnostics.
4. Make cancellation interrupt generation and release resources predictably.
5. Bound concurrent loads and sessions. Avoid silently duplicating weights or caches.
6. Test warm and cold execution. Keep model download time, runtime startup, prompt processing, and generation time distinguishable.
7. Compare behavior with the training/export runner on fixed contexts. Investigate any changes in serialization, parsing, or stopping.

Choose one engine for the first working path. Add another backend when a concrete device or format requirement justifies it. Keep hardware-specific kernels isolated from session and product logic.

**Output:** runnable intact-model baseline and measured load/generation report.

**Advance when:** a valid model executes repeatedly without resource leaks, cancellation works, and generation semantics match the selected bundle policy closely enough for product evaluation.

## 4. Assemble model-visible context

Public context includes the product's supported scope, permitted tool definitions, authorized observations, and relevant user history. The host app determines visibility.

Build an explicit serializer shared in behavior with training. Preserve model-specific message order, call IDs, tool arguments, result encoding, and stopping rules. Verify parity with known serialized examples.

Keep private fixture state, expected outcomes, judge output, task/split labels, seeds, and reward metadata outside the context. Treat text contained in furniture names, documents, or tool results as product data; it doesn't grant new authority or expand scope.

Define truncation and summarization rules before enabling long sessions. Preserve unresolved references, current user intent, required confirmation, pending operations, and the state needed to avoid duplicate writes. If essential context can't fit, fail or ask a useful clarification under the product's policy.

Select tools from the application's actual capabilities. A short catalog can reduce input size, but filtering tools doesn't replace authorization or prove that a request is supported.

**Output:** context builder, serialization fixtures, privacy-boundary tests, and explicit overflow behavior.

**Advance when:** training and runtime agree on test conversations; no known private metadata enters the policy; context limits produce predictable behavior without silently dropping necessary information.

## 5. Connect product tools and enforce execution rules

Inspect BuildUp's actual integration boundary before writing its adapter. Record whether the model calls a local MCP, browser-exposed tools, or another host API, and where authorization and state changes occur.

For each generated call:

1. Parse the structured action without executing arbitrary generated code.
2. Validate the tool name and arguments against the current public contract.
3. Check the current user's permissions and product state at execution time.
4. Apply required confirmation and transaction rules.
5. Execute through the application's supported API.
6. Record the actual result/error and resulting authorized observation.
7. Return that evidence to the model before dependent actions continue.

Treat malformed calls, unavailable tools, denied permissions, and tool failures as distinct outcomes. Bound repair attempts. A fallback parser mustn't broaden the accepted action language beyond the product contract.

For multiple calls in one user request, preserve the application's intended transaction and undo behavior. Batch only actions that the product allows to execute independently.

**Output:** product adapter, tool validator, result serializer, error mapping, and application-boundary tests.

**Advance when:** every supported tool has end-to-end success and failure coverage; denied actions remain denied even with malicious arguments; unintended calls produce no effects; results are grounded in actual execution.

## 6. Implement the session loop

The basic cycle is user request → public observation → model output → validated tool execution → updated observation → grounded response.

Keep session state separate from the inference backend. Track request ID, pending calls, user confirmation, cancellation, interaction budget, and the installed bundle/product version.

Handle these cases explicitly:

| Case | Expected behavior |
| --- | --- |
| Supported complete request | Execute permitted changes and report observed results |
| Ambiguous supported request | Ask for the missing decision; preserve state |
| Pure unrelated request | Brief scope explanation, zero product actions |
| Product-related unavailable action | Explain the unavailable capability accurately |
| Permission restriction | Explain the access limit; preserve protected state |
| Independent mixed request | Complete the valid portion and identify the unsupported part |
| Unsupported precondition | Clarify before taking dependent action |
| Tool failure | Report or recover using actual evidence |
| Timeout after possible commit | Reconcile effects before retrying |
| User cancellation | Stop further actions; describe any effects already committed |
| New valid request after rejection | Evaluate and act normally within scope |
| Quoted instructions in product data | Treat them as data, preserving the user's actual intent |

Don't display speculative success while an operation is pending. Prefer short confirmations tied to completed tool results. Keep the UI responsive while generation or product operations run.

**Output:** bounded interaction loop and scenario-based session tests.

**Advance when:** each case behaves correctly under the defined product policy, including interruption between tool calls and truthful reporting of partial completion.

## 7. Measure the intact and trained artifacts

Before specialization, collect an intact-model baseline. Repeat the same measurement procedure on the trained, pruned, and quantized candidates.

Record:

- Total downloaded bytes and cached bytes, including adapters and tokenizer assets.
- Cold start with a clearly defined cache state.
- Time to first visible action and complete task.
- Prompt/context length, output length, tool count, and retries.
- Peak RAM and, where available, accelerator memory.
- Sustained latency and resource use across repeated requests.
- Failure, cancellation, and resource-cleanup behavior.

Use representative product requests and state sizes. Report p50 and p95 latency with sample counts, device details, power mode, engine settings, and cache conditions. Keep model latency distinct from product-tool latency so an expensive mesh operation doesn't masquerade as slow token generation.

Verify tests on the actual target host. Native backend results don't establish WebGPU/browser behavior. Record memory pressure, blocked capabilities, or unsupported packed shapes as deployment failures.

**Output:** comparable device report for each exact bundle/runtime combination.

**Advance when:** every agreed resource and latency limit passes. If it fails, return measured evidence to training about context, output length, model size, format, or adapter overhead.

## 8. Run behavioral acceptance on the delivered system

Freeze the model bundle, engine, product adapter, context policy, decoding, retry policy, and evaluation cases before acceptance.

Execute tasks through the same path the user will run. Use the generator's independent judges and real-product conformance checks to assess outcomes. Mesh rendering and valid tool syntax are diagnostic checks; the complete request and protected state determine success.

For a reversible internal pilot, the starting gates are:

| Metric | Proposed gate |
| --- | --- |
| Complete supported-task success | At least 95% overall and 90% per adequately sampled core family |
| Correct unrelated-request rejection | At least 95% |
| False rejection of supported requests | At most 2% |
| Forbidden effects or collateral changes | Zero observed |
| Resource/latency budget | Every agreed limit passes |

These gates are provisional. BuildUp's actual actions, geometry constraints, and intended customer use need their own acceptance policy. Report permission attempts, false success claims, ambiguity, mixed requests, recovery, cancellation, and undo separately.

Report denominators and uncertainty. Use the realistic request mix alongside balanced boundary diagnostics. Protect acceptance from tuning; refresh evidence after any acceptance case becomes repair data.

**Output:** signed version tuple, acceptance report, device report, known limitations, and release decision.

**Advance when:** product and device gates both pass on the exact delivered system. A different engine or decoding policy requires the affected checks again.

## 9. Package, update, and support

Deliver installation instructions, bundle identity, supported product version, scope description, engine/device requirements, permissions behavior, and rollback instructions.

Verify downloads before installing them. Keep the last working bundle available until the new version passes its load and integration checks. Switch versions between sessions; avoid changing weights or tool semantics mid-request.

Product contract changes trigger compatibility checks and relevant regression tests. A mismatch must fail clearly or follow an explicitly supported compatibility path. Request new training data or a new bundle when the supported behavior changes.

Define diagnostics before collecting them. Default local processing should keep prompts, context, and model outputs on the device. Any uploaded telemetry, customer examples, or cloud fallback requires an explicit product/data-flow decision.

**Success criteria:** another engineer can install the bundle, complete supported tasks, inspect its version/limits, handle failures, and roll back without needing the training repository.

## First deliverable checklist

- [ ] Select one reference device, engine, and explicit resource budget.
- [ ] Load an intact candidate with the exact tokenizer/template.
- [ ] Complete one actual BuildUp tool round trip after inspecting its interface.
- [ ] Exercise permission/error/cancellation behavior.
- [ ] Measure cold load, bytes, memory, and task latency.
- [ ] Consume a trained bundle independently.
- [ ] Pass behavioral and device acceptance before delivery.

The contract checker validates the scaffold's examples. The inference, product integration, and measurements above remain implementation work.
