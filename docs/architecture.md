# Runtime architecture

The artifact declares what the weights mean. The runtime selects an implementation that can execute them on the chosen platform within the caller's limits.

## Dependency direction

```text
native host / browser bindings / qualification
                 |
      generation + tokenizer + grammar
                 |
         model executor + loader
                 |
              engine-api
              /        \
     backend-cpu      backend-wgpu

offline converters -> versioned files -> loader
```

The executor is generic over `InferenceOps`. Backend crates implement that finite contract; the core never imports a concrete backend. A host constructs CPU or WebGPU, then the same loader and model executor run against it. `decoding-protocol` is a separate schema/framing component for callers.

This is the strategy boundary. Keep model equations, stored encodings, platform capabilities, and individual kernels separate. A new kernel shouldn't require a new model implementation or a larger model file.

| Decision | Source | Owner |
| --- | --- | --- |
| Architecture and tensor roles | Validated config and inventory | Model loader/executor |
| Dense, ternary, or NF4 representation | Tensor descriptors and format metadata | Loader/encoding contract |
| CPU or WebGPU implementation | Host platform and explicit backend construction | Host |
| Kernel and optional repack | Encoding, shape, capabilities, resource policy | Backend and executor |

The host currently selects a backend explicitly. Within it, dispatch selects compatible implementations. There is no calibrated cross-device autotuner or universal claim that one kernel is fastest.

Keep the existing trait and typed enums until a real extension needs another interface. Each strategy needs a tested capability boundary and an independent correctness reference.

## Stored formats and internal layouts

Dense F32/BF16 assets become F32 backend values. Packed ternary and NF4 store U8 code streams with F16 group-128 scales. Ternary uses 2 bits per code; NF4 uses 4. Reserved ternary codes, invalid shapes/scales, unknown formats, and incompatible inventories reject during admission.

Known safetensors `format=pt` and absent format metadata admit dense assets. Custom product markers aren't inferred as dense. Such exports need explicit offline normalization into a supported format.

Each packed matrix resolves its format from validated descriptors. Mixed ternary/NF4 pairs use independent operations when a fused kernel can't share one format. Classifier heads remain distinct from vocabulary-sized language-model heads.

Canonical packed bytes are portable. LUT2 and PN4 are backend-private representations with explicit layout tags. Raw operations reject repacked buffers. Serialized artifacts must never depend on an experimental shader name or a developer's environment variable.

## Execution policy

`Lfm2ExecutionOptions` makes ternary LUT2 policy explicit:

- `Off` uses canonical streams and performs no repacking.
- `DownOnly` admits FFN down projections first.
- `Auto` also admits eligible gate/up pairs.
- `max_lut2_bytes` caps duplicate repack storage, defaulting to 64 MiB.

The constructor validates the model/backend before repacking. Down projections receive priority; gate/up streams are admitted together. Unsupported or resource-limited optimizations keep canonical execution available. Other backend failures propagate. Requested mode, admitted streams, and skipped roles are inspectable.

Changing dispatch mode later only selects already admitted streams. It never allocates a representation. The repack allowance doesn't reserve every later cache or activation allocation; backend admission still applies.

`WgpuOptions` carries diagnostics and experimental NF4 staging precision. Environment parsing belongs in host executables. Numerical core crates don't read process-global configuration.

## Ownership and resource limits

Every buffer belongs to one backend identity and generation. Cross-backend or stale values reject. Completions and abandoned tasks retain storage until the last submitted GPU consumer finishes.

WebGPU accounts physical buffer classes, alignment padding, pooled allocations, readback staging, retained results, and fixed host/device uniform storage. A dropped allocation remains charged while pending work or a pool owns it. Safe completed pools may be evicted under budget pressure.

Resource reports classify logical live weights/caches and place physical overhead in scratch. Their total describes backend-accounted storage at that moment. Driver memory, pipelines, process RSS, and an overall process peak need separate instrumentation.

Input capacity, selector width, and constraint masks are checked before recording work where possible. Invalid caller input returns an error while preserving the usable executor. A backend failure after partial recording may quarantine it.

## Code organization

`engine-api` separates errors, tensors, capabilities, resources/completion, operations, assets, and tokens. CPU/GPU modules follow operation families. The executor separates construction, storage, execution, dispatch, prefix tasks, readback, and scoring.

GPU shader bodies live in `crates/backend-wgpu/src/shaders`. `kernels.rs` is the registry. Research shaders live under `shaders/experimental`, exposed through the explicit `experimental-kernels` feature and experimental API. Default execution stays on qualified production paths.

Split modules when ownership or invariants become hard to inspect. A file-length target alone doesn't justify more abstraction.

## Offline converters

Converters belong here because they produce runtime input formats. They live in an independent Python package under `tools/converters`, with their own lockfile and tests.

Structural `convert` copies canonical dense weights unchanged. `quantize` is explicitly lossy, with a named algorithm, source identity, and source/output hashes. QAT, checkpoint recovery, optimizer state, and product-specific training logic stay with training.

The converter's `minifield.converter-bundle/1` manifest records inference assets and provenance. Training's `model-bundle/0.1.0` release contract also requires product identities, a chat template, and a product contract. The low-level loader accepts assets; the host must verify any complete release envelope it claims to support.

Cross-language checks generate synthetic LM and classifier bundles, convert each to dense/ternary/NF4, compare exact decoded weights, then run the actual Rust CPU loader and inference. No sibling import or Python subprocess sits in the Rust inference path.

## Scope control

Product adapters, old JavaScript lifecycle prototypes, static report renderers, and the subprocess-based external engine adapter have been retired from the active tree. Historical reasoning remains in `docs/research`; generated evidence belongs outside the source repository.

Add another architecture, backend, or external converter for a named use case, with contract fixtures and qualification evidence. Interfaces shouldn't imply support that hasn't been implemented.
