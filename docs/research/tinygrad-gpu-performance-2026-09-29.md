# Tinygrad concepts for Minifield GPU inference

Research date: September 29, 2026. Status: source investigation, baseline measurements, and mechanism probes. This document proposes experiments; it doesn't change runtime behavior or establish a candidate speedup.

## Recommendation

There's useful performance work here. Start with **short-prefill ternary kernels and coherent weight loading**, then tackle **attention tiling and a reusable layer workspace**. Build a small, qualified kernel catalog around those experiments so selection can depend on shape and device.

Keep the existing operation families, Rust executor, representation contracts, and WebGPU delivery path. Tinygrad supplies good examples of scheduling, specialization, and lifetime planning. Its general compiler and native hardware queues would bring a much larger maintenance project.

| Order | Experiment | Concrete opportunity | Confidence in mechanism | Portability |
| --- | --- | --- | --- | --- |
| 0 | Better timing and shape evidence | Separate host preparation, GPU execution, cold pipelines, and complete inference | High; prerequisite for accepting gains | Native and browser, with optional timing features |
| 1 | Ternary MT2/MT4/MT8 | Reuse decoded weights across short token batches; today's ternary path repeats single-row GEMV below 96 rows | High that the path is missing; gain unmeasured | Ordinary WGSL |
| 1 | Coherent GEMM weight loader | Cooperatively load consecutive K elements, then transpose in workgroup memory | High that current indices are strided across x; hardware effect unmeasured | Ordinary WGSL |
| 1 | Kernel descriptors and bounded offline tuning | Replace scattered variant metadata and fixed cutoffs with inspectable, qualified choices | High organizational value; speed depends on winning candidates | Native and browser |
| 2 | Layer workspace | Reuse temporary buffers after their final scheduled consumer within a pass | High memory opportunity; latency benefit unmeasured | Rust executor plus backend ownership checks |
| 2 | F32 tiled online attention | Remove global score storage, reuse K/V, and distribute head-dimension work | Strong algorithmic basis; WGSL implementation required | Ordinary WGSL baseline |
| 2 | Split-KV decode | Launch more independent work for long contexts, then combine partials | Strong for underoccupied long-context decode; cutoff unknown | Ordinary WGSL baseline |
| 3 | Stable host execution plans | Reuse bindings and prepared dispatch descriptions after workspace addresses stabilize | CPU overhead hypothesis; profile first | Portable host replay |
| 3 | Subgroup reduction variants | Reduce normalization and sampling barriers | Optional capability; numerical and mapping checks required | Native wgpu 30 path initially |

All proposed performance changes need same-weight parity, full/cached bundle qualification, matched timing, and the platforms they claim to support. The existing [procedure](../procedure.md) already defines these gates.

## Versions, scope, and evidence

| Component | Identity |
| --- | --- |
| Tinygrad | [`7bfaa18daaec8382f6656d46745a424b053cdc0a`](https://github.com/tinygrad/tinygrad/commit/7bfaa18daaec8382f6656d46745a424b053cdc0a), `uopfunc sugar (#18511)` |
| Runtime source and measurements | `e74ec20243405fae91e053198e2f23af8783dd05`, existing telemetry feature branch |
| Report's base | `bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f`, remote main at investigation time |
| Relationship between runtime revisions | `crates/backend-wgpu` is identical; executor execution changes are 8 lines of work accounting |
| Dependencies | Locked wgpu/naga family 30.0.1, Rust toolchain 1.89.0 |
| Measured adapter | Apple M1 Max, native Metal, integrated GPU |
| Advertised limits/features | 32 KiB workgroup storage, 1,024 workgroup invocations; F16, subgroups, and timestamp queries advertised |

The source review covered the scheduling and code-generation path, optimization heuristics/search, memory planner, JIT, WebGPU/Metal launch paths, tensor-core layouts, custom AMD inference kernels, and related correctness tests. On Minifield's side it traced dense/packed projections, normalization, attention, dispatch/pooling, executor scratch lifetimes, prefix/cache ownership, and qualification. This is a targeted GPU inference investigation, not an audit of every tinygrad device driver or training operation.

Primary source links below pin the tinygrad commit and runtime base. Dependency links use Cargo's recorded wgpu source commit, `40f4a34ebaf56f9a046231f54125ad046239d3f3`. Recommendations are explicitly inferences from those sources and local evidence.

### Checks actually performed

- Tinygrad's CPU-only memory-planner tests: **16 passed**. No tinygrad GPU throughput comparison was run.
- A synthetic planner probe: 4 internal buffers of 4 MiB each became an 8 MiB arena on the NULL backend, with held inputs/outputs preserved and simultaneous operands nonoverlapping. The identical WEBGPU schedule stayed at 16 MiB with 0 buffers rewritten.
- Online-softmax/split-combine mathematical probe: **400 comparisons**, maximum absolute difference `8.88e-16` against a stable float64 reference. It covered tile boundaries, causal prefixes, excluded NaN padding, and large finite scores. This validates the recurrence, not WGSL/F32 implementation parity.
- Required native GPU tier, `scripts/check.sh gpu`: **48 passed**, 2 developer benchmarks intentionally ignored. That is 4 allocation/layout tests, 14 low-bit tests, and 30 parity tests, run serially on a real adapter.
- Release parity, default parallel: 29 passed, 1 readback poll-budget failure in `ternary_gemm_matches_nf4_on_shared_weights`. Release parity, serial: 29 passed, 1 readback poll-budget failure in `causal_gqa_matches_cpu_and_appends_cache`. Both failures are preserved. The debug GPU tier passed both tests.
- Native synthetic component benchmarks: 5 fresh-process samples per component, each containing 2 warmup repetitions and 6 measured repetitions. Dense/packed projection benchmarks and feature probes were also run.

The synchronous read helper has a fixed 600,000-poll budget. A faster optimized polling loop can exhaust a count budget before a long operation finishes; the observed failures are consistent with that mechanism. A dedicated tracing experiment is still needed to establish the cause conclusively. No numerical assertion failed in those 2 runs. Keep benchmark fence timing and readback reliability as separate measurements. [Runtime readback][rt-lib]

Raw runs, scripts, and provenance are outside Git in the workspace experiment `experiments/2026-09-29-tinygrad-runtime-research/`. This report contains the reviewed interpretation; generated output remains in that evidence folder. No model weights or customer inputs were needed.

### Fresh component baseline

Times are reported by the existing benchmark's wall-clock wait after recording. They include submission/completion overhead and aren't hardware timestamp durations.

| Synthetic workload | Median µs | Min–max µs | Process samples |
| --- | ---: | ---: | ---: |
| GQA, M=346, Q heads=16, KV heads=8, D=64 | 5,118.3 | 4,689.5–5,167.4 | 5 |
| NF4 gate/up pair, M=346, K=1,024, N=2,560 | 3,511.1 | 3,461.0–3,619.8 | 5 |
| NF4 down with input SwiGLU, M=346, K=2,560, N=1,024 | 2,917.8 | 2,873.1–2,968.0 | 5 |
| Standalone SwiGLU plus plain NF4 down | 2,889.4 | 2,822.2–3,036.1 | 5 |
| Fused pair/SwiGLU producer | 3,554.0 | 3,456.2–3,640.8 | 5 |
| Fused producer plus plain NF4 down | 6,278.8 | 6,241.0–6,374.0 | 5 |

The attention component is substantial enough to investigate. These rows don't establish its fraction of end-to-end latency: convolution layers, attention layers, final-layer row reduction, actual representations, and cached-prefix execution change the mix.

One dense baseline sample at M=346 measured 4,033.8 µs for K=N=1,024, 12,041.4 µs for K=1,024/N=3,072, and 10,086.2 µs for K=2,560/N=1,024. Canonical packed baselines were roughly 1,113–1,299, 3,044–3,050, and 2,734–2,764 µs respectively. Those compare different weight representations and memory footprints; they don't isolate a loader optimization.

Repeated hot buffers, simple synthetic values, only 5 process samples, and unrecorded power/thermal controls limit interpretation. The first exploratory attention sample was also much slower than the repeated set. We haven't derived speedup claims from either sample set. [Benchmark implementation][rt-bench]

## What tinygrad's compiler actually does

The relevant path has distinct decisions:

```text
Tensor / UOp expression
    -> range preparation, buffer materialization, kernel boundaries
    -> dependency-ordered CALL schedule
    -> kernel axis/tile selection, heuristics or search
    -> index simplification, memory coalescing, reduction/barrier lowering
    -> device renderer, compilation, program
    -> ordinary execution or captured/JIT execution
```

Kernel boundaries and a kernel's internal schedule are separate concerns. Tinygrad's schedule tracks producer dependencies and writes that supersede previously read buffer states. Its compiler then chooses which axes become threads, register fragments, or reduction loops. That separation is useful for Minifield's explicit model executor. [Schedule dependencies][tg-schedule], [compiler passes][tg-codegen]

At this commit, `OptOps` contains `TC`, `SPLIT`, `PADTO`, and `SWAP`. Tutorials describing independent `LOCAL`, `UPCAST`, or `UNROLL` optimization enums describe older interfaces. Those concepts now appear as axis types targeted by `SPLIT`. [Current option definitions][tg-opt]

Translate the concepts into finite shader variants:

| Tinygrad concept | Runtime equivalent |
| --- | --- |
| Global axes | Workgroup grid and output tiles |
| Local/warp axes | Invocations cooperating on a tile or reduction |
| Upcast axes | Several output accumulators per invocation |
| Unroll axes | Compile-time reduction fragments |
| Padding | Qualified aligned fast path plus masked boundary handling |
| Tensor-core layout | Device-specific fragment mapping and dtype contract |
| Schedule/materialization | Explicit executor fusion choice and temporary lifetime |

The heuristics consider matvec threads per row, rows per thread, stride, reduction shape, shared storage, and available tensor-core layouts. The optimizer imposes bounds on register expansion, reduction transformations, and shared memory. These are useful knobs for our catalog; their default values aren't portable performance guarantees. [Heuristics][tg-heuristic], [scheduler constraints][tg-postrange]

## What the runtime already does well

Several tempting recommendations would duplicate existing work:

| Capability | Existing implementation | Implication |
| --- | --- | --- |
| Fused residual add/RMS norm | `normalization.rs`, executor's next-layer norm | Preserve the current fused boundary |
| Fused Q/K normalization/RoPE | `qk_norm_rope` | Focus on attention itself before bolting on another fusion |
| Packed prefill tiling | Shared 64×32 output, K16 shader body | Extend the template system selectively |
| Gate/up and SwiGLU fusion | Packed pair and producer/consumer variants | Measure register and recomputation costs before further fusion |
| Dense/packed GEMV row groups | 8/16 lanes per output, several outputs per workgroup | Tune existing grouping; avoid resurrecting an obsolete reduction tree |
| Packed weight layouts | Canonical ternary/NF4 plus admitted LUT2 | Retain explicit format/layout tags and budgeted duplicates |
| Batched submissions | Consecutive dispatches share a compute pass | Per-kernel queue batching is already solved |
| Uniform staging | Bounded dynamic-offset ring with safe wrap submission | Keep safe uniform lifetime behavior |
| Pipeline reuse | In-memory cache keyed by kernel | Cold specialization needs a richer key, not a first cache |
| Buffer pooling | Submission-serial quarantine and physical accounting | Reuse between submissions is already safe |
| Uninitialized scratch | Executor calls `allocate_f32_uninit` for scratch | Explicit scratch clears are already removed |
| Prefix/cache execution | Retained base state, branching, cached-tail qualification | Include both full and cached workloads in acceptance |

The backend has 50 kernel IDs and 53 WGSL files, including experiments and shared bodies. Its families are already split into dense, packed, normalization, attention/convolution, and sampling. The problem is the number of independent places that describe and select a variant. [Kernel registry][rt-kernels], [backend guide](../../crates/backend-wgpu/README.md)

Tinygrad's WebGPU `Program.__call__` creates layouts, bindings, a compute pipeline, a pass, and a submission for each invocation, then releases those objects. Minifield caches pipelines/layouts and batches commands. Tinygrad's fast native queue machinery shouldn't be attributed to this WebGPU path. [Tinygrad WebGPU launch][tg-webgpu], [Minifield submission][rt-device]

## Experiment 1: short-prefill ternary reuse

`packed_linear` chooses tiled GEMM at rows ≥96. Below that cutoff, NF4 batches with rows >1 get an 8-token GEMV tile. Ternary uses the single-row GEMV path for each input row. Paired and input-SwiGLU dispatch have corresponding format-specific decisions. [Packed dispatch][rt-packed]

This makes a small, concrete experiment: add canonical ternary multi-token variants with MT2, MT4, and MT8. Reuse each decoded weight and group scale across several activation rows. Start with ordinary linear, then pair/down variants only if measurements justify their larger accumulator footprint.

Tinygrad's matvec scheduling balances lanes, output rows, and register work, while its custom inference tests explicitly exercise token counts and row groups. Borrow that search space and boundary discipline. Our existing NF4 implementation is the closest template because it already fits our binding ABI and completion model. [Matvec heuristics][tg-heuristic], [inference boundary tests][tg-llm-tests]

Qualification must cover rows 1, 2, 7, 8, 9, 31, 32, 63, 64, 95, 96, and 97; admitted K/N values; canonical reserved codes; every group-scale boundary; and paired/fused output equivalence. Packed K admission rules remain intact. Awkward output widths need masked stores, even where production models use aligned shapes.

Tune lanes with K, N, representation, and token tile. The current helper considers only output width: 16 lanes below N=4,096, otherwise 8. Its explanatory comment mentions 32-lane groups although the implementation selects 8 for wide outputs. Fix that comment when changing selection so future work follows the actual schedule. [Lane helper][rt-lanes]

Expected benefit: less repeated weight decode/load work in short prefill. Risk: extra accumulators can spill or reduce occupancy. Single-token decode stays on its qualified baseline unless a lane variant wins there too.

## Experiment 2: coherent weight loads and register tiles

### Dense GEMM

Our dense shader produces a 16×16 output tile with 256 invocations and 1 accumulator each. Its right-hand load is `w[col * k + l_w]`, where col changes with local x and l_w changes with local y. Adjacent x positions therefore access rows separated by K elements. [Dense shader][rt-gemm]

Try a cooperative loader whose neighboring logical load indices traverse K, then transpose into the shared layout consumed by multiplication. Initially preserve the output tile, arithmetic order, bounds, and F32 representation. This isolates the load mapping from register-tile changes.

WebGPU doesn't define a universal mapping between local invocation IDs and hardware subgroup lanes. The strided index pattern is a source fact; the generated memory transactions and resulting gain need Metal/Vulkan/DX12/browser measurements. Inspection of generated native shaders or a device profiler is preferable to assuming that source-level x adjacency equals a fixed warp layout. [WGSL subgroup semantics][wgsl-subgroups]

After the loader experiment, try a small set of output fragments per invocation, such as 2×2 or 4×2, with shared storage held within device limits. This trades more accumulators for reuse of loaded operands. Change 1 dimension at a time and retain the baseline as fallback.

### Packed GEMM

The shared NF4/ternary prefill body already computes a 4×2 fragment per invocation. It stages 1,024 activation floats and up to 2 sets of 512 decoded weights. Its declared shared arrays total 8 KiB; a single-stream compiler may remove the unused second array. Each projection keeps 8 output accumulators, or 16 for a pair. [Packed prefill body][rt-prefill]

Its loader also changes output column across x while reading K at y. For canonical row-major packed streams, this suggests cooperatively loading consecutive packed words along K, decoding a word into its constituent values once, and transposing the decoded tile in shared storage. Treat NF4, canonical ternary, and LUT2 as separate layouts with separate proof obligations.

Group scales span 128 weights while the loop advances K by 16. Explore loading a scale for its live group and reusing it across fragments/iterations, with guards at group boundaries. The compiler may already remove some repeated reads, and carrying scales longer consumes registers. Count instructions and benchmark before retaining an explicit hoist.

Tinygrad's late coalescing pass checks buffer identity, contiguous offsets, validity, type, and alignment before combining accesses. Its WGSL renderer sets `supports_float4=False`, so the generic renderer's vectorization gains can't simply be assumed for WGSL. Our deliberate `vec4` bindings are already further along on that point. [Coalescing][tg-coalesce], [WGSL renderer][tg-wgsl]

A `vec4` expression also doesn't prove a single 128-bit machine load. Distinguish vector arithmetic, vector binding alignment, and emitted load instructions in experiment notes.

### Avoid a full dequantization detour

Resident GPU dense weights cost 4 bytes per value. Canonical ternary plus F32 group-128 scales costs 0.28125 bytes/value; NF4 costs 0.53125, before allocation padding. Expanding all weights would erase much of our bandwidth and resident-memory advantage. Temporary shared decoding preserves compact global storage.

LUT2 duplicates are already explicit and budgeted. A transposed or interleaved repack must receive its own backend layout tag, resource charge, and admission decision. Equal byte counts don't establish compatible layouts. [Layout/accounting guide](../../crates/backend-wgpu/README.md)

### Shape specialization and address arithmetic

Tinygrad simplifies symbolic indexing before rendering and makes tile/reduction choices visible to code generation. Our hot shaders receive K, N, and other dimensions through uniforms, so the compiler can't treat every model dimension as a compile-time constant. A finite set of specialized variants could expose those constants and simplify loops, offsets, and aligned boundary checks. [Compiler passes][tg-codegen], [speed/indexing discussion][tg-speed]

Try WGSL override constants or generated constants for a few frequently used K/N shapes, with a generic fallback. Specialize only dimensions that remain fixed across calls; cached-tail token count and visible context often change. Cache identity and pipeline admission must include the chosen constants.

First inspect emitted code to see which address operations survived compilation. Constant division by packing/group widths may already become shifts. Manual bit tricks add little if the compiler already emits the same instructions. Reduction unrolling can also explode code size and registers, so compare small fragments before fully unrolling K.

This is a separate experiment from changing arithmetic precision. Measure compilation/cold-start cost alongside steady inference, and cap the number of resident specialized pipelines.

## Experiment 3: a reusable layer workspace

The executor allocates buffers throughout a layer and appends prior activations, projections, attention output, residuals, FFN input, and FFN temporaries to the pass's scratch vector. They survive to completion even after their final logical consumer. The final layer uses only 1 FFN row, another detail a plan must preserve. [Executor loop][rt-execution], [prefix-task ownership][rt-prefix]

This is a strong memory opportunity. If an intermediate has shape M×H, its logical F32 size is `4*M*H` bytes. At M=346/H=1,024, that is 1.35 MiB. A 346×2,560 FFN intermediate is 3.38 MiB. Retaining several such buffers across many layers grows with depth, whereas a stage workspace can grow with the maximum simultaneously live set.

Tinygrad derives first/last appearances, allocates shared arenas, preserves held buffers, and extends copy lifetimes/separates copy lanes to prevent false dependencies. The synthetic 16-to-8 MiB probe demonstrated actual reuse in that planner. However, `_can_plan` explicitly excludes WEBGPU, CL, and DISK because the required views aren't supported. The transferable concept is lifetime planning; its arena rewrite isn't a drop-in WebGPU implementation. [Planner][tg-memory], [planner tests][tg-memory-tests]

### Ownership matters more than shortening Rust lifetimes

Dropping or clearing our scratch vector earlier won't make those allocations available inside the recorded pass. `PooledBuf` returns go into a pending queue, and reuse waits for submission completion. Bypassing that quarantine could give a later operation storage still referenced by earlier work. [Pool/submission lifetime][rt-device]

Start with explicit reusable slots owned by an inference pass:

1. Plan shape classes for operator inputs/outputs, residuals, and FFN intermediates across the model's layer types. Reuse slots between layers with the same logical shape; give different attention/FFN shapes and the final 1-row stage separate slots initially.
2. Lease buffers once per active pass or branch. Use ping-pong slots where an operation reads the prior activation and writes the next.
3. Rebind the same slot only after every earlier scheduled consumer has executed. Preserve dispatch/copy ordering and access barriers.
4. Keep model weights, KV/convolution state, retained prefix state, public results, and readback staging outside disposable workspace slots.
5. Hold the lease through completion or cancellation quarantine. Concurrent or branched work needs separate ownership or explicit serialization.

Whole-buffer slots fit the current backend better than arbitrary arena views: dispatch uses entire-buffer bindings, and F32 admission expects contiguous layouts at offset 0. Reusing an oversized allocation for a different logical shape still needs an admitted descriptor/capacity mechanism; exact-shape slot reuse avoids introducing that in the first change. General offset/stride support would expand the engine contract and every affected operation. [Binding construction][rt-device]

Report peak owned bytes, physical size-class padding, pool residency, allocations, and copies alongside time. This could make larger prompts admissible even if latency barely moves. Don't report the synthetic planner's 50% reduction as a forecast for an actual model.

Uninitialized scratch allocation already removes explicit runtime fills. WebGPU still enforces its own initialization/security rules, so allocator API names don't imply zero driver initialization cost. [Executor storage](../../crates/executor-core/src/lfm2/executor/storage.rs)

## Experiment 4: tiled F32 online attention

### Current work

Our GQA shaders assign a workgroup to each query-head/token row. They write scores to a storage buffer, scan them for a maximum, replace them with exponentials, reduce a denominator, then scan those probabilities for each output dimension. K dot products loop over head dimensions serially within each score lane. Related query heads repeat K/V reads. [Batch attention shader][rt-gqa-batch], [decode shader][rt-gqa]

For D=64 and a 256-invocation group, only 64 invocations compute the final value sum. During the score phase, neighboring score lanes access different K rows separated by KV width, suggesting another cooperative-load opportunity. These are static schedule observations, not profiler measurements.

The host allocates `block_tokens * query_heads * new_cache_len * 4` score bytes. It aims for approximately 64 MiB blocks, although a single query row can exceed that budget at extreme admitted dimensions. The M=346/QH=16/L=346 benchmark requests 7,661,824 logical score bytes, or 7.31 MiB before size-class rounding. [Attention admission/allocation][rt-attention]

### Proposed portable baseline

Keep F32 Q, K, V, statistics, and output accumulation. Tile visible K/V into workgroup storage, cooperate over D, reuse K/V across a small group of queries or related heads, and keep softmax statistics/output accumulators live while streaming tiles. Size the query tile against registers and shared storage; the shortest useful first variant may process only a few queries.

For scores in a new tile, maintain a running maximum m, denominator l, and unnormalized output o:

```text
m_next = max(m, max(scores))
alpha  = exp(m - m_next)
p      = exp(scores - m_next)
l_next = alpha*l + sum(p)
o_next = alpha*o + sum(p*V)
output = o/l after the final tile
```

This is the online-softmax mechanism behind FlashAttention. It computes the same real-arithmetic attention function while changing floating-point reduction order. The first tile and all-masked tiles need explicit handling to avoid undefined differences between infinities. [FlashAttention paper](https://arxiv.org/abs/2205.14135), [FlashAttention-2 recurrence](https://tridao.me/publications/flash2/flash2.pdf)

Tinygrad contains a concrete fused attention implementation with shared tiling and running statistics. Its custom code is restricted to RDNA3/4-style AMD/HIP paths and uses F16 storage/WMMA in relevant prefill paths. Transfer the schedule and recurrence to our F32 baseline; preserve the runtime's numerical contract before exploring lower precision. [AMD inference kernels][tg-amd]

Our cached execution reads old K/V from cache and new K/V from separate buffers, then appends new rows. A new shader must honor that split without an unnecessary concatenation pass. Causal visibility is `old_cache_len + token_index + 1`, including ragged tiles and nonzero cached-prefix offsets.

The CPU mechanism probe excluded NaNs beyond the visible prefix before arithmetic. GPU code must also prevent invalid lanes from contaminating dot products or value accumulation: multiplying a masked probability by NaN can still produce NaN. Tinygrad tests nonfinite unused cache tails, symbolic lengths, large GQA groups, and unaligned starts. Port those test ideas in addition to our existing cache-append parity. [Attention edge-case tests][tg-llm-tests]

Acceptance requires finite outputs, existing declared CPU/GPU tolerances, full/cached logits and stable decisions, correct cache contents, allocation-limit behavior, and native/browser execution. A smaller score buffer alone doesn't prove a faster kernel.

## Experiment 5: split-KV decode

For 1-token decode, the current grid has approximately 1 workgroup per query head. A long context makes each group do more serial work without increasing that group count. Split the visible KV sequence into several independent intervals, compute each interval's local maximum/denominator/output, then launch a small combine kernel. Tinygrad has a partial/combine implementation of this pattern. [AMD decode implementation][tg-amd]

Combine partials with their maxima in the same way the online recurrence rescales tiles. The extra scratch scales with splits × heads × (D+2), rather than storing all scores. Empty splits contribute no mass. The final denominator covers every visible key exactly once. [Flash-Decoding explanation](https://princeton-nlp.github.io/flash-decoding/)

This adds a dispatch, partial writes, and a reduction. Short contexts may lose. Start with split counts 1, 2, 4, and 8 at several context lengths, retain direct decode below a measured cutoff, and cap partial workspace through existing resource policies.

Don't copy tinygrad's chunk sizes or maximum split counts as device-independent defaults. Its waves, shared-memory padding, and fragment mappings target particular hardware. Our dimensions, GQA ratio, cache representation, and browser launch overhead need their own tuning.

This research measured prefill components only. Decode prioritization needs actual generation traces, especially long-context single-token sampling and grammar-constrained execution.

## Kernel organization: operation, variant, plan

Keep the current module boundaries. Introduce 3 explicit layers within them:

| Layer | Responsibility | Examples |
| --- | --- | --- |
| Operation | Validate semantic inputs/outputs and representation compatibility | Dense linear, packed linear, causal GQA |
| Kernel variant | Describe a particular implementation and its ABI | Canonical ternary MT4, NF4 M64/N32/K16, online F32 GQA |
| Execution plan | Choose qualified variants, temporary slots, and ordered calls | Shape-specific FFN path; attention direct/split; layer workspace |

Today `Kernel`, `ALL`, names, source composition, binding counts, and access masks occupy separate registry matches. Selection lives in operation modules, a shared lane helper, and the executor's FFN crossover. New variants will multiply those coordination points. [Registry][rt-kernels], [packed selection][rt-packed], [executor fusion selection][rt-execution]

### A single kernel descriptor

Use a typed descriptor per variant, initially covering existing implementations:

| Field | Why it belongs together |
| --- | --- |
| Stable variant ID and operation family | Correlate plans, dispatch telemetry, and qualification |
| Source factory/template and source hash | Identify the exact compiled shader |
| Binding ABI and access roles | Keep layouts, read-only masks, shader declarations, and alias rules aligned |
| Parameter ABI and byte size | Prevent silent host/shader parameter mismatches |
| Workgroup/grid calculation and tile shape | Make launch math inspectable and testable |
| Representation/layout requirements | Distinguish canonical codes, LUT2, and future repacks |
| Capability and shared-storage requirements | Reject unsupported F16/subgroup/matrix choices before compilation |
| Arithmetic/staging/output modes | Keep F32 and experimental precision explicit |
| Shape predicates and complete-write behavior | Preserve boundary masks and safe uninitialized outputs |
| Production/experimental status | Keep research choices out of automatic default selection |

The pipeline key must include variant ID plus specialization, numerical mode, source identity, and device generation. A cache keyed only by today's enum becomes insufficient when 1 ID can compile several tile/shape versions.

Keep source bodies grouped by operation and share only real common structure. The existing packed prefill body plus representation/epilogue accessors is a good model. Separate layout decoding, tile traversal, and output epilogue where their ABI permits reuse; don't force every kernel into 1 template.

Avoid changing serialized model files to name a GPU kernel. They describe canonical representations. Backend-local repacks and plan choices remain implementation decisions with explicit memory charges.

### Put shape policy behind a narrow planning boundary

First centralize the backend's repeated cutoff/lane/tile choices and attach diagnostics explaining each selection. Then expose a narrow typed planning/capability query only where the executor must choose different compositions, such as fused-producer FFN versus input-fused down.

The executor should still own model math, state changes, and final-layer behavior. A backend plan can describe available composition and workspace requirements without importing WebGPU constants into generic execution. Preserve the CPU implementation and representation checks while moving the current ≥96 crossover behind that boundary.

A general graph compiler would be a large detour for a fixed model executor. Small descriptors and plans capture the useful separation with much less surface area.

## Bounded tuning rather than unrestricted runtime search

Tinygrad's beam search compiles alternative schedules, deduplicates compiled programs, times them, prunes candidates, and caches selected options. It limits compile time, expanded instructions, local work, and register fragments. The search can time a reduced global workload and scale the result, then choose a best observed timing. Those are search heuristics, not release acceptance. [Search implementation][tg-search]

The inspected search loop doesn't perform per-candidate numerical validation. Tinygrad has separate correctness suites. Our tuning runner should explicitly reject a candidate before timing if it fails parity or produces nonfinite output.

Start with 12–30 variants per operation family across a limited set of knobs: lane group, token/output tile, reduction tile, load mapping, shared padding, and epilogue. Several independent experiments can share that runner without requiring UOp, Python in deployment, or a new compiler.

Use this loop:

1. Enumerate only variants compatible with device limits, representation, and arithmetic mode.
2. Compile/warm separately from steady timing; save compile failures and rejected configurations.
3. Check correctness at the candidate's admitted boundaries.
4. Benchmark full target shapes with matched buffers and rotated weight working sets.
5. Interleave baseline and candidate runs; retain raw observations, medians, dispersion, and host/GPU timing.
6. Confirm the winner in full/cached bundle inference, then publish a small typed selection table.

Selection-table identity should include device family/backend, available driver/browser/runtime identity, wgpu/naga versions, shader hash, shape bucket, representation/layout, fusion, numerical mode, and relevant resource policy. Where browser identity is coarse or unavailable, use a conservative qualified default. A missing table entry should fall back predictably.

Don't spend users' first inference searching kernels. Offline tuning and optional engineering tools can produce the tables. Host configuration remains typed; tinygrad's environment-variable controls belong in experiments, not runtime core APIs.

## Host plans, JIT lessons, and native-only paths

Tinygrad's JIT captures a schedule, plans memory, compiles, and binds later arguments against expected shapes/dtypes/devices. It also protects live buffers and mutated inputs. Our compiled Rust executor already avoids Python tensor-graph construction, so any host-plan gain must come from measured allocation, binding, validation, or dispatch-description work. [JIT][tg-jit]

Current dispatch reaps completed work, constructs bindings, and stages parameters for every operation. Stable workspace buffers could support a bounded bind-group cache and prepared call list. Key bindings by actual buffer allocation generation, layout, offset/size, and device generation; retain safe ownership and evict boundedly. Otherwise a cache can retain every historical allocation and defeat pooling. [Dispatch path][rt-device]

Warm only the variants selected for an admitted model, rather than every experimental shader. Report initialization separately. wgpu's optional pipeline-cache feature is implemented for Vulkan and lists Metal/DX12 as unimplemented at this version. A disk-cache switch isn't an M1 cold-start solution. [wgpu pipeline-cache support][wgpu-cache]

Tinygrad's Metal queue builds indirect command buffers with patched arguments and dispatch sizes. That is a native Metal opportunity, requiring a dedicated backend and its own ownership/qualification work. WebGPU offers no equivalent reusable compute-command bundle through our API; portable plans still record new command buffers. [Metal queue][tg-metal], [wgpu command-buffer API][wgpu-command]

Custom AMD/NVIDIA queues, driver mappings, cache modifiers, and low-level instructions require their native backends. They can't be expressed by importing tinygrad's Python helpers into WGSL. A native fork becomes worthwhile only if profiling shows a portable ceiling that matters to shipped workloads.

## Optional subgroups and matrix instructions

Subgroup reductions are plausible for RMS norm, attention statistics, and argmax. Begin with 1 isolated reduction variant. Preserve stable lowest-index ties and existing nonfinite behavior for sampling, and combine subgroup partials correctly when a workgroup spans several subgroups.

Never assume subgroup size 32 or assume a fixed relationship between subgroup IDs and local invocation IDs. WGSL specifies variable supported sizes and leaves their mapping undefined. [WGSL subgroup semantics][wgsl-subgroups]

At our locked dependency version, `wgpu::Features::SUBGROUP` remains native-only even though subgroups have entered WebGPU's standard. Browser adoption requires the actual wgpu/browser path and executed qualification. Adapter advertisement on the M1 establishes native availability, not browser support. [wgpu subgroup contract][wgpu-subgroups]

Tinygrad's tensor-core support maps fragments to each target's lane structure. Its Metal renderer emits simdgroup matrix operations on supported Apple hardware, while the generic WGSL renderer advertises no tensor-core layouts. [Tensor-core layouts][tg-tc], [native renderers][tg-cstyle], [WGSL renderer][tg-wgsl]

Our adapter probe advertises 8×8×8 F32 and F16 combinations. wgpu 30's cooperative-matrix feature is experimental/native-only, and its documented implementation currently supports 8×8 F32. Hardware-reported combinations don't establish WGSL/Naga support for every dtype. [wgpu cooperative matrices][wgpu-matrix]

The existing native matrix experiment regressed on measured FFN shapes. Revisit only with a new load/fragment schedule, generated-code inspection, and full numerical checks. Introducing F16 activations, Q8 activation quantization, or integer-dot accumulation changes the numerical contract and needs separate task-quality evidence. Those are future precision experiments.

## Previous experiments we should learn from

The repository's [FFN experiment log](../ffn-prefill-experiments.md) is valuable evidence against repeating attractive ideas blindly:

| Prior experiment | Recorded result | Consequence for this plan |
| --- | --- | --- |
| Materialized SwiGLU versus recomputation | Little consistent latency separation | Measure traffic and arithmetic together |
| Fused gate/up/SwiGLU producer | Useful dispatch/buffer reduction; mixed timing | Keep it, tune its register footprint |
| Larger K64 shared tile | Strong regression on tested shapes | Shared-memory capacity alone doesn't predict occupancy |
| Wider output tile | Early gains contradicted by later aligned-shape comparisons | Current narrow 64×32/K16 remains baseline |
| Cooperative F32 matrices | About 19–28% slower in that experiment | Hardware acceleration needs a good feed/fragment schedule |
| Simple workgroup swizzle | Roughly ±2%, within noise | Low priority without locality evidence |
| F16 staging | Small plain-kernel gains, drift and fused regressions | Separate precision investigation |
| Ternary LUT2 | Qualified local and bundle gains with resident duplicate cost | Preserve explicit layout/admission and current successful path |

Those results belong to their recorded revision, model, and device. The fresh synthetic baseline doesn't requalify those historical speedups or prove the same crossover elsewhere.

Fusion can also recompute expensive input transformations or expand registers until occupancy falls. Tinygrad's materialization pass conservatively keeps some intermediates when several inputs or reductions are involved. Its WebGPU buffer limit is another constraint. Use fusion where a producer/consumer schedule is demonstrably cheaper and valid; evaluate competing compositions. [Materialization and binding limits][tg-rangeify]

Split-K GEMM is a secondary candidate when the output grid has too few tiles. Typical N=1,024–3,072 prefill already supplies many output workgroups, and split reduction adds work and changes arithmetic order. Long-context attention has the clearer underoccupancy case.

Persistent GPU decode loops would also interact with host grammar, cancellation, and action/session semantics. Profile those transitions before attempting a GPU-resident control loop. Removing every CPU interaction isn't implied by kernel tuning.

## Implementation sequence and acceptance

Each row is independently reviewable. Make production changes only after its experiment earns a result.

| Step | Primary ownership | Deliverable | Acceptance evidence |
| --- | --- | --- | --- |
| A | `backend-wgpu/tests/kernel_bench.rs`, qualification tooling | Shape/layout-rich benchmark output; optional GPU timestamps; rotating working sets | Timing self-checks, overhead measured, raw samples and exact identities |
| B | `backend-wgpu/src/kernels.rs` and operation selectors | Descriptors for existing variants, no initial selection change | Same dispatch counts/ABI, registry consistency, native tier and browser qualification if behavior changes |
| C | Packed shaders/`packed.rs` | Canonical ternary MT variants and lane sweep | Boundary parity; full/cached same-bundle win; single-token and memory regressions checked |
| D | Dense/shared packed shader loaders | Coherent-load experiment, then bounded register tiles | Same representation/math mode; mask/stride checks; generated-code inspection; native/browser timing |
| E | Executor storage/execution/prefix tasks | Pass-owned reusable workspace with explicit live intervals | Full/cached/branch equality, cancellation/quarantine, limits/accounting, peak-memory evidence |
| F | Attention shaders/`attention_convolution.rs` | F32 tiled online prefill | Causal/nonfinite-tail/cache tests, logits/decisions, score-memory elimination, bundle timing |
| G | Attention dispatch and sampling/generation tooling | Split-KV decode with typed cutoff | Long-context generation, split boundaries, short-context fallback, grammar and cancellation |
| H | Device dispatch/completion | Bounded stable bindings and prepared plans | Host-time gain, no retained-memory growth, uniform-wrap/device-loss/cancellation checks |

For every candidate, use existing declared tolerances and add checks for changed arithmetic boundaries. Record maximum absolute/relative differences and argmax, not only top-1 agreement. Implementation parity and model-quality evaluation answer different questions.

A suggested performance policy is to require a repeatable end-to-end gain larger than measured dispersion, with no material regression in the accepted shape/platform set. Set numeric acceptance thresholds in the experiment profile before looking at results. A memory-only change may be accepted for reduced peak bytes with neutral latency if that is its stated objective.

Browser claims require actual browser execution. Native M1 results don't establish NVIDIA/AMD/DX12 behavior or browser callback scheduling. No browser execution, full-bundle benchmark, or candidate implementation was performed for this research report.

### Commands for a candidate change

From the runtime repository root:

```sh
scripts/check.sh quick
scripts/check.sh gpu
scripts/check.sh ci
scripts/check.sh wasm
```

Run the browser gate and bundle qualification with their explicit existing profiles as documented in [procedure](../procedure.md), [qualification](../../tools/qualification/README.md), and [web](../../web/README.md). The commands above are the future acceptance sequence; only the research checks listed earlier were executed for this document.

## Source map and reading order

Read the paired runtime/upstream sources for each proposed experiment. This keeps ideas attached to the actual missing work.

| Question | Tinygrad source | Runtime source |
| --- | --- | --- |
| Where do kernel boundaries and dependencies come from? | [Materialization][tg-rangeify], [schedule][tg-schedule] | [Executor execution][rt-execution] |
| Which threads compute which outputs? | [Options][tg-opt], [heuristics][tg-heuristic], [constraints][tg-postrange] | [GEMV][rt-gemv], [GEMM][rt-gemm], [packed body][rt-prefill] |
| How are choices searched and cached? | [Beam search][tg-search], [compiler][tg-codegen] | [Registry][rt-kernels], [device/pipelines][rt-device] |
| How are memory accesses simplified? | [Coalescing][tg-coalesce], [WGSL renderer][tg-wgsl] | [Dense][rt-gemm], [packed dispatch][rt-packed] |
| Which buffers can alias safely? | [Planner][tg-memory], [16 tests][tg-memory-tests] | [Scratch retention][rt-execution], [pool/quarantine][rt-device] |
| What survives capture and replay? | [JIT][tg-jit], [Metal queue][tg-metal] | [Prefix ownership][rt-prefix], [dispatch][rt-device] |
| How do attention and quantized edge cases get tested? | [Inference tests][tg-llm-tests], [custom inference][tg-amd] | [Attention][rt-attention], [benchmark][rt-bench] |
| Are profiler bandwidth numbers measured? | [Estimates][tg-estimates], [realization/timing][tg-realize] | [Qualification](../../tools/qualification/README.md) |

Tinygrad's `ops`, `lds`, and `mem` are modeled estimates. `lds` counts non-register accesses, including workgroup memory; `mem` caps repeated accesses to a buffer footprint. Neither is measured DRAM traffic. Our diagnostics should label estimated FLOPs/bytes separately from GPU timestamps and hardware-counter data. [Estimate definitions][tg-estimates]

Tinygrad is MIT-licensed. Retain its copyright/license notice when copying substantial implementation code. This report's proposals can be implemented against our own ABI; no upstream production code was copied. [License][tg-license]

[tg-schedule]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/schedule/__init__.py#L32-L85
[tg-rangeify]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/schedule/rangeify.py#L48-L195
[tg-codegen]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/__init__.py#L275-L377
[tg-opt]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/opt/__init__.py#L6-L8
[tg-heuristic]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/opt/heuristic.py#L62-L190
[tg-postrange]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/opt/postrange.py#L106-L165
[tg-search]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/opt/search.py#L15-L172
[tg-coalesce]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/codegen/late/coalesce.py#L109-L168
[tg-wgsl]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/renderer/wgsl.py#L53-L65
[tg-memory]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/schedule/memory.py#L12-L62
[tg-memory-tests]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/test/null/test_memory_planner.py
[tg-jit]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/engine/jit.py#L126-L175
[tg-webgpu]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/runtime/ops_webgpu.py#L69-L140
[tg-metal]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/runtime/ops_metal.py#L103-L163
[tg-amd]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/llm/kernels/amd.py
[tg-llm-tests]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/test/device/amd/test_llm.py#L286-L430
[tg-tc]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/renderer/tc.py
[tg-cstyle]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/renderer/cstyle.py#L361-L402
[tg-estimates]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/renderer/__init__.py#L17-L64
[tg-realize]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/tinygrad/engine/realize.py
[tg-license]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/LICENSE
[rt-kernels]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/kernels.rs#L233-L586
[rt-lib]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/lib.rs#L685-L702
[rt-device]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/device.rs#L639-L979
[rt-packed]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/packed.rs#L100-L180
[rt-execution]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/executor-core/src/lfm2/executor/execution.rs#L281-L526
[rt-prefix]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/executor-core/src/lfm2/executor/prefix_task.rs
[rt-attention]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/attention_convolution.rs#L14-L195
[rt-gqa]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/shaders/gqa.wgsl
[rt-gqa-batch]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/shaders/gqa_batch.wgsl
[rt-gemm]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/shaders/gemm.wgsl
[rt-gemv]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/shaders/gemv.wgsl
[rt-prefill]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/shaders/nf4_prefill.wgsl
[rt-bench]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/tests/kernel_bench.rs
[wgpu-subgroups]: https://github.com/gfx-rs/wgpu/blob/40f4a34ebaf56f9a046231f54125ad046239d3f3/wgpu-types/src/features.rs#L1126-L1142
[wgpu-matrix]: https://github.com/gfx-rs/wgpu/blob/40f4a34ebaf56f9a046231f54125ad046239d3f3/wgpu-types/src/features.rs#L1406-L1425
[wgpu-cache]: https://github.com/gfx-rs/wgpu/blob/40f4a34ebaf56f9a046231f54125ad046239d3f3/wgpu-types/src/features.rs#L1164-L1172
[wgpu-command]: https://github.com/gfx-rs/wgpu/blob/40f4a34ebaf56f9a046231f54125ad046239d3f3/wgpu/src/api/command_buffer.rs
[wgsl-subgroups]: https://www.w3.org/TR/WGSL/#subgroups
[rt-lanes]: https://github.com/Minifield-Labs/runtime/blob/bb3830aa29b7f6af1c0bf14bd8ef034ed4d1b79f/crates/backend-wgpu/src/lib.rs#L87-L101
[tg-speed]: https://github.com/tinygrad/tinygrad/blob/7bfaa18daaec8382f6656d46745a424b053cdc0a/docs/developer/speed.md#L47-L71
