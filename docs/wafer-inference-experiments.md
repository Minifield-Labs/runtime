# Wafer inference reading: proposed Minifield experiments

Reviewed September 12, 2026. Status: research reference; experiment protocols now live in `experiments/`. No model or performance benchmark was run for this note.

Updated September 14, 2026: local inference targets batch 1. General request batching and concurrent candidate generation are outside this plan. Parallel test-time search would require a separate decision; longer sequential reasoning can still run at batch 1. Prefill can process many prompt tokens together within that one sequence.

Prioritize the first working Rust decoder, explicit memory accounting, repeated-context reuse, and reliable action generation. Keep each optimization behind a comparison with the same bundle, product requests, and target device. The [runtime procedure](procedure.md) still owns delivery and acceptance.

Added September 14, 2026: the [xn optimization reference](xn-optimization-reference.md) supplies pinned Rust code for profiling, compact cache access, single-query attention, quantized kernels, and scheduling. Its proposed trials fit experiments 0001–0004 in the linked protocols; model compatibility and browser support need separate validation.

## What the post and replies contribute

Wafer's [September 11 post](https://x.com/wafer_ai/status/2098550437364560336) introduces a curated [performance-engineering resource list](https://github.com/wafer-ai/gpu-perf-engineering-resources). Its first reading is the February 2025 chapter [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/). The list provides references across kernels, profiling, caching, quantization, and serving; its breadth doesn't establish a measured Minifield speedup.

The chapter separates prompt processing (prefill) from incremental token generation (decode). It connects their costs to computation, weight traffic, KV-cache traffic, and workload shape. Its server examples need new measurements on our target devices.

I read the main conversation and the replies exposed through X's probable-spam expansion, including the expanded technical replies. X can hide or omit replies, so this is a reading of the accessible conversation.

| Reply | What we take from it |
| --- | --- |
| [Loopsaaage on bandwidth](https://x.com/daotagoto/status/2098624575143866449), read through X's English translation, and [Murali on data movement](https://x.com/NMuraliRama/status/2098775329984872688) | Measure the bytes crossing each boundary. Treat bandwidth as a hypothesis to test for decode; prefill, unpacking, and launch overhead can require different work. |
| [Sebastian on workload/batch benchmarks](https://x.com/sebuzdugan/status/2098788984302309839) and [Yokush on sustained latency](https://x.com/YokushObiwan/status/2098732592015241430) | Define the actual request mix and latency objective. Start local runtime at batch 1; benchmark bulk generation separately. |
| [EYOV asking about MLX](https://x.com/everolivares/status/2098571545840955874) | Apple silicon is a useful comparison surface. MLX measurements would establish native behavior on that device; browser qualification remains its own gate. |

The replies supply useful prompts for investigation. The technical replies inspected don't include reproducible benchmark artifacts supporting their implied gains.

## Falcon and MiniCPM5 cache arithmetic

The pinned [Falcon-E-1B-Instruct configuration](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct/blob/e013cb099d5fe02a4ac46555971b152deb2d7cb1/config.json), checked in the browser, has 24 layers, 16 query heads, 2 KV heads, and head dimension 128. It already uses grouped-query attention. Preserve the 2-head KV representation in runtime storage.

The [MiniCPM5-1B configuration](https://huggingface.co/openbmb/MiniCPM5-1B/blob/main/config.json), checked September 14, has the same 24 layers, 2 KV heads, and head dimension 128. Both candidates therefore have the same KV payload per token at the same cache precision. Cache precision is independent of weight quantization.

For one sequence with a 16-bit cache, applying the chapter's cache formula gives:

```text
KV bytes = 2 (K and V) × layers × KV heads × head dimension × tokens × bytes/element
         = 2 × 24 × 2 × 128 × tokens × 2
         = 24,576 bytes per token = 24 KiB per token
```

| Total cached tokens | 16-bit KV | 8-bit payload | 4-bit payload |
| --- | ---: | ---: | ---: |
| 512 | 12 MiB | 6 MiB | 3 MiB |
| 1,024 | 24 MiB | 12 MiB | 6 MiB |
| 2,048 | 48 MiB | 24 MiB | 12 MiB |
| 4,096 | 96 MiB | 48 MiB | 24 MiB |
| 8,192 | 192 MiB | 96 MiB | 48 MiB |
| 16,384 | 384 MiB | 192 MiB | 96 MiB |
| 32,768 | 768 MiB | 384 MiB | 192 MiB |

These are calculated storage payloads, before scales, alignment, allocation slack, or temporary buffers. Prompt and generated tokens both count. Cache capacity and actively read cache length must be reported separately.

For prefix caching, apply the table to the retained prefix length. A 2,048-token app prefix retains 48 MiB at FP16/BF16. A CPU backend storing FP32 doubles these payloads. Keeping the prefix on both CPU and GPU also introduces another copy; unified physical memory only avoids duplication when the implementation shares the allocation.

Our local base-model assessment records a published inference weight file of 665,041,488 bytes. That historical file size needs rechecking for the actual delivered export. At short contexts, this gives us reason to examine weight representation and prefill before investing in low-bit KV kernels.

There is another concrete design constraint: the workspace's `training/src/minifield_training/sft/falcon.py`, in `layer_forward`, expresses repeated K/V heads and a full float32 attention-score tensor. At batch 1 and 4,096 tokens, that tensor's logical shape `[1, 16, 4096, 4096]` is 1 GiB for one layer. This is shape arithmetic, not a measured allocation; compiler fusion can change actual memory use. Keep that training path as a numerical reference and design the runtime's attention memory explicitly. [FlashAttention](https://arxiv.org/abs/2205.14135) supplies the primary reference for tiled exact attention.

## Experiments, in order

The [experiment index](research/proposals/README.md) now owns order and status. Each file contains the description, hypothesis, prerequisites, comparison, measurements, decision criteria, and result record. The links below preserve the original reading's experiment numbers.

### 1. Establish one complete decoder and a device report

[0001: Complete decoder and device baseline](research/proposals/0001-decoder-baseline.md).

### 2. Reuse app context and session history

[0002: Prefix and session reuse](research/proposals/0002-prefix-session-reuse.md).

### 3. Keep prefill and decode allocations bounded

[0003: Bounded prefill and decode allocations](research/proposals/0003-bounded-attention.md).

### 4. Measure actual packed-weight execution

[0004: Packed-weight execution](research/proposals/0004-packed-weight-execution.md).

### 5. Constrain action syntax and reduce repair cycles

[0005: Constrained action generation](research/proposals/0005-constrained-action-generation.md).

### 6. Compress KV only when its cost warrants it

[0006: KV-cache compression](research/proposals/0006-kv-cache-compression.md), conditional on measured cache cost.

### 7. Evaluate speculative decoding when decode cost warrants it

[0007: Speculative decoding, including DSpark](research/proposals/0007-speculative-decoding.md), conditional on measured task latency and resource headroom.

## Additional resource: Inside vLLM

Added September 14, 2026: Aleksa Gordić's [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm), published August 29, 2025. It explains vLLM V1 using commit `42172ad` from August 9, 2025. Treat API names, flags, and support claims as historical; verify upstream before implementation.

Our reading priorities for batch 1:

| Article topic | Connection to our plan |
| --- | --- |
| Prefix caching | Experiment 2: exact-prefix reuse and cache invalidation. Keep one bounded reusable allocation. |
| Chunked prefill | Experiment 3: measure memory, responsiveness, and cancellation between chunks. |
| Guided decoding | Experiment 5: include grammar compilation and token-masking overhead. |
| Latency benchmarks | Experiment 1: separate first-token, per-token, and complete-action latency. |
| Prompt-lookup speculation | Conditional [experiment 0007](research/proposals/0007-speculative-decoding.md): measure acceptance and verification cost. |

Continuous batching, distributed serving, and paged allocation across concurrent requests remain outside our current scope. CUDA graph replay requires a compatible native backend; browser execution needs separate validation. This resource adds reading and comparison ideas, with no measured Minifield speedup.

## Work to defer

| Technique | Trigger to reconsider |
| --- | --- |
| Continuous batching and sophisticated paged KV allocation | An explicit new requirement beyond the current batch-1 runtime; bulk generation belongs to a separate worker experiment |
| Parallel test-time candidate generation | A deliberate product/quality experiment with its own memory and latency budget |
| Distributed KV stores, tensor parallel serving, separate prefill/decode servers | A defined server workload under platform/data-generation with measured scale and cost |
| Draft-model speculative decoding | See the measured latency and resource triggers in [0007](research/proposals/0007-speculative-decoding.md) |
| CUDA-specific kernel ports | A measured NVIDIA worker bottleneck, or an explicit NVIDIA native target |
| Changes to attention architecture | A separate training/recovery experiment with product evaluations and deployment support |

Bulk teacher generation and RL rollout throughput belong to their own worker experiments. Keep the local user latency objective explicit.

## First backlog slice

Start with [0001: Complete decoder and device baseline](research/proposals/0001-decoder-baseline.md), including the reference device and declared limits. Follow the [experiment index](research/proposals/README.md) for subsequent comparisons, conditional triggers, and result status.

The experiment index also owns shared measurement rules and evidence storage. The [runtime procedure](procedure.md) continues to own product and release acceptance.
