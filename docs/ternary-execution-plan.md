# Ternary LFM2.5 execution plan

Created September 18, 2026. Status: plan agreed, no implementation started.

Goal: run LiquidAI LFM2.5-230M in this runtime with ternary weights packed as
GGUF-compatible Q2_0 (group 128), optimized for speed on Metal-class GPUs (via
wgpu) and wasm32 SIMD128, then fast multi-token generation with a KV cache.
Accuracy of the ternary model is explicitly not a gate; speed and correctness
of the math are.

Ownership boundary: this repo owns the packed format contract and the
consumption path (loader, ops, kernels). The quantizer used here is throwaway
scaffolding that produces packed checkpoints for speed iteration. The
production producer is training-side QAT export implementing the same format
spec; nothing quantization-related ships inside runtime.

## Locked decisions

- **Packing:** Bonsai-style Q2_0 g128, verified from source. Each weight
  `w = (q - 1) * scale`, `q` a 2-bit code, one FP16 scale per 128 weights,
  34 bytes per block (2.125 bits/weight). Bit order: weight `j` sits at bits
  `2*(j%4)` of byte `j/4`, little-endian. `q = 3` decodes to `+2*scale` but is
  unreachable under the reference quantizer; assert it never appears.
  ~61 MB for 230M parameters. References: `prism-ml/Ternary-Bonsai-1.7B-gguf`,
  Prism's `PrismML-Eng/llama.cpp` fork `prism` branch (NEON + Metal kernels).
- **Quantizer (scaffolding):** absmax per group to mirror the reference
  (`d = max|w|`, `q = clamp(round(w/d)+1, 0, 3)`); absmean kept as an option.
  Lossless either way on already-ternary inputs. A working-reference Python
  script on a branch in `training/` — the deliverable is the packed model
  files it emits for runtime development, not the tool itself. Disposable
  once QAT export exists.
- **Container:** our own safetensors stores split streams per packed weight —
  a `U8` code tensor plus a separate scales tensor — which fixes u32 alignment
  for WGSL and passes existing manifest validation unchanged. The Prism
  interleaved `[scale][codes]` layout is emitted only in the GGUF oracle
  artifact. Type ID 42 is g128 in the fork, g64 upstream, with no in-file
  marker; emit g128 and pin a fork commit (watch the `PQ2_0` rename).
- **GPU backend:** wgpu. One WGSL codebase serves native Metal and browser
  WebGPU later. Pure wasm32 SIMD128 kernels remain a separate CPU surface.
- **Oracle:** llama.cpp GGUF. `LiquidAI/LFM2.5-230M-GGUF` is published with a
  BF16 variant (bit-identical weights, no conversion). Prism's fork runs our
  exported Q2_0 GGUF as a packed-format end-to-end check.
- **Activation path:** f32 activations are the correctness anchor everywhere
  (matches the reference Metal path). int8 activation quantization is a later
  CPU kernel optimization, not part of the first parity target.

## Validation strategy

1. Existing F32 CPU executor vs llama.cpp logits on identical prompts:
   proves architecture math. Tolerance is declared, not zero — summation
   order and RoPE trig paths differ. Top-1 agreement is required.
   Oracle mechanics: CPU-only build (`-DGGML_METAL=OFF -DGGML_NATIVE=OFF`),
   `-t 1`, `--temp 0 --top-k 1 --seed 42`, `-ctk f32 -ctv f32` (KV defaults
   to f16 — mandatory flags), `--verbose-prompt` to record token IDs.
   `--save-all-logits` writes u16 log-probs, adequate for top-1 only; a tight
   tolerance gate needs a small libllama tool writing raw f32 logits.
   Fixture prompts must explicitly begin with token 1 (BOS is not auto-added).
2. Fused ternary kernel vs dequantized-ternary weights through the F32 path:
   proves kernel math. Dequantized ternary values are exact in F32, so this
   comparison is near-exact and isolates kernel bugs from quantization error.
3. Ternary model vs BF16 model: top-1 agreement / KL divergence reported as
   information only, never a gate.
4. Exported Q2_0 GGUF through Prism's llama.cpp fork: external check on both
   packing and kernel semantics.

## Current foundation

Already built in this repository: backend-neutral LFM2 executor over
`InferenceOps` (crates/executor-core), scalar CPU backend (crates/backend-cpu),
byte-level tokenizer, generation loop, infer-cli, prefix snapshots with
per-layer conv history and KV cache, safetensors loader (F32/BF16 only),
audited Q4/Q8 references (engines/quant-reference), xn review
(docs/xn-optimization-reference.md), ordered experiment protocols
(experiments/0001-0007).

Known gap for speed: the executor appends one token at a time through ~217
backend op calls plus ~21 cache copies per token (~240 interactions total at
230M's layer count). At the measured ~32-71 us per wgpu dispatch on Metal that
is 5-18 ms of CPU time per token — roughly 10x the ~0.6 ms GPU weight-streaming
cost at Q2_0. Decode fusion to ~20 dispatches is the dominant speed task, not
optional. All portable ops are already multi-token capable, so prefill batching
is executor-side work only.

## Research findings (September 18, five-agent sweep)

Verified facts: the real checkpoint is one safetensors file, 132 tensors, all
BF16, 229,693,184 params, pinned rev `40cb2ad3`. Names match `weights.rs`
exactly; the config parses today. Executor math matches both HF
`modeling_lfm2.py` and llama.cpp `lfm2.cpp`. Pinned xn source is checked out
at `runtime/tmp/xn-src` (gitignored) for kernel reference.

### Spec items resolved for T0

- Packed container = U8 code tensor + separate scales tensor per weight (not a
  new `DType`; `byte_width` math assumes whole bytes everywhere). New
  `PackedLinear` op in `InferenceOps`; executor branches per-role at the ~10
  `linear` callsites. Norms and the [1024,1,3] conv kernel stay dense; the tied
  embedding stays a gather while lm_head takes the packed path.
- Q2_0 block stride is 34 B, not u32-aligned — split streams also serve the
  WGSL binding problem.
- lm_head grid: 65,535 workgroups/dim overflows at n=65536 by one — use a 2D
  grid or multi-column workgroups.
- Uniform buffer ring (256 B dynamic offsets), not push constants (native-only).
- `map_async` polling only; a blocking recv deadlocks a browser tab.
- wgpu handles are not Send/Sync on wasm32; the backend stays on its worker
  thread. Request adapter-max storage-buffer/buffer limits (defaults 128/256
  MiB fail the F32 lm_head at 268 MB; packed it is ~18 MB).

### Pitfalls logged for T3

- Tokenizer admission is pinned to an LFM2-derived profile (exact JSON equality
  on pre_tokenizer/decoder, `[left,right]` merge arrays, contiguous vocab) —
  highest-risk silent blocker; check the real tokenizer.json first.
- Exact-inventory manifest strictness (extra tensor fails) and uniform-BF16
  requirement — real file verified at exactly 132 BF16 tensors.
- Tokenizer rejects emitted IDs >= 64402 though model vocab is 65536 —
  mid-stream error edge case to handle or document.
- Executor eps routing: `block_norm_epsilon` feeds layer/head norms but HF uses
  `norm_eps` for all; equal at 1e-5 so numerically harmless — fix or document.
- Verify conv history buffers are zero-initialized in the backend.
- 128K context exceeds the 2 GiB default (KV ~3.2 GB); run bounded context.

### Kernel decisions from the survey

- Decode is bandwidth-bound (Prism ~99% of bound on Metal at 8B); ceiling at
  61 MB is ~2,000-4,500 tok/s GPU, ~1,000-1,300 multi-core CPU.
- NEON: port ggml's Q2_0 kernel to g128 (4x32 sub-blocks, int8 activations +
  SDOT). AVX2: i16-lane shift + madd/maddubs (no upstream exists). WASM:
  baseline `i32x4.dot_i16x8_s`, `dot_i8x16_i7x16` gated behind a probe.
  WGSL: dequant-to-f32 + FMA with shared-mem reduction; DP4a/subgroups gated.
- xn templates to copy: `vec_dot_q2k_q8k` unpack pattern, `sdpa_decode`
  online-softmax shape + `qh/group` head mapping, `threadpool`/`par_units`,
  buffer pool + lazy pipeline cache + single-encoder batching. xn's
  `poll(Wait)` sync, per-read fresh staging, and push-constants requirement
  are the parts to avoid.
- LUT (TL1/T-MAC) and 5-trits packing: deferred unless profiling shows
  compute-bound.

## Tasks

| ID | Task | Depends on | Status | Acceptance |
| --- | --- | --- | --- | --- |
| T0 | Format + validation spec: model rev `40cb2ad3` pinned, tensor coverage, split code/scale layout in safetensors + GGUF g128 export, tolerance gates, fixture policy. Resolved inputs in Research findings | none | Planned | Spec doc + hand-packed tiny fixture |
| T1 | Sample exporter on a branch in `training/` (Python): BF16 safetensors to 2-bit codes + FP16 group scales, absmax RTN, split-stream safetensors out + GGUF out via gguf-py. Working reference only; the deliverable is packed model files for runtime development | T0 | Planned | Produces a packed LFM2.5-230M that loads in runtime; round-trip + q=3-assertion sanity checks pass |
| T2 | Oracle capture: published `LFM2.5-230M-BF16.gguf`, CPU llama.cpp build, logits and greedy sequences on fixed prompts (f32 KV, `-t 1`, temp 0), hashed fixtures outside Git | none | Done (llama.cpp b11046, `llama-completion`, 6 prompts; tooling committed under `scripts/oracle/`, artifacts in `tmp/oracle/`) | Fixture set with recorded hashes; oracle reproducible |
| T3 | F32 real-weight validation: existing CPU executor on actual 230M vs T2 logits. Check tokenizer profile + inventory first; fixture prompts include BOS 1; eps-routing fix | T2 | Done (`tests/oracle_lfm25.rs`, env-gated): 118/118 positions top-1, max\|Δ\| 0.33, mean 0.021; 330/332 generated ids, both misses near-ties | Logit parity within declared tolerance, top-1 agreement |
| T4 | Ternary CPU path: loader accepts U8 codes + scales tensors, `packed_linear`/`packed_gather_rows` ops, scalar ternary matvec, per-role executor wiring via `resolve`. Escape hatch realized as bitwise dequantized-reference comparison (in-loader dequant mode deferred; it needs a 2-tensor→1-buffer binding pass) | T1, T3 | **Done** | Packed executor bit-identical to dequantized dense executor (synthetic model + fixture vectors); real packed model runs end to end |
| T5 | CPU SIMD ternary matvec: NEON done in `crates/kernels-simd` (vld4-deinterleaved f32-activation path, scoped-unsafe crate); AVX2 untested on dev hardware; thread pool and int8-activation path remain open | T4 | **Done (NEON); pool + AVX2 pending** | Parity vs scalar within declared reorder tolerance; packed 26-token run 6.74s -> 0.96s (7x) on M1 Max |
| T6 | wasm32 SIMD128 kernels + single-thread harness: baseline `dot_i16x8_s`, relaxed-dot gated by probe; threads deferred | T4 | Planned | Native-vs-WASM output parity (check-web pattern); measured speedup |
| T7a | wgpu InferenceOps backend, F32: single-encoder batching, SubmissionIndex fences, pooled staging, uniform-ring params, WGSL kernels, buffer pool | none | Planned | F32 executor on wgpu matches backend-cpu outputs |
| T7b | wgpu ternary matvec + fused decode kernels (Prism MSL Q2_0 kernel adapted to WGSL; separate codes/scales bindings, 2D lm_head grid) | T4, T7a | Planned | Ternary forward on GPU matches T4 CPU reference; faster than CPU |
| T8 | Decode fusion (dominant speed item): ~240 interactions/token to ~20 dispatches; fused GQA online-softmax decode, fused conv/MLP, on-device logits/argmax, one readback per token | T4 + one backend | Planned | Dispatch count per token reduced; measured decode latency gain |
| T9 | Prefill batching: multi-token rows through existing shape-generic ops | T4 | Planned | Identical outputs vs per-token path; measured prefill speedup |
| T10 | KV work: F16 KV option, active views over preallocated cache, prefix reuse measurement | T8 | Planned | Cache bytes and latency measured; parity preserved |
| T11 | Bench harness: per-op attribution, cold/warm, memory, p50/p95 per run-record template | T3 | Planned | Reproducible reports for each backend and format |

Experiment mapping: T3/T11 feed experiments/0001 (decoder baseline); T10 maps
to 0002/0003; T5/T6/T7b to 0004 (packed-weight execution); constrained actions
(0005), KV compression (0006), and speculative decoding (0007) stay on their
own triggers.

## Critical path and parallelism

T0 -> {T1, T2} -> T3 -> T4 -> {T5, T6, T7b} -> {T8, T9, T10}.

Startable today with no dependencies: T0, T2, T7a. T7a is the largest
independent chunk and needs no oracle or packing work.

## Per-task workflow

Every task runs the same loop so `main` is always usable and never drifts:

1. Branch off current main (`ternary/<task>` naming).
2. Implement.
3. Test: run the task's acceptance gate from the table above, fresh.
4. Report: append the iteration to the relevant
   `reports/<model>-<platform>/metrics.json` and re-render `report.html`.
   Perf-bearing tasks record real measurements; correctness gates still add a
   row with whatever was measured (unmeasured fields render N/A).
5. Code review the full diff before PR. Subagent-produced diffs get a
   fresh-eyes pass from the orchestrating session.
6. Docs: docstrings on new public items; AGENTS.md/README updates wherever
   behavior, commands, or crate inventory changed.
7. PR via `gh`, merged into main per repo convention. No task stacks on an
   unmerged branch unless the dependency table already says so.
8. Repo returns clean — `git status` empty — before the next task starts.

Parallel streams still apply: kernel agents work disjoint crates on their own
branches and merge in completion order. In-progress work lives on branches
and worktrees only; main stays clean throughout.

## Open items deferred by this plan

- wasm32 threads (SharedArrayBuffer) need COOP/COEP headers; evaluate after
  single-threaded SIMD results.
- GGUF as the primary bundle container vs safetensors-with-packed-blobs.
  Current plan keeps the existing safetensors loader and emits GGUF only for
  oracle use. Revisit if GGUF interop becomes a delivery requirement.
- Browser Worker host and product integration stay under docs/procedure.md;
  this plan covers inference only.
