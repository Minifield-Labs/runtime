# Bidirectional pointer execution prerequisite

## Hypothesis

The supplied encoder checkpoint needs its own complete-sequence equations before it can judge GPU optimizations. Sending it through causal LFM2 inference would change its attention, convolution boundaries, final feed-forward rows, and output heads. A speed measurement from that path would reward different predictions.

Training revision [`96fd486`](https://github.com/Minifield-Labs/minifield-training/blob/96fd486d1601bb910b390652bb21d53463e8237a/src/minifield_training/models/magicbox/pointer.py) defines the encoder and four pointer projections. Packed-segment behavior is pinned to [`736fe88`](https://github.com/Minifield-Labs/minifield-training/blob/736fe888a6754d0f80657879e080ac02dbfc154d/src/minifield_training/kernels/bidirectional.py). The runtime ports these equations and consumes serialized assets; it doesn't import training source or call Python during inference.

## Fix

`engine-api::EncoderOps` extends the existing finite operation contract with 2 operations: bidirectional GQA and centered gated convolution. They consume checked `EncoderSegments`, with `0` for padding and a nonzero label for each contiguous request. Positions restart at `0` in each active segment. A label can't leave its run and later reappear.

The scalar CPU implementation provides the arithmetic reference. Its output staging and score region share one checked temporary allocation, so concurrent mathematical scratch must fit the admitted budget. A query attends to all active keys in its own segment, including future tokens. Each KV head serves its existing contiguous query-head group. Masked rows are skipped before reading their values, so unused nonfinite storage can't contaminate an active output through `0 * NaN`.

The centered convolution reads `B[t + j - floor(width/2)] * V[t + j - floor(width/2)]`, multiplies by tap `j`, sums, then multiplies by `C[t]`. Out-of-range rows, padding, and other segments contribute zero. This also preserves the source's even-width convention: left padding of `floor(width/2)` and right cropping.

The encoder weight plan reuses validated LFM2 backbone roles and representations. It removes the vocabulary head and admits 4 dense pointer matrices: start/end query and start/end key. The checkpoint's backbone has 16 layers, hidden width `1024`, FFN width `4608`, `16` query heads, `8` KV heads, head dimension `64`, and pointer width `256`. Geometry remains config-owned.

The executor keeps every token through the final FFN and final RMS normalization. It gathers question marker states for query projections and projects all token states for keys. The resulting start/end logits are `Q Kᵀ / sqrt(pointer_width)`, with shape `[questions, tokens]`.

Operator, Q/K and FFN normalization use `block_norm_eps`; final embedding normalization uses `norm_eps`. A synthetic checkpoint with distinct values verifies that execution preserves both settings. The supplied checkpoint sets both to `1e-5`.

Task policy stays caller-owned. Choice, ordinal, binary, and extraction questions supply marker positions and selectable token metadata. Options average separate start/end softmax probabilities using FP64 host arithmetic, matching the pinned decoder. Span scores also add F32 logits in FP64, so close candidates retain the source's decision order. Extraction computes presence as `1 - p(absent)` and finds a maximum start-plus-end score within a contiguous selectable run. Strict comparisons preserve the source's deterministic scan order for ties. Returned spans use half-open source-relative token positions; hosts own text and character mapping.

Softmax denominators and ordinal expectations use fixed Neumaier FP64 compensation in candidate order. This matches the pinned decoder's Python `sum()` float path in its supported CPython 3.12/3.13 environments. Tiny candidates therefore survive threshold decisions. The [summation correction](research/runtime-pointer-host-summation-2026-09-29.md) records the primary source, numerical policy, boundary regressions, and independent parity evidence.

`PointerTask` records operations, submits a fence, then polls asynchronous readback. Its scratch arena owns every recorded output, including partial passes. Dropped fenced tasks transfer storage to the existing retirement queue. A failed fence submission quarantines the encoder and retains unfenced storage. Backend instance and generation checks match the existing causal executor's ownership boundary.

## Test

Synthetic CPU tests cover future-token attention, KV head geometry, isolated segments, inactive rows, odd/even convolution taps, restarting positions, stable option/span ties, rejected cross-segment candidates, loader admission, full final-layer FFN rows, all 4 pointer projections, repeated inference, task cancellation/drop, resource release, stale backend generations, combined temporary-storage admission before output mutation, adversarial FP64 probability thresholds and span-score ties, and compensated softmax/ordinal sums.

After review, all `15` portable encoder tests passed, including the distinct block/final epsilon fixture. The executor crate passed Clippy with warnings denied. Raw portable output is stored outside Git in `experiments/2026-09-29-runtime-gpu-hill-climb/notes/encoder-reviewed-epsilon-gate.log`.

GPU operation tests compare the new shaders with the scalar reference over small and multi-workgroup contexts, including the production `16`/`8` head ratio and dimension `64`. They cover odd/even convolution, padding, adjacent segments, extreme finite attention scores, overflowed masked value storage, rejected geometry, and recovery. These tests require a real adapter for qualification.

Review also found that flattened dispatch grids round up beyond `65,535` workgroups. Encoder attention now passes the logical group count and returns extra groups before any storage access or barrier. A `128`-token, `512`-query-head fixture crosses that boundary and compares the admitted output with the scalar reference.

The reviewed encoder hardware gate passed `5` tests with `0` failures or skips in `2.21 s`. All registered shader variants passed portable Naga validation, and the WGPU crate passed Clippy with warnings denied. Raw hardware output is stored outside Git in `experiments/2026-09-29-runtime-gpu-hill-climb/notes/encoder-reviewed-boundary-gate.log`.

Actual-checkpoint qualification must compare each backend with an independent reference on the same decoded representation and frozen pointer inputs. Stored-weight quantization quality is a separate question. The encoder path adds no measured performance claim by itself.

## Resolution

The runtime now has an explicit bidirectional pointer path and a scalar reference rather than relying on causal model execution to approximate this checkpoint. The causal executor and its last-row FFN optimization remain separate. Hardware qualification and supplied-model references determine acceptance before this path is used as a hill-climbing baseline.
