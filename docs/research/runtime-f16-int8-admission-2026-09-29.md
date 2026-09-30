# Give every precision lane an exact meaning

Date: September 29, 2026. This is model admission and backend bring-up evidence.
Performance claims require a frozen campaign after the prerequisites pass.

## Hypothesis

The supplied checkpoints need 4 independent storage lanes: FP16, INT8, NF4, and
ternary. Embeddings and task heads carry product information that the caller has
explicitly asked us to protect, so the preparation policy keeps those roles dense.

The existing loader expands F16 values into F32, but the typed LFM2 inventory
only admitted F32/BF16. INT8 also had no model representation or GPU operation.
Adding explicit representations should let the same finite executor run every
lane while preserving the exact weights used by its correctness reference.

## Fix

Added F16 to typed LFM2 configuration admission. Stored FP16 weights expand to
F32 backend values. The storage label describes the artifact; arithmetic remains
F32 throughout this baseline. Half arithmetic needs its own qualified policy.

Added `minifield.int8.v1` and mixed-role `int8-v1` admission. Each matrix stores
U8 `[rows,K]` two's-complement codes and F16 `[rows,K/128]` scales. Decoding is
`signed(code) * scale[group]`. Quantization permits codes from -127 through 127;
-128 is reserved and the loader rejects it before upload.

The protected preparation tool uses mixed metadata so embeddings, classification
heads, pointer projections, normalization weights, and convolution taps retain
their declared dense precision. The supplied QAT file keeps its original bytes.
Its reference occupies a separate cell because its pruned vocabulary and mixed
weights differ from the full classifier checkpoint.

Packed operand widths select 1, 2, or 4 weights per byte. They share the group-128
scale layout, so INT8, NF4, and ternary can't alias accidentally when dimensions
are valid. Backend-private LUT2 repacks stay restricted to ternary.

CPU INT8 uses a signed-byte group dot. WGPU INT8 uses explicit sign extension
and the existing F32 64-by-32 GEMM template through its own decode headers. This
establishes the numerical path at every row count; choosing faster short-row
kernels is a later candidate that must earn a win under the frozen host.

Backend resource counters now retain their highest accounted total. WGPU covers
physical buffers, alignment/pools, staging, uniform storage, and completion-owned
results. CPU covers classified resident buffers and completion-owned results.
CPU's internal temporary vectors, process RSS, pipelines, and driver allocations
sit outside this counter. The field is `peak_accounted_bytes`; it isn't a total
process or driver-memory measurement.

## Test

The signed-byte GPU fixture constructs codes independently of the exporter. It
covers -127, zero, 127, distinct group scales, K=256, 17 output rows, and input
rows 1, 7, 95, 96, and 97. It checks linear, paired projection, fused SwiGLU pair,
and fused SwiGLU input against independent scalar math.

The actual Metal adapter passed this test on September 29, 2026:

```sh
MINIFIELD_REQUIRE_GPU=1 cargo test -p minifield-backend-wgpu \
  --test int8 --locked -- --test-threads=1
```

2 loader regressions also passed. A reserved signed byte rejects before the
first weight upload. A mixed FP16/INT8 classifier plan keeps embedding/head
roles dense and binds the INT8 matrix to the exact signed-byte layout.

```sh
cargo test -p minifield-executor-core --test int8 --locked
```

The converter's independent encoding tests and complete checkpoint preparation
are recorded in [the preparation log](runtime-protected-precision-preparation-2026-09-29.md).
Actual model/backend parity, portable CI, peak-counter regression checks, and
full hardware qualification must pass before this foundation is considered ready.

## Resolution

FP16 storage and signed INT8 now have concrete, inspectable paths. The metadata,
loader, operand geometry, CPU decoder, and GPU decoder agree on the encoding.
The frozen references keep each precision's numerical behavior separate.

These changes supply missing evaluation prerequisites. Their speed remains
unclaimed until the real deployment workloads run through the frozen campaign.
