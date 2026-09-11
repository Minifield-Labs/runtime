# MFQ8 matrix record, version 1

Contract package: quant-matrix/0.1.0. Owner: training. Runtime consumes a pinned snapshot and the 60-byte synthetic fixture.

MFQ8 is a single-matrix interchange record for quantization and kernel parity tests. Full model bundles continue using their existing contract and engine-specific weight format.

## Byte layout

All metadata is little-endian. The header is exactly 28 bytes.

| Offset | Field | Type |
| --- | --- | --- |
| 0 | Magic MFQ8 | 4 bytes |
| 4 | Format version, 1 | u32 |
| 8 | Rows | u32 |
| 12 | Columns | u32 |
| 16 | Group width | u32 |
| 20 | Code count | u32 |
| 24 | Scale count | u32 |
| 28 | Row-major signed INT8 codes | code_count bytes |
| Next | Zero padding to the next 4-byte boundary | 0–3 bytes |
| Next | Row-major FP32 scales | scale_count × 4 bytes |

Rows, columns, and group width must be positive. code_count = rows × columns. scale_count = rows × ceil(columns/group_width). A group wider than its row has one scale. Padding must be zero and trailing bytes are rejected.

The reader validates checked dimensions, exact payload size, and a caller-supplied element budget before allocating matrix storage. Every scale must be finite, positive, and at least the smallest normal FP32 value.

## Quantization semantics

The audited exporter first converts finite real weights to FP32 and flushes subnormal weights to zero. For each row-local group, it uses max(abs(group))/127 with a minimum scale of FP32's smallest normal value; an all-zero group uses scale 1.

Values round half away from zero and clip to [-127, 127]. Imported signed-byte records may contain -128. Dequantization is code × group_scale. Biases and model graph semantics belong to the surrounding engine format.

The CPU oracle accumulates code/input products in FP32 and applies each group scale once. Compare with an explicitly chosen tolerance: accumulation order can differ across NumPy, JAX, Rust, and GPU implementations.

## Golden fixture

examples/quant-matrix-v0.1.0/ contains one synthetic 2 × 7 matrix with group width 4, half-way values, a zero group, partial final groups, and 2 input vectors. fixture.json records source weights, codes/scales, inputs, expected output, and the binary digest.

Refresh through training/scripts/write_quant_fixture.py, review any byte changes, and explicitly update runtime's snapshot. The record and examples contain no trained model.
