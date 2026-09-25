# NF4 packed format v1

Created September 25, 2026. Status: v1 contract, first consumer the
polyomino classifier bundle. Format identifier: `minifield.nf4.v1`.

This document is the versioned contract between the weight producer
(export tooling) and the consumer (this repository's loader, ops, and
kernels). Once a packed artifact ships, fields defined here are immutable;
changes land as a new format version.

## Packed representation (safetensors container)

For each packed source tensor `X` of shape `[N, K]` (row-major, `K` the
reduction dimension), the artifact stores two tensors:

- `X.codes`: dtype `U8`, shape `[N, K/2]`. Each byte holds two 4-bit
  codebook indices. Weight `j` of a row is coded by the low nibble of byte
  `j/2` when `j` is even, the high nibble when `j` is odd: weight 0 occupies
  the lowest four bits of byte 0.
- `X.scales`: dtype `F16`, shape `[N, K/128]`. One scale per contiguous
  128-weight group within the row. Grouping is per-row only.

Dequantization:

```
w[j] = NF4[q[j]] * scales[j / 128]
q in {0..15} indexes the codebook below.
```

Codebook (sorted, bitsandbytes-compatible normal-float levels for
zero-mean data):

```
NF4 = [-1.0,
       -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
        0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023,  1.0]
```

- All 16 codes are valid in v1.
- Scales are stored IEEE-754 binary16 and dequantized after conversion to
  F32. The stored fp16 value is the semantic value.
- v1 requires `K % 128 == 0` for every packed tensor; no tail handling
  exists.
- v1 requires two-dimensional packed tensors.

Container metadata: the safetensors `__metadata__` block must contain:

| Key | Value |
| --- | --- |
| `format` | `minifield.nf4.v1` |
| `group_size` | `128` |
| `code_order` | `little-endian-byte-sequential` |
| `quantizer` | producer identifier, e.g. `nf4-v1` |
| `source_model` | producer's source model identifier |
| `producer` | tool name + version string |

Dense tensors keep their original names and may be BF16 or F32. The runtime
loader treats `X.codes`/`X.scales` pairs as one packed logical tensor `X`.

## Relationship to `minifield.ternary.v1`

Both formats share the split-stream safetensors container and the
per-128-group fp16 scales layout; they differ only in codes width and
decode. Consumers that support both derive the decode unambiguously from
the operand shapes: `scales` fixes `K` (scale count x 128), and the codes
width is `K/4` for ternary or `K/2` for NF4. A packed artifact must not mix
code widths inside one `X.codes`/`X.scales` pair.

## Reference quantizer (producer-side)

The format defines the code semantics, not the quantization algorithm. The
polyomino packer implements the reference rule:

- Per 128-weight group: `d = max |w|` computed in fp32, stored fp16.
- `q = argmin_c |NF4[c] - w / d|` (nearest codebook level), ties to the
  lower index.
- All-zero group: `d = 0`, every `q = 7` (NF4 index of 0.0).
- NF4 has an asymmetric codebook with a dedicated zero level, so exact
  zeros are representable.

## Validation spec

1. **Kernel gate.** Fused packed-linear output vs dequantize-then-F32-
   linear on identical inputs: `|d| <= 1e-4` absolute plus `1e-4` relative
   for float-accumulation kernels; the row gather is bitwise.
2. **Operand gate.** Consumers reject codes/scales pairs whose widths
   don't describe one of the supported formats (`K/4` or `K/2` against the
   scale-derived `K`), non-rank-2 streams, or non-U8 codes buffers.
3. **Model gate.** Producer reports argmax agreement and max `|d logit|`
   against the source model on its validation corpus; thresholds are
   declared per artifact, not by this contract.

## Out of scope for v1

Activation quantization, KV cache compression, tail groups (`K % 128`),
non-2D packed tensors, alternate group sizes, per-channel scales, and
alternate codebooks (e.g. FP4). Each would be a new format version.
