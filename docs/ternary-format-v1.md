# Ternary packed format v1

Created September 18, 2026. Status: v1 contract, pending first consumer
(T4 packed loader). Format identifier: `minifield.ternary.v1`.

This document is the versioned contract between the weight producer
(export tooling in `training/`) and the consumer (this repository's loader,
ops, and kernels). Once a packed artifact ships, fields defined here are
immutable; changes land as a new format version.

## Source model pin

- Model: `LiquidAI/LFM2.5-230M`
- Revision: `40cb2ad3b3044d5a41eee083a6103c8b523afa45`
- File: single `model.safetensors`, 132 tensors, all BF16,
  229,693,184 parameters
- Vocabulary: 65,536. Tied embedding / LM head.

## Tensor coverage

Packed (ternary codes + scales):

| Tensor pattern | Shape | Rows N | K |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | [65536, 1024] | 65536 | 1024 |
| `model.layers.N.self_attn.q_proj.weight` | [1024, 1024] | 1024 | 1024 |
| `model.layers.N.self_attn.k_proj.weight` | [512, 1024] | 512 | 1024 |
| `model.layers.N.self_attn.v_proj.weight` | [512, 1024] | 512 | 1024 |
| `model.layers.N.self_attn.out_proj.weight` | [1024, 1024] | 1024 | 1024 |
| `model.layers.N.conv.in_proj.weight` | [3072, 1024] | 3072 | 1024 |
| `model.layers.N.conv.out_proj.weight` | [1024, 1024] | 1024 | 1024 |
| `model.layers.N.feed_forward.w1.weight` | [2560, 1024] | 2560 | 1024 |
| `model.layers.N.feed_forward.w2.weight` | [1024, 2560] | 1024 | 2560 |
| `model.layers.N.feed_forward.w3.weight` | [2560, 1024] | 2560 | 1024 |

Dense (unchanged, BF16 or F32 as produced):

| Tensor pattern | Shape |
| --- | --- |
| `model.embedding_norm.weight` | [1024] |
| `model.layers.N.operator_norm.weight` | [1024] |
| `model.layers.N.ffn_norm.weight` | [1024] |
| `model.layers.N.self_attn.q_layernorm.weight` | [64] |
| `model.layers.N.self_attn.k_layernorm.weight` | [64] |
| `model.layers.N.conv.conv.weight` | [1024, 1, 3] |

Notes:

- `embed_tokens` serves two consumers: row gather (embedding lookup) and
  full matvec (tied LM head). It is packed once; the gather path dequantizes
  the selected row, the LM head uses the packed matvec.
- Attention layers: indices 2, 4, 6, 8, 10, 12. Conv layers: 0, 1, 3, 5, 7,
  9, 11, 13. Per `layer_types` in the model config.
- 104 projection tensors pack; 28 dense tensors remain. Packed weight bytes
  are approximately 61 MB versus approximately 920 MB expanded F32.

## Packed representation (safetensors container)

For each packed source tensor `X` of shape `[N, K]` (row-major, `K` the
reduction dimension), the artifact stores two tensors:

- `X.codes`: dtype `U8`, shape `[N, K/4]`. Each byte holds four 2-bit codes.
  Weight `j` of a row is coded by bits `[2*(j%4), 2*(j%4)+1]` of byte `j/4`,
  little-endian: weight 0 occupies the lowest two bits of byte 0.
- `X.scales`: dtype `F16`, shape `[N, K/128]`. One scale per contiguous
  128-weight group within the row. Grouping is per-row only.

Dequantization:

```
w[j] = (q[j] - 1) * scales[j / 128]
q in {0, 1, 2} decodes to {-1, 0, +1} times the group scale.
```

- `q = 3` is reserved in v1. It would decode to `+2 * scale`; consumers must
  reject a packed artifact containing code 3 rather than decode it.
- Scales are stored IEEE-754 binary16 and dequantized after conversion to
  F32. The stored fp16 value is the semantic value.
- v1 requires `K % 128 == 0` for every packed tensor. All packed tensors in
  the pinned model satisfy this (K in {1024, 2560}); no tail handling exists.
- v1 requires two-dimensional packed tensors. The LM head shares the packed
  embedding tensor; there is no separate `lm_head` entry.

Container metadata: the safetensors `__metadata__` block must contain:

| Key | Value |
| --- | --- |
| `format` | `minifield.ternary.v1` |
| `group_size` | `128` |
| `code_order` | `little-endian-byte-sequential` |
| `quantizer` | producer identifier, e.g. `absmax-v1` |
| `source_model` | `LiquidAI/LFM2.5-230M` |
| `source_revision` | `40cb2ad3b3044d5a41eee083a6103c8b523afa45` |
| `producer` | tool name + version string |

Dense tensors keep their original names and may be BF16 or F32. The runtime
loader treats `X.codes`/`X.scales` pairs as one packed logical tensor `X`.

## Reference quantizer (producer-side)

The format defines the code semantics, not the quantization algorithm. The
development exporter implements the reference rule, matching the Prism
quantizer so artifacts stay comparable:

- Per 128-weight group: `d = max |w|` computed in fp32, stored fp16.
- `q = clamp(round_half_away_from_zero(w / d) + 1, 0, 2)`. The producer never
  emits code 3 in v1.
- All-zero group: `d = 0`, every `q = 1`.
- On already-ternary inputs with an fp16-representable scale this rule is
  lossless.

## GGUF oracle variant (external validation only)

For cross-checking in Prism's `PrismML-Eng/llama.cpp` fork, an artifact may
be emitted in the fork's Q2_0 group-128 layout: per row, contiguous blocks of
`[2-byte fp16 scale][32 bytes of codes]` (34 bytes per 128 weights).

- GGUF type id 42 denotes g128 in the fork but g64 upstream; the two are not
  interchangeable and carry no in-file marker. Emit g128 only, and record the
  exact Prism fork commit used for validation (watch the planned `PQ2_0`
  rename).
- The GGUF variant is a validation artifact. The delivery container is the
  split-stream safetensors described above.

## Validation spec

Ordered gates; each comparison has exactly one moving part.

1. **Architecture gate (T3).** Existing F32 executor vs llama.cpp BF16
   oracle on the fixture prompts. Required: top-1 agreement at every
   evaluated position. Report max `|Δ logit|` and mean `|Δ|`; values are
   recorded in the run evidence. A max `|Δ|` above 0.5 stops the line for
   diagnosis (provisional threshold, tightened after measurement).
   Oracle capture rules: CPU-only build, `-t 1`, `--temp 0 --top-k 1`,
   `-ctk f32 -ctv f32`, prompts fed to our executor with the same token IDs
   llama.cpp used, including leading BOS id 1.
2. **Packing gate (T1/T4).** pack then dequant on ternary inputs with
   fp16-representable scales must be exact: codes preserved, dequantized
   values equal to inputs. Worked vectors live in
   `crates/executor-core/tests/fixtures/ternary-v1-001/`.
3. **Kernel gate (T4+).** Fused packed-linear output vs dequantize-then-F32-
   linear on identical inputs: near-exact. Integer-accumulation kernels are
   held to exact equality; float-accumulation paths to `|Δ| <= 1e-5`
   relative to output scale, declared per op in its test.
4. **Activation-quantization gate (deferred).** When int8 activation paths
   land (T5 follow-on), a separate declared tolerance applies versus the
   f32-activation packed path. Not part of v1 acceptance.

Accuracy of the ternary model versus BF16 (KL, top-1 drift) is reported as
information only and never gates a merge.

## Worked examples

`d` = group scale.

- `[d, -d, 0, 0.5d]`: `w/d = [1, -1, 0, 0.5]`, round-half-away gives
  `[1, -1, 0, 1]`, so `q = [2, 0, 1, 2]`, byte `0x92`. Dequantizes to
  `[d, -d, 0, d]` — the 0.5d case shows the rounding rule, not an error.
- All-zero group of 4: `q = [1, 1, 1, 1]`, byte `0x55`, scale `0`.
- `[-0.49d, -0.51d]`: `w/d = [-0.49, -0.51]` rounds to `[0, -1]`,
  `q = [1, 0]`. The -0.51 case flips to -d; ties and near-ties follow the
  same rule.

A full set of vectors including a complete 128-weight group is in the
`ternary-v1-001` fixture directory with a manifest.

## Out of scope for v1

Int8 activation quantization, KV cache compression, tail groups (`K % 128`),
non-2D packed tensors, alternate group sizes, and alternate code orders
(e.g. TQ2_0 digit-major). Each would be a new format version.
