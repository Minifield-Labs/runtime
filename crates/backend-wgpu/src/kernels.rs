//! Portable WGSL kernels for the F32 wgpu backend.
//!
//! Every kernel takes kernel parameters through binding 0 as a `var<uniform>`
//! block of `vec4<u32>` words. The backend binds a 256-byte slot of a shared
//! uniform ring with a dynamic offset, so no push constants or immediates are
//! required. All tensor bindings are `read_write` storage buffers so one bind
//! group layout per binding count serves every kernel. Grids that can exceed
//! `65_535` workgroups flatten `wid.z * (nx * ny) + wid.y * nx + wid.x` through
//! `@builtin(num_workgroups)` and bounds-check inside the shader.

/// Flattened workgroup index for 1D-element grids, plus early-out guard.
const WGSL_INDEX: &str = include_str!("shaders/wgsl_index.wgsl");

/// Zero-fill used by `allocate_f32` (device buffers are not zero-initialized).
const FILL: &str = include_str!("shaders/fill.wgsl");

/// Word-wise repack of a `minifield.ternary.v1` code stream into the LUT2
/// pair-nibble layout. Each raw u32 carries sixteen two-bit codes, read as
/// eight consecutive pairs `c0 | (c1 << 2)`; each pair becomes one nibble
/// indexing [0, x0, x1, x0+x1, x0-x1] with bit 3 as the negate flag. Raw
/// code 3 has no LUT2 representation; it maps to the zero nibble because
/// valid streams never carry it. Same byte count in and out.
const REPACK_LUT2: &str = include_str!("shaders/repack_lut2.wgsl");

/// Elementwise binary op over equal contiguous layouts. `op`: 0 = add, 1 = multiply.
const BINARY: &str = include_str!("shaders/binary.wgsl");

/// Checked row-major rectangle copy between distinct packed rank-two buffers.
/// dst[(dr+r)*dw + dc + c] = src[(sr+r)*sw + sc + c] for r < rows, c < cols.
const COPY2D: &str = include_str!("shaders/copy2d.wgsl");

/// Gather selected rows from a packed [rows, columns] table.
/// dst[r*cols + c] = table[ids[r]*cols + c]
const GATHER: &str = include_str!("shaders/gather.wgsl");

/// Gather selected columns from a contiguous [rows, width] f32 input:
/// dst[r*count + k] = src[r*width + cols[k]]. `cols` carries staged host u32
/// selectors, already range-checked against `width` on the host; caller
/// order and duplicates are preserved.
const GATHER_COLUMNS: &str = include_str!("shaders/gather_columns.wgsl");

/// GEMV for the m == 1 decode path: dst[j] = `sum_l` x[l] * w[j,l].
/// One workgroup per output column; 256 threads tree-reduce over k in shared
/// memory. `w4` binds the same weight buffer as a vec4 view: a weight row that
/// starts 16-byte aligned (j*k % 4 == 0) streams one 128-bit load per four
/// elements; unaligned rows and the k % 4 tail use the scalar path.
const GEMV: &str = include_str!("shaders/gemv.wgsl");

/// Packed ternary GEMV for `minifield.ternary.v1` weights:
/// dst[i*n + j] = `sum_l` x[i*k + l] * (code(j,l) - 1) * scale(j, l/128).
/// One workgroup per output element (the flat grid covers every m, m == 1
/// being the decode case); 256 threads tree-reduce over k in shared memory.
/// Codes bind as u32 words over the byte stream: weight l lives in byte l/4
/// of the row, i.e. bits [8*((l/4)%4) + 2*(l%4)] of word l/16.
const PACKED_GEMV: &str = include_str!("shaders/packed_gemv.wgsl");

/// Packed ternary gather: dst[r*k + l] = (code(ids[r], l) - 1) *
/// scale(ids[r], l/128). One thread per output element; the code decode is
/// the same byte/bit scheme as `PACKED_GEMV`.
const PACKED_GATHER: &str = include_str!("shaders/packed_gather.wgsl");

/// NF4 codebook for `minifield.nf4.v1` streams: decode
/// `w = NF4[code] * scale` where `code` indexes this sorted 16-entry
/// normal-float table (bitsandbytes-compatible levels for zero-mean data).
/// Each u32 codes word carries eight 4-bit indices, low nibble first, so a
/// word covers 8 weights and a 128-weight group spans 16 words.
const NF4_LUT: &str = include_str!("shaders/nf4_lut.wgsl");

/// Packed NF4 GEMV for `minifield.nf4.v1` weights:
/// dst[i*n + j] = `sum_l` x[i*k + l] * NF4[code(j,l)] * scale(j, l/128).
/// Same lane-grouped reduce layout as `PACKED_GEMV`; each u32 word decodes
/// eight 4-bit NF4 level indices instead of sixteen 2-bit ternary codes.
const PACKED_GEMV_NF4: &str = include_str!("shaders/packed_gemv_nf4.wgsl");

/// Multi-token packed NF4 GEMV: same lane-grouped reduce as
/// `PACKED_GEMV_NF4` but each workgroup covers MT input rows, so every codes
/// word is decoded once and reused for all MT dots. Use for prefill-width
/// inputs where per-row re-decode dominates; m=1 keeps the single-token
/// kernel. `pc.q` = [m, `m_tiles`, `vec_ok`, 0].
const PACKED_GEMV_MT_NF4: &str = include_str!("shaders/packed_gemv_mt_nf4.wgsl");

/// Packed NF4 gather: dst[r*k + l] = NF4[code(ids[r], l)] *
/// scale(ids[r], l/128). One thread per output element.
const PACKED_GATHER_NF4: &str = include_str!("shaders/packed_gather_nf4.wgsl");

/// Row-wise argmax over `[T, V]` f32 logits: one workgroup per row, 256
/// threads tree-reduce (value, index) pairs keeping the smallest index on
/// ties. `dst[r]` is the winning index as an exact f32 integer, or NaN when
/// the row contains any non-finite element.
const ARGMAX: &str = include_str!("shaders/argmax.wgsl");

/// Argmax stage 1 for wide rows: each workgroup reduces one 2048-element
/// block of a row to a `(value, first-index)` partial in `partials`. A
/// non-finite element in the block writes NaN, which stage 2 propagates. The
/// single-workgroup `ARGMAX` path is kept for narrow rows.
const ARGMAX_BLOCKS: &str = include_str!("shaders/argmax_blocks.wgsl");

/// Argmax stage 2: one workgroup per row reduces the stage-1 partials to the
/// row's first strict maximum index, or NaN when any partial is NaN or every
/// block reported the 2^24 empty-candidate sentinel.
const ARGMAX_FINAL: &str = include_str!("shaders/argmax_final.wgsl");

/// Paired packed ternary GEMV over one shared input: workgroup (i, j) computes
/// `dst_a[i,j]` and `dst_b[i,j]` with independent accumulators over the same
/// x row. Weight decode matches `PACKED_GEMV`; outputs share one `[m, n]`
/// shape.
const PACKED_GEMV_PAIR: &str = include_str!("shaders/packed_gemv_pair.wgsl");

/// Packed ternary GEMV over an on-the-fly SiLU(gate) * up activation:
/// dst[i*n + j] = `sum_l` (silu(gate[i,l]) * up[i,l]) * (code(j,l) - 1) *
/// scale(j, l/128). Same lane-grouped layout as `PACKED_GEMV`; the activation
/// is computed inline from coalesced gate/up loads so no shared staging or
/// extra barrier is needed.
const PACKED_SWIGLU_GEMV: &str = include_str!("shaders/packed_swiglu_gemv.wgsl");

/// Paired packed NF4 GEMV over one shared input: same lane-grouped layout as
/// `PACKED_GEMV_PAIR` with the `minifield.nf4.v1` nibble decode.
const PACKED_GEMV_PAIR_NF4: &str = include_str!("shaders/packed_gemv_pair_nf4.wgsl");

/// Packed NF4 GEMV over an on-the-fly SiLU(gate) * up activation: same
/// lane-grouped layout as `PACKED_SWIGLU_GEMV` with the `minifield.nf4.v1`
/// nibble decode.
const PACKED_SWIGLU_GEMV_NF4: &str = include_str!("shaders/packed_swiglu_gemv_nf4.wgsl");

/// Multi-token packed NF4 GEMV over two shared-input weights: decode each
/// codes pair once, reuse across MT input rows. `pc.q` = [m, `m_tiles`, `vec_ok`,
/// 0]. Partial buffers hold MT accumulators per thread.
const PACKED_GEMV_PAIR_MT_NF4: &str = include_str!("shaders/packed_gemv_pair_mt_nf4.wgsl");

/// Multi-token packed NF4 GEMV over an on-the-fly `SiLU(gate) * up`
/// activation: decode once per word, reuse across MT input rows. `pc.q` =
/// [m, `m_tiles`, `vec_ok`, 0].
const PACKED_SWIGLU_GEMV_MT_NF4: &str = include_str!("shaders/packed_swiglu_gemv_mt_nf4.wgsl");

/// Fused residual add plus row RMS norm: `sum[r,c] = a[r,c] + b[r,c]` then
/// `normed[r,c] = sum[r,c] * rsqrt(mean(sum[r,:]^2) + eps) * alpha[c]`.
/// One workgroup per row; the two passes share one dispatch.
const ADD_NORM: &str = include_str!("shaders/add_norm.wgsl");

/// Fused per-head RMS norm plus split-half rotary for query and key rows.
/// Workgroup `w` handles one [token, head] slice: `w / heads_total` is the
/// token, heads `0..q_heads` map to query, the rest to key. The normed head
/// is staged in shared memory so both rotation halves read final values.
/// `table` is the same host cos/sin layout `ROTARY` uses; `head_dim <= 512`.
const QK_NORM_ROPE: &str = include_str!("shaders/qk_norm_rope.wgsl");

/// Tiled shared-memory GEMM for m > 1: dst[i,j] = `sum_l` x[i,l] * w[j,l].
/// Tiles are 16x16; the weight tile is transposed on load because W is packed
/// [n, k] while the tile needs [k, n]. Grid: (ceil(n/16), ceil(m/16) split over
/// y/z); each thread produces one output element.
const GEMM: &str = include_str!("shaders/gemm.wgsl");

/// Row RMS norm: dst[r,c] = src[r,c] * rsqrt(mean(src[r,:]^2) + eps) * alpha[c].
/// One workgroup per row; also serves per-head norms by treating each
/// [token, head] slice as a row.
const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");

/// Split-half `RoPE` over packed `[tokens, heads * head_dim]` rows. Host-computed
/// cos/sin tables (f64 evaluation cast to f32, matching the scalar reference)
/// are bound as two contiguous regions of one table buffer.
/// dst[i1] = a*cos - b*sin ; dst[i2] = b*cos + a*sin
const ROTARY: &str = include_str!("shaders/rotary.wgsl");

/// Causal grouped-query attention for one appended token.
/// One workgroup per query head. Stage 1 computes scaled dot scores for every
/// visible key into a scratch row, stage 2 tree-reduces the max and the exp
/// denominator, stage 3 writes the weighted value sum per head dimension.
/// Keys/values before `init` come from the caches; later ones from the
/// appended key/value rows.
const GQA: &str = include_str!("shaders/gqa.wgsl");

/// Batched causal GQA for multi-token calls: identical math to `GQA` but one
/// workgroup covers one (query head, token) pair of a token block, so a
/// prefill-width batch dispatches once per block instead of once per token.
/// Grid = `block_tokens` * `query_heads`; `p2` = [`query_heads`, `block_tokens`,
/// `token_base`, 0]. Score scratch is one `new_cache_len` row per (t, h) pair
/// in the block.
const GQA_BATCH: &str = include_str!("shaders/gqa_batch.wgsl");

/// Gate + history concatenation for the gated short convolution.
/// `u_ext[i]` is `history[i]` for `i < hist_elems`, else `b[j] * v[j]` read
/// from the fused `[tokens, 3 * hidden]` projection at offsets `{0, 2h}`.
const CONV_GATE: &str = include_str!("shaders/conv_gate.wgsl");

/// Depthwise gated short convolution over the concatenated history + gate rows.
/// `dst[t*hidden + c]` is `proj[t, h + c]` (the fused C gate) times the tapped
/// sum over `u_ext` rows.
const CONV: &str = include_str!("shaders/conv.wgsl");

/// Fused single-token gated short convolution. Computes u = B * V for the new
/// token, the tapped conv sum scaled by the C gate, and writes the shifted
/// history `hist_out = [hist rows 1..m | u]` so the caller can swap it into the
/// history buffer without same-buffer copies.
const CONV_STEP: &str = include_str!("shaders/conv_step.wgsl");

/// `SiLU`(gate) * up over equal contiguous layouts.
const SWIGLU: &str = include_str!("shaders/swiglu.wgsl");

const NF4_LINEAR_HEADER: &str = include_str!("shaders/nf4_linear_header.wgsl");

/// `NF4_LINEAR_HEADER` with the `minifield.ternary.v1` two-bit decode:
/// sixteen codes per word, values -1/0/+1 times the per-128 scale.
const TERNARY_LINEAR_HEADER: &str = include_str!("shaders/ternary_linear_header.wgsl");

const NF4_PAIR_HEADER: &str = include_str!("shaders/nf4_pair_header.wgsl");

/// `NF4_PAIR_HEADER` with the `SwiGLU` epilogue folded into the store: each
/// invocation already owns the finished 2x2 fragments of both projections,
/// so `silu(gate) * up` writes one hidden value instead of two buffers.
const NF4_PAIR_SWIGLU_HEADER: &str = include_str!("shaders/nf4_pair_swiglu_header.wgsl");

/// Same fusion for `minifield.ternary.v1` streams: two-bit codes, 16 weights
/// per u32 word, `code - 1` times the group scale.
const TERNARY_PAIR_SWIGLU_HEADER: &str = include_str!("shaders/ternary_pair_swiglu_header.wgsl");

/// `NF4_PAIR_HEADER` with the `minifield.ternary.v1` two-bit decode.
const TERNARY_PAIR_HEADER: &str = include_str!("shaders/ternary_pair_header.wgsl");

/// `NF4_SWIGLU_HEADER` with the `minifield.ternary.v1` two-bit decode.
const TERNARY_SWIGLU_HEADER: &str = include_str!("shaders/ternary_swiglu_header.wgsl");

/// Group-partial control headers: the weight accessor returns the UNSCALED
/// decoded code value and `scale_*` exposes the per-128 group scale, so the
/// `lowbit_scale128` template can defer scaling to group boundaries.
const TERNARY_SCALE128_HEADER: &str = include_str!("shaders/ternary_scale128_header.wgsl");

const NF4_SCALE128_HEADER: &str = include_str!("shaders/nf4_scale128_header.wgsl");

const NF4_SWIGLU_HEADER: &str = include_str!("shaders/nf4_swiglu_header.wgsl");

/// Kernel identifiers in lazy-pipeline-cache order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Kernel {
    Fill,
    Binary,
    Copy2d,
    Gather,
    Gemv,
    Gemm,
    RmsNorm,
    Rotary,
    Gqa,
    GqaBatch,
    ConvGate,
    Conv,
    ConvStep,
    SwiGlu,
    PackedGemv,
    PackedGather,
    PackedGemvPair,
    PackedSwigluGemv,
    PackedGemvNf4,
    PackedGatherNf4,
    PackedGemvPairNf4,
    PackedSwigluGemvNf4,
    PackedGemvMtNf4,
    PackedGemmNf4,
    PackedGemmTernary,
    PackedGemvPairMtNf4,
    PackedGemmPairNf4,
    PackedGemmPairTernary,
    PackedSwigluGemvMtNf4,
    PackedSwigluGemmNf4,
    PackedSwigluGemmTernary,
    PackedGemmPairSwigluNf4,
    PackedGemmPairSwiglu,
    PackedGemmTernaryScale128,
    PackedGemmNf4Scale128,
    PackedGemmTernarySign,
    PackedGemmTernarySignSel,
    PackedGemmTernaryLut2,
    PackedGemmTernaryLut2Alt,
    PackedGemmTernaryPn4,
    PackedGemmNf4Register,
    PackedGemmNf4Product,
    PackedGemmPairSwigluLut2,
    RepackTernaryLut2,
    AddNorm,
    QkNormRope,
    Argmax,
    ArgmaxBlocks,
    ArgmaxFinal,
    GatherColumns,
}

impl Kernel {
    pub const ALL: &[Self] = &[
        Self::Fill,
        Self::Binary,
        Self::Copy2d,
        Self::Gather,
        Self::Gemv,
        Self::Gemm,
        Self::RmsNorm,
        Self::Rotary,
        Self::Gqa,
        Self::GqaBatch,
        Self::ConvGate,
        Self::Conv,
        Self::ConvStep,
        Self::SwiGlu,
        Self::PackedGemv,
        Self::PackedGather,
        Self::PackedGemvPair,
        Self::PackedSwigluGemv,
        Self::PackedGemvNf4,
        Self::PackedGatherNf4,
        Self::PackedGemvPairNf4,
        Self::PackedSwigluGemvNf4,
        Self::PackedGemvMtNf4,
        Self::PackedGemmNf4,
        Self::PackedGemmTernary,
        Self::PackedGemvPairMtNf4,
        Self::PackedGemmPairNf4,
        Self::PackedGemmPairTernary,
        Self::PackedSwigluGemvMtNf4,
        Self::PackedSwigluGemmNf4,
        Self::PackedSwigluGemmTernary,
        Self::PackedGemmPairSwigluNf4,
        Self::PackedGemmPairSwiglu,
        Self::PackedGemmTernaryScale128,
        Self::PackedGemmNf4Scale128,
        Self::PackedGemmTernarySign,
        Self::PackedGemmTernarySignSel,
        Self::PackedGemmTernaryLut2,
        Self::PackedGemmTernaryLut2Alt,
        Self::PackedGemmTernaryPn4,
        Self::PackedGemmNf4Register,
        Self::PackedGemmNf4Product,
        Self::PackedGemmPairSwigluLut2,
        Self::RepackTernaryLut2,
        Self::AddNorm,
        Self::QkNormRope,
        Self::Argmax,
        Self::ArgmaxBlocks,
        Self::ArgmaxFinal,
        Self::GatherColumns,
    ];

    /// Bitmask over storage binding positions (1..=`storage_bindings`) the
    /// shader declares `read` rather than `read_write`. Dawn validates
    /// binding access in both directions, so the bind group layout must
    /// declare the same positions read-only.
    pub const fn read_only_mask(self) -> u32 {
        match self {
            Self::Gemv => 0b11100, // x, w, w4
            Self::PackedGemv
            | Self::PackedGemvNf4
            | Self::PackedGemvMtNf4
            | Self::PackedGemmNf4
            | Self::PackedGemmTernary
            | Self::PackedGemmTernaryScale128
            | Self::PackedGemmNf4Scale128
            | Self::PackedGemmTernarySign
            | Self::PackedGemmTernarySignSel
            | Self::PackedGemmTernaryLut2
            | Self::PackedGemmTernaryLut2Alt
            | Self::PackedGemmTernaryPn4
            | Self::PackedGemmNf4Register
            | Self::PackedGemmNf4Product => 0b11_1100, // x, codes, scales, x4
            Self::PackedGemvPair
            | Self::PackedGemvPairNf4
            | Self::PackedGemvPairMtNf4
            | Self::PackedGemmPairNf4
            | Self::PackedGemmPairTernary => {
                0b1_1111_1000 // x, a+b, x4
            }
            Self::PackedSwigluGemv
            | Self::PackedSwigluGemvNf4
            | Self::PackedSwigluGemvMtNf4
            | Self::PackedSwigluGemmNf4
            | Self::PackedSwigluGemmTernary
            | Self::PackedGemmPairSwigluNf4
            | Self::PackedGemmPairSwiglu
            | Self::PackedGemmPairSwigluLut2 => {
                0b1111_1100 // x, a+b streams, x4
            }
            Self::RepackTernaryLut2 => 0b10,             // src
            Self::Argmax | Self::ArgmaxBlocks => 0b1100, // src, allow
            Self::ArgmaxFinal => 0b100,                  // partials
            Self::GatherColumns => 0b110,                // src, cols
            _ => 0,
        }
    }

    /// WGSL source with the shared flattened-index helper prepended.
    #[allow(clippy::too_many_lines)] // Registry maps each finite operation to its shader body.
    pub fn source(self, staging: crate::Nf4Staging) -> String {
        let body = match self {
            Self::PackedGemmNf4 => NF4_LINEAR_HEADER,
            Self::PackedGemmTernary => TERNARY_LINEAR_HEADER,
            Self::PackedGemmPairNf4 => NF4_PAIR_HEADER,
            Self::PackedGemmPairTernary => TERNARY_PAIR_HEADER,
            Self::PackedSwigluGemmNf4 => NF4_SWIGLU_HEADER,
            Self::PackedSwigluGemmTernary => TERNARY_SWIGLU_HEADER,
            Self::PackedGemmPairSwigluNf4 => NF4_PAIR_SWIGLU_HEADER,
            Self::PackedGemmPairSwiglu => TERNARY_PAIR_SWIGLU_HEADER,
            Self::PackedGemmTernaryScale128 => TERNARY_SCALE128_HEADER,
            Self::PackedGemmNf4Scale128 => NF4_SCALE128_HEADER,
            // Standalone experiment shaders carry their own bindings.
            Self::PackedGemmTernarySign
            | Self::PackedGemmTernarySignSel
            | Self::PackedGemmTernaryLut2
            | Self::PackedGemmTernaryLut2Alt
            | Self::PackedGemmTernaryPn4
            | Self::PackedGemmNf4Register
            | Self::PackedGemmNf4Product
            | Self::PackedGemmPairSwigluLut2 => "",
            Self::RepackTernaryLut2 => REPACK_LUT2,
            Self::Fill => FILL,
            Self::Binary => BINARY,
            Self::Copy2d => COPY2D,
            Self::Gather => GATHER,
            Self::Gemv => GEMV,
            Self::Gemm => GEMM,
            Self::RmsNorm => RMSNORM,
            Self::Rotary => ROTARY,
            Self::Gqa => GQA,
            Self::GqaBatch => GQA_BATCH,
            Self::ConvGate => CONV_GATE,
            Self::Conv => CONV,
            Self::ConvStep => CONV_STEP,
            Self::SwiGlu => SWIGLU,
            Self::PackedGemv => PACKED_GEMV,
            Self::PackedGather => PACKED_GATHER,
            Self::PackedGemvPair => PACKED_GEMV_PAIR,
            Self::PackedSwigluGemv => PACKED_SWIGLU_GEMV,
            Self::PackedGemvNf4 => PACKED_GEMV_NF4,
            Self::PackedGatherNf4 => PACKED_GATHER_NF4,
            Self::PackedGemvPairNf4 => PACKED_GEMV_PAIR_NF4,
            Self::PackedSwigluGemvNf4 => PACKED_SWIGLU_GEMV_NF4,
            Self::PackedGemvMtNf4 => PACKED_GEMV_MT_NF4,
            Self::PackedGemvPairMtNf4 => PACKED_GEMV_PAIR_MT_NF4,
            Self::PackedSwigluGemvMtNf4 => PACKED_SWIGLU_GEMV_MT_NF4,
            Self::AddNorm => ADD_NORM,
            Self::QkNormRope => QK_NORM_ROPE,
            Self::Argmax => ARGMAX,
            Self::ArgmaxBlocks => ARGMAX_BLOCKS,
            Self::ArgmaxFinal => ARGMAX_FINAL,
            Self::GatherColumns => GATHER_COLUMNS,
        };
        if matches!(
            self,
            Self::PackedGemmNf4
                | Self::PackedGemmTernary
                | Self::PackedGemmPairNf4
                | Self::PackedGemmPairTernary
                | Self::PackedSwigluGemmNf4
                | Self::PackedSwigluGemmTernary
                | Self::PackedGemmPairSwigluNf4
                | Self::PackedGemmPairSwiglu
        ) {
            let lut = if matches!(
                self,
                Self::PackedGemmPairSwiglu
                    | Self::PackedGemmTernary
                    | Self::PackedGemmPairTernary
                    | Self::PackedSwigluGemmTernary
            ) {
                ""
            } else {
                NF4_LUT
            };
            if staging != crate::Nf4Staging::F32 {
                let (stg_x, stg_w) = staging.types();
                let tile = include_str!("shaders/experimental/nf4_prefill_f16.wgsl")
                    .replace("__STGX__", stg_x)
                    .replace("__STGW__", stg_w);
                return ["enable f16;\n", WGSL_INDEX, lut, body, &tile].concat();
            }
            let tile = include_str!("shaders/nf4_prefill.wgsl");
            return [WGSL_INDEX, lut, body, tile].concat();
        }
        // Low-bit arithmetic experiments selected by MINI_LOWBITS_EXPERIMENT.
        // Scale-control variants reuse the control template with an unscaled
        // decode header; the rest are self-contained shaders.
        match self {
            Self::PackedGemmTernaryScale128 | Self::PackedGemmNf4Scale128 => {
                let lut = if matches!(self, Self::PackedGemmNf4Scale128) {
                    NF4_LUT
                } else {
                    ""
                };
                return [
                    WGSL_INDEX,
                    lut,
                    body,
                    include_str!("shaders/experimental/lowbit_scale128.wgsl"),
                ]
                .concat();
            }
            Self::PackedGemmTernarySign | Self::PackedGemmTernarySignSel => {
                let sel = matches!(self, Self::PackedGemmTernarySignSel);
                let tile = include_str!("shaders/experimental/ternary_sign.wgsl")
                    .replace("__SEL__", if sel { "true" } else { "false" });
                return [WGSL_INDEX, &tile].concat();
            }
            Self::PackedGemmTernaryLut2 | Self::PackedGemmTernaryLut2Alt => {
                let (bm, bn) = if matches!(self, Self::PackedGemmTernaryLut2) {
                    ("32", "64")
                } else {
                    ("64", "32")
                };
                let tile = include_str!("shaders/ternary_lut2.wgsl")
                    .replace("__BM__", bm)
                    .replace("__BN__", bn);
                return [WGSL_INDEX, &tile].concat();
            }
            Self::PackedGemmTernaryPn4 => {
                return [
                    WGSL_INDEX,
                    include_str!("shaders/experimental/ternary_pn4.wgsl"),
                ]
                .concat();
            }
            Self::PackedGemmNf4Register => {
                return [
                    WGSL_INDEX,
                    NF4_LUT,
                    include_str!("shaders/experimental/nf4_register.wgsl"),
                ]
                .concat();
            }
            Self::PackedGemmNf4Product => {
                return [
                    WGSL_INDEX,
                    NF4_LUT,
                    include_str!("shaders/experimental/nf4_product.wgsl"),
                ]
                .concat();
            }
            Self::PackedGemmPairSwigluLut2 => {
                return [
                    WGSL_INDEX,
                    include_str!("shaders/ternary_pair_swiglu_lut2.wgsl"),
                ]
                .concat();
            }
            _ => {}
        }
        let mut source = String::with_capacity(WGSL_INDEX.len() + body.len() + 1);
        source.push_str(WGSL_INDEX);
        if matches!(
            self,
            Self::PackedGemvNf4
                | Self::PackedGatherNf4
                | Self::PackedGemvPairNf4
                | Self::PackedSwigluGemvNf4
                | Self::PackedGemvMtNf4
                | Self::PackedGemvPairMtNf4
                | Self::PackedSwigluGemvMtNf4
        ) {
            source.push_str(NF4_LUT);
        }
        source.push_str(body);
        source
    }

    /// Storage-buffer bindings after the uniform params binding.
    pub const fn storage_bindings(self) -> u32 {
        match self {
            Self::Fill => 1,
            Self::ArgmaxFinal | Self::Copy2d | Self::RepackTernaryLut2 => 2,
            Self::Argmax
            | Self::ArgmaxBlocks
            | Self::Binary
            | Self::Gather
            | Self::GatherColumns
            | Self::Gemm
            | Self::RmsNorm
            | Self::Rotary
            | Self::ConvGate
            | Self::SwiGlu => 3,
            Self::Gemv | Self::Conv | Self::PackedGather | Self::PackedGatherNf4 => 4,
            Self::AddNorm
            | Self::ConvStep
            | Self::PackedGemv
            | Self::PackedGemvNf4
            | Self::PackedGemvMtNf4
            | Self::PackedGemmNf4
            | Self::PackedGemmTernary
            | Self::PackedGemmTernaryScale128
            | Self::PackedGemmNf4Scale128
            | Self::PackedGemmTernarySign
            | Self::PackedGemmTernarySignSel
            | Self::PackedGemmTernaryLut2
            | Self::PackedGemmTernaryLut2Alt
            | Self::PackedGemmTernaryPn4
            | Self::PackedGemmNf4Register
            | Self::PackedGemmNf4Product => 5,
            Self::Gqa
            | Self::GqaBatch
            | Self::PackedSwigluGemv
            | Self::PackedSwigluGemvNf4
            | Self::PackedSwigluGemvMtNf4
            | Self::PackedSwigluGemmNf4
            | Self::PackedSwigluGemmTernary
            | Self::QkNormRope
            | Self::PackedGemmPairSwigluNf4
            | Self::PackedGemmPairSwiglu
            | Self::PackedGemmPairSwigluLut2 => 7,
            Self::PackedGemvPair
            | Self::PackedGemvPairNf4
            | Self::PackedGemvPairMtNf4
            | Self::PackedGemmPairNf4
            | Self::PackedGemmPairTernary => 8,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Fill => "fill",
            Self::Binary => "binary",
            Self::Copy2d => "copy2d",
            Self::Gather => "gather",
            Self::Gemv => "gemv",
            Self::Gemm => "gemm",
            Self::RmsNorm => "rmsnorm",
            Self::Rotary => "rotary",
            Self::Gqa => "gqa",
            Self::GqaBatch => "gqa_batch",
            Self::ConvGate => "conv_gate",
            Self::Conv => "conv",
            Self::ConvStep => "conv_step",
            Self::SwiGlu => "swiglu",
            Self::PackedGemv => "packed_gemv",
            Self::PackedGather => "packed_gather",
            Self::PackedGemvPair => "packed_gemv_pair",
            Self::PackedSwigluGemv => "packed_swiglu_gemv",
            Self::PackedGemvNf4 => "packed_gemv_nf4",
            Self::PackedGatherNf4 => "packed_gather_nf4",
            Self::PackedGemvPairNf4 => "packed_gemv_pair_nf4",
            Self::PackedSwigluGemvNf4 => "packed_swiglu_gemv_nf4",
            Self::PackedGemmNf4 => "packed_gemm_nf4",
            Self::PackedGemmTernary => "packed_gemm_ternary",
            Self::PackedGemvMtNf4 => "packed_gemv_mt_nf4",
            Self::PackedGemmPairNf4 => "packed_gemm_pair_nf4",
            Self::PackedGemmPairTernary => "packed_gemm_pair_ternary",
            Self::PackedGemvPairMtNf4 => "packed_gemv_pair_mt_nf4",
            Self::PackedSwigluGemmNf4 => "packed_swiglu_gemm_nf4",
            Self::PackedSwigluGemmTernary => "packed_swiglu_gemm_ternary",
            Self::PackedSwigluGemvMtNf4 => "packed_swiglu_gemv_mt_nf4",
            Self::PackedGemmPairSwigluNf4 => "packed_gemm_pair_swiglu_nf4",
            Self::PackedGemmPairSwiglu => "packed_gemm_pair_swiglu",
            Self::PackedGemmTernaryScale128 => "packed_gemm_ternary_scale128",
            Self::PackedGemmNf4Scale128 => "packed_gemm_nf4_scale128",
            Self::PackedGemmTernarySign => "packed_gemm_ternary_sign",
            Self::PackedGemmTernarySignSel => "packed_gemm_ternary_sign_sel",
            Self::PackedGemmTernaryLut2 => "packed_gemm_ternary_lut2",
            Self::PackedGemmTernaryLut2Alt => "packed_gemm_ternary_lut2_alt",
            Self::PackedGemmTernaryPn4 => "packed_gemm_ternary_pn4",
            Self::PackedGemmNf4Register => "packed_gemm_nf4_register",
            Self::PackedGemmNf4Product => "packed_gemm_nf4_product",
            Self::PackedGemmPairSwigluLut2 => "packed_gemm_pair_swiglu_lut2",
            Self::RepackTernaryLut2 => "repack_ternary_lut2",
            Self::AddNorm => "add_norm",
            Self::QkNormRope => "qk_norm_rope",
            Self::Argmax => "argmax",
            Self::ArgmaxBlocks => "argmax_blocks",
            Self::ArgmaxFinal => "argmax_final",
            Self::GatherColumns => "gather_columns",
        }
    }
}
