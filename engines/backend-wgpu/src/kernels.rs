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
const WGSL_INDEX: &str = r"
fn flat_wg(wid: vec3<u32>, numw: vec3<u32>) -> u32 {
    return wid.z * numw.y * numw.x + wid.y * numw.x + wid.x;
}
";

/// Zero-fill used by `allocate_f32` (device buffers are not zero-initialized).
const FILL: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    dst[i] = bitcast<f32>(pc.p.y);
}
";

/// Elementwise binary op over equal contiguous layouts. `op`: 0 = add, 1 = multiply.
const BINARY: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    let a = lhs[i];
    let b = rhs[i];
    var r: f32;
    if pc.p.y == 0u {
        r = a + b;
    } else {
        r = a * b;
    }
    dst[i] = r;
}
";

/// Checked row-major rectangle copy between distinct packed rank-two buffers.
/// dst[(dr+r)*dw + dc + c] = src[(sr+r)*sw + sc + c] for r < rows, c < cols.
const COPY2D: &str = r"
struct Params { p0: vec4<u32>, p1: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let rows = pc.p0.z;
    let cols = pc.p0.w;
    if i >= rows * cols { return; }
    let r = i / cols;
    let c = i - r * cols;
    let src_index = (pc.p0.x + r) * pc.p1.x + pc.p0.y + c;
    let dst_index = (pc.p1.y + r) * pc.p1.z + pc.p1.w + c;
    dst[dst_index] = src[src_index];
}
";

/// Gather selected rows from a packed [rows, columns] table.
/// dst[r*cols + c] = table[ids[r]*cols + c]
const GATHER: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

// ids are f32 so a device-side argmax output feeds this kernel directly. A
// non-finite, fractional, or out-of-range selector poisons its output row
// with NaN, which downstream finiteness checks surface as a failure.
fn row_id(r: u32) -> u32 {
    let idf = ids[r];
    if idf < 0.0 || idf >= 16777216.0 || fract(idf) != 0.0 {
        return 0xFFFFFFFFu;
    }
    return u32(idf);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let r = i / pc.p.y;
    let c = i - r * pc.p.y;
    let id = row_id(r);
    if id >= pc.p.z {
        // NaN arrives through the params uniform: Dawn rejects non-finite
        // bitcast constants during shader const-eval.
        dst[i] = bitcast<f32>(pc.p.w);
        return;
    }
    dst[i] = table[id * pc.p.y + c];
}
";

/// Gather selected columns from a contiguous [rows, width] f32 input:
/// dst[r*count + k] = src[r*width + cols[k]]. `cols` carries staged host u32
/// selectors, already range-checked against `width` on the host; caller
/// order and duplicates are preserved.
const GATHER_COLUMNS: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let r = i / pc.p.y;
    let k = i - r * pc.p.y;
    dst[i] = src[r * pc.p.z + cols[k]];
}
";

/// GEMV for the m == 1 decode path: dst[j] = `sum_l` x[l] * w[j,l].
/// One workgroup per output column; 256 threads tree-reduce over k in shared
/// memory. `w4` binds the same weight buffer as a vec4 view: a weight row that
/// starts 16-byte aligned (j*k % 4 == 0) streams one 128-bit load per four
/// elements; unaligned rows and the k % 4 tail use the scalar path.
const GEMV: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> w4: array<vec4<f32>>;

// Groups of `G` lanes each reduce one output row; the workgroup covers
// `256/G` rows so narrow matrices still launch enough workgroups to fill the
// GPU. Each lane strides the row's vec4 words so weight loads stay coalesced,
// then one barrier hands each row's G partials to a single thread for a short
// serial reduce. Replaces the per-output workgroup and its barrier tree.
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let rows_per_wg = 256u / lanes;
    let row0 = flat_wg(wid, numw) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let k4 = k >> 2u;
    let kbulk = k4 << 2u;
    var acc = 0.0;
    if (j < n) {
        let rbase = j * k;
        if (rbase & 3u) == 0u {
            let base4 = rbase >> 2u;
            for (var g = lane; g < k4; g = g + lanes) {
                let rv = w4[base4 + g];
                let l = g << 2u;
                acc = acc + rv.x * x[l];
                acc = acc + rv.y * x[l + 1u];
                acc = acc + rv.z * x[l + 2u];
                acc = acc + rv.w * x[l + 3u];
            }
            for (var l = kbulk + lane; l < k; l = l + lanes) {
                acc = acc + x[l] * w[rbase + l];
            }
        } else {
            for (var l = lane; l < k; l = l + lanes) {
                acc = acc + x[l] * w[rbase + l];
            }
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[o] = s;
        }
    }
}
";

/// Packed ternary GEMV for `minifield.ternary.v1` weights:
/// dst[i*n + j] = `sum_l` x[i*k + l] * (code(j,l) - 1) * scale(j, l/128).
/// One workgroup per output element (the flat grid covers every m, m == 1
/// being the decode case); 256 threads tree-reduce over k in shared memory.
/// Codes bind as u32 words over the byte stream: weight l lives in byte l/4
/// of the row, i.e. bits [8*((l/4)%4) + 2*(l%4)] of word l/16.
const PACKED_GEMV: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

// Groups of `G` lanes each reduce one output row; the workgroup covers
// `256/G` rows so narrow matrices still launch enough workgroups to fill the
// GPU. Each lane strides the row's u32 code words (16 weights per word), so
// code loads are coalesced and each lane's activation reads form a contiguous
// 64-byte chunk. The word's scale applies to its 16-weight dot once, matching
// the CPU group-then-scale accumulation. One barrier, then each row's G
// partials reduce serially on a single thread. Replaces the per-output
// workgroup and its barrier tree.
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 15u) >> 4u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 3u)];
            let base = w << 4u;
            var dot = 0.0;
            if (vec_ok && base + 16u <= k) {
                // Whole word: four 128-bit activation loads cover 16 elements.
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 3u;
                    dot = dot
                        + xv.x * f32(i32((word >> sh) & 3u) - 1)
                        + xv.y * f32(i32((word >> (sh + 2u)) & 3u) - 1)
                        + xv.z * f32(i32((word >> (sh + 4u)) & 3u) - 1)
                        + xv.w * f32(i32((word >> (sh + 6u)) & 3u) - 1);
                }
            } else {
                let count = k - base;
                for (var e = 0u; e < count; e = e + 1u) {
                    let code = (word >> (e * 2u)) & 3u;
                    dot = dot + x[xbase + base + e] * f32(i32(code) - 1);
                }
            }
            acc = acc + dot * scale;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[i * n + o] = s;
        }
    }
}
";

/// Packed ternary gather: dst[r*k + l] = (code(ids[r], l) - 1) *
/// scale(ids[r], l/128). One thread per output element; the code decode is
/// the same byte/bit scheme as `PACKED_GEMV`.
const PACKED_GATHER: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> scales: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let k = pc.p.y;
    let r = i / k;
    let l = i - r * k;
    let srcf = ids[r];
    var src = 0xFFFFFFFFu;
    if srcf >= 0.0 && srcf < 16777216.0 && fract(srcf) == 0.0 {
        src = u32(srcf);
    }
    if src >= pc.p.z {
        dst[i] = bitcast<f32>(pc.p.w);
        return;
    }
    let word = codes[src * (k >> 4u) + (l >> 4u)];
    let byte = (word >> (((l >> 2u) & 3u) << 3u)) & 0xFFu;
    let code = (byte >> ((l & 3u) << 1u)) & 3u;
    dst[i] = f32(i32(code) - 1) * scales[src * (k >> 7u) + (l >> 7u)];
}
";

/// NF4 codebook for `minifield.nf4.v1` streams: decode
/// `w = NF4[code] * scale` where `code` indexes this sorted 16-entry
/// normal-float table (bitsandbytes-compatible levels for zero-mean data).
/// Each u32 codes word carries eight 4-bit indices, low nibble first, so a
/// word covers 8 weights and a 128-weight group spans 16 words.
const NF4_LUT: &str = "
const NF4: array<f32, 16> = array<f32, 16>(
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
);
";

/// Packed NF4 GEMV for `minifield.nf4.v1` weights:
/// dst[i*n + j] = `sum_l` x[i*k + l] * NF4[code(j,l)] * scale(j, l/128).
/// Same lane-grouped reduce layout as `PACKED_GEMV`; each u32 word decodes
/// eight 4-bit NF4 level indices instead of sixteen 2-bit ternary codes.
const PACKED_GEMV_NF4: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 4u)];
            let base = w << 3u;
            var dot = 0.0;
            if (vec_ok && base + 8u <= k) {
                // Whole word: two 128-bit activation loads cover 8 elements.
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 2u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 4u;
                    dot = dot
                        + xv.x * NF4[(word >> sh) & 0xFu]
                        + xv.y * NF4[(word >> (sh + 4u)) & 0xFu]
                        + xv.z * NF4[(word >> (sh + 8u)) & 0xFu]
                        + xv.w * NF4[(word >> (sh + 12u)) & 0xFu];
                }
            } else {
                let count = min(8u, k - base);
                for (var e = 0u; e < count; e = e + 1u) {
                    let nib = (word >> (e * 4u)) & 0xFu;
                    dot = dot + x[xbase + base + e] * NF4[nib];
                }
            }
            acc = acc + dot * scale;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[i * n + o] = s;
        }
    }
}
";

/// Multi-token packed NF4 GEMV: same lane-grouped reduce as
/// `PACKED_GEMV_NF4` but each workgroup covers MT input rows, so every codes
/// word is decoded once and reused for all MT dots. Use for prefill-width
/// inputs where per-row re-decode dominates; m=1 keeps the single-token
/// kernel. `pc.q` = [m, `m_tiles`, `vec_ok`, 0].
const PACKED_GEMV_MT_NF4: &str = r"
const MT: u32 = 8u;
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let m = pc.q.x;
    let rows_per_wg = 256u / lanes;
    let i0 = (flat / tiles) * MT;
    let row0 = (flat - (flat / tiles) * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let vec_ok = pc.q.z != 0u;
    var acc: array<f32, 8>;
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        acc[ml] = 0.0;
    }
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 4u)];
            let base = w << 3u;
            var dw: array<f32, 8>;
            for (var e = 0u; e < 8u; e = e + 1u) {
                dw[e] = NF4[(word >> (e * 4u)) & 0xFu];
            }
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    let xbase = i * k + base;
                    var dot = 0.0;
                    if (vec_ok && base + 8u <= k) {
                        let b4 = xbase >> 2u;
                        let xa = x4[b4];
                        let xb = x4[b4 + 1u];
                        dot = xa.x * dw[0] + xa.y * dw[1] + xa.z * dw[2] + xa.w * dw[3]
                            + xb.x * dw[4] + xb.y * dw[5] + xb.z * dw[6] + xb.w * dw[7];
                    } else {
                        let count = min(8u, k - base);
                        for (var e = 0u; e < count; e = e + 1u) {
                            dot = dot + x[xbase + e] * dw[e];
                        }
                    }
                    acc[ml] = acc[ml] + dot * scale;
                }
            }
        }
    }
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        part[ml * 256u + tid] = acc[ml];
    }
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            let pbase = tid * lanes;
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    var s = 0.0;
                    for (var l = 0u; l < lanes; l = l + 1u) {
                        s = s + part[ml * 256u + pbase + l];
                    }
                    dst[i * n + o] = s;
                }
            }
        }
    }
}
";

/// Packed NF4 gather: dst[r*k + l] = NF4[code(ids[r], l)] *
/// scale(ids[r], l/128). One thread per output element.
const PACKED_GATHER_NF4: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> scales: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let k = pc.p.y;
    let r = i / k;
    let l = i - r * k;
    let srcf = ids[r];
    var src = 0xFFFFFFFFu;
    if srcf >= 0.0 && srcf < 16777216.0 && fract(srcf) == 0.0 {
        src = u32(srcf);
    }
    if src >= pc.p.z {
        dst[i] = bitcast<f32>(pc.p.w);
        return;
    }
    let word = codes[src * (k >> 3u) + (l >> 3u)];
    let nib = (word >> ((l & 7u) << 2u)) & 0xFu;
    dst[i] = NF4[nib] * scales[src * (k >> 7u) + (l >> 7u)];
}
";

/// Row-wise argmax over `[T, V]` f32 logits: one workgroup per row, 256
/// threads tree-reduce (value, index) pairs keeping the smallest index on
/// ties. `dst[r]` is the winning index as an exact f32 integer, or NaN when
/// the row contains any non-finite element.
const ARGMAX: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> allow: array<u32>;

var<workgroup> sh_v: array<f32, 256>;
var<workgroup> sh_i: array<u32, 256>;
var<workgroup> sh_b: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let cols = pc.p.y;
    if row >= pc.p.x { return; }
    let tid = lid.x;
    let base = row * cols;
    // Dawn rejects -inf/NaN bitcast constants; lowest f32 plus a sentinel
    // index preserves first-strict-max semantics, and NaN rides pc.p.w.
    // `allow` gates candidacy: masked-out elements are skipped, so their
    // non-finite values cannot poison the row.
    var best = -3.4028234663852886e38;
    var idx = 0xFFFFFFFFu;
    var bad = 0u;
    let use_mask = pc.p.z != 0u;
    for (var c = tid; c < cols; c = c + 256u) {
        if use_mask && (allow[c >> 5u] & (1u << (c & 31u))) == 0u { continue; }
        let v = src[base + c];
        if v == v && abs(v) <= 3.4028234663852886e38 {
            if v > best || (v == best && c < idx) {
                best = v;
                idx = c;
            }
        } else {
            bad = 1u;
        }
    }
    sh_v[tid] = best;
    sh_i[tid] = idx;
    sh_b[tid] = bad;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            let ov = sh_v[tid + s];
            let oi = sh_i[tid + s];
            if ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid]) {
                sh_v[tid] = ov;
                sh_i[tid] = oi;
            }
            sh_b[tid] = sh_b[tid] | sh_b[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        if sh_b[0] != 0u || sh_i[0] == 0xFFFFFFFFu {
            dst[row] = bitcast<f32>(pc.p.w);
        } else {
            dst[row] = f32(sh_i[0]);
        }
    }
}
";

/// Argmax stage 1 for wide rows: each workgroup reduces one 2048-element
/// block of a row to a `(value, first-index)` partial in `partials`. A
/// non-finite element in the block writes NaN, which stage 2 propagates. The
/// single-workgroup `ARGMAX` path is kept for narrow rows.
const ARGMAX_BLOCKS: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> allow: array<u32>;

var<workgroup> sh_v: array<f32, 256>;
var<workgroup> sh_i: array<u32, 256>;
var<workgroup> sh_b: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let cols = pc.p.y;
    let blocks = pc.p.z;
    let row = flat / blocks;
    let block = flat - row * blocks;
    if row >= pc.p.x { return; }
    let tid = lid.x;
    let row_base = row * cols;
    let start = block * 2048u;
    let stop = min(start + 2048u, cols);
    // `allow` gates candidacy per element; a fully masked block emits the
    // f32-safe index sentinel (2^24) that stage 2 never selects.
    var best = -3.4028234663852886e38;
    var idx = 0xFFFFFFFFu;
    var bad = 0u;
    let use_mask = pc.q.x != 0u;
    for (var c = start + tid; c < stop; c = c + 256u) {
        if use_mask && (allow[c >> 5u] & (1u << (c & 31u))) == 0u { continue; }
        let v = src[row_base + c];
        if v == v && abs(v) <= 3.4028234663852886e38 {
            if v > best || (v == best && c < idx) {
                best = v;
                idx = c;
            }
        } else {
            bad = 1u;
        }
    }
    sh_v[tid] = best;
    sh_i[tid] = idx;
    sh_b[tid] = bad;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            let ov = sh_v[tid + s];
            let oi = sh_i[tid + s];
            if ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid]) {
                sh_v[tid] = ov;
                sh_i[tid] = oi;
            }
            sh_b[tid] = sh_b[tid] | sh_b[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        if sh_b[0] != 0u {
            partials[flat] = vec2<f32>(bitcast<f32>(pc.p.w), 0.0);
        } else {
            partials[flat] = vec2<f32>(sh_v[0], min(f32(sh_i[0]), 16777216.0));
        }
    }
}
";

/// Argmax stage 2: one workgroup per row reduces the stage-1 partials to the
/// row's first strict maximum index, or NaN when any partial is NaN or every
/// block reported the 2^24 empty-candidate sentinel.
const ARGMAX_FINAL: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> partials: array<vec2<f32>>;

var<workgroup> sh_v: array<f32, 256>;
var<workgroup> sh_i: array<u32, 256>;
var<workgroup> sh_b: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let blocks = pc.p.z;
    if row >= pc.p.x { return; }
    let tid = lid.x;
    let base = row * blocks;
    var best = -3.4028234663852886e38;
    var idx = 0xFFFFFFFFu;
    var bad = 0u;
    for (var b = tid; b < blocks; b = b + 256u) {
        let p = partials[base + b];
        if p.x != p.x {
            bad = 1u;
        } else if p.x > best || (p.x == best && u32(p.y) < idx) {
            best = p.x;
            idx = u32(p.y);
        }
    }
    sh_v[tid] = best;
    sh_i[tid] = idx;
    sh_b[tid] = bad;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            let ov = sh_v[tid + s];
            let oi = sh_i[tid + s];
            if ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid]) {
                sh_v[tid] = ov;
                sh_i[tid] = oi;
            }
            sh_b[tid] = sh_b[tid] | sh_b[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        if sh_b[0] != 0u || sh_i[0] >= 16777216u {
            dst[row] = bitcast<f32>(pc.p.w);
        } else {
            dst[row] = f32(sh_i[0]);
        }
    }
}
";

/// Paired packed ternary GEMV over one shared input: workgroup (i, j) computes
/// `dst_a[i,j]` and `dst_b[i,j]` with independent accumulators over the same
/// x row. Weight decode matches `PACKED_GEMV`; outputs share one `[m, n]`
/// shape.
const PACKED_GEMV_PAIR: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;

// Same lane-grouped layout as `PACKED_GEMV`: each group of `G` lanes reduces
// one output row for both weight sets, so the shared x stream is read once
// per element for both dots. `part` holds the A partials in [0, 256) and the
// B partials in [256, 512); one barrier, then each row's thread reduces both.
var<workgroup> part: array<f32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 15u) >> 4u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc_a = 0.0;
    var acc_b = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word_a = codes_a[cbase + w];
            let word_b = codes_b[cbase + w];
            let scale_a = scales_a[sbase + (w >> 3u)];
            let scale_b = scales_b[sbase + (w >> 3u)];
            let base = w << 4u;
            var dot_a = 0.0;
            var dot_b = 0.0;
            if (vec_ok && base + 16u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 3u;
                    let ca0 = f32(i32((word_a >> sh) & 3u) - 1);
                    let ca1 = f32(i32((word_a >> (sh + 2u)) & 3u) - 1);
                    let ca2 = f32(i32((word_a >> (sh + 4u)) & 3u) - 1);
                    let ca3 = f32(i32((word_a >> (sh + 6u)) & 3u) - 1);
                    let cb0 = f32(i32((word_b >> sh) & 3u) - 1);
                    let cb1 = f32(i32((word_b >> (sh + 2u)) & 3u) - 1);
                    let cb2 = f32(i32((word_b >> (sh + 4u)) & 3u) - 1);
                    let cb3 = f32(i32((word_b >> (sh + 6u)) & 3u) - 1);
                    dot_a = dot_a + xv.x * ca0 + xv.y * ca1 + xv.z * ca2 + xv.w * ca3;
                    dot_b = dot_b + xv.x * cb0 + xv.y * cb1 + xv.z * cb2 + xv.w * cb3;
                }
            } else {
                let count = k - base;
                for (var e = 0u; e < count; e = e + 1u) {
                    let xv = x[xbase + base + e];
                    let code_a = (word_a >> (e * 2u)) & 3u;
                    let code_b = (word_b >> (e * 2u)) & 3u;
                    dot_a = dot_a + xv * f32(i32(code_a) - 1);
                    dot_b = dot_b + xv * f32(i32(code_b) - 1);
                }
            }
            acc_a = acc_a + dot_a * scale_a;
            acc_b = acc_b + dot_b * scale_b;
        }
    }
    part[tid] = acc_a;
    part[256u + tid] = acc_b;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s_a = 0.0;
            var s_b = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s_a = s_a + part[pbase + l];
                s_b = s_b + part[256u + pbase + l];
            }
            dst_a[i * n + o] = s_a;
            dst_b[i * n + o] = s_b;
        }
    }
}
";

/// Packed ternary GEMV over an on-the-fly SiLU(gate) * up activation:
/// dst[i*n + j] = `sum_l` (silu(gate[i,l]) * up[i,l]) * (code(j,l) - 1) *
/// scale(j, l/128). Same lane-grouped layout as `PACKED_GEMV`; the activation
/// is computed inline from coalesced gate/up loads so no shared staging or
/// extra barrier is needed.
const PACKED_SWIGLU_GEMV: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 15u) >> 4u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 3u)];
            let base = w << 4u;
            var dot = 0.0;
            if (vec_ok && base + 16u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let g = gate4[b4 + q];
                    let sigmoid = vec4(1.0) / (vec4(1.0) + exp(-g));
                    let xv = (g * sigmoid) * up4[b4 + q];
                    let sh = q << 3u;
                    dot = dot
                        + xv.x * f32(i32((word >> sh) & 3u) - 1)
                        + xv.y * f32(i32((word >> (sh + 2u)) & 3u) - 1)
                        + xv.z * f32(i32((word >> (sh + 4u)) & 3u) - 1)
                        + xv.w * f32(i32((word >> (sh + 6u)) & 3u) - 1);
                }
            } else {
                let count = k - base;
                for (var e = 0u; e < count; e = e + 1u) {
                    let l = xbase + base + e;
                    let g = gate[l];
                    let sigmoid = 1.0 / (1.0 + exp(-g));
                    let xv = (g * sigmoid) * up[l];
                    let code = (word >> (e * 2u)) & 3u;
                    dot = dot + xv * f32(i32(code) - 1);
                }
            }
            acc = acc + dot * scale;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[i * n + o] = s;
        }
    }
}
";

/// Paired packed NF4 GEMV over one shared input: same lane-grouped layout as
/// `PACKED_GEMV_PAIR` with the `minifield.nf4.v1` nibble decode.
const PACKED_GEMV_PAIR_NF4: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;

var<workgroup> part: array<f32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc_a = 0.0;
    var acc_b = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word_a = codes_a[cbase + w];
            let word_b = codes_b[cbase + w];
            let scale_a = scales_a[sbase + (w >> 4u)];
            let scale_b = scales_b[sbase + (w >> 4u)];
            let base = w << 3u;
            var dot_a = 0.0;
            var dot_b = 0.0;
            if (vec_ok && base + 8u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 2u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 4u;
                    let ca0 = NF4[(word_a >> sh) & 0xFu];
                    let ca1 = NF4[(word_a >> (sh + 4u)) & 0xFu];
                    let ca2 = NF4[(word_a >> (sh + 8u)) & 0xFu];
                    let ca3 = NF4[(word_a >> (sh + 12u)) & 0xFu];
                    let cb0 = NF4[(word_b >> sh) & 0xFu];
                    let cb1 = NF4[(word_b >> (sh + 4u)) & 0xFu];
                    let cb2 = NF4[(word_b >> (sh + 8u)) & 0xFu];
                    let cb3 = NF4[(word_b >> (sh + 12u)) & 0xFu];
                    dot_a = dot_a + xv.x * ca0 + xv.y * ca1 + xv.z * ca2 + xv.w * ca3;
                    dot_b = dot_b + xv.x * cb0 + xv.y * cb1 + xv.z * cb2 + xv.w * cb3;
                }
            } else {
                let count = min(8u, k - base);
                for (var e = 0u; e < count; e = e + 1u) {
                    let xv = x[xbase + base + e];
                    let nib_a = (word_a >> (e * 4u)) & 0xFu;
                    let nib_b = (word_b >> (e * 4u)) & 0xFu;
                    dot_a = dot_a + xv * NF4[nib_a];
                    dot_b = dot_b + xv * NF4[nib_b];
                }
            }
            acc_a = acc_a + dot_a * scale_a;
            acc_b = acc_b + dot_b * scale_b;
        }
    }
    part[tid] = acc_a;
    part[256u + tid] = acc_b;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s_a = 0.0;
            var s_b = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s_a = s_a + part[pbase + l];
                s_b = s_b + part[256u + pbase + l];
            }
            dst_a[i * n + o] = s_a;
            dst_b[i * n + o] = s_b;
        }
    }
}
";

/// Packed NF4 GEMV over an on-the-fly SiLU(gate) * up activation: same
/// lane-grouped layout as `PACKED_SWIGLU_GEMV` with the `minifield.nf4.v1`
/// nibble decode.
const PACKED_SWIGLU_GEMV_NF4: &str = r"
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 4u)];
            let base = w << 3u;
            var dot = 0.0;
            if (vec_ok && base + 8u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 2u; q = q + 1u) {
                    let g = gate4[b4 + q];
                    let sigmoid = vec4(1.0) / (vec4(1.0) + exp(-g));
                    let xv = (g * sigmoid) * up4[b4 + q];
                    let sh = q << 4u;
                    dot = dot
                        + xv.x * NF4[(word >> sh) & 0xFu]
                        + xv.y * NF4[(word >> (sh + 4u)) & 0xFu]
                        + xv.z * NF4[(word >> (sh + 8u)) & 0xFu]
                        + xv.w * NF4[(word >> (sh + 12u)) & 0xFu];
                }
            } else {
                let count = min(8u, k - base);
                for (var e = 0u; e < count; e = e + 1u) {
                    let l = xbase + base + e;
                    let g = gate[l];
                    let sigmoid = 1.0 / (1.0 + exp(-g));
                    let xv = (g * sigmoid) * up[l];
                    let nib = (word >> (e * 4u)) & 0xFu;
                    dot = dot + xv * NF4[nib];
                }
            }
            acc = acc + dot * scale;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[i * n + o] = s;
        }
    }
}
";

/// Multi-token packed NF4 GEMV over two shared-input weights: decode each
/// codes pair once, reuse across MT input rows. `pc.q` = [m, `m_tiles`, `vec_ok`,
/// 0]. Partial buffers hold MT accumulators per thread.
const PACKED_GEMV_PAIR_MT_NF4: &str = r"
const MT: u32 = 8u;
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;

var<workgroup> part_a: array<f32, 2048>;
var<workgroup> part_b: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let m = pc.q.x;
    let rows_per_wg = 256u / lanes;
    let i0 = (flat / tiles) * MT;
    let row0 = (flat - (flat / tiles) * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let vec_ok = pc.q.z != 0u;
    var acc_a: array<f32, 8>;
    var acc_b: array<f32, 8>;
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        acc_a[ml] = 0.0;
        acc_b[ml] = 0.0;
    }
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        for (var w = lane; w < words; w = w + lanes) {
            let word_a = codes_a[cbase + w];
            let word_b = codes_b[cbase + w];
            let scale_a = scales_a[sbase + (w >> 4u)];
            let scale_b = scales_b[sbase + (w >> 4u)];
            let base = w << 3u;
            var dw_a: array<f32, 8>;
            var dw_b: array<f32, 8>;
            for (var e = 0u; e < 8u; e = e + 1u) {
                let sh = e * 4u;
                dw_a[e] = NF4[(word_a >> sh) & 0xFu];
                dw_b[e] = NF4[(word_b >> sh) & 0xFu];
            }
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    let xbase = i * k + base;
                    var dot_a = 0.0;
                    var dot_b = 0.0;
                    if (vec_ok && base + 8u <= k) {
                        let b4 = xbase >> 2u;
                        let xa = x4[b4];
                        let xb = x4[b4 + 1u];
                        dot_a = xa.x * dw_a[0] + xa.y * dw_a[1] + xa.z * dw_a[2] + xa.w * dw_a[3]
                              + xb.x * dw_a[4] + xb.y * dw_a[5] + xb.z * dw_a[6] + xb.w * dw_a[7];
                        dot_b = xa.x * dw_b[0] + xa.y * dw_b[1] + xa.z * dw_b[2] + xa.w * dw_b[3]
                              + xb.x * dw_b[4] + xb.y * dw_b[5] + xb.z * dw_b[6] + xb.w * dw_b[7];
                    } else {
                        let count = min(8u, k - base);
                        for (var e = 0u; e < count; e = e + 1u) {
                            let xv = x[xbase + e];
                            dot_a = dot_a + xv * dw_a[e];
                            dot_b = dot_b + xv * dw_b[e];
                        }
                    }
                    acc_a[ml] = acc_a[ml] + dot_a * scale_a;
                    acc_b[ml] = acc_b[ml] + dot_b * scale_b;
                }
            }
        }
    }
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        part_a[ml * 256u + tid] = acc_a[ml];
        part_b[ml * 256u + tid] = acc_b[ml];
    }
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            let pbase = tid * lanes;
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    var s_a = 0.0;
                    var s_b = 0.0;
                    for (var l = 0u; l < lanes; l = l + 1u) {
                        s_a = s_a + part_a[ml * 256u + pbase + l];
                        s_b = s_b + part_b[ml * 256u + pbase + l];
                    }
                    dst_a[i * n + o] = s_a;
                    dst_b[i * n + o] = s_b;
                }
            }
        }
    }
}
";

/// Multi-token packed NF4 GEMV over an on-the-fly `SiLU(gate) * up`
/// activation: decode once per word, reuse across MT input rows. `pc.q` =
/// [m, `m_tiles`, `vec_ok`, 0].
const PACKED_SWIGLU_GEMV_MT_NF4: &str = r"
const MT: u32 = 8u;
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;

var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let tiles = pc.p.w;
    let m = pc.q.x;
    let rows_per_wg = 256u / lanes;
    let i0 = (flat / tiles) * MT;
    let row0 = (flat - (flat / tiles) * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let vec_ok = pc.q.z != 0u;
    var acc: array<f32, 8>;
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        acc[ml] = 0.0;
    }
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 4u)];
            let base = w << 3u;
            var dw: array<f32, 8>;
            for (var e = 0u; e < 8u; e = e + 1u) {
                dw[e] = NF4[(word >> (e * 4u)) & 0xFu];
            }
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    let abase = i * k + base;
                    var dot = 0.0;
                    if (vec_ok && base + 8u <= k) {
                        let b4 = abase >> 2u;
                        let ga = gate4[b4];
                        let gb = gate4[b4 + 1u];
                        let sa = vec4(1.0) / (vec4(1.0) + exp(-ga));
                        let sb = vec4(1.0) / (vec4(1.0) + exp(-gb));
                        let xa = (ga * sa) * up4[b4];
                        let xb = (gb * sb) * up4[b4 + 1u];
                        dot = xa.x * dw[0] + xa.y * dw[1] + xa.z * dw[2] + xa.w * dw[3]
                            + xb.x * dw[4] + xb.y * dw[5] + xb.z * dw[6] + xb.w * dw[7];
                    } else {
                        let count = min(8u, k - base);
                        for (var e = 0u; e < count; e = e + 1u) {
                            let l = abase + e;
                            let g = gate[l];
                            let sigmoid = 1.0 / (1.0 + exp(-g));
                            let xv = (g * sigmoid) * up[l];
                            dot = dot + xv * dw[e];
                        }
                    }
                    acc[ml] = acc[ml] + dot * scale;
                }
            }
        }
    }
    for (var ml = 0u; ml < MT; ml = ml + 1u) {
        part[ml * 256u + tid] = acc[ml];
    }
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            let pbase = tid * lanes;
            for (var ml = 0u; ml < MT; ml = ml + 1u) {
                let i = i0 + ml;
                if (i < m) {
                    var s = 0.0;
                    for (var l = 0u; l < lanes; l = l + 1u) {
                        s = s + part[ml * 256u + pbase + l];
                    }
                    dst[i * n + o] = s;
                }
            }
        }
    }
}
";

/// Fused residual add plus row RMS norm: `sum[r,c] = a[r,c] + b[r,c]` then
/// `normed[r,c] = sum[r,c] * rsqrt(mean(sum[r,:]^2) + eps) * alpha[c]`.
/// One workgroup per row; the two passes share one dispatch.
const ADD_NORM: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> sum: array<f32>;
@group(0) @binding(2) var<storage, read_write> normed: array<f32>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> alpha: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let tid = lid.x;
    let ncols = pc.p.x;
    let nrows = pc.p.y;
    if row >= nrows { return; }
    let eps = bitcast<f32>(pc.p.z);
    let base = row * ncols;

    var acc = 0.0;
    for (var c = tid; c < ncols; c = c + 256u) {
        let v = a[base + c] + b[base + c];
        sum[base + c] = v;
        acc = acc + v * v;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let mean = sh[0] / f32(ncols);
    let scale = inverseSqrt(mean + eps);
    workgroupBarrier();

    for (var c = tid; c < ncols; c = c + 256u) {
        normed[base + c] = scale * sum[base + c] * alpha[c];
    }
}
";

/// Fused per-head RMS norm plus split-half rotary for query and key rows.
/// Workgroup `w` handles one [token, head] slice: `w / heads_total` is the
/// token, heads `0..q_heads` map to query, the rest to key. The normed head
/// is staged in shared memory so both rotation halves read final values.
/// `table` is the same host cos/sin layout `ROTARY` uses; `head_dim <= 512`.
const QK_NORM_ROPE: &str = r"
struct Params { p0: vec4<u32>, p1: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> k: array<f32>;
@group(0) @binding(6) var<storage, read_write> qw: array<f32>;
@group(0) @binding(7) var<storage, read_write> kw: array<f32>;

var<workgroup> sh: array<f32, 256>;
var<workgroup> nd: array<f32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = flat_wg(wid, numw);
    let tid = lid.x;
    let q_heads = pc.p0.x;
    let heads_total = pc.p0.y;
    let head_dim = pc.p0.z;
    let half = head_dim >> 1u;
    let q_width = pc.p0.w;
    let kv_width = pc.p1.x;
    let tokens = pc.p1.y;
    let eps = bitcast<f32>(pc.p1.z);
    if w >= tokens * heads_total { return; }
    let t = w / heads_total;
    let h = w - t * heads_total;
    let is_q = h < q_heads;
    let head = select(h - q_heads, h, is_q);
    let width = select(kv_width, q_width, is_q);
    let base = t * width + head * head_dim;

    var acc = 0.0;
    for (var d = tid; d < head_dim; d = d + 256u) {
        let v = select(k[base + d], q[base + d], is_q);
        acc = acc + v * v;
        nd[d] = v;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(sh[0] / f32(head_dim) + eps);
    workgroupBarrier();
    for (var d = tid; d < head_dim; d = d + 256u) {
        let wv = select(kw[d], qw[d], is_q);
        nd[d] = nd[d] * scale * wv;
    }
    workgroupBarrier();

    let table_len = tokens * half;
    for (var c = tid; c < half; c = c + 256u) {
        let cs = t * half + c;
        let co = table[cs];
        let si = table[table_len + cs];
        let a = nd[c];
        let bv = nd[half + c];
        let i1 = base + c;
        let i2 = base + half + c;
        let r1 = a * co - bv * si;
        let r2 = bv * co + a * si;
        if is_q {
            q_out[i1] = r1;
            q_out[i2] = r2;
        } else {
            k_out[i1] = r1;
            k_out[i2] = r2;
        }
    }
}
";

/// Tiled shared-memory GEMM for m > 1: dst[i,j] = `sum_l` x[i,l] * w[j,l].
/// Tiles are 16x16; the weight tile is transposed on load because W is packed
/// [n, k] while the tile needs [k, n]. Grid: (ceil(n/16), ceil(m/16) split over
/// y/z); each thread produces one output element.
const GEMM: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;

const TILE: u32 = 16u;
var<workgroup> lt: array<array<f32, 16>, 16>;
var<workgroup> rt: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = pc.p.x;
    let n = pc.p.y;
    let k = pc.p.z;
    let row_tile = wid.z * numw.y + wid.y;
    let row = row_tile * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let ly = lid.y;
    let lx = lid.x;

    var acc = 0.0;
    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let l_x = t * TILE + lx;
        let l_w = t * TILE + ly;
        if row < m && l_x < k {
            lt[ly][lx] = x[row * k + l_x];
        } else {
            lt[ly][lx] = 0.0;
        }
        if col < n && l_w < k {
            rt[ly][lx] = w[col * k + l_w];
        } else {
            rt[ly][lx] = 0.0;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + lt[ly][kk] * rt[kk][lx];
        }
        workgroupBarrier();
    }

    if row < m && col < n {
        dst[row * n + col] = acc;
    }
}
";

/// Row RMS norm: dst[r,c] = src[r,c] * rsqrt(mean(src[r,:]^2) + eps) * alpha[c].
/// One workgroup per row; also serves per-head norms by treating each
/// [token, head] slice as a row.
const RMSNORM: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> alpha: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let tid = lid.x;
    let ncols = pc.p.x;
    let nrows = pc.p.y;
    if row >= nrows { return; }
    let eps = bitcast<f32>(pc.p.z);
    let base = row * ncols;

    var acc = 0.0;
    for (var c = tid; c < ncols; c = c + 256u) {
        let v = src[base + c];
        acc = acc + v * v;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let mean = sh[0] / f32(ncols);
    let scale = inverseSqrt(mean + eps);
    workgroupBarrier();

    for (var c = tid; c < ncols; c = c + 256u) {
        dst[base + c] = scale * src[base + c] * alpha[c];
    }
}
";

/// Split-half `RoPE` over packed `[tokens, heads * head_dim]` rows. Host-computed
/// cos/sin tables (f64 evaluation cast to f32, matching the scalar reference)
/// are bound as two contiguous regions of one table buffer.
/// dst[i1] = a*cos - b*sin ; dst[i2] = b*cos + a*sin
const ROTARY: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let heads = pc.p.x;
    let half = pc.p.y;
    let tokens = pc.p.z;
    let span = heads * half;
    if i >= tokens * span { return; }
    let t = i / span;
    let rem = i - t * span;
    let h = rem / half;
    let c = rem - h * half;
    let head_dim = 2u * half;
    let packed = heads * head_dim;
    let i1 = t * packed + h * head_dim + c;
    let i2 = i1 + half;
    let cs = t * half + c;
    let co = table[cs];
    let si = table[tokens * half + cs];
    let a = src[i1];
    let b = src[i2];
    dst[i1] = a * co - b * si;
    dst[i2] = b * co + a * si;
}
";

/// Causal grouped-query attention for one appended token.
/// One workgroup per query head. Stage 1 computes scaled dot scores for every
/// visible key into a scratch row, stage 2 tree-reduces the max and the exp
/// denominator, stage 3 writes the weighted value sum per head dimension.
/// Keys/values before `init` come from the caches; later ones from the
/// appended key/value rows.
const GQA: &str = r"
struct Params { p0: vec4<u32>, p1: vec4<u32>, p2: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qh = wid.x;
    let tid = lid.x;
    let group_size = pc.p0.x;
    let head_dim = pc.p0.y;
    let kv_width = pc.p0.z;
    let q_width = pc.p0.w;
    let init = pc.p1.x;
    let token = pc.p1.y;
    let score_stride = pc.p1.z;
    let scale = bitcast<f32>(pc.p1.w);
    let kvh = qh / group_size;
    let visible = init + token + 1u;
    let srow = qh * score_stride;
    let qbase = token * q_width + qh * head_dim;

    // Scaled dot scores for every visible key.
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            var kv: f32;
            if ki < init {
                kv = kcache[ki * kv_width + kvh * head_dim + d];
            } else {
                kv = k[(ki - init) * kv_width + kvh * head_dim + d];
            }
            dot = dot + q[qbase + d] * kv;
        }
        scores[srow + ki] = dot * scale;
    }
    workgroupBarrier();

    // Row max.
    var partial = -3.402823466e+38;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        partial = max(partial, scores[srow + ki]);
    }
    sh[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = max(sh[tid], sh[tid + s]); }
        workgroupBarrier();
    }
    let maximum = sh[0];
    workgroupBarrier();

    // exp(score - max) in place, plus its denominator.
    var denom_part = 0.0;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        let e = exp(scores[srow + ki] - maximum);
        scores[srow + ki] = e;
        denom_part = denom_part + e;
    }
    sh[tid] = denom_part;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let denom = sh[0];

    storageBarrier();

    // Weighted value sum per dimension.
    for (var d = tid; d < head_dim; d = d + 256u) {
        var acc = 0.0;
        for (var ki = 0u; ki < visible; ki = ki + 1u) {
            var vv: f32;
            if ki < init {
                vv = vcache[ki * kv_width + kvh * head_dim + d];
            } else {
                vv = v[(ki - init) * kv_width + kvh * head_dim + d];
            }
            acc = acc + scores[srow + ki] * vv;
        }
        dst[qbase + d] = acc / denom;
    }
}
";

/// Batched causal GQA for multi-token calls: identical math to `GQA` but one
/// workgroup covers one (query head, token) pair of a token block, so a
/// prefill-width batch dispatches once per block instead of once per token.
/// Grid = `block_tokens` * `query_heads`; `p2` = [`query_heads`, `block_tokens`,
/// `token_base`, 0]. Score scratch is one `new_cache_len` row per (t, h) pair
/// in the block.
const GQA_BATCH: &str = r"
struct Params { p0: vec4<u32>, p1: vec4<u32>, p2: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let tid = lid.x;
    let group_size = pc.p0.x;
    let head_dim = pc.p0.y;
    let kv_width = pc.p0.z;
    let q_width = pc.p0.w;
    let init = pc.p1.x;
    let score_stride = pc.p1.z;
    let scale = bitcast<f32>(pc.p1.w);
    let query_heads = pc.p2.x;
    let block_tokens = pc.p2.y;
    if flat >= block_tokens * query_heads { return; }
    let token = pc.p2.z + flat / query_heads;
    let qh = flat - (flat / query_heads) * query_heads;
    let kvh = qh / group_size;
    let visible = init + token + 1u;
    let srow = flat * score_stride;
    let qbase = token * q_width + qh * head_dim;

    // Scaled dot scores for every visible key.
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            var kv: f32;
            if ki < init {
                kv = kcache[ki * kv_width + kvh * head_dim + d];
            } else {
                kv = k[(ki - init) * kv_width + kvh * head_dim + d];
            }
            dot = dot + q[qbase + d] * kv;
        }
        scores[srow + ki] = dot * scale;
    }
    workgroupBarrier();

    // Row max.
    var partial = -3.402823466e+38;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        partial = max(partial, scores[srow + ki]);
    }
    sh[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = max(sh[tid], sh[tid + s]); }
        workgroupBarrier();
    }
    let maximum = sh[0];
    workgroupBarrier();

    // exp(score - max) in place, plus its denominator.
    var denom_part = 0.0;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        let e = exp(scores[srow + ki] - maximum);
        scores[srow + ki] = e;
        denom_part = denom_part + e;
    }
    sh[tid] = denom_part;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let denom = sh[0];

    storageBarrier();

    // Weighted value sum per dimension.
    for (var d = tid; d < head_dim; d = d + 256u) {
        var acc = 0.0;
        for (var ki = 0u; ki < visible; ki = ki + 1u) {
            var vv: f32;
            if ki < init {
                vv = vcache[ki * kv_width + kvh * head_dim + d];
            } else {
                vv = v[(ki - init) * kv_width + kvh * head_dim + d];
            }
            acc = acc + scores[srow + ki] * vv;
        }
        dst[qbase + d] = acc / denom;
    }
}
";

/// Gate + history concatenation for the gated short convolution.
/// `u_ext[i]` is `history[i]` for `i < hist_elems`, else `b[j] * v[j]` read
/// from the fused `[tokens, 3 * hidden]` projection at offsets `{0, 2h}`.
const CONV_GATE: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_ext: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let hist_elems = pc.p.x;
    let total = pc.p.y;
    let hidden = pc.p.z;
    let stride = pc.p.w;
    if i >= total { return; }
    if i < hist_elems {
        u_ext[i] = hist[i];
    } else {
        let j = i - hist_elems;
        let t = j / hidden;
        let c = j - t * hidden;
        let base = t * stride + c;
        u_ext[i] = proj[base] * proj[base + 2u * hidden];
    }
}
";

/// Depthwise gated short convolution over the concatenated history + gate rows.
/// `dst[t*hidden + c]` is `proj[t, h + c]` (the fused C gate) times the tapped
/// sum over `u_ext` rows.
const CONV: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> u_ext: array<f32>;
@group(0) @binding(2) var<storage, read_write> kernel: array<f32>;
@group(0) @binding(3) var<storage, read_write> proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let tokens = pc.p.x;
    let hidden = pc.p.y;
    let width = pc.p.z;
    let stride = pc.p.w;
    if i >= tokens * hidden { return; }
    let t = i / hidden;
    let c = i - t * hidden;
    var acc = 0.0;
    for (var tap = 0u; tap < width; tap = tap + 1u) {
        acc = acc + kernel[c * width + tap] * u_ext[(t + tap) * hidden + c];
    }
    dst[i] = acc * proj[t * stride + hidden + c];
}
";

/// Fused single-token gated short convolution. Computes u = B * V for the new
/// token, the tapped conv sum scaled by the C gate, and writes the shifted
/// history `hist_out = [hist rows 1..m | u]` so the caller can swap it into the
/// history buffer without same-buffer copies.
const CONV_STEP: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> kern: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let c = flat_wg(wid, numw) * 256u + lid.x;
    let hidden = pc.p.x;
    let m = pc.p.y;
    let stride = pc.p.z;
    if c >= hidden { return; }
    let u = proj[c] * proj[c + 2u * hidden];
    var acc = kern[c * (m + 1u) + m] * u;
    for (var j = 0u; j < m; j = j + 1u) {
        acc = acc + kern[c * (m + 1u) + j] * hist[j * hidden + c];
    }
    dst[c] = acc * proj[hidden + c];
    for (var r = 0u; r + 1u < m; r = r + 1u) {
        hist_out[r * hidden + c] = hist[(r + 1u) * hidden + c];
    }
    hist_out[(m - 1u) * hidden + c] = u;
}
";

/// `SiLU`(gate) * up over equal contiguous layouts.
const SWIGLU: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    let g = gate[i];
    let sigmoid = 1.0 / (1.0 + exp(-g));
    dst[i] = g * sigmoid * up[i];
}
";

const NF4_LINEAR_HEADER: &str = r"
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
";

/// `NF4_LINEAR_HEADER` with the `minifield.ternary.v1` two-bit decode:
/// sixteen codes per word, values -1/0/+1 times the per-128 scale.
const TERNARY_LINEAR_HEADER: &str = r"
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
";

const NF4_PAIR_HEADER: &str = r"
const PAIR: bool = true;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_a[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales_a[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_b[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales_b[row * (k / 128u) + col / 128u];
}
fn store_output(i: u32, a: f32, b: f32) { dst_a[i] = a; dst_b[i] = b; }
";

/// `NF4_PAIR_HEADER` with the SwiGLU epilogue folded into the store: each
/// invocation already owns the finished 2x2 fragments of both projections,
/// so `silu(gate) * up` writes one hidden value instead of two buffers.
const NF4_PAIR_SWIGLU_HEADER: &str = r"
const PAIR: bool = true;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes_a: array<u32>;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;
@group(0) @binding(5) var<storage, read> codes_b: array<u32>;
@group(0) @binding(6) var<storage, read> scales_b: array<f32>;
@group(0) @binding(7) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_a[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales_a[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_b[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales_b[row * (k / 128u) + col / 128u];
}
fn store_output(i: u32, a: f32, b: f32) { dst[i] = (a / (1.0 + exp(-a))) * b; }
";

/// Same fusion for `minifield.ternary.v1` streams: two-bit codes, 16 weights
/// per u32 word, `code - 1` times the group scale.
const TERNARY_PAIR_SWIGLU_HEADER: &str = r"
const PAIR: bool = true;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes_a: array<u32>;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;
@group(0) @binding(5) var<storage, read> codes_b: array<u32>;
@group(0) @binding(6) var<storage, read> scales_b: array<f32>;
@group(0) @binding(7) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_a[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales_a[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_b[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales_b[row * (k / 128u) + col / 128u];
}
fn store_output(i: u32, a: f32, b: f32) { dst[i] = (a / (1.0 + exp(-a))) * b; }
";

/// `NF4_PAIR_HEADER` with the `minifield.ternary.v1` two-bit decode.
const TERNARY_PAIR_HEADER: &str = r"
const PAIR: bool = true;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_a[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales_a[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_b[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales_b[row * (k / 128u) + col / 128u];
}
fn store_output(i: u32, a: f32, b: f32) { dst_a[i] = a; dst_b[i] = b; }
";

/// `NF4_SWIGLU_HEADER` with the `minifield.ternary.v1` two-bit decode.
const TERNARY_SWIGLU_HEADER: &str = r"
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { let g = gate[i]; return (g / (1.0 + exp(-g))) * up[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 16u) + col / 16u];
    return f32(i32((word >> ((col % 16u) * 2u)) & 3u) - 1) * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
";

const NF4_SWIGLU_HEADER: &str = r"
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { let g = gate[i]; return (g / (1.0 + exp(-g))) * up[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
";

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
            | Self::PackedGemmTernary => 0b11_1100, // x, codes, scales, x4
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
            | Self::PackedSwigluGemmTernary => {
                0b1111_1100 // gate..up4
            }
            Self::PackedGemmPairSwigluNf4 | Self::PackedGemmPairSwiglu => {
                0b1111_1100 // x, a+b streams, x4
            }
            Self::Argmax | Self::ArgmaxBlocks => 0b1100, // src, allow
            Self::ArgmaxFinal => 0b100,                  // partials
            Self::GatherColumns => 0b110,                // src, cols
            _ => 0,
        }
    }

    /// WGSL source with the shared flattened-index helper prepended.
    pub fn source(self) -> String {
        let body = match self {
            Self::PackedGemmNf4 => NF4_LINEAR_HEADER,
            Self::PackedGemmTernary => TERNARY_LINEAR_HEADER,
            Self::PackedGemmPairNf4 => NF4_PAIR_HEADER,
            Self::PackedGemmPairTernary => TERNARY_PAIR_HEADER,
            Self::PackedSwigluGemmNf4 => NF4_SWIGLU_HEADER,
            Self::PackedSwigluGemmTernary => TERNARY_SWIGLU_HEADER,
            Self::PackedGemmPairSwigluNf4 => NF4_PAIR_SWIGLU_HEADER,
            Self::PackedGemmPairSwiglu => TERNARY_PAIR_SWIGLU_HEADER,
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
            // MINI_NF4_STAGE_F16 experiment: x|w|wx selects which workgroup
            // staging arrays store f16 (decode and accumulation stay f32).
            // Requires SHADER_F16, requested by the device under the same knob.
            if let Some(mode) = nf4_f16_stage_mode() {
                let stg_x = if mode.contains('x') { "f16" } else { "f32" };
                let stg_w = if mode.contains('w') { "f16" } else { "f32" };
                let tile = include_str!("nf4_prefill_f16.wgsl")
                    .replace("__STGX__", stg_x)
                    .replace("__STGW__", stg_w);
                return ["enable f16;\n", WGSL_INDEX, lut, body, &tile].concat();
            }
            let tile = include_str!("nf4_prefill.wgsl");
            return [WGSL_INDEX, lut, body, tile].concat();
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
            Self::ArgmaxFinal | Self::Copy2d => 2,
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
            | Self::PackedGemmTernary => 5,
            Self::Gqa
            | Self::GqaBatch
            | Self::PackedSwigluGemv
            | Self::PackedSwigluGemvNf4
            | Self::PackedSwigluGemvMtNf4
            | Self::PackedSwigluGemmNf4
            | Self::PackedSwigluGemmTernary
            | Self::QkNormRope => 7,
            Self::PackedGemvPair
            | Self::PackedGemvPairNf4
            | Self::PackedGemvPairMtNf4
            | Self::PackedGemmPairNf4
            | Self::PackedGemmPairTernary => 8,
            Self::PackedGemmPairSwigluNf4 | Self::PackedGemmPairSwiglu => 7,
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
            Self::AddNorm => "add_norm",
            Self::QkNormRope => "qk_norm_rope",
            Self::Argmax => "argmax",
            Self::ArgmaxBlocks => "argmax_blocks",
            Self::ArgmaxFinal => "argmax_final",
            Self::GatherColumns => "gather_columns",
        }
    }
}

/// Experimental f16 staging selector for the NF4 GEMM tile.
/// `MINI_NF4_STAGE_F16=w|x|wx` marks weights / activations / both for f16
/// workgroup storage; unset means the production f32 tile. Paired with the
/// matching SHADER_F16 device request so both sides flip together.
pub fn nf4_f16_stage_mode() -> Option<String> {
    let mode = std::env::var("MINI_NF4_STAGE_F16").ok()?;
    (mode.chars().all(|c| matches!(c, 'w' | 'x')) && !mode.is_empty()).then_some(mode)
}
