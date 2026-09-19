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
        dst[i] = bitcast<f32>(0x7FC00000u);
        return;
    }
    dst[i] = table[id * pc.p.y + c];
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
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> w4: array<vec4<f32>>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j = flat_wg(wid, numw);
    let tid = lid.x;
    let n = pc.p.x;
    let k = pc.p.y;
    if j >= n { return; }

    let rbase = j * k;
    var acc = 0.0;
    if k >= 4u && (rbase & 3u) == 0u {
        let k4 = k >> 2u;
        let kbulk = k4 << 2u;
        let base4 = rbase >> 2u;
        for (var g = tid; g < k4; g = g + 256u) {
            let rv = w4[base4 + g];
            let l = g * 4u;
            acc = acc + rv.x * x[l + 0u];
            acc = acc + rv.y * x[l + 1u];
            acc = acc + rv.z * x[l + 2u];
            acc = acc + rv.w * x[l + 3u];
        }
        for (var l = kbulk + tid; l < k; l = l + 256u) {
            acc = acc + x[l] * w[rbase + l];
        }
    } else {
        for (var l = tid; l < k; l = l + 256u) {
            acc = acc + x[l] * w[rbase + l];
        }
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[j] = sh[0];
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
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> scales: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let m = pc.p.z;
    if flat >= m * n { return; }
    let i = flat / n;
    let j = flat - i * n;
    let tid = lid.x;
    let code_words = k >> 4u;
    let groups = k >> 7u;
    let cbase = j * code_words;
    let sbase = j * groups;
    let xbase = i * k;
    var acc = 0.0;
    for (var l = tid; l < k; l = l + 256u) {
        let word = codes[cbase + (l >> 4u)];
        let byte = (word >> (((l >> 2u) & 3u) << 3u)) & 0xFFu;
        let code = (byte >> ((l & 3u) << 1u)) & 3u;
        acc = acc + x[xbase + l] * (f32(i32(code) - 1) * scales[sbase + (l >> 7u)]);
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[i * n + j] = sh[0];
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
        dst[i] = bitcast<f32>(0x7FC00000u);
        return;
    }
    let word = codes[src * (k >> 4u) + (l >> 4u)];
    let byte = (word >> (((l >> 2u) & 3u) << 3u)) & 0xFFu;
    let code = (byte >> ((l & 3u) << 1u)) & 3u;
    dst[i] = f32(i32(code) - 1) * scales[src * (k >> 7u) + (l >> 7u)];
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
@group(0) @binding(2) var<storage, read_write> src: array<f32>;

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
    var best = bitcast<f32>(0xFF800000u);
    var idx = 0u;
    var bad = 0u;
    for (var c = tid; c < cols; c = c + 256u) {
        let v = src[base + c];
        if v == v && abs(v) <= 3.4028234663852886e38 {
            if v > best {
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
            dst[row] = bitcast<f32>(0x7FC00000u);
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
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read_write> scales_b: array<f32>;

var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let m = pc.p.z;
    if flat >= m * n { return; }
    let i = flat / n;
    let j = flat - i * n;
    let tid = lid.x;
    let code_words = k >> 4u;
    let groups = k >> 7u;
    let cabase = j * code_words;
    let cbbase = j * code_words;
    let sbase = j * groups;
    let xbase = i * k;
    var acc_a = 0.0;
    var acc_b = 0.0;
    for (var l = tid; l < k; l = l + 256u) {
        let xv = x[xbase + l];
        let word_a = codes_a[cabase + (l >> 4u)];
        let word_b = codes_b[cbbase + (l >> 4u)];
        let shift = ((l >> 2u) & 3u) << 3u;
        let lane = (l & 3u) << 1u;
        let code_a = (((word_a >> shift) & 0xFFu) >> lane) & 3u;
        let code_b = (((word_b >> shift) & 0xFFu) >> lane) & 3u;
        acc_a = acc_a + xv * (f32(i32(code_a) - 1) * scales_a[sbase + (l >> 7u)]);
        acc_b = acc_b + xv * (f32(i32(code_b) - 1) * scales_b[sbase + (l >> 7u)]);
    }
    sh_a[tid] = acc_a;
    sh_b[tid] = acc_b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            sh_a[tid] = sh_a[tid] + sh_a[tid + s];
            sh_b[tid] = sh_b[tid] + sh_b[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        dst_a[i * n + j] = sh_a[0];
        dst_b[i * n + j] = sh_b[0];
    }
}
";

/// Packed ternary GEMV over an on-the-fly SiLU(gate) * up activation:
/// dst[i*n + j] = `sum_l` (silu(gate[i,l]) * up[i,l]) * (code(j,l) - 1) *
/// scale(j, l/128). Same workgroup layout and decode as `PACKED_GEMV`.
const PACKED_SWIGLU_GEMV: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> up: array<f32>;
@group(0) @binding(4) var<storage, read_write> codes: array<u32>;
@group(0) @binding(5) var<storage, read_write> scales: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let n = pc.p.x;
    let k = pc.p.y;
    let m = pc.p.z;
    if flat >= m * n { return; }
    let i = flat / n;
    let j = flat - i * n;
    let tid = lid.x;
    let code_words = k >> 4u;
    let groups = k >> 7u;
    let cbase = j * code_words;
    let sbase = j * groups;
    let xbase = i * k;
    var acc = 0.0;
    for (var l = tid; l < k; l = l + 256u) {
        let g = gate[xbase + l];
        let sigmoid = 1.0 / (1.0 + exp(-g));
        let xv = (g * sigmoid) * up[xbase + l];
        let word = codes[cbase + (l >> 4u)];
        let byte = (word >> (((l >> 2u) & 3u) << 3u)) & 0xFFu;
        let code = (byte >> ((l & 3u) << 1u)) & 3u;
        acc = acc + xv * (f32(i32(code) - 1) * scales[sbase + (l >> 7u)]);
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[i * n + j] = sh[0];
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
    ConvGate,
    Conv,
    SwiGlu,
    PackedGemv,
    PackedGather,
    PackedGemvPair,
    PackedSwigluGemv,
    AddNorm,
    QkNormRope,
    Argmax,
}

impl Kernel {
    /// WGSL source with the shared flattened-index helper prepended.
    pub fn source(self) -> String {
        let body = match self {
            Self::Fill => FILL,
            Self::Binary => BINARY,
            Self::Copy2d => COPY2D,
            Self::Gather => GATHER,
            Self::Gemv => GEMV,
            Self::Gemm => GEMM,
            Self::RmsNorm => RMSNORM,
            Self::Rotary => ROTARY,
            Self::Gqa => GQA,
            Self::ConvGate => CONV_GATE,
            Self::Conv => CONV,
            Self::SwiGlu => SWIGLU,
            Self::PackedGemv => PACKED_GEMV,
            Self::PackedGather => PACKED_GATHER,
            Self::PackedGemvPair => PACKED_GEMV_PAIR,
            Self::PackedSwigluGemv => PACKED_SWIGLU_GEMV,
            Self::AddNorm => ADD_NORM,
            Self::QkNormRope => QK_NORM_ROPE,
            Self::Argmax => ARGMAX,
        };
        let mut source = String::with_capacity(WGSL_INDEX.len() + body.len() + 1);
        source.push_str(WGSL_INDEX);
        source.push_str(body);
        source
    }

    /// Storage-buffer bindings after the uniform params binding.
    pub const fn storage_bindings(self) -> u32 {
        match self {
            Self::Fill => 1,
            Self::Argmax | Self::Copy2d => 2,
            Self::Binary
            | Self::Gather
            | Self::Gemm
            | Self::RmsNorm
            | Self::Rotary
            | Self::ConvGate
            | Self::SwiGlu => 3,
            Self::Gemv | Self::Conv | Self::PackedGemv | Self::PackedGather => 4,
            Self::AddNorm | Self::PackedSwigluGemv => 5,
            Self::Gqa | Self::PackedGemvPair | Self::QkNormRope => 7,
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
            Self::ConvGate => "conv_gate",
            Self::Conv => "conv",
            Self::SwiGlu => "swiglu",
            Self::PackedGemv => "packed_gemv",
            Self::PackedGather => "packed_gather",
            Self::PackedGemvPair => "packed_gemv_pair",
            Self::PackedSwigluGemv => "packed_swiglu_gemv",
            Self::AddNorm => "add_norm",
            Self::QkNormRope => "qk_norm_rope",
            Self::Argmax => "argmax",
        }
    }
}
