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
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
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
    let c = i - r * pc.p.y;
    dst[i] = table[ids[r] * pc.p.y + c];
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
/// `u_ext[i]` is `history[i]` for `i < hist_elems`, else `b[j] * v[j]`.
const CONV_GATE: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> u_ext: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let hist_elems = pc.p.x;
    let total = pc.p.y;
    if i >= total { return; }
    if i < hist_elems {
        u_ext[i] = hist[i];
    } else {
        let j = i - hist_elems;
        u_ext[i] = b[j] * v[j];
    }
}
";

/// Depthwise gated short convolution over the concatenated history + gate rows.
/// `dst[t*hidden + c]` is `cgate[t,c]` times the tapped sum over `u_ext` rows.
const CONV: &str = r"
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> u_ext: array<f32>;
@group(0) @binding(2) var<storage, read_write> kernel: array<f32>;
@group(0) @binding(3) var<storage, read_write> cgate: array<f32>;
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
    if i >= tokens * hidden { return; }
    let t = i / hidden;
    let c = i - t * hidden;
    var acc = 0.0;
    for (var tap = 0u; tap < width; tap = tap + 1u) {
        acc = acc + kernel[c * width + tap] * u_ext[(t + tap) * hidden + c];
    }
    dst[i] = acc * cgate[i];
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
            Self::Copy2d => 2,
            Self::Binary
            | Self::Gather
            | Self::Gemm
            | Self::RmsNorm
            | Self::Rotary
            | Self::SwiGlu => 3,
            Self::Gemv | Self::ConvGate | Self::Conv => 4,
            Self::Gqa => 7,
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
        }
    }
}
