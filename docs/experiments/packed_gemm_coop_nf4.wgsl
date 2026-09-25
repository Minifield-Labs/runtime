const PAIR: bool = false;
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
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

// 32x64 output tile, 16 reduction elements staged per iteration. Full tiles
// finish with cooperative 8x8 matrices: each subgroup owns an 8-row x 32-col
// slab (eight tasks across the tile), loading A and B fragments from the
// staged workgroup tiles. Partial tiles fall back to the scalar 2x4-fragment
// path so output stores stay bounds-checked; `coopStoreT` cannot clip.
var<workgroup> inputs: array<f32, 512>;
var<workgroup> weights_a: array<f32, 1024>;
var<workgroup> zeros: array<f32, 64>;
var<workgroup> nsg_cell: u32;
var<workgroup> full_cell: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
) {
    // Inline flat_wg: helper-function parameters carry no builtin binding,
    // so naga marks call results non-uniform, and `if full` would poison the
    // cooperative ops under it.
    let flat = wid.z * numw.y * numw.x + wid.y * numw.x + wid.x;
    let m = pc.p.x;
    let n = pc.p.y;
    let k = pc.p.z;
    let columns = pc.p.w;
    let row0 = (flat / columns) * 32u;
    let col0 = (flat % columns) * 64u;
    let lx = lid.x % 16u;
    let ly = lid.x / 16u;
    // No per-lane conditionals in this block: naga's uniformity analysis
    // conservatively treats a non-uniform `if` as tainting the rest of the
    // block, which would reject the cooperative ops below. Mask with select
    // and clamp addresses instead.
    zeros[lid.x % 64u] = 0.0;
    // `num_subgroups` is not on naga's uniform builtin whitelist, so it cannot
    // appear in control flow around cooperative ops. Every subgroup holds the
    // same value; bounce it through workgroup memory for a provably uniform
    // loop bound.
    nsg_cell = nsg;
    full_cell = select(0u, 1u, (row0 + 32u <= m) && (col0 + 64u <= n));

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    workgroupBarrier();
    let nsg_u = workgroupUniformLoad(&nsg_cell);
    let full = workgroupUniformLoad(&full_cell) == 1u;
    let tasks_per_sg = (8u + nsg_u - 1u) / nsg_u;
    var c: array<coop_mat8x8<f32, C>, 8>;
    for (var i = 0u; i < 8u; i += 1u) {
        c[i] = coopLoad<coop_mat8x8<f32, C>>(&zeros[0], 8u);
    }
    for (var base = 0u; base < k; base += 16u) {
        for (var hi = 0u; hi < 2u; hi += 1u) {
            let r = ly + hi * 16u;
            let rr = min(row0 + r, m - 1u);
            let ck = min(base + lx, k - 1u);
            let ok_x = row0 + r < m && base + lx < k;
            inputs[r * 16u + lx] = select(0.0, input_value(rr * k + ck), ok_x);
        }
        for (var lo = 0u; lo < 4u; lo += 1u) {
            let cw = lx + lo * 16u;
            let wk = min(base + ly, k - 1u);
            let ok_w = col0 + cw < n && base + ly < k;
            weights_a[ly * 64u + cw] = select(0.0, weight_a(min(col0 + cw, n - 1u), wk, k), ok_w);
        }
        workgroupBarrier();
        if full {
            for (var t = 0u; t < tasks_per_sg; t += 1u) {
                let task = min(t * nsg_u + sid, 7u);
                let rs = (task % 4u) * 8u;
                let ch = (task / 4u) * 32u;
                for (var kk = 0u; kk < 2u; kk += 1u) {
                    let am = coopLoadT<coop_mat8x8<f32, A>>(
                        &inputs[rs * 16u + kk * 8u], 16u);
                    for (var j = 0u; j < 4u; j += 1u) {
                        let bm = coopLoadT<coop_mat8x8<f32, B>>(
                            &weights_a[kk * 8u * 64u + ch + j * 8u], 64u);
                        c[t * 4u + j] = coopMultiplyAdd(am, bm, c[t * 4u + j]);
                    }
                }
            }
        } else {
            for (var d = 0u; d < 16u; d += 1u) {
                let w = vec4<f32>(
                    weights_a[d * 64u + lx],
                    weights_a[d * 64u + lx + 16u],
                    weights_a[d * 64u + lx + 32u],
                    weights_a[d * 64u + lx + 48u],
                );
                a0 += inputs[ly * 16u + d] * w;
                a1 += inputs[(ly + 16u) * 16u + d] * w;
            }
        }
        workgroupBarrier();
    }
    if full {
        for (var t = 0u; t < tasks_per_sg; t += 1u) {
            let task = min(t * nsg_u + sid, 7u);
            let rs = (task % 4u) * 8u;
            let ch = (task / 4u) * 32u;
            for (var j = 0u; j < 4u; j += 1u) {
                coopStoreT(
                    c[t * 4u + j],
                    &dst[(row0 + rs) * n + col0 + ch + j * 8u],
                    n,
                );
            }
        }
    } else {
        for (var hi = 0u; hi < 2u; hi += 1u) {
            for (var lo = 0u; lo < 4u; lo += 1u) {
                let r = row0 + ly + hi * 16u;
                let cw = col0 + lx + lo * 16u;
                let fa = select(a1[lo], a0[lo], hi == 0u);
                if r < m && cw < n {
                    dst[r * n + cw] = fa;
                }
            }
        }
    }
}
