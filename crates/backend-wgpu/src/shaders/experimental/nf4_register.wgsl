// E19 NF4 register decode: production 64x32 K16 tile minus the decoded
// FP32 weight workgroup array. Each thread loads the four packed u32 words
// covering its two output columns for the K16 slice (eight 4-bit codes per
// word), decodes each centroid with its group scale in E14 order, and
// reuses the value across the thread's four token rows. Decoded values are
// shared across four rows instead of the staged tile's 64: the trade is
// more decode instructions against zero weight staging.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

var<workgroup> inputs: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let m = pc.p.x;
    let n = pc.p.y;
    let k = pc.p.z;
    let columns = pc.p.w;
    let row0 = (flat / columns) * 64u;
    let col0 = (flat % columns) * 32u;
    if row0 >= m { return; }
    let x = lid.x;
    let y = lid.y;
    let words = k / 8u;
    let gk = k / 128u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 16u) {
        for (var hi = 0u; hi < 4u; hi += 1u) {
            let r = y + hi * 16u;
            var xv = 0.0;
            if row0 + r < m && base + x < k {
                xv = x_in[(row0 + r) * k + base + x];
            }
            inputs[r * 16u + x] = xv;
        }
        // Two u32 words per column cover the K16 slice (8 codes each).
        var w00 = 0u;
        var w01 = 0u;
        var w10 = 0u;
        var w11 = 0u;
        let wbase = base / 8u;
        if col0 + x < n {
            w00 = codes[(col0 + x) * words + wbase];
            w01 = codes[(col0 + x) * words + wbase + 1u];
        }
        if col0 + x + 16u < n {
            w10 = codes[(col0 + x + 16u) * words + wbase];
            w11 = codes[(col0 + x + 16u) * words + wbase + 1u];
        }
        let g = base / 128u;
        let s0 = scales[min(col0 + x, n - 1u) * gk + g];
        let s1 = scales[min(col0 + x + 16u, n - 1u) * gk + g];
        workgroupBarrier();
        for (var d = 0u; d < 16u; d += 1u) {
            let wlo0 = select(w01, w00, d < 8u);
            let wlo1 = select(w11, w10, d < 8u);
            let shift = (d % 8u) * 4u;
            let c0v = NF4[(wlo0 >> shift) & 15u] * s0;
            let c1v = NF4[(wlo1 >> shift) & 15u] * s1;
            let xv = vec4<f32>(
                inputs[y * 16u + d],
                inputs[(y + 16u) * 16u + d],
                inputs[(y + 32u) * 16u + d],
                inputs[(y + 48u) * 16u + d],
            );
            a0 += xv * c0v;
            a1 += xv * c1v;
        }
        workgroupBarrier();
    }
    for (var hi = 0u; hi < 4u; hi += 1u) {
        let r = row0 + y + hi * 16u;
        let c0 = col0 + x;
        let c1 = col0 + x + 16u;
        if r < m {
            if c0 < n { dst[r * n + c0] = a0[hi]; }
            if c1 < n { dst[r * n + c1] = a1[hi]; }
        }
    }
}
