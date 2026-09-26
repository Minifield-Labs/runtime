// E15 ternary sign/select control: 64x32 output tile, K16 slices, but no
// decoded FP32 weight tile. Each thread reads the packed u32 word for each
// of its two output columns (one word covers a K16 slice at two bits per
// weight), extracts the code once, and reuses it across four token rows.
// __SEL__ expands to "true" (select form) or "false" (sign/zero bitmask).
// Group-partial schedule: unscaled contributions accumulate within each
// 128-weight group, then the group scale applies once.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

const USE_SELECT: bool = __SEL__;
var<workgroup> inputs: array<f32, 1024>;

// code 0 -> -x, 1 -> 0, 2 -> +x across four token values.
fn signed4(xv: vec4<f32>, code: u32) -> vec4<f32> {
    if USE_SELECT {
        return select(select(xv, -xv, code == 0u), vec4<f32>(0.0), code == 1u);
    }
    let sign = select(0u, 0x80000000u, code == 0u);
    let keep = select(0xffffffffu, 0u, code == 1u);
    return bitcast<vec4<f32>>(
        (bitcast<vec4<u32>>(xv) ^ vec4<u32>(sign)) & vec4<u32>(keep)
    );
}

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
    let words = k / 16u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var p0 = vec4<f32>(0.0);
    var p1 = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 16u) {
        for (var hi = 0u; hi < 4u; hi += 1u) {
            let r = y + hi * 16u;
            var xv = 0.0;
            if row0 + r < m && base + x < k {
                xv = x_in[(row0 + r) * k + base + x];
            }
            inputs[r * 16u + x] = xv;
        }
        // One u32 per column covers this K16 slice; out-of-range columns
        // produce code 0 contributions (-x is never stored, partials stay 0
        // because the epilogue masks c >= n).
        var w0 = 0u;
        var w1 = 0u;
        if col0 + x < n {
            w0 = codes[(col0 + x) * words + base / 16u];
        }
        if col0 + x + 16u < n {
            w1 = codes[(col0 + x + 16u) * words + base / 16u];
        }
        workgroupBarrier();
        for (var d = 0u; d < 16u; d += 1u) {
            let xv = vec4<f32>(
                inputs[y * 16u + d],
                inputs[(y + 16u) * 16u + d],
                inputs[(y + 32u) * 16u + d],
                inputs[(y + 48u) * 16u + d],
            );
            p0 += signed4(xv, (w0 >> (d * 2u)) & 3u);
            p1 += signed4(xv, (w1 >> (d * 2u)) & 3u);
        }
        workgroupBarrier();
        if (base + 16u) % 128u == 0u {
            let g = base / 128u;
            let gk = k / 128u;
            let sa0 = scales[min(col0 + x, n - 1u) * gk + g];
            let sa1 = scales[min(col0 + x + 16u, n - 1u) * gk + g];
            a0 += p0 * sa0;
            a1 += p1 * sa1;
            p0 = vec4<f32>(0.0);
            p1 = vec4<f32>(0.0);
        }
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
