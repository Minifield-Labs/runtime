// Group-partial control for the packed 64x32 K16 tile. Same staging and
// fragment ownership as the production kernel, but weights stage UNSCALED
// codes and each thread accumulates a per-128-group partial that the group
// scale multiplies once. inner is always a multiple of 128 (defined by the
// scale shape), so a group spans exactly eight K16 steps.
// Headers provide code_a/code_b (unscaled decoded value), scale_a/scale_b,
// input_value, and store_output.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
var<workgroup> inputs: array<f32, 1024>;
var<workgroup> weights_a: array<f32, 512>;
var<workgroup> weights_b: array<f32, 512>;

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
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0);
    var b1 = vec4<f32>(0.0);
    var p0 = vec4<f32>(0.0);
    var p1 = vec4<f32>(0.0);
    var q0 = vec4<f32>(0.0);
    var q1 = vec4<f32>(0.0);
    let groups = k / 128u;
    for (var base = 0u; base < k; base += 16u) {
        for (var hi = 0u; hi < 4u; hi += 1u) {
            let r = y + hi * 16u;
            var xv = 0.0;
            if row0 + r < m && base + x < k {
                xv = input_value((row0 + r) * k + base + x);
            }
            inputs[r * 16u + x] = xv;
        }
        for (var lo = 0u; lo < 2u; lo += 1u) {
            let c = x + lo * 16u;
            var wa = 0.0;
            var wb = 0.0;
            if col0 + c < n {
                wa = code_a(col0 + c, base + y, k);
                if PAIR { wb = code_b(col0 + c, base + y, k); }
            }
            weights_a[y * 32u + c] = wa;
            if PAIR { weights_b[y * 32u + c] = wb; }
        }
        workgroupBarrier();
        for (var d = 0u; d < 16u; d += 1u) {
            let w0 = weights_a[d * 32u + x];
            let w1 = weights_a[d * 32u + x + 16u];
            let xv0 = inputs[y * 16u + d];
            let xv1 = inputs[(y + 16u) * 16u + d];
            p0 += vec4(xv0 * w0, xv0 * w1, xv1 * w0, xv1 * w1);
            let xv2 = inputs[(y + 32u) * 16u + d];
            let xv3 = inputs[(y + 48u) * 16u + d];
            p1 += vec4(xv2 * w0, xv2 * w1, xv3 * w0, xv3 * w1);
            if PAIR {
                let v0 = weights_b[d * 32u + x];
                let v1 = weights_b[d * 32u + x + 16u];
                q0 += vec4(xv0 * v0, xv0 * v1, xv1 * v0, xv1 * v1);
                q1 += vec4(xv2 * v0, xv2 * v1, xv3 * v0, xv3 * v1);
            }
        }
        workgroupBarrier();
        if (base + 16u) % 128u == 0u {
            let g = base / 128u;
            // Clamp out-of-range columns: their partials are zero, but an
            // out-of-bounds scale read must not inject NaN or trap.
            let sa0 = scale_a(min(col0 + x, n - 1u), g, k);
            let sa1 = scale_a(min(col0 + x + 16u, n - 1u), g, k);
            a0 += vec4(p0.x * sa0, p0.y * sa1, p0.z * sa0, p0.w * sa1);
            a1 += vec4(p1.x * sa0, p1.y * sa1, p1.z * sa0, p1.w * sa1);
            p0 = vec4<f32>(0.0);
            p1 = vec4<f32>(0.0);
            if PAIR {
                let sb0 = scale_b(min(col0 + x, n - 1u), g, k);
                let sb1 = scale_b(min(col0 + x + 16u, n - 1u), g, k);
                b0 += vec4(q0.x * sb0, q0.y * sb1, q0.z * sb0, q0.w * sb1);
                b1 += vec4(q1.x * sb0, q1.y * sb1, q1.z * sb0, q1.w * sb1);
                q0 = vec4<f32>(0.0);
                q1 = vec4<f32>(0.0);
            }
        }
    }
    for (var hi = 0u; hi < 4u; hi += 1u) {
        for (var lo = 0u; lo < 2u; lo += 1u) {
            let r = row0 + y + hi * 16u;
            let c = col0 + x + lo * 16u;
            let fa = select(a1[hi * 2u + lo - 4u], a0[hi * 2u + lo], hi < 2u);
            let fb = select(b1[hi * 2u + lo - 4u], b0[hi * 2u + lo], hi < 2u);
            if r < m && c < n {
                store_output(r * n + c, fa, fb);
            }
        }
    }
}
