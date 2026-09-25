// 32x32 output tile, 32 reduction elements per iteration. Each of the
// 16x16 invocations loads four activations and four decoded weights, then
// accumulates a 2x2 output fragment. Input and weight accessors are supplied
// by the linear, paired-linear, or fused SwiGLU binding header.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
var<workgroup> inputs: array<f32, 1024>;
var<workgroup> weights_a: array<f32, 1024>;
var<workgroup> weights_b: array<f32, 1024>;

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
    let row0 = (flat / columns) * 32u;
    let col0 = (flat % columns) * 32u;
    if row0 >= m { return; }
    let x = lid.x;
    let y = lid.y;
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 32u) {
        for (var hi = 0u; hi < 2u; hi += 1u) {
            for (var lo = 0u; lo < 2u; lo += 1u) {
                let r = y + hi * 16u;
                let c = x + lo * 16u;
                var xv = 0.0;
                if row0 + r < m && base + c < k {
                    xv = input_value((row0 + r) * k + base + c);
                }
                inputs[r * 32u + c] = xv;
                var wa = 0.0;
                var wb = 0.0;
                if col0 + c < n && base + r < k {
                    wa = weight_a(col0 + c, base + r, k);
                    if PAIR { wb = weight_b(col0 + c, base + r, k); }
                }
                weights_a[r * 32u + c] = wa;
                if PAIR { weights_b[r * 32u + c] = wb; }
            }
        }
        workgroupBarrier();
        for (var d = 0u; d < 32u; d += 1u) {
            let xv0 = inputs[y * 32u + d];
            let xv1 = inputs[(y + 16u) * 32u + d];
            let w0 = weights_a[d * 32u + x];
            let w1 = weights_a[d * 32u + x + 16u];
            a += vec4(xv0 * w0, xv0 * w1, xv1 * w0, xv1 * w1);
            if PAIR {
                let v0 = weights_b[d * 32u + x];
                let v1 = weights_b[d * 32u + x + 16u];
                b += vec4(xv0 * v0, xv0 * v1, xv1 * v0, xv1 * v1);
            }
        }
        workgroupBarrier();
    }
    for (var hi = 0u; hi < 2u; hi += 1u) {
        for (var lo = 0u; lo < 2u; lo += 1u) {
            let r = row0 + y + hi * 16u;
            let c = col0 + x + lo * 16u;
            if r < m && c < n {
                store_output(r * n + c, a[hi * 2u + lo], b[hi * 2u + lo]);
            }
        }
    }
}
