// 32x64 output tile, 16 reduction elements per iteration. Each of the
// 16x16 invocations loads two activations and four decoded weights, then
// accumulates a 2x4 output fragment. Input and weight accessors are supplied
// by the binding header.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
var<workgroup> inputs: array<f32, 512>;
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
    let col0 = (flat % columns) * 64u;
    if row0 >= m { return; }
    let x = lid.x;
    let y = lid.y;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0);
    var b1 = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 16u) {
        for (var hi = 0u; hi < 2u; hi += 1u) {
            let r = y + hi * 16u;
            var xv = 0.0;
            if row0 + r < m && base + x < k {
                xv = input_value((row0 + r) * k + base + x);
            }
            inputs[r * 16u + x] = xv;
        }
        for (var lo = 0u; lo < 4u; lo += 1u) {
            let c = x + lo * 16u;
            var wa = 0.0;
            var wb = 0.0;
            if col0 + c < n && base + y < k {
                wa = weight_a(col0 + c, base + y, k);
                if PAIR { wb = weight_b(col0 + c, base + y, k); }
            }
            weights_a[y * 64u + c] = wa;
            if PAIR { weights_b[y * 64u + c] = wb; }
        }
        workgroupBarrier();
        for (var d = 0u; d < 16u; d += 1u) {
            let w = vec4<f32>(
                weights_a[d * 64u + x],
                weights_a[d * 64u + x + 16u],
                weights_a[d * 64u + x + 32u],
                weights_a[d * 64u + x + 48u],
            );
            a0 += inputs[y * 16u + d] * w;
            a1 += inputs[(y + 16u) * 16u + d] * w;
            if PAIR {
                let v = vec4<f32>(
                    weights_b[d * 64u + x],
                    weights_b[d * 64u + x + 16u],
                    weights_b[d * 64u + x + 32u],
                    weights_b[d * 64u + x + 48u],
                );
                b0 += inputs[y * 16u + d] * v;
                b1 += inputs[(y + 16u) * 16u + d] * v;
            }
        }
        workgroupBarrier();
    }
    for (var hi = 0u; hi < 2u; hi += 1u) {
        for (var lo = 0u; lo < 4u; lo += 1u) {
            let r = row0 + y + hi * 16u;
            let c = col0 + x + lo * 16u;
            let fa = select(a1[lo], a0[lo], hi == 0u);
            let fb = select(b1[lo], b0[lo], hi == 0u);
            if r < m && c < n {
                store_output(r * n + c, fa, fb);
            }
        }
    }
}
