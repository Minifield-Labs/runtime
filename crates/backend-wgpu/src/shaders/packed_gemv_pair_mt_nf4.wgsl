
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
