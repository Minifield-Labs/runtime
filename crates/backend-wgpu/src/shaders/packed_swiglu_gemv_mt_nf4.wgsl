
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
