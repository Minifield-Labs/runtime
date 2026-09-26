
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;

var<workgroup> part: array<f32, 256>;

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
    let rows_per_wg = 256u / lanes;
    let i = flat / tiles;
    let row0 = (flat - i * tiles) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let words = (k + 7u) >> 3u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 4u)];
            let base = w << 3u;
            var dot = 0.0;
            if (vec_ok && base + 8u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 2u; q = q + 1u) {
                    let g = gate4[b4 + q];
                    let sigmoid = vec4(1.0) / (vec4(1.0) + exp(-g));
                    let xv = (g * sigmoid) * up4[b4 + q];
                    let sh = q << 4u;
                    dot = dot
                        + xv.x * NF4[(word >> sh) & 0xFu]
                        + xv.y * NF4[(word >> (sh + 4u)) & 0xFu]
                        + xv.z * NF4[(word >> (sh + 8u)) & 0xFu]
                        + xv.w * NF4[(word >> (sh + 12u)) & 0xFu];
                }
            } else {
                let count = min(8u, k - base);
                for (var e = 0u; e < count; e = e + 1u) {
                    let l = xbase + base + e;
                    let g = gate[l];
                    let sigmoid = 1.0 / (1.0 + exp(-g));
                    let xv = (g * sigmoid) * up[l];
                    let nib = (word >> (e * 4u)) & 0xFu;
                    dot = dot + xv * NF4[nib];
                }
            }
            acc = acc + dot * scale;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s = s + part[pbase + l];
            }
            dst[i * n + o] = s;
        }
    }
}
