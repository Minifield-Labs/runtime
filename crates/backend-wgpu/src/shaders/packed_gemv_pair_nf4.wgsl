
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

var<workgroup> part: array<f32, 512>;

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
    var acc_a = 0.0;
    var acc_b = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word_a = codes_a[cbase + w];
            let word_b = codes_b[cbase + w];
            let scale_a = scales_a[sbase + (w >> 4u)];
            let scale_b = scales_b[sbase + (w >> 4u)];
            let base = w << 3u;
            var dot_a = 0.0;
            var dot_b = 0.0;
            if (vec_ok && base + 8u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 2u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 4u;
                    let ca0 = NF4[(word_a >> sh) & 0xFu];
                    let ca1 = NF4[(word_a >> (sh + 4u)) & 0xFu];
                    let ca2 = NF4[(word_a >> (sh + 8u)) & 0xFu];
                    let ca3 = NF4[(word_a >> (sh + 12u)) & 0xFu];
                    let cb0 = NF4[(word_b >> sh) & 0xFu];
                    let cb1 = NF4[(word_b >> (sh + 4u)) & 0xFu];
                    let cb2 = NF4[(word_b >> (sh + 8u)) & 0xFu];
                    let cb3 = NF4[(word_b >> (sh + 12u)) & 0xFu];
                    dot_a = dot_a + xv.x * ca0 + xv.y * ca1 + xv.z * ca2 + xv.w * ca3;
                    dot_b = dot_b + xv.x * cb0 + xv.y * cb1 + xv.z * cb2 + xv.w * cb3;
                }
            } else {
                let count = min(8u, k - base);
                for (var e = 0u; e < count; e = e + 1u) {
                    let xv = x[xbase + base + e];
                    let nib_a = (word_a >> (e * 4u)) & 0xFu;
                    let nib_b = (word_b >> (e * 4u)) & 0xFu;
                    dot_a = dot_a + xv * NF4[nib_a];
                    dot_b = dot_b + xv * NF4[nib_b];
                }
            }
            acc_a = acc_a + dot_a * scale_a;
            acc_b = acc_b + dot_b * scale_b;
        }
    }
    part[tid] = acc_a;
    part[256u + tid] = acc_b;
    workgroupBarrier();
    if (tid < rows_per_wg) {
        let o = row0 + tid;
        if (o < n) {
            var s_a = 0.0;
            var s_b = 0.0;
            let pbase = tid * lanes;
            for (var l = 0u; l < lanes; l = l + 1u) {
                s_a = s_a + part[pbase + l];
                s_b = s_b + part[256u + pbase + l];
            }
            dst_a[i * n + o] = s_a;
            dst_b[i * n + o] = s_b;
        }
    }
}
