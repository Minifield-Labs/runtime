
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

// Same lane-grouped layout as `PACKED_GEMV`: each group of `G` lanes reduces
// one output row for both weight sets, so the shared x stream is read once
// per element for both dots. `part` holds the A partials in [0, 256) and the
// B partials in [256, 512); one barrier, then each row's thread reduces both.
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
    let words = (k + 15u) >> 4u;
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
            let scale_a = scales_a[sbase + (w >> 3u)];
            let scale_b = scales_b[sbase + (w >> 3u)];
            let base = w << 4u;
            var dot_a = 0.0;
            var dot_b = 0.0;
            if (vec_ok && base + 16u <= k) {
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 3u;
                    let ca0 = f32(i32((word_a >> sh) & 3u) - 1);
                    let ca1 = f32(i32((word_a >> (sh + 2u)) & 3u) - 1);
                    let ca2 = f32(i32((word_a >> (sh + 4u)) & 3u) - 1);
                    let ca3 = f32(i32((word_a >> (sh + 6u)) & 3u) - 1);
                    let cb0 = f32(i32((word_b >> sh) & 3u) - 1);
                    let cb1 = f32(i32((word_b >> (sh + 2u)) & 3u) - 1);
                    let cb2 = f32(i32((word_b >> (sh + 4u)) & 3u) - 1);
                    let cb3 = f32(i32((word_b >> (sh + 6u)) & 3u) - 1);
                    dot_a = dot_a + xv.x * ca0 + xv.y * ca1 + xv.z * ca2 + xv.w * ca3;
                    dot_b = dot_b + xv.x * cb0 + xv.y * cb1 + xv.z * cb2 + xv.w * cb3;
                }
            } else {
                let count = k - base;
                for (var e = 0u; e < count; e = e + 1u) {
                    let xv = x[xbase + base + e];
                    let code_a = (word_a >> (e * 2u)) & 3u;
                    let code_b = (word_b >> (e * 2u)) & 3u;
                    dot_a = dot_a + xv * f32(i32(code_a) - 1);
                    dot_b = dot_b + xv * f32(i32(code_b) - 1);
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
