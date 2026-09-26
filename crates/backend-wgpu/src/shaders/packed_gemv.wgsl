
struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

// Groups of `G` lanes each reduce one output row; the workgroup covers
// `256/G` rows so narrow matrices still launch enough workgroups to fill the
// GPU. Each lane strides the row's u32 code words (16 weights per word), so
// code loads are coalesced and each lane's activation reads form a contiguous
// 64-byte chunk. The word's scale applies to its 16-weight dot once, matching
// the CPU group-then-scale accumulation. One barrier, then each row's G
// partials reduce serially on a single thread. Replaces the per-output
// workgroup and its barrier tree.
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
    let words = (k + 15u) >> 4u;
    let groups = (k + 127u) >> 7u;
    let xbase = i * k;
    var acc = 0.0;
    if (j < n) {
        let cbase = j * words;
        let sbase = j * groups;
        let vec_ok = pc.q.x != 0u;
        for (var w = lane; w < words; w = w + lanes) {
            let word = codes[cbase + w];
            let scale = scales[sbase + (w >> 3u)];
            let base = w << 4u;
            var dot = 0.0;
            if (vec_ok && base + 16u <= k) {
                // Whole word: four 128-bit activation loads cover 16 elements.
                let b4 = (xbase + base) >> 2u;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let xv = x4[b4 + q];
                    let sh = q << 3u;
                    dot = dot
                        + xv.x * f32(i32((word >> sh) & 3u) - 1)
                        + xv.y * f32(i32((word >> (sh + 2u)) & 3u) - 1)
                        + xv.z * f32(i32((word >> (sh + 4u)) & 3u) - 1)
                        + xv.w * f32(i32((word >> (sh + 6u)) & 3u) - 1);
                }
            } else {
                let count = k - base;
                for (var e = 0u; e < count; e = e + 1u) {
                    let code = (word >> (e * 2u)) & 3u;
                    dot = dot + x[xbase + base + e] * f32(i32(code) - 1);
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
