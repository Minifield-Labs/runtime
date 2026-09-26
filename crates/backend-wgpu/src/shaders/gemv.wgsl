
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> w4: array<vec4<f32>>;

// Groups of `G` lanes each reduce one output row; the workgroup covers
// `256/G` rows so narrow matrices still launch enough workgroups to fill the
// GPU. Each lane strides the row's vec4 words so weight loads stay coalesced,
// then one barrier hands each row's G partials to a single thread for a short
// serial reduce. Replaces the per-output workgroup and its barrier tree.
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = pc.p.x;
    let k = pc.p.y;
    let lanes = pc.p.z;
    let rows_per_wg = 256u / lanes;
    let row0 = flat_wg(wid, numw) * rows_per_wg;
    let tid = lid.x;
    let group = tid / lanes;
    let lane = tid - group * lanes;
    let j = row0 + group;
    let k4 = k >> 2u;
    let kbulk = k4 << 2u;
    var acc = 0.0;
    if (j < n) {
        let rbase = j * k;
        if (rbase & 3u) == 0u {
            let base4 = rbase >> 2u;
            for (var g = lane; g < k4; g = g + lanes) {
                let rv = w4[base4 + g];
                let l = g << 2u;
                acc = acc + rv.x * x[l];
                acc = acc + rv.y * x[l + 1u];
                acc = acc + rv.z * x[l + 2u];
                acc = acc + rv.w * x[l + 3u];
            }
            for (var l = kbulk + lane; l < k; l = l + lanes) {
                acc = acc + x[l] * w[rbase + l];
            }
        } else {
            for (var l = lane; l < k; l = l + lanes) {
                acc = acc + x[l] * w[rbase + l];
            }
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
            dst[o] = s;
        }
    }
}
