
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> allow: array<u32>;

var<workgroup> sh_v: array<f32, 256>;
var<workgroup> sh_i: array<u32, 256>;
var<workgroup> sh_b: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let cols = pc.p.y;
    if row >= pc.p.x { return; }
    let tid = lid.x;
    let base = row * cols;
    // Dawn rejects -inf/NaN bitcast constants; lowest f32 plus a sentinel
    // index preserves first-strict-max semantics, and NaN rides pc.p.w.
    // `allow` gates candidacy: masked-out elements are skipped, so their
    // non-finite values cannot poison the row.
    var best = -3.4028234663852886e38;
    var idx = 0xFFFFFFFFu;
    var bad = 0u;
    let use_mask = pc.p.z != 0u;
    for (var c = tid; c < cols; c = c + 256u) {
        if use_mask && (allow[c >> 5u] & (1u << (c & 31u))) == 0u { continue; }
        let v = src[base + c];
        if v == v && abs(v) <= 3.4028234663852886e38 {
            if v > best || (v == best && c < idx) {
                best = v;
                idx = c;
            }
        } else {
            bad = 1u;
        }
    }
    sh_v[tid] = best;
    sh_i[tid] = idx;
    sh_b[tid] = bad;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            let ov = sh_v[tid + s];
            let oi = sh_i[tid + s];
            if ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid]) {
                sh_v[tid] = ov;
                sh_i[tid] = oi;
            }
            sh_b[tid] = sh_b[tid] | sh_b[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        if sh_b[0] != 0u || sh_i[0] == 0xFFFFFFFFu {
            dst[row] = bitcast<f32>(pc.p.w);
        } else {
            dst[row] = f32(sh_i[0]);
        }
    }
}
