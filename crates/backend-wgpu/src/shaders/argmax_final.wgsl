
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> partials: array<vec2<f32>>;

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
    let blocks = pc.p.z;
    if row >= pc.p.x { return; }
    let tid = lid.x;
    let base = row * blocks;
    var best = -3.4028234663852886e38;
    var idx = 0xFFFFFFFFu;
    var bad = 0u;
    for (var b = tid; b < blocks; b = b + 256u) {
        let p = partials[base + b];
        if p.x != p.x {
            bad = 1u;
        } else if p.x > best || (p.x == best && u32(p.y) < idx) {
            best = p.x;
            idx = u32(p.y);
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
        if sh_b[0] != 0u || sh_i[0] >= 16777216u {
            dst[row] = bitcast<f32>(pc.p.w);
        } else {
            dst[row] = f32(sh_i[0]);
        }
    }
}
