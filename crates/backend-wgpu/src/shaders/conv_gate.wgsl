
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_ext: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let hist_elems = pc.p.x;
    let total = pc.p.y;
    let hidden = pc.p.z;
    let stride = pc.p.w;
    if i >= total { return; }
    if i < hist_elems {
        u_ext[i] = hist[i];
    } else {
        let j = i - hist_elems;
        let t = j / hidden;
        let c = j - t * hidden;
        let base = t * stride + c;
        u_ext[i] = proj[base] * proj[base + 2u * hidden];
    }
}
