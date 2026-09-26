
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let r = i / pc.p.y;
    let k = i - r * pc.p.y;
    dst[i] = src[r * pc.p.z + cols[k]];
}
