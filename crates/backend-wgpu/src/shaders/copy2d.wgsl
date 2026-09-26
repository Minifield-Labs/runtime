
struct Params { p0: vec4<u32>, p1: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let rows = pc.p0.z;
    let cols = pc.p0.w;
    if i >= rows * cols { return; }
    let r = i / cols;
    let c = i - r * cols;
    let src_index = (pc.p0.x + r) * pc.p1.x + pc.p0.y + c;
    let dst_index = (pc.p1.y + r) * pc.p1.z + pc.p1.w + c;
    dst[dst_index] = src[src_index];
}
