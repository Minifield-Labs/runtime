
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    let a = lhs[i];
    let b = rhs[i];
    var r: f32;
    if pc.p.y == 0u {
        r = a + b;
    } else {
        r = a * b;
    }
    dst[i] = r;
}
