
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    let g = gate[i];
    let sigmoid = 1.0 / (1.0 + exp(-g));
    dst[i] = g * sigmoid * up[i];
}
