
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> u_ext: array<f32>;
@group(0) @binding(2) var<storage, read_write> kernel: array<f32>;
@group(0) @binding(3) var<storage, read_write> proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let tokens = pc.p.x;
    let hidden = pc.p.y;
    let width = pc.p.z;
    let stride = pc.p.w;
    if i >= tokens * hidden { return; }
    let t = i / hidden;
    let c = i - t * hidden;
    var acc = 0.0;
    for (var tap = 0u; tap < width; tap = tap + 1u) {
        acc = acc + kernel[c * width + tap] * u_ext[(t + tap) * hidden + c];
    }
    dst[i] = acc * proj[t * stride + hidden + c];
}
