
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let heads = pc.p.x;
    let half = pc.p.y;
    let tokens = pc.p.z;
    let span = heads * half;
    if i >= tokens * span { return; }
    let t = i / span;
    let rem = i - t * span;
    let h = rem / half;
    let c = rem - h * half;
    let head_dim = 2u * half;
    let packed = heads * head_dim;
    let i1 = t * packed + h * head_dim + c;
    let i2 = i1 + half;
    let cs = t * half + c;
    let co = table[cs];
    let si = table[tokens * half + cs];
    let a = src[i1];
    let b = src[i2];
    dst[i1] = a * co - b * si;
    dst[i2] = b * co + a * si;
}
