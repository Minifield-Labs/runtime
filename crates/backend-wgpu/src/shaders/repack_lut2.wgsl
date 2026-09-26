
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

const REMAP: array<u32, 16> = array<u32, 16>(
    11u, 10u, 4u, 0u, 9u, 0u, 1u, 0u, 12u, 2u, 3u, 0u, 0u, 0u, 0u, 0u
);

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    if i >= pc.p.x { return; }
    let w = src[i];
    var o = 0u;
    for (var j = 0u; j < 8u; j += 1u) {
        o |= REMAP[(w >> (j * 4u)) & 15u] << (j * 4u);
    }
    dst[i] = o;
}
