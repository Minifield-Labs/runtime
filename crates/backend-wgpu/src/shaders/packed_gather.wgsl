
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> scales: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let total = pc.p.x * pc.p.y;
    if i >= total { return; }
    let k = pc.p.y;
    let r = i / k;
    let l = i - r * k;
    let srcf = ids[r];
    var src = 0xFFFFFFFFu;
    if srcf >= 0.0 && srcf < 16777216.0 && fract(srcf) == 0.0 {
        src = u32(srcf);
    }
    if src >= pc.p.z {
        dst[i] = bitcast<f32>(pc.p.w);
        return;
    }
    let word = codes[src * (k >> 4u) + (l >> 4u)];
    let byte = (word >> (((l >> 2u) & 3u) << 3u)) & 0xFFu;
    let code = (byte >> ((l & 3u) << 1u)) & 3u;
    dst[i] = f32(i32(code) - 1) * scales[src * (k >> 7u) + (l >> 7u)];
}
