
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> scales: array<f32>;

fn signed_byte(word: u32, shift: u32) -> f32 {
    return f32(bitcast<i32>(((word >> shift) & 255u) << 24u) >> 24u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_wg(wid, numw) * 256u + lid.x;
    let words_per_row = pc.p.y >> 2u;
    let total = pc.p.x * words_per_row;
    if i >= total { return; }
    let k = pc.p.y;
    let r = i / words_per_row;
    let word_in_row = i - r * words_per_row;
    let srcf = ids[r];
    var src = 0xFFFFFFFFu;
    if srcf >= 0.0 && srcf < 16777216.0 && fract(srcf) == 0.0 {
        src = u32(srcf);
    }
    if src >= pc.p.z {
        dst[i] = vec4<f32>(bitcast<f32>(pc.p.w));
        return;
    }
    let word = codes[src * words_per_row + word_in_row];
    let scale = scales[src * (k >> 7u) + (word_in_row >> 5u)];
    dst[i] = vec4<f32>(
        signed_byte(word, 0u),
        signed_byte(word, 8u),
        signed_byte(word, 16u),
        signed_byte(word, 24u),
    ) * vec4<f32>(scale);
}
