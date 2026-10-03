
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
    let k = pc.p.y;
    let words_per_row = k >> 4u;
    let total = pc.p.x * words_per_row;
    if i >= total { return; }
    let r = i / words_per_row;
    let word_column = i - r * words_per_row;
    let l = word_column << 4u;
    let base = r * k + l;
    let srcf = ids[r];
    var src = 0xFFFFFFFFu;
    if srcf >= 0.0 && srcf < 16777216.0 && fract(srcf) == 0.0 {
        src = u32(srcf);
    }
    if src >= pc.p.z {
        let fill = bitcast<f32>(pc.p.w);
        for (var j = 0u; j < 16u; j += 1u) {
            dst[base + j] = fill;
        }
        return;
    }
    let word = codes[src * words_per_row + word_column];
    let scale = scales[src * (k >> 7u) + (l >> 7u)];
    for (var j = 0u; j < 16u; j += 1u) {
        let code = (word >> (j << 1u)) & 3u;
        dst[base + j] = f32(i32(code) - 1) * scale;
    }
}
