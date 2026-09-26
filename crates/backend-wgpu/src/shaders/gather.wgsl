
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

// ids are f32 so a device-side argmax output feeds this kernel directly. A
// non-finite, fractional, or out-of-range selector poisons its output row
// with NaN, which downstream finiteness checks surface as a failure.
fn row_id(r: u32) -> u32 {
    let idf = ids[r];
    if idf < 0.0 || idf >= 16777216.0 || fract(idf) != 0.0 {
        return 0xFFFFFFFFu;
    }
    return u32(idf);
}

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
    let c = i - r * pc.p.y;
    let id = row_id(r);
    if id >= pc.p.z {
        // NaN arrives through the params uniform: Dawn rejects non-finite
        // bitcast constants during shader const-eval.
        dst[i] = bitcast<f32>(pc.p.w);
        return;
    }
    dst[i] = table[id * pc.p.y + c];
}
