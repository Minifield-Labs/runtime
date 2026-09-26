
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> kern: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let c = flat_wg(wid, numw) * 256u + lid.x;
    let hidden = pc.p.x;
    let m = pc.p.y;
    let stride = pc.p.z;
    if c >= hidden { return; }
    let u = proj[c] * proj[c + 2u * hidden];
    var acc = kern[c * (m + 1u) + m] * u;
    for (var j = 0u; j < m; j = j + 1u) {
        acc = acc + kern[c * (m + 1u) + j] * hist[j * hidden + c];
    }
    dst[c] = acc * proj[hidden + c];
    for (var r = 0u; r + 1u < m; r = r + 1u) {
        hist_out[r * hidden + c] = hist[(r + 1u) * hidden + c];
    }
    hist_out[(m - 1u) * hidden + c] = u;
}
