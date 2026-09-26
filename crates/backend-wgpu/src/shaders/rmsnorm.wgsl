
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> alpha: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = flat_wg(wid, numw);
    let tid = lid.x;
    let ncols = pc.p.x;
    let nrows = pc.p.y;
    if row >= nrows { return; }
    let eps = bitcast<f32>(pc.p.z);
    let base = row * ncols;

    var acc = 0.0;
    for (var c = tid; c < ncols; c = c + 256u) {
        let v = src[base + c];
        acc = acc + v * v;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let mean = sh[0] / f32(ncols);
    let scale = inverseSqrt(mean + eps);
    workgroupBarrier();

    for (var c = tid; c < ncols; c = c + 256u) {
        dst[base + c] = scale * src[base + c] * alpha[c];
    }
}
