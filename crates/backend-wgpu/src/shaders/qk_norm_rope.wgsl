
struct Params { p0: vec4<u32>, p1: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> k: array<f32>;
@group(0) @binding(6) var<storage, read_write> qw: array<f32>;
@group(0) @binding(7) var<storage, read_write> kw: array<f32>;

var<workgroup> sh: array<f32, 256>;
var<workgroup> nd: array<f32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = flat_wg(wid, numw);
    let tid = lid.x;
    let q_heads = pc.p0.x;
    let heads_total = pc.p0.y;
    let head_dim = pc.p0.z;
    let half = head_dim >> 1u;
    let q_width = pc.p0.w;
    let kv_width = pc.p1.x;
    let tokens = pc.p1.y;
    let eps = bitcast<f32>(pc.p1.z);
    if w >= tokens * heads_total { return; }
    let t = w / heads_total;
    let h = w - t * heads_total;
    let is_q = h < q_heads;
    let head = select(h - q_heads, h, is_q);
    let width = select(kv_width, q_width, is_q);
    let base = t * width + head * head_dim;

    var acc = 0.0;
    for (var d = tid; d < head_dim; d = d + 256u) {
        let v = select(k[base + d], q[base + d], is_q);
        acc = acc + v * v;
        nd[d] = v;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(sh[0] / f32(head_dim) + eps);
    workgroupBarrier();
    for (var d = tid; d < head_dim; d = d + 256u) {
        let wv = select(kw[d], qw[d], is_q);
        nd[d] = nd[d] * scale * wv;
    }
    workgroupBarrier();

    let table_len = tokens * half;
    for (var c = tid; c < half; c = c + 256u) {
        let cs = t * half + c;
        let co = table[cs];
        let si = table[table_len + cs];
        let a = nd[c];
        let bv = nd[half + c];
        let i1 = base + c;
        let i2 = base + half + c;
        let r1 = a * co - bv * si;
        let r2 = bv * co + a * si;
        if is_q {
            q_out[i1] = r1;
            q_out[i2] = r2;
        } else {
            k_out[i1] = r1;
            k_out[i2] = r2;
        }
    }
}
