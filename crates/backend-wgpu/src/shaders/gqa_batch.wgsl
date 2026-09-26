
struct Params { p0: vec4<u32>, p1: vec4<u32>, p2: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let tid = lid.x;
    let group_size = pc.p0.x;
    let head_dim = pc.p0.y;
    let kv_width = pc.p0.z;
    let q_width = pc.p0.w;
    let init = pc.p1.x;
    let score_stride = pc.p1.z;
    let scale = bitcast<f32>(pc.p1.w);
    let query_heads = pc.p2.x;
    let block_tokens = pc.p2.y;
    if flat >= block_tokens * query_heads { return; }
    let token = pc.p2.z + flat / query_heads;
    let qh = flat - (flat / query_heads) * query_heads;
    let kvh = qh / group_size;
    let visible = init + token + 1u;
    let srow = flat * score_stride;
    let qbase = token * q_width + qh * head_dim;

    // Scaled dot scores for every visible key.
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            var kv: f32;
            if ki < init {
                kv = kcache[ki * kv_width + kvh * head_dim + d];
            } else {
                kv = k[(ki - init) * kv_width + kvh * head_dim + d];
            }
            dot = dot + q[qbase + d] * kv;
        }
        scores[srow + ki] = dot * scale;
    }
    workgroupBarrier();

    // Row max.
    var partial = -3.402823466e+38;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        partial = max(partial, scores[srow + ki]);
    }
    sh[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = max(sh[tid], sh[tid + s]); }
        workgroupBarrier();
    }
    let maximum = sh[0];
    workgroupBarrier();

    // exp(score - max) in place, plus its denominator.
    var denom_part = 0.0;
    for (var ki = tid; ki < visible; ki = ki + 256u) {
        let e = exp(scores[srow + ki] - maximum);
        scores[srow + ki] = e;
        denom_part = denom_part + e;
    }
    sh[tid] = denom_part;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let denom = sh[0];

    storageBarrier();

    // Weighted value sum per dimension.
    for (var d = tid; d < head_dim; d = d + 256u) {
        var acc = 0.0;
        for (var ki = 0u; ki < visible; ki = ki + 1u) {
            var vv: f32;
            if ki < init {
                vv = vcache[ki * kv_width + kvh * head_dim + d];
            } else {
                vv = v[(ki - init) * kv_width + kvh * head_dim + d];
            }
            acc = acc + scores[srow + ki] * vv;
        }
        dst[qbase + d] = acc / denom;
    }
}
