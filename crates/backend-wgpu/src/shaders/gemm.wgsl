
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;

const TILE: u32 = 16u;
var<workgroup> lt: array<array<f32, 16>, 16>;
var<workgroup> rt: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = pc.p.x;
    let n = pc.p.y;
    let k = pc.p.z;
    let row_tile = wid.z * numw.y + wid.y;
    let row = row_tile * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let ly = lid.y;
    let lx = lid.x;

    var acc = 0.0;
    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let l_x = t * TILE + lx;
        let l_w = t * TILE + ly;
        if row < m && l_x < k {
            lt[ly][lx] = x[row * k + l_x];
        } else {
            lt[ly][lx] = 0.0;
        }
        if col < n && l_w < k {
            rt[ly][lx] = w[col * k + l_w];
        } else {
            rt[ly][lx] = 0.0;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + lt[ly][kk] * rt[kk][lx];
        }
        workgroupBarrier();
    }

    if row < m && col < n {
        dst[row * n + col] = acc;
    }
}
