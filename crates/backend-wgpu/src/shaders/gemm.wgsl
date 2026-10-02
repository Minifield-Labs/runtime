
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;

const TILE: u32 = 16u;
const OUTPUT_TILE: u32 = 32u;
var<workgroup> lt: array<array<f32, 16>, 32>;
var<workgroup> rt: array<array<f32, 32>, 16>;

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
    let row = row_tile * OUTPUT_TILE + lid.y;
    let col = wid.x * OUTPUT_TILE + lid.x;
    let ly = lid.y;
    let lx = lid.x;

    // Each invocation owns rows y/y+16 and columns x/x+16.
    var acc = vec4<f32>(0.0);
    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let l_x = t * TILE + lx;
        let l_w = t * TILE + ly;
        for (var offset = 0u; offset < 2u; offset += 1u) {
            let r = row + offset * TILE;
            let c = col + offset * TILE;
            if r < m && l_x < k {
                lt[ly + offset * TILE][lx] = x[r * k + l_x];
            } else {
                lt[ly + offset * TILE][lx] = 0.0;
            }
            if c < n && l_w < k {
                rt[ly][lx + offset * TILE] = w[c * k + l_w];
            } else {
                rt[ly][lx + offset * TILE] = 0.0;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            let x0 = lt[ly][kk];
            let x1 = lt[ly + TILE][kk];
            let w0 = rt[kk][lx];
            let w1 = rt[kk][lx + TILE];
            acc += vec4(x0 * w0, x0 * w1, x1 * w0, x1 * w1);
        }
        workgroupBarrier();
    }

    if row < m && col < n {
        dst[row * n + col] = acc.x;
    }
    if row < m && col + TILE < n {
        dst[row * n + col + TILE] = acc.y;
    }
    if row + TILE < m && col < n {
        dst[(row + TILE) * n + col] = acc.z;
    }
    if row + TILE < m && col + TILE < n {
        dst[(row + TILE) * n + col + TILE] = acc.w;
    }
}
