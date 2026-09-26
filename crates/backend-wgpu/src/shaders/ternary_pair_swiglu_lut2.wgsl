// Paired ternary LUT2 projection with the SwiGLU epilogue: one workgroup
// builds the five-entry activation table per K16 tile, then consumes it for
// both the gate and up code streams before folding silu(a) * b into the
// output. Codes carry LUT2 pair nibbles (bits 0..2 index [0, x0, x1,
// x0+x1, x0-x1], bit 3 negates); scales apply per 128-weight group. Tile
// is 32 tokens x 64 output columns per stream, 256 threads, each owning
// four token rows and two columns of each stream.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes_a: array<u32>;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;
@group(0) @binding(5) var<storage, read> codes_b: array<u32>;
@group(0) @binding(6) var<storage, read> scales_b: array<f32>;
@group(0) @binding(7) var<storage, read> x4: array<vec4<f32>>;

const RG: u32 = 8u;
const CPAIRS: u32 = 32u;
const PAIRS: u32 = 8u;
const ENTRIES: u32 = 5u;
const TABLE_LEN: u32 = PAIRS * ENTRIES * RG;

var<workgroup> inputs: array<f32, 512>;
var<workgroup> table: array<vec4<f32>, TABLE_LEN>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) numw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = flat_wg(wid, numw);
    let m = pc.p.x;
    let n = pc.p.y;
    let k = pc.p.z;
    let columns = pc.p.w;
    let row0 = (flat / columns) * 32u;
    let col0 = (flat % columns) * 64u;
    if row0 >= m { return; }
    let tid = lid.x;
    let rg = tid / CPAIRS;
    let cp = tid % CPAIRS;
    let words = k / 16u;
    let gk = k / 128u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0);
    var b1 = vec4<f32>(0.0);
    var pa0 = vec4<f32>(0.0);
    var pa1 = vec4<f32>(0.0);
    var pb0 = vec4<f32>(0.0);
    var pb1 = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 16u) {
        for (var i = tid; i < 512u; i += 256u) {
            let r = i / 16u;
            let kc = i % 16u;
            var xv = 0.0;
            if row0 + r < m && base + kc < k {
                xv = x_in[(row0 + r) * k + base + kc];
            }
            inputs[i] = xv;
        }
        workgroupBarrier();
        for (var e = tid; e < TABLE_LEN; e += 256u) {
            let er = e % RG;
            let rem = e / RG;
            let entry = rem % ENTRIES;
            let pair = rem / ENTRIES;
            var val = vec4<f32>(0.0);
            if entry != 0u {
                let k0 = pair * 2u;
                let x0v = vec4<f32>(
                    inputs[(er * 4u) * 16u + k0],
                    inputs[(er * 4u + 1u) * 16u + k0],
                    inputs[(er * 4u + 2u) * 16u + k0],
                    inputs[(er * 4u + 3u) * 16u + k0],
                );
                let x1v = vec4<f32>(
                    inputs[(er * 4u) * 16u + k0 + 1u],
                    inputs[(er * 4u + 1u) * 16u + k0 + 1u],
                    inputs[(er * 4u + 2u) * 16u + k0 + 1u],
                    inputs[(er * 4u + 3u) * 16u + k0 + 1u],
                );
                if entry == 1u { val = x0v; }
                else if entry == 2u { val = x1v; }
                else if entry == 3u { val = x0v + x1v; }
                else { val = x0v - x1v; }
            }
            table[e] = val;
        }
        workgroupBarrier();
        let n0 = col0 + cp * 2u;
        var wa0 = 0u;
        var wa1 = 0u;
        var wb0 = 0u;
        var wb1 = 0u;
        if n0 < n {
            wa0 = codes_a[n0 * words + base / 16u];
            wb0 = codes_b[n0 * words + base / 16u];
        }
        if n0 + 1u < n {
            wa1 = codes_a[(n0 + 1u) * words + base / 16u];
            wb1 = codes_b[(n0 + 1u) * words + base / 16u];
        }
        for (var j = 0u; j < PAIRS; j += 1u) {
            let nib = (wa0 >> (j * 4u)) & 15u;
            pa0 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(table[(j * ENTRIES + (nib & 7u)) * RG + rg])
                    ^ vec4<u32>((nib & 8u) << 28u)
            );
            let nib1 = (wa1 >> (j * 4u)) & 15u;
            pa1 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(table[(j * ENTRIES + (nib1 & 7u)) * RG + rg])
                    ^ vec4<u32>((nib1 & 8u) << 28u)
            );
            let nib2 = (wb0 >> (j * 4u)) & 15u;
            pb0 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(table[(j * ENTRIES + (nib2 & 7u)) * RG + rg])
                    ^ vec4<u32>((nib2 & 8u) << 28u)
            );
            let nib3 = (wb1 >> (j * 4u)) & 15u;
            pb1 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(table[(j * ENTRIES + (nib3 & 7u)) * RG + rg])
                    ^ vec4<u32>((nib3 & 8u) << 28u)
            );
        }
        workgroupBarrier();
        if (base + 16u) % 128u == 0u {
            let g = base / 128u;
            let cn0 = min(n0, n - 1u);
            let cn1 = min(n0 + 1u, n - 1u);
            a0 += pa0 * scales_a[cn0 * gk + g];
            a1 += pa1 * scales_a[cn1 * gk + g];
            b0 += pb0 * scales_b[cn0 * gk + g];
            b1 += pb1 * scales_b[cn1 * gk + g];
            pa0 = vec4<f32>(0.0);
            pa1 = vec4<f32>(0.0);
            pb0 = vec4<f32>(0.0);
            pb1 = vec4<f32>(0.0);
        }
    }
    for (var hi = 0u; hi < 4u; hi += 1u) {
        let r = row0 + rg * 4u + hi;
        let c0 = col0 + cp * 2u;
        if r < m {
            if c0 < n {
                let g0 = a0[hi];
                dst[r * n + c0] = (g0 / (1.0 + exp(-g0))) * b0[hi];
            }
            if c0 + 1u < n {
                let g1 = a1[hi];
                dst[r * n + c0 + 1u] = (g1 / (1.0 + exp(-g1))) * b1[hi];
            }
        }
    }
}
