// E16 ternary two-weight canonical lookup. Two ternary weights map to one
// nibble: bits 0..2 select a table entry, bit 3 negates it. The five-entry
// table over each K-pair is [0, x0, x1, x0+x1, x0-x1] built from staged
// FP32 activations, so a consumer replaces two decoded-weight FMAs with one
// indexed vec4 read plus a sign correction. Tile geometry is token-folded
// at build time: __BM__/__BN__ in {32x64, 64x32}; 256 threads, each owning
// four token rows and two output columns. Eight nibbles per u32 cover a
// K16 slice; the per-128 group scale applies after eight steps.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

const BM: u32 = __BM__;
const BN: u32 = __BN__;
const RG: u32 = BM / 4u;
const CPAIRS: u32 = BN / 2u;
const PAIRS: u32 = 8u;
const ENTRIES: u32 = 5u;
const TABLE_LEN: u32 = PAIRS * ENTRIES * RG;

var<workgroup> inputs: array<f32, BM * 16u>;
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
    let row0 = (flat / columns) * BM;
    let col0 = (flat % columns) * BN;
    if row0 >= m { return; }
    let tid = lid.x;
    let rg = tid / CPAIRS;
    let cp = tid % CPAIRS;
    let words = k / 16u;
    let gk = k / 128u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var p0 = vec4<f32>(0.0);
    var p1 = vec4<f32>(0.0);
    for (var base = 0u; base < k; base += 16u) {
        for (var i = tid; i < BM * 16u; i += 256u) {
            let r = i / 16u;
            let kc = i % 16u;
            var xv = 0.0;
            if row0 + r < m && base + kc < k {
                xv = x_in[(row0 + r) * k + base + kc];
            }
            inputs[i] = xv;
        }
        workgroupBarrier();
        // Build the pair tables. Each of the PAIRS*RG tasks produces five
        // vec4 entries; entries are flattened across threads.
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
        var w0 = 0u;
        var w1 = 0u;
        let n0 = col0 + cp * 2u;
        if n0 < n { w0 = codes[n0 * words + base / 16u]; }
        if n0 + 1u < n { w1 = codes[(n0 + 1u) * words + base / 16u]; }
        for (var j = 0u; j < PAIRS; j += 1u) {
            let nib0 = (w0 >> (j * 4u)) & 15u;
            let t0 = table[(j * ENTRIES + (nib0 & 7u)) * RG + rg];
            p0 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(t0) ^ vec4<u32>((nib0 & 8u) << 28u)
            );
            let nib1 = (w1 >> (j * 4u)) & 15u;
            let t1 = table[(j * ENTRIES + (nib1 & 7u)) * RG + rg];
            p1 += bitcast<vec4<f32>>(
                bitcast<vec4<u32>>(t1) ^ vec4<u32>((nib1 & 8u) << 28u)
            );
        }
        workgroupBarrier();
        if (base + 16u) % 128u == 0u {
            let g = base / 128u;
            let sa0 = scales[min(n0, n - 1u) * gk + g];
            let sa1 = scales[min(n0 + 1u, n - 1u) * gk + g];
            a0 += p0 * sa0;
            a1 += p1 * sa1;
            p0 = vec4<f32>(0.0);
            p1 = vec4<f32>(0.0);
        }
    }
    for (var hi = 0u; hi < 4u; hi += 1u) {
        let r = row0 + rg * 4u + hi;
        let c0 = col0 + cp * 2u;
        if r < m {
            if c0 < n { dst[r * n + c0] = a0[hi]; }
            if c0 + 1u < n { dst[r * n + c0 + 1u] = a1[hi]; }
        }
    }
}
