// E20 NF4 activation-product lookup, lower-confidence variant. The table
// holds X[m,k] * NF4[q] for all sixteen centroids, so a consumer replaces
// decode+multiply with an indexed add of an activation-derived product.
// Group scales still apply per output channel at each 128-weight boundary.
// BM16 x BN64 x BK8: 256 threads, two token rows and two output columns
// per thread, vec2 token-contiguous table entries. NF4 needs no repack:
// the standard four-bit stream is read directly.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

const BM: u32 = 16u;
const BN: u32 = 64u;
const BK: u32 = 8u;
const RG: u32 = BM / 2u;
const CPAIRS: u32 = BN / 2u;
const TABLE_LEN: u32 = BK * 16u * RG;

var<workgroup> inputs: array<f32, BM * BK>;
var<workgroup> table: array<vec2<f32>, TABLE_LEN>;

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
    let words = k / 8u;
    let gk = k / 128u;
    var a0 = vec2<f32>(0.0);
    var a1 = vec2<f32>(0.0);
    var p0 = vec2<f32>(0.0);
    var p1 = vec2<f32>(0.0);
    for (var base = 0u; base < k; base += BK) {
        for (var i = tid; i < BM * BK; i += 256u) {
            let r = i / BK;
            let kc = i % BK;
            var xv = 0.0;
            if row0 + r < m && base + kc < k {
                xv = x_in[(row0 + r) * k + base + kc];
            }
            inputs[i] = xv;
        }
        workgroupBarrier();
        // Table entry e -> (rg, centroid q, local k). Only the threads whose
        // entry index fits build entries.
        for (var e = tid; e < TABLE_LEN; e += 256u) {
            let er = e % RG;
            let rem = e / RG;
            let q = rem % 16u;
            let kl = rem / 16u;
            table[e] = vec2<f32>(
                inputs[(er * 2u) * BK + kl],
                inputs[(er * 2u + 1u) * BK + kl],
            ) * NF4[q];
        }
        workgroupBarrier();
        var w0 = 0u;
        var w1 = 0u;
        let n0 = col0 + cp * 2u;
        if n0 < n { w0 = codes[n0 * words + base / 8u]; }
        if n0 + 1u < n { w1 = codes[(n0 + 1u) * words + base / 8u]; }
        for (var j = 0u; j < BK; j += 1u) {
            p0 += table[(j * 16u + ((w0 >> (j * 4u)) & 15u)) * RG + rg];
            p1 += table[(j * 16u + ((w1 >> (j * 4u)) & 15u)) * RG + rg];
        }
        workgroupBarrier();
        if (base + BK) % 128u == 0u {
            let g = base / 128u;
            let sa0 = scales[min(n0, n - 1u) * gk + g];
            let sa1 = scales[min(n0 + 1u, n - 1u) * gk + g];
            a0 += p0 * sa0;
            a1 += p1 * sa1;
            p0 = vec2<f32>(0.0);
            p1 = vec2<f32>(0.0);
        }
    }
    for (var hi = 0u; hi < 2u; hi += 1u) {
        let r = row0 + rg * 2u + hi;
        let c0 = col0 + cp * 2u;
        if r < m {
            if c0 < n { dst[r * n + c0] = a0[hi]; }
            if c0 + 1u < n { dst[r * n + c0 + 1u] = a1[hi]; }
        }
    }
}
