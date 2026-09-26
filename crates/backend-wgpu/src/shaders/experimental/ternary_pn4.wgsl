// E17 ternary four-weight positive/negative subset lookup. Four ternary
// weights encode as one byte: low nibble marks +1 positions, high nibble
// marks -1 positions (P & N == 0). The consumer builds sixteen subset sums
// over each activation quartet and computes dot4 = L[P] - L[N], replacing
// four decoded-weight FMAs with two indexed vec4 reads, a subtract, and an
// accumulate. BM32 x BN64 x BK16, 256 threads, four rows and two output
// columns per thread. Per-128 group scale applies after eight steps.
struct Params { p: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x_in: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;

const BM: u32 = 32u;
const BN: u32 = 64u;
const RG: u32 = BM / 4u;
const CPAIRS: u32 = BN / 2u;
const QUARTETS: u32 = 4u;
const SUBSETS: u32 = 16u;
const TABLE_LEN: u32 = QUARTETS * SUBSETS * RG;

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
        // Build the subset-sum tables: entry e -> (rg, subset, quartet).
        // Each entry sums the x vec4s whose position bit is set.
        for (var e = tid; e < TABLE_LEN; e += 256u) {
            let er = e % RG;
            let rem = e / RG;
            let subset = rem % SUBSETS;
            let q = rem / SUBSETS;
            var val = vec4<f32>(0.0);
            for (var j = 0u; j < 4u; j += 1u) {
                if ((subset >> j) & 1u) == 1u {
                    let kc = q * 4u + j;
                    val += vec4<f32>(
                        inputs[(er * 4u) * 16u + kc],
                        inputs[(er * 4u + 1u) * 16u + kc],
                        inputs[(er * 4u + 2u) * 16u + kc],
                        inputs[(er * 4u + 3u) * 16u + kc],
                    );
                }
            }
            table[e] = val;
        }
        workgroupBarrier();
        var w0 = 0u;
        var w1 = 0u;
        let n0 = col0 + cp * 2u;
        if n0 < n { w0 = codes[n0 * words + base / 16u]; }
        if n0 + 1u < n { w1 = codes[(n0 + 1u) * words + base / 16u]; }
        for (var j = 0u; j < QUARTETS; j += 1u) {
            let b0 = (w0 >> (j * 8u)) & 255u;
            p0 += table[(j * SUBSETS + (b0 & 15u)) * RG + rg]
                - table[(j * SUBSETS + (b0 >> 4u)) * RG + rg];
            let b1 = (w1 >> (j * 8u)) & 255u;
            p1 += table[(j * SUBSETS + (b1 & 15u)) * RG + rg]
                - table[(j * SUBSETS + (b1 >> 4u)) * RG + rg];
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
