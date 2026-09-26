//! Low-bit grouped-arithmetic experiment tests
//! (`WgpuBackend::experimental_packed_linear`, an explicit research API).
//!
//! Covers the variant encodings exhaustively on CPU, then verifies each GPU
//! candidate kernel against the standard CPU decode of the same weights.
//! The selector lives on the backend instance, so tests need no global env.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names
)]

mod common;

use minifield_backend_cpu::CpuBackend;
use minifield_backend_wgpu::experimental::LowbitsKernel;
use minifield_backend_wgpu::{WgpuBackend, WgpuBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, InferenceCompletion, ResourceLimits, Shape,
};

type Packer = dyn Fn(&[f32], usize, usize) -> (Vec<u8>, Vec<f32>);

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 1 << 34,
        max_pending_operations: 1_024,
    }
}

fn gpu() -> Option<WgpuBackend> {
    common::gpu(limits())
}

fn cpu() -> CpuBackend {
    CpuBackend::new(7, limits())
}

/// Deterministic xorshift values in [-1, 1).
fn values(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state % 1_000_003) as f32 / 1_000_003.0;
        out.push(unit.mul_add(2.0, -1.0));
    }
    out
}

fn read(backend: &WgpuBackend, buffer: &WgpuBuffer) -> Vec<f32> {
    backend.read_f32(buffer).expect("wgpu readback")
}

fn assert_close_ctx(left: &[f32], right: &[f32], abs: f32, rel: f32, ctx: &str) {
    assert_eq!(left.len(), right.len(), "length mismatch ({ctx})");
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        let diff = (a - b).abs();
        let scale = b.abs().mul_add(rel, abs);
        assert!(
            a.is_finite() && b.is_finite() && diff <= scale,
            "{ctx} element {index}: {a} vs {b} exceeds tolerance (diff {diff})"
        );
    }
}

/// Pack one 128-weight group per `minifield.ternary.v1` (mirrors parity.rs).
fn pack_group(row: &[f32]) -> ([u8; 32], f32) {
    let scale = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let mut codes = [0_u8; 32];
    for (j, w) in row.iter().enumerate() {
        let q = if scale == 0.0 {
            1_u8
        } else {
            ((w / scale).round() + 1.0).clamp(0.0, 3.0) as u8
        };
        codes[j / 4] |= q << (2 * (j % 4));
    }
    (codes, scale)
}

fn pack_ternary(weights: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(rows * k / 4);
    let mut scales = Vec::with_capacity(rows * k / 128);
    for chunk in weights.as_chunks::<128>().0 {
        let (group_codes, scale) = pack_group(chunk);
        codes.extend_from_slice(&group_codes);
        scales.push(scale);
    }
    (codes, scales)
}

/// Decode one ternary code word (16 weights) into code values 0..3.
fn ternary_codes(word: u32) -> [u8; 16] {
    let mut out = [0_u8; 16];
    for (j, c) in out.iter_mut().enumerate() {
        *c = ((word >> (2 * j)) & 3) as u8;
    }
    out
}

/// E16 remap: index `c0 | (c1 << 2)` over original two-bit codes -> nibble
/// (index 0..4 in bits 0..2, negate in bit 3). 0xFF marks unused codes 3.
const LUT2_REMAP: [u8; 16] = [
    11, 10, 4, 0xFF, 9, 0, 1, 0xFF, 12, 2, 3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// Repack a ternary code stream into E16 pair nibbles: same byte length,
/// eight four-bit pair entries per u32 word.
fn repack_lut2(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len());
    for chunk in codes.as_chunks::<4>().0 {
        let word = u32::from_le_bytes(*chunk);
        let mut packed = 0_u32;
        for j in 0..8 {
            let idx = (word >> (4 * j)) & 15;
            let nib = LUT2_REMAP[idx as usize];
            assert_ne!(nib, 0xFF, "lut2 repack sees invalid ternary code");
            packed |= u32::from(nib) << (4 * j);
        }
        out.extend_from_slice(&packed.to_le_bytes());
    }
    out
}

/// Repack a ternary code stream into E17 P/N bytes: byte per quartet,
/// low nibble marks +1 positions, high nibble marks -1 positions.
fn repack_pn4(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len());
    for chunk in codes.as_chunks::<4>().0 {
        let word = u32::from_le_bytes(*chunk);
        let t = ternary_codes(word);
        let mut packed = 0_u32;
        for q in 0..4 {
            let mut p = 0_u32;
            let mut n = 0_u32;
            for j in 0..4 {
                match t[q * 4 + j] {
                    0 => n |= 1 << j,
                    2 => p |= 1 << j,
                    1 => {}
                    c => panic!("pn4 repack sees invalid ternary code {c}"),
                }
            }
            packed |= (p | (n << 4)) << (8 * q);
        }
        out.extend_from_slice(&packed.to_le_bytes());
    }
    out
}

/// CPU-side E16 consumer: pair nibble -> signed table lookup.
fn lut2_pair_dot(x0: f32, x1: f32, nib: u8) -> f32 {
    let entry = nib & 7;
    let base = match entry {
        0 => 0.0,
        1 => x0,
        2 => x1,
        3 => x0 + x1,
        _ => x0 - x1,
    };
    if nib & 8 != 0 { -base } else { base }
}

/// CPU-side E17 consumer: byte -> subset sums difference.
fn pn4_quartet_dot(x: [f32; 4], byte: u8) -> f32 {
    let subset = |mask: u8| {
        (0..4)
            .filter(|j| mask & (1 << j) != 0)
            .fold(0.0, |a, j| a + x[j])
    };
    subset(byte & 15) - subset(byte >> 4)
}

#[test]
fn lut2_encoding_covers_all_pairs() {
    // All nine valid (c0, c1) code pairs: decoded sign*entry equals the
    // represented dot product for arbitrary activations.
    let mut covered = 0;
    for c0 in 0u8..4 {
        for c1 in 0u8..4 {
            let idx = c0 | (c1 << 2);
            let nib = LUT2_REMAP[idx as usize];
            if c0 == 3 || c1 == 3 {
                assert_eq!(nib, 0xFF);
                continue;
            }
            covered += 1;
            for (x0, x1) in [(0.3_f32, -0.7_f32), (-1.25, 0.9), (2.0, 2.0)] {
                let t0 = f32::from(c0) - 1.0;
                let t1 = f32::from(c1) - 1.0;
                assert_eq!(lut2_pair_dot(x0, x1, nib), t0 * x0 + t1 * x1);
            }
        }
    }
    assert_eq!(covered, 9);
}

#[test]
fn pn4_encoding_covers_all_quartets() {
    // All 81 ternary quartets: P/N masks disjoint and L[P] - L[N] exact.
    let mut covered = 0;
    for flat in 0..256_u16 {
        let t = [
            (flat & 3) as u8,
            ((flat >> 2) & 3) as u8,
            ((flat >> 4) & 3) as u8,
            ((flat >> 6) & 3) as u8,
        ];
        if t.contains(&3) {
            continue;
        }
        covered += 1;
        let mut p = 0_u8;
        let mut n = 0_u8;
        for (j, c) in t.iter().enumerate() {
            match c {
                0 => n |= 1 << j,
                2 => p |= 1 << j,
                _ => {}
            }
        }
        assert_eq!(p & n, 0);
        let byte = p | (n << 4);
        for x in [[0.3_f32, -0.7, 1.1, -0.2], [-0.9, 0.4, 0.4, 0.4]] {
            let dot: f32 = (0..4).map(|j| (f32::from(t[j]) - 1.0) * x[j]).sum();
            // Subset-sum grouping reorders the adds; compare to rounding.
            let got = pn4_quartet_dot(x, byte);
            assert!((got - dot).abs() <= 1e-6, "{byte:08b}: {got} vs {dot}");
        }
    }
    assert_eq!(covered, 81);
}

#[test]
fn repackers_preserve_stream_length() {
    let weights = values(43, 128 * 4);
    let (codes, _) = pack_ternary(&weights, 4, 128);
    assert_eq!(repack_lut2(&codes).len(), codes.len());
    assert_eq!(repack_pn4(&codes).len(), codes.len());
}

/// Random ternary-valued weights; every 128-group carries an absmax member
/// so its scale is nonzero.
fn ternary_grid(n: u64, k: u64) -> Vec<f32> {
    let mut w = vec![0.0_f32; (n * k) as usize];
    for r in 0..n as usize {
        for g in 0..(k / 128) as usize {
            let s = 0.5 + ((r + g) % 3) as f32 * 0.25;
            for i in 0..128_usize {
                let c = ((r * 7 + g + i) % 3) as i32 - 1;
                w[r * k as usize + g * 128 + i] = c as f32 * s;
            }
        }
    }
    w
}

fn pack_ternary_direct(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    pack_ternary(w, n, k)
}

fn pack_lut2_stream(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (codes, scales) = pack_ternary(w, n, k);
    (repack_lut2(&codes), scales)
}

fn pack_pn4_stream(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (codes, scales) = pack_ternary(w, n, k);
    (repack_pn4(&codes), scales)
}

const NF4: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

fn pack_group_nf4(row: &[f32]) -> ([u8; 64], f32) {
    let scale = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let mut codes = [0_u8; 64];
    for (j, w) in row.iter().enumerate() {
        let v = if scale == 0.0 { 0.0 } else { w / scale };
        let code = NF4
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - v).abs().total_cmp(&(*b - v).abs()))
            .map(|(i, _)| i as u8)
            .expect("nonempty codebook");
        codes[j / 2] |= code << (4 * (j % 2));
    }
    (codes, scale)
}

fn pack_nf4_stream(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(n * k / 2);
    let mut scales = Vec::with_capacity(n * k / 128);
    for chunk in w.as_chunks::<128>().0 {
        let (group_codes, scale) = pack_group_nf4(chunk);
        codes.extend_from_slice(&group_codes);
        scales.push(scale);
    }
    (codes, scales)
}

// All variants share one physical Metal adapter; parallel submissions starve
// the fixed readback poll budget, so GPU parity runs serialize here.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// GPU-vs-CPU parity for one variant on the same represented weights; the
/// GPU codes may be repacked while the CPU decodes the standard stream.
fn run_variant(
    variant: &str,
    shapes: &[(u64, u64, u64)],
    code_bytes_per_k: u64,
    weights: &dyn Fn(u64, u64) -> Vec<f32>,
    pack_gpu: &Packer,
    pack_cpu: &Packer,
) {
    let _serial = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = gpu() else { return };
    let experiment: LowbitsKernel = variant.parse().expect("known variant");
    let mut reference = cpu();
    for &(m, k, n) in shapes {
        let w = weights(n, k);
        let (g_codes, g_scales) = pack_gpu(&w, n as usize, k as usize);
        let (c_codes, c_scales) = pack_cpu(&w, n as usize, k as usize);
        assert_eq!(g_scales, c_scales, "scale streams diverge");
        run_single(
            &mut backend,
            &mut reference,
            m,
            k,
            n,
            k / code_bytes_per_k,
            &g_codes,
            &c_codes,
            &g_scales,
            experiment,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_single(
    backend: &mut WgpuBackend,
    reference: &mut CpuBackend,
    m: u64,
    k: u64,
    n: u64,
    code_width: u64,
    gpu_codes: &[u8],
    cpu_codes: &[u8],
    scales: &[f32],
    experiment: LowbitsKernel,
) {
    let input = values(41, (m * k) as usize);
    let gpu_in = backend
        .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
        .expect("gpu in");
    let cpu_in = reference
        .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
        .expect("cpu in");
    let gpu_c = backend
        .upload_experimental_codes(
            Shape::new(&[n, code_width]).expect("codes"),
            gpu_codes,
            experiment,
        )
        .expect("gpu codes");
    let cpu_c = reference
        .upload_u8_classified(
            Shape::new(&[n, code_width]).expect("codes"),
            cpu_codes,
            AllocationClass::Weight,
        )
        .expect("cpu codes");
    let gpu_s = backend
        .upload_f32_classified(
            Shape::new(&[n, k / 128]).expect("scales"),
            scales,
            AllocationClass::Weight,
        )
        .expect("gpu scales");
    let cpu_s = reference
        .upload_f32_classified(
            Shape::new(&[n, k / 128]).expect("scales"),
            scales,
            AllocationClass::Weight,
        )
        .expect("cpu scales");
    let mut gpu_out = backend
        .allocate_f32(Shape::new(&[m, n]).expect("out"))
        .expect("gpu out");
    let mut cpu_out = reference
        .allocate_f32(Shape::new(&[m, n]).expect("out"))
        .expect("cpu out");
    backend
        .experimental_packed_linear(&mut gpu_out, &gpu_in, &gpu_c, &gpu_s, experiment)
        .expect("gpu packed linear");
    reference
        .packed_linear(&mut cpu_out, &cpu_in, &cpu_c, &cpu_s)
        .expect("cpu packed linear");
    assert_close_ctx(
        &read(backend, &gpu_out),
        cpu_out.as_slice(),
        1e-4,
        1e-4,
        &format!("m={m} k={k} n={n}"),
    );
}

// GPU shapes: m=95/96/97 brackets the tile dispatch boundary (only for
// variants on the standard encodings), 299/346 are the classifier's
// cached-tail and full-prefill lengths, 320/384 aligned padding controls.
const STD_SHAPES: &[(u64, u64, u64)] = &[
    (95, 256, 64),
    (96, 256, 64),
    (97, 384, 64),
    (299, 512, 256),
    (346, 2560, 1024),
    (320, 1024, 2560),
];
// Repacked encodings cannot ride the m<96 fallback (it reads raw two-bit
// codes), so lookup variants only cover m>=96.
const LUT_SHAPES: &[(u64, u64, u64)] = &[
    (96, 256, 64),
    (97, 384, 64),
    (299, 512, 256),
    (346, 2560, 1024),
    (320, 1024, 2560),
];

#[test]
fn scale128_control_ternary_matches_cpu() {
    run_variant(
        "scale128_control",
        STD_SHAPES,
        4,
        &ternary_grid,
        &pack_ternary_direct,
        &pack_ternary_direct,
    );
}

#[test]
fn scale128_control_nf4_matches_cpu() {
    run_variant(
        "scale128_control",
        STD_SHAPES,
        2,
        &|n, k| values(43, (n * k) as usize),
        &pack_nf4_stream,
        &pack_nf4_stream,
    );
}

#[test]
fn ternary_sign_matches_cpu() {
    run_variant(
        "ternary_sign",
        STD_SHAPES,
        4,
        &ternary_grid,
        &pack_ternary_direct,
        &pack_ternary_direct,
    );
}

#[test]
fn ternary_sign_sel_matches_cpu() {
    run_variant(
        "ternary_sign_sel",
        STD_SHAPES,
        4,
        &ternary_grid,
        &pack_ternary_direct,
        &pack_ternary_direct,
    );
}

#[test]
fn ternary_lut2_matches_cpu() {
    run_variant(
        "ternary_lut2",
        LUT_SHAPES,
        4,
        &ternary_grid,
        &pack_lut2_stream,
        &pack_ternary_direct,
    );
}

#[test]
fn ternary_lut2_alt_matches_cpu() {
    run_variant(
        "ternary_lut2_64x32",
        LUT_SHAPES,
        4,
        &ternary_grid,
        &pack_lut2_stream,
        &pack_ternary_direct,
    );
}

#[test]
fn ternary_pn4_matches_cpu() {
    run_variant(
        "ternary_pn4",
        LUT_SHAPES,
        4,
        &ternary_grid,
        &pack_pn4_stream,
        &pack_ternary_direct,
    );
}

/// Production path parity: the load-time repack kernel is the only producer
/// of LUT2 streams, so this consumes `repack_ternary_lut2` output through
/// `packed_linear_lut2` and compares against the CPU decode of the raw
/// stream the repack read.
#[test]
fn lut2_repack_kernel_matches_cpu() {
    let _serial = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    for &(m, k, n) in LUT_SHAPES {
        let w = ternary_grid(n, k);
        let (raw_codes, scales) = pack_ternary_direct(&w, n as usize, k as usize);
        let input = values(41, (m * k) as usize);
        let x = backend
            .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
            .expect("gpu in");
        let raw = backend
            .upload_u8_classified(
                Shape::new(&[n, k / 4]).expect("codes"),
                &raw_codes,
                AllocationClass::Weight,
            )
            .expect("raw codes");
        let lut2 = backend.repack_ternary_lut2(&raw).expect("repack");
        let gs = backend
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &scales,
                AllocationClass::Weight,
            )
            .expect("gpu scales");
        let mut out = backend
            .allocate_f32(Shape::new(&[m, n]).expect("out"))
            .expect("gpu out");
        backend
            .packed_linear_lut2(&mut out, &x, &lut2, &gs)
            .expect("gpu lut2 linear");

        let cx = reference
            .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
            .expect("cpu in");
        let cc = reference
            .upload_u8_classified(
                Shape::new(&[n, k / 4]).expect("codes"),
                &raw_codes,
                AllocationClass::Weight,
            )
            .expect("cpu codes");
        let cs = reference
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &scales,
                AllocationClass::Weight,
            )
            .expect("cpu scales");
        let mut cout = reference
            .allocate_f32(Shape::new(&[m, n]).expect("out"))
            .expect("cpu out");
        reference
            .packed_linear(&mut cout, &cx, &cc, &cs)
            .expect("cpu packed linear");
        assert_close_ctx(
            &read(&backend, &out),
            cout.as_slice(),
            1e-4,
            1e-4,
            &format!("m={m} k={k} n={n}"),
        );
    }
}

/// Paired gate/up LUT2 + fused `SwiGLU` against the CPU pair decode of the
/// same raw streams. n=1024/k=2560 is the classifier's FFN shape rotated
/// (the producer reads k=1024 inputs to n=2560 outputs); both orientations
/// are covered so the shared-table path is checked at production scale.
const PAIR_SHAPES: &[(u64, u64, u64)] = &[
    (96, 256, 64),
    (97, 384, 64),
    (299, 512, 256),
    (346, 1024, 2560),
];

#[test]
fn pair_swiglu_lut2_matches_cpu() {
    let _serial = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    for &(m, k, n) in PAIR_SHAPES {
        let (gate_codes, gate_scales) =
            pack_ternary_direct(&ternary_grid(n, k), n as usize, k as usize);
        let (up_codes, up_scales) =
            pack_ternary(&values(43, (n * k) as usize), n as usize, k as usize);
        let input = values(41, (m * k) as usize);

        let x = backend
            .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
            .expect("gpu in");
        let upload_codes = |backend: &mut WgpuBackend, codes: &[u8]| {
            let raw = backend
                .upload_u8_classified(
                    Shape::new(&[n, k / 4]).expect("codes"),
                    codes,
                    AllocationClass::Weight,
                )
                .expect("raw codes");
            backend.repack_ternary_lut2(&raw).expect("repack")
        };
        let gate_lut2 = upload_codes(&mut backend, &gate_codes);
        let up_lut2 = upload_codes(&mut backend, &up_codes);
        let gsa = backend
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &gate_scales,
                AllocationClass::Weight,
            )
            .expect("gate scales");
        let usa = backend
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &up_scales,
                AllocationClass::Weight,
            )
            .expect("up scales");
        let mut out = backend
            .allocate_f32(Shape::new(&[m, n]).expect("out"))
            .expect("gpu out");
        backend
            .packed_swiglu_pair_lut2(&mut out, &x, &gate_lut2, &gsa, &up_lut2, &usa)
            .expect("gpu lut2 pair swiglu");

        let cx = reference
            .upload_f32(Shape::new(&[m, k]).expect("in"), &input)
            .expect("cpu in");
        let cg = reference
            .upload_u8_classified(
                Shape::new(&[n, k / 4]).expect("codes"),
                &gate_codes,
                AllocationClass::Weight,
            )
            .expect("cpu gate codes");
        let cu = reference
            .upload_u8_classified(
                Shape::new(&[n, k / 4]).expect("codes"),
                &up_codes,
                AllocationClass::Weight,
            )
            .expect("cpu up codes");
        let cgs = reference
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &gate_scales,
                AllocationClass::Weight,
            )
            .expect("cpu gate scales");
        let cus = reference
            .upload_f32_classified(
                Shape::new(&[n, k / 128]).expect("scales"),
                &up_scales,
                AllocationClass::Weight,
            )
            .expect("cpu up scales");
        let mut cout = reference
            .allocate_f32(Shape::new(&[m, n]).expect("out"))
            .expect("cpu out");
        reference
            .packed_swiglu_pair(&mut cout, &cx, &cg, &cgs, &cu, &cus)
            .expect("cpu pair swiglu");
        assert_close_ctx(
            &read(&backend, &out),
            cout.as_slice(),
            1e-4,
            1e-4,
            &format!("pair swiglu m={m} k={k} n={n}"),
        );
    }
}

#[test]
fn nf4_register_matches_cpu() {
    run_variant(
        "nf4_register",
        STD_SHAPES,
        2,
        &|n, k| values(43, (n * k) as usize),
        &pack_nf4_stream,
        &pack_nf4_stream,
    );
}

#[test]
fn nf4_product_matches_cpu() {
    run_variant(
        "nf4_product",
        STD_SHAPES,
        2,
        &|n, k| values(43, (n * k) as usize),
        &pack_nf4_stream,
        &pack_nf4_stream,
    );
}

// ---------------------------------------------------------------------------
// A-B benchmark: production tile vs each experiment on identical inputs.
// Run: cargo test --release -p minifield-backend-wgpu --test lowbits
//      ab_bench -- --ignored --nocapture
// ---------------------------------------------------------------------------

const DISPATCHES: u32 = 30;
const BLOCKS: usize = 12;

/// Median of a mutable copy.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// Mean +/- a ~95% paired interval (t~2.2 at 12 samples).
fn paired_report(a: &[f64], b: &[f64]) -> String {
    let n = a.len() as f64;
    let (ma, mb) = (median(a), median(b));
    let diffs: Vec<f64> = a.iter().zip(b).map(|(a, b)| b / a - 1.0).collect();
    let mean = diffs.iter().sum::<f64>() / n;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let ci = 2.2 * (var / n).sqrt();
    format!(
        "A={ma:>7.1}us B={mb:>7.1}us delta={:>+6.2}% +/-{:>5.2}%",
        mean * 100.0,
        ci * 100.0
    )
}

/// Encode `DISPATCHES` complete `packed_linear` ops, wait once, return us/op.
fn bench_block(backend: &WgpuBackend, call: &mut dyn FnMut()) -> f64 {
    let start = std::time::Instant::now();
    for _ in 0..DISPATCHES {
        call();
    }
    fence_wait(backend);
    start.elapsed().as_secs_f64() * 1e6 / f64::from(DISPATCHES)
}

#[allow(clippy::too_many_arguments)]
fn ab_compare(
    backend: &mut WgpuBackend,
    label: &str,
    m: u64,
    k: u64,
    n: u64,
    weights: &[f32],
    pack_a: &Packer,
    pack_b: &Packer,
    experiment: &str,
) {
    // Ternary packs 128 weights to 32 B/row, NF4 to 64 B/row.
    let ternary = pack_a(&[0.0; 128], 1, 128).0.len() == 32;
    let code_width = if ternary { k / 4 } else { k / 2 };
    let input = values(41, (m * k) as usize);
    let x = backend
        .upload_f32(Shape::new(&[m, k]).expect("x"), &input)
        .expect("x");
    let mut out = backend
        .allocate_f32(Shape::new(&[m, n]).expect("out"))
        .expect("out");
    let experiment: LowbitsKernel = experiment.parse().expect("known variant");
    let bufs = [(pack_a, LowbitsKernel::Baseline), (pack_b, experiment)].map(|(pack, kernel)| {
        let (codes, _) = pack(weights, n as usize, k as usize);
        backend
            .upload_experimental_codes(Shape::new(&[n, code_width]).expect("codes"), &codes, kernel)
            .expect("codes")
    });
    let (_, scales) = pack_a(weights, n as usize, k as usize);
    let s = backend
        .upload_f32_classified(
            Shape::new(&[n, k / 128]).expect("scales"),
            &scales,
            AllocationClass::Scratch,
        )
        .expect("scales");
    let sides = [(LowbitsKernel::Baseline, &bufs[0]), (experiment, &bufs[1])];
    // Warm both pipelines (compile + first dispatch) before timing.
    for (experiment, codes) in &sides {
        for _ in 0..4 {
            backend
                .experimental_packed_linear(&mut out, &x, codes, &s, *experiment)
                .expect("warm dispatch");
        }
        fence_wait(backend);
    }
    let mut a = Vec::with_capacity(BLOCKS);
    let mut b = Vec::with_capacity(BLOCKS);
    for block in 0..BLOCKS {
        // A-B-B-A ordering cancels slow drift within each matched pair.
        let order = if block % 2 == 0 { [0usize, 1] } else { [1, 0] };
        for which in order {
            let (experiment, codes) = sides[which];
            let us = bench_block(backend, &mut || {
                backend
                    .experimental_packed_linear(&mut out, &x, codes, &s, experiment)
                    .expect("bench dispatch");
            });
            if which == 0 {
                a.push(us);
            } else {
                b.push(us);
            }
        }
    }
    eprintln!("{label:<52} {}", paired_report(&a, &b));
}

fn fence_wait(backend: &WgpuBackend) {
    let mut f = backend.fence().expect("fence");
    loop {
        match f.poll_step() {
            CompletionPoll::Pending => std::hint::spin_loop(),
            CompletionPoll::Ready(r) => {
                r.expect("fence result");
                return;
            }
        }
    }
}

/// One benchmark pass: both classifier projections at m=346 plus the cached
/// tail at m=299 for every experiment of the matching format.
#[test]
#[ignore = "dev benchmark; run explicitly with --ignored --nocapture"]
fn ab_bench() {
    let _serial = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let shapes = [
        (346u64, 1024u64, 2560u64),
        (346, 2560, 1024),
        (299, 2560, 1024),
    ];
    let ternary = &pack_ternary_direct as &Packer;
    let nf4 = &pack_nf4_stream as &Packer;
    for &(m, k, n) in &shapes {
        let tw = ternary_grid(n, k);
        let nw = values(43, (n * k) as usize);
        for (name, pack_b) in [
            ("scale128_control", ternary),
            ("ternary_sign", ternary),
            ("ternary_sign_sel", ternary),
            ("ternary_lut2", &pack_lut2_stream as &Packer),
            ("ternary_lut2_64x32", &pack_lut2_stream as &Packer),
            ("ternary_pn4", &pack_pn4_stream as &Packer),
        ] {
            ab_compare(
                &mut backend,
                &format!("ternary m={m} k={k} n={n} vs {name}"),
                m,
                k,
                n,
                &tw,
                ternary,
                pack_b,
                name,
            );
        }
        for (name, pack_b) in [
            ("scale128_control", nf4),
            ("nf4_register", nf4),
            ("nf4_product", nf4),
        ] {
            ab_compare(
                &mut backend,
                &format!("nf4     m={m} k={k} n={n} vs {name}"),
                m,
                k,
                n,
                &nw,
                nf4,
                pack_b,
                name,
            );
        }
    }
}

/// One full FFN step (gate/up projection + `SwiGLU` + down projection) under
/// each integration config. The pair producer keeps its fused form in every
/// arm; only which kernels consume which code streams changes.
struct FfnBuffers {
    x: WgpuBuffer,
    gate_raw: WgpuBuffer,
    up_raw: WgpuBuffer,
    down_raw: WgpuBuffer,
    gate_lut2: WgpuBuffer,
    up_lut2: WgpuBuffer,
    down_lut2: WgpuBuffer,
    gate_scales: WgpuBuffer,
    up_scales: WgpuBuffer,
    down_scales: WgpuBuffer,
}

fn ffn_dispatch(
    backend: &WgpuBackend,
    config: u8,
    b: &FfnBuffers,
    hidden: &mut WgpuBuffer,
    out: &mut WgpuBuffer,
) {
    match config {
        0 => {
            backend
                .packed_swiglu_pair(
                    hidden,
                    &b.x,
                    &b.gate_raw,
                    &b.gate_scales,
                    &b.up_raw,
                    &b.up_scales,
                )
                .expect("raw pair");
            backend
                .packed_linear(out, hidden, &b.down_raw, &b.down_scales)
                .expect("raw down");
        }
        1 => {
            backend
                .packed_swiglu_pair(
                    hidden,
                    &b.x,
                    &b.gate_raw,
                    &b.gate_scales,
                    &b.up_raw,
                    &b.up_scales,
                )
                .expect("raw pair");
            backend
                .packed_linear_lut2(out, hidden, &b.down_lut2, &b.down_scales)
                .expect("lut2 down");
        }
        _ => {
            backend
                .packed_swiglu_pair_lut2(
                    hidden,
                    &b.x,
                    &b.gate_lut2,
                    &b.gate_scales,
                    &b.up_lut2,
                    &b.up_scales,
                )
                .expect("lut2 pair");
            backend
                .packed_linear_lut2(out, hidden, &b.down_lut2, &b.down_scales)
                .expect("lut2 down");
        }
    }
}

/// Whole-FFN comparison of the three integration configs:
///   0 baseline = fused raw pair+`SwiGLU`, raw ternary down
///   1 integration A = fused raw pair+`SwiGLU`, LUT2 down
///   2 integration B = paired LUT2+`SwiGLU` (one shared activation table),
///     LUT2 down
/// Order rotates per block so position effects cancel across configs.
const FFN_K_IN: u64 = 1024;
const FFN_N_MID: u64 = 2560;

#[test]
#[ignore = "dev benchmark; run explicitly with --ignored --nocapture"]
fn ffn_lut2_bench() {
    let _serial = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let up_gate_w = ternary_grid(FFN_N_MID, FFN_K_IN);
    let up_up_w = values(43, (FFN_N_MID * FFN_K_IN) as usize);
    let down_w = ternary_grid(FFN_K_IN, FFN_N_MID);
    for m in [346u64, 299] {
        let (gate_c, gate_s) =
            pack_ternary_direct(&up_gate_w, FFN_N_MID as usize, FFN_K_IN as usize);
        let (up_c, up_s) = pack_ternary(&up_up_w, FFN_N_MID as usize, FFN_K_IN as usize);
        let (down_c, down_s) = pack_ternary_direct(&down_w, FFN_K_IN as usize, FFN_N_MID as usize);
        let input = values(41, (m * FFN_K_IN) as usize);
        let upload_codes = |backend: &mut WgpuBackend, codes: &[u8], rows: u64, k: u64| {
            backend
                .upload_u8_classified(
                    Shape::new(&[rows, k / 4]).expect("codes"),
                    codes,
                    AllocationClass::Scratch,
                )
                .expect("codes")
        };
        let upload_scales = |backend: &mut WgpuBackend, scales: &[f32], rows: u64, k: u64| {
            backend
                .upload_f32_classified(
                    Shape::new(&[rows, k / 128]).expect("scales"),
                    scales,
                    AllocationClass::Scratch,
                )
                .expect("scales")
        };
        let b = FfnBuffers {
            x: backend
                .upload_f32(Shape::new(&[m, FFN_K_IN]).expect("x"), &input)
                .expect("x"),
            gate_raw: upload_codes(&mut backend, &gate_c, FFN_N_MID, FFN_K_IN),
            up_raw: upload_codes(&mut backend, &up_c, FFN_N_MID, FFN_K_IN),
            down_raw: upload_codes(&mut backend, &down_c, FFN_K_IN, FFN_N_MID),
            gate_lut2: upload_codes(&mut backend, &repack_lut2(&gate_c), FFN_N_MID, FFN_K_IN),
            up_lut2: upload_codes(&mut backend, &repack_lut2(&up_c), FFN_N_MID, FFN_K_IN),
            down_lut2: upload_codes(&mut backend, &repack_lut2(&down_c), FFN_K_IN, FFN_N_MID),
            gate_scales: upload_scales(&mut backend, &gate_s, FFN_N_MID, FFN_K_IN),
            up_scales: upload_scales(&mut backend, &up_s, FFN_N_MID, FFN_K_IN),
            down_scales: upload_scales(&mut backend, &down_s, FFN_K_IN, FFN_N_MID),
        };
        let mut hidden = backend
            .allocate_f32(Shape::new(&[m, FFN_N_MID]).expect("hidden"))
            .expect("hidden");
        let mut out = backend
            .allocate_f32(Shape::new(&[m, FFN_K_IN]).expect("out"))
            .expect("out");
        // Warm all pipelines before timing.
        for config in 0..3u8 {
            for _ in 0..4 {
                ffn_dispatch(&backend, config, &b, &mut hidden, &mut out);
            }
            fence_wait(&backend);
        }
        let mut times: [Vec<f64>; 3] = [
            Vec::with_capacity(BLOCKS),
            Vec::with_capacity(BLOCKS),
            Vec::with_capacity(BLOCKS),
        ];
        for block in 0..BLOCKS {
            // Rotate starting config per block so ordering effects cancel.
            for step in 0..3usize {
                let config = ((block + step) % 3) as u8;
                let start = std::time::Instant::now();
                for _ in 0..DISPATCHES {
                    ffn_dispatch(&backend, config, &b, &mut hidden, &mut out);
                }
                fence_wait(&backend);
                times[usize::from(config)]
                    .push(start.elapsed().as_secs_f64() * 1e6 / f64::from(DISPATCHES));
            }
        }
        let names = ["base pair+down", "raw pair, lut2 down", "lut2 pair+down"];
        let base = median(&times[0]);
        eprintln!("ffn m={m}:");
        for (i, name) in names.iter().enumerate() {
            let med = median(&times[i]);
            eprintln!(
                "  {name:<22} {med:>9.1}us  {:>+6.2}% vs base",
                (med / base - 1.0) * 100.0
            );
        }
    }
}
