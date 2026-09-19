//! Op-level parity tests: `minifield-backend-wgpu` versus the scalar CPU
//! baseline on synthetic tensors.
//!
//! Every test skips cleanly when no wgpu adapter exists (headless CI without
//! a GPU driver), so the suite is safe everywhere but verifies on Metal,
//! Vulkan, DX12, and WebGPU-capable drivers.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]

use minifield_backend_cpu::CpuBackend;
use minifield_backend_wgpu::{WgpuBackend, WgpuBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, GatedShortConvSpec, GqaSpec, InferenceCompletion,
    PackedHeadSpec, RectCopy2d, ResourceLimits, RotarySpec, Shape, TokenIds,
};

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 1 << 34,
        max_pending_operations: 1_024,
    }
}

/// A backend, or `None` when this machine has no usable adapter.
fn gpu() -> Option<WgpuBackend> {
    WgpuBackend::new(0x77, limits()).ok()
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

/// Exact comparison for ops whose arithmetic is identical (copies, gathers,
/// single-op elementwise).
fn assert_exact(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index}: {a} != {b}");
    }
}

/// Tolerant comparison for ops with reordered reductions (dot products,
/// softmax, norms).
fn assert_close(left: &[f32], right: &[f32], abs: f32, rel: f32) {
    assert_eq!(left.len(), right.len(), "length mismatch");
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        let diff = (a - b).abs();
        let scale = b.abs().mul_add(rel, abs);
        assert!(
            a.is_finite() && b.is_finite() && diff <= scale,
            "element {index}: {a} vs {b} exceeds tolerance (diff {diff})"
        );
    }
}

#[test]
fn upload_allocate_and_readback_roundtrip() {
    let Some(mut backend) = gpu() else { return };
    let data = values(11, 513);
    let buffer = backend
        .upload_f32(Shape::new(&[513]).expect("shape"), &data)
        .expect("upload");
    assert_exact(&read(&backend, &buffer), &data);

    let zeroed = backend
        .allocate_f32(Shape::new(&[64]).expect("shape"))
        .expect("allocate");
    assert!(read(&backend, &zeroed).iter().all(|v| *v == 0.0));
}

#[test]
fn fence_and_readback_completions() {
    let Some(mut backend) = gpu() else { return };
    let buffer = backend
        .upload_f32(Shape::new(&[4]).expect("shape"), &[1.0, 2.0, 3.0, 4.0])
        .expect("upload");
    let mut fence = backend.fence().expect("fence");
    for _ in 0..600_000 {
        if let CompletionPoll::Ready(result) = fence.poll_step() {
            result.expect("fence result");
            let mut readback = backend.read_f32_async(&buffer).expect("readback");
            for _ in 0..600_000 {
                if let CompletionPoll::Ready(result) = readback.poll_step() {
                    assert_exact(&result.expect("readback result"), &[1.0, 2.0, 3.0, 4.0]);
                    return;
                }
            }
            panic!("readback did not complete");
        }
    }
    panic!("fence did not complete");
}

#[test]
fn elementwise_add_multiply_match_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let shape = Shape::new(&[6, 17]).expect("shape");
    let a = values(3, 102);
    let b = values(5, 102);
    let (gpu_a, gpu_b) = (
        backend.upload_f32(shape, &a).expect("upload a"),
        backend.upload_f32(shape, &b).expect("upload b"),
    );
    let (cpu_a, cpu_b) = (
        reference.upload_f32(shape, &a).expect("cpu a"),
        reference.upload_f32(shape, &b).expect("cpu b"),
    );
    let mut gpu_out = backend.allocate_f32(shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(shape).expect("cpu out");

    backend.add(&mut gpu_out, &gpu_a, &gpu_b).expect("gpu add");
    reference
        .add(&mut cpu_out, &cpu_a, &cpu_b)
        .expect("cpu add");
    assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());

    backend
        .multiply(&mut gpu_out, &gpu_a, &gpu_b)
        .expect("gpu multiply");
    reference
        .multiply(&mut cpu_out, &cpu_a, &cpu_b)
        .expect("cpu multiply");
    assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
}

#[test]
fn elementwise_grid_splits_past_one_dimension() {
    // 70_000 elements exceeds 65_535 workgroups-per-dimension only if the
    // dispatch forgot to split; here it just exercises the 2D flatten path at
    // 256 threads per workgroup.
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let count = 70_000_usize;
    let shape = Shape::new(&[count as u64]).expect("shape");
    let a = values(9, count);
    let b = values(13, count);
    let gpu_a = backend.upload_f32(shape, &a).expect("upload a");
    let gpu_b = backend.upload_f32(shape, &b).expect("upload b");
    let cpu_a = reference.upload_f32(shape, &a).expect("cpu a");
    let cpu_b = reference.upload_f32(shape, &b).expect("cpu b");
    let mut gpu_out = backend.allocate_f32(shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(shape).expect("cpu out");
    backend.add(&mut gpu_out, &gpu_a, &gpu_b).expect("gpu add");
    reference
        .add(&mut cpu_out, &cpu_a, &cpu_b)
        .expect("cpu add");
    assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
}

#[test]
fn copy_and_rect_2d_match_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let shape = Shape::new(&[8, 6]).expect("shape");
    let src = values(21, 48);
    let gpu_src = backend.upload_f32(shape, &src).expect("gpu src");
    let cpu_src = reference.upload_f32(shape, &src).expect("cpu src");
    let mut gpu_dst = backend.allocate_f32(shape).expect("gpu dst");
    let mut cpu_dst = reference.allocate_f32(shape).expect("cpu dst");
    backend.copy(&mut gpu_dst, &gpu_src).expect("gpu copy");
    reference.copy(&mut cpu_dst, &cpu_src).expect("cpu copy");
    assert_exact(&read(&backend, &gpu_dst), cpu_dst.as_slice());

    let rect = RectCopy2d::new(1, 2, 4, 1, 3, 4);
    let mut gpu_dst2 = backend.allocate_f32(shape).expect("gpu dst2");
    let mut cpu_dst2 = reference.allocate_f32(shape).expect("cpu dst2");
    backend
        .copy_rect_2d(&mut gpu_dst2, &gpu_src, rect)
        .expect("gpu rect copy");
    reference
        .copy_rect_2d(&mut cpu_dst2, &cpu_src, rect)
        .expect("cpu rect copy");
    assert_exact(&read(&backend, &gpu_dst2), cpu_dst2.as_slice());
}

#[test]
fn gather_rows_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let table_shape = Shape::new(&[64, 7]).expect("table shape");
    let table = values(31, 448);
    let ids = [63_u32, 0, 17, 17, 2, 41];
    let out_shape = Shape::new(&[6, 7]).expect("out shape");
    let gpu_table = backend.upload_f32(table_shape, &table).expect("gpu table");
    let cpu_table = reference
        .upload_f32(table_shape, &table)
        .expect("cpu table");
    let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
    backend
        .gather_rows(&mut gpu_out, &gpu_table, TokenIds::Host(&ids))
        .expect("gpu gather");
    reference
        .gather_rows(&mut cpu_out, &cpu_table, TokenIds::Host(&ids))
        .expect("cpu gather");
    assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
}

#[test]
fn linear_gemv_and_gemm_match_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();

    // m == 1 GEMV path, k divisible by 4 (vec4 fast path) and not (scalar tail).
    for k in [64_u64, 65] {
        let input_shape = Shape::new(&[1, k]).expect("input shape");
        let weight_shape = Shape::new(&[37, k]).expect("weight shape");
        let out_shape = Shape::new(&[1, 37]).expect("out shape");
        let input = values(41, k as usize);
        let weight = values(43, (37 * k) as usize);
        let gpu_in = backend.upload_f32(input_shape, &input).expect("gpu in");
        let gpu_w = backend.upload_f32(weight_shape, &weight).expect("gpu w");
        let cpu_in = reference.upload_f32(input_shape, &input).expect("cpu in");
        let cpu_w = reference.upload_f32(weight_shape, &weight).expect("cpu w");
        let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
        backend
            .linear(&mut gpu_out, &gpu_in, &gpu_w)
            .expect("gpu linear");
        reference
            .linear(&mut cpu_out, &cpu_in, &cpu_w)
            .expect("cpu linear");
        assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-4, 1e-4);
    }

    // m > 1 tiled GEMM path, including tile-edge sizes.
    for (m, n, k) in [(5_u64, 37_u64, 65_u64), (17, 16, 48)] {
        let input_shape = Shape::new(&[m, k]).expect("input shape");
        let weight_shape = Shape::new(&[n, k]).expect("weight shape");
        let out_shape = Shape::new(&[m, n]).expect("out shape");
        let input = values(47, (m * k) as usize);
        let weight = values(53, (n * k) as usize);
        let gpu_in = backend.upload_f32(input_shape, &input).expect("gpu in");
        let gpu_w = backend.upload_f32(weight_shape, &weight).expect("gpu w");
        let cpu_in = reference.upload_f32(input_shape, &input).expect("cpu in");
        let cpu_w = reference.upload_f32(weight_shape, &weight).expect("cpu w");
        let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
        backend
            .linear(&mut gpu_out, &gpu_in, &gpu_w)
            .expect("gpu linear");
        reference
            .linear(&mut cpu_out, &cpu_in, &cpu_w)
            .expect("cpu linear");
        assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-4, 1e-4);
    }
}

#[test]
fn rms_norms_match_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let input_shape = Shape::new(&[4, 24]).expect("input shape");
    let weight_shape = Shape::new(&[24]).expect("weight shape");
    let input = values(59, 96);
    let weight: Vec<f32> = values(61, 24).iter().map(|v| v.abs() + 0.25).collect();
    let gpu_in = backend.upload_f32(input_shape, &input).expect("gpu in");
    let gpu_w = backend.upload_f32(weight_shape, &weight).expect("gpu w");
    let cpu_in = reference.upload_f32(input_shape, &input).expect("cpu in");
    let cpu_w = reference.upload_f32(weight_shape, &weight).expect("cpu w");
    let mut gpu_out = backend.allocate_f32(input_shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(input_shape).expect("cpu out");
    backend
        .row_rms_norm(&mut gpu_out, &gpu_in, &gpu_w, 1e-5)
        .expect("gpu rms");
    reference
        .row_rms_norm(&mut cpu_out, &cpu_in, &cpu_w, 1e-5)
        .expect("cpu rms");
    assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-5, 1e-4);

    // Per-head rows: [tokens, heads * head_dim] with head_dim-wide norms.
    let heads = PackedHeadSpec::new(3, 8).expect("heads");
    let head_weight_shape = Shape::new(&[8]).expect("head weight shape");
    let head_weight: Vec<f32> = weight[..8].to_vec();
    let gpu_hw = backend
        .upload_f32(head_weight_shape, &head_weight)
        .expect("gpu hw");
    let cpu_hw = reference
        .upload_f32(head_weight_shape, &head_weight)
        .expect("cpu hw");
    backend
        .head_rms_norm(&mut gpu_out, &gpu_in, &gpu_hw, heads, 1e-5)
        .expect("gpu head rms");
    reference
        .head_rms_norm(&mut cpu_out, &cpu_in, &cpu_hw, heads, 1e-5)
        .expect("cpu head rms");
    assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-5, 1e-4);
}

#[test]
fn split_half_rotary_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let spec =
        RotarySpec::new(PackedHeadSpec::new(4, 8).expect("heads"), 10_000.0).expect("rotary spec");
    let shape = Shape::new(&[3, 32]).expect("shape");
    let input = values(67, 96);
    let positions = [0_u64, 5, 17];
    let gpu_in = backend.upload_f32(shape, &input).expect("gpu in");
    let cpu_in = reference.upload_f32(shape, &input).expect("cpu in");
    let mut gpu_out = backend.allocate_f32(shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(shape).expect("cpu out");
    backend
        .split_half_rotary(&mut gpu_out, &gpu_in, &positions, spec)
        .expect("gpu rotary");
    reference
        .split_half_rotary(&mut cpu_out, &cpu_in, &positions, spec)
        .expect("cpu rotary");
    assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-6, 1e-5);
}

#[test]
fn causal_gqa_matches_cpu_and_appends_cache() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let spec = GqaSpec::new(4, 2, 8).expect("gqa spec");
    let capacity = 8_u64;
    let cache_shape = Shape::new(&[capacity, 16]).expect("cache shape");
    let q_shape = Shape::new(&[3, 32]).expect("q shape");
    let kv_shape = Shape::new(&[3, 16]).expect("kv shape");
    let query = values(71, 96);
    let key = values(73, 48);
    let value = values(79, 48);
    let gpu_q = backend.upload_f32(q_shape, &query).expect("gpu q");
    let gpu_k = backend.upload_f32(kv_shape, &key).expect("gpu k");
    let gpu_v = backend.upload_f32(kv_shape, &value).expect("gpu v");
    let cpu_q = reference.upload_f32(q_shape, &query).expect("cpu q");
    let cpu_k = reference.upload_f32(kv_shape, &key).expect("cpu k");
    let cpu_v = reference.upload_f32(kv_shape, &value).expect("cpu v");
    let mut gpu_kc = backend
        .allocate_f32_classified(cache_shape, AllocationClass::Cache)
        .expect("gpu kc");
    let mut gpu_vc = backend
        .allocate_f32_classified(cache_shape, AllocationClass::Cache)
        .expect("gpu vc");
    let mut cpu_kc = reference
        .allocate_f32_classified(cache_shape, AllocationClass::Cache)
        .expect("cpu kc");
    let mut cpu_vc = reference
        .allocate_f32_classified(cache_shape, AllocationClass::Cache)
        .expect("cpu vc");
    let mut gpu_out = backend.allocate_f32(q_shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(q_shape).expect("cpu out");
    let mut gpu_len = 0_u64;
    let mut cpu_len = 0_u64;
    backend
        .causal_gqa(
            &mut gpu_out,
            &gpu_q,
            &gpu_k,
            &gpu_v,
            &mut gpu_kc,
            &mut gpu_vc,
            &mut gpu_len,
            spec,
        )
        .expect("gpu gqa");
    reference
        .causal_gqa(
            &mut cpu_out,
            &cpu_q,
            &cpu_k,
            &cpu_v,
            &mut cpu_kc,
            &mut cpu_vc,
            &mut cpu_len,
            spec,
        )
        .expect("cpu gqa");
    assert_eq!(gpu_len, cpu_len);
    assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-4, 1e-4);
    assert_close(&read(&backend, &gpu_kc), cpu_kc.as_slice(), 1e-6, 1e-6);
    assert_close(&read(&backend, &gpu_vc), cpu_vc.as_slice(), 1e-6, 1e-6);

    // A second decode token attends over the appended cache.
    let q1 = Shape::new(&[1, 32]).expect("q1");
    let kv1 = Shape::new(&[1, 16]).expect("kv1");
    let query2 = values(83, 32);
    let key2 = values(89, 16);
    let value2 = values(97, 16);
    let gpu_q2 = backend.upload_f32(q1, &query2).expect("gpu q2");
    let gpu_k2 = backend.upload_f32(kv1, &key2).expect("gpu k2");
    let gpu_v2 = backend.upload_f32(kv1, &value2).expect("gpu v2");
    let cpu_q2 = reference.upload_f32(q1, &query2).expect("cpu q2");
    let cpu_k2 = reference.upload_f32(kv1, &key2).expect("cpu k2");
    let cpu_v2 = reference.upload_f32(kv1, &value2).expect("cpu v2");
    let mut gpu_out2 = backend.allocate_f32(q1).expect("gpu out2");
    let mut cpu_out2 = reference.allocate_f32(q1).expect("cpu out2");
    backend
        .causal_gqa(
            &mut gpu_out2,
            &gpu_q2,
            &gpu_k2,
            &gpu_v2,
            &mut gpu_kc,
            &mut gpu_vc,
            &mut gpu_len,
            spec,
        )
        .expect("gpu gqa 2");
    reference
        .causal_gqa(
            &mut cpu_out2,
            &cpu_q2,
            &cpu_k2,
            &cpu_v2,
            &mut cpu_kc,
            &mut cpu_vc,
            &mut cpu_len,
            spec,
        )
        .expect("cpu gqa 2");
    assert_eq!(gpu_len, cpu_len);
    assert_close(&read(&backend, &gpu_out2), cpu_out2.as_slice(), 1e-4, 1e-4);
    assert_close(&read(&backend, &gpu_kc), cpu_kc.as_slice(), 1e-6, 1e-6);
}

#[test]
fn gated_short_convolution_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    // tokens < history rows and tokens >= history rows both exercise the
    // rolling-history assembly paths.
    for tokens in [1_u64, 5] {
        let spec = GatedShortConvSpec::new(16, 3).expect("conv spec");
        let token_shape = Shape::new(&[tokens, 16]).expect("token shape");
        let projection_shape = Shape::new(&[tokens, 48]).expect("projection shape");
        let kernel_shape = Shape::new(&[16, 3]).expect("kernel shape");
        let history_shape = Shape::new(&[2, 16]).expect("history shape");
        let count = (tokens * 48) as usize;
        let projection = values(101, count);
        let kernel = values(109, 48);
        let history = values(113, 32);
        let gpu_p = backend
            .upload_f32(projection_shape, &projection)
            .expect("gpu projection");
        let gpu_k = backend.upload_f32(kernel_shape, &kernel).expect("gpu k");
        let mut gpu_h = backend
            .upload_f32_classified(history_shape, &history, AllocationClass::Cache)
            .expect("gpu history");
        let cpu_p = reference
            .upload_f32(projection_shape, &projection)
            .expect("cpu projection");
        let cpu_k = reference.upload_f32(kernel_shape, &kernel).expect("cpu k");
        let mut cpu_h = reference
            .upload_f32_classified(history_shape, &history, AllocationClass::Cache)
            .expect("cpu history");
        let mut gpu_out = backend.allocate_f32(token_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(token_shape).expect("cpu out");
        backend
            .gated_short_convolution(&mut gpu_out, &gpu_p, &gpu_k, &mut gpu_h, spec)
            .expect("gpu conv");
        reference
            .gated_short_convolution(&mut cpu_out, &cpu_p, &cpu_k, &mut cpu_h, spec)
            .expect("cpu conv");
        assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-5, 1e-5);
        assert_close(&read(&backend, &gpu_h), cpu_h.as_slice(), 1e-6, 1e-6);
    }
}

#[test]
fn swiglu_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let shape = Shape::new(&[5, 13]).expect("shape");
    let gate = values(127, 65);
    let up = values(131, 65);
    let gpu_g = backend.upload_f32(shape, &gate).expect("gpu gate");
    let gpu_u = backend.upload_f32(shape, &up).expect("gpu up");
    let cpu_g = reference.upload_f32(shape, &gate).expect("cpu gate");
    let cpu_u = reference.upload_f32(shape, &up).expect("cpu up");
    let mut gpu_out = backend.allocate_f32(shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(shape).expect("cpu out");
    backend
        .swiglu(&mut gpu_out, &gpu_g, &gpu_u)
        .expect("gpu swiglu");
    reference
        .swiglu(&mut cpu_out, &cpu_g, &cpu_u)
        .expect("cpu swiglu");
    assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-6, 1e-6);
}

/// Pack one 128-weight group per `minifield.ternary.v1`: absmax scale,
/// round-to-nearest code, byte `j/4` bits `2*(j%4)`. Returns (code bytes,
/// f32 scale). A zero row encodes all codes 1 (weight 0) with scale 0.
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

/// Build the (codes, scales) streams for `rows` x `k` weights, `k % 128 == 0`.
fn pack_weights(weights: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(rows * k / 4);
    let mut scales = Vec::with_capacity(rows * k / 128);
    for chunk in weights.as_chunks::<128>().0 {
        let (group_codes, scale) = pack_group(chunk);
        codes.extend_from_slice(&group_codes);
        scales.push(scale);
    }
    (codes, scales)
}

#[test]
fn packed_linear_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();

    // m == 1 decode path and a multi-row batch; k spans several 128-weight
    // groups. n = 70_000 exercises the flattened-grid split past one
    // workgroup dimension (the lm_head shape is 65_536 rows).
    for (m, n, k) in [(1_u64, 37_u64, 256_u64), (5, 37, 384), (1, 70_000, 128)] {
        let input_shape = Shape::new(&[m, k]).expect("input shape");
        let codes_shape = Shape::new(&[n, k / 4]).expect("codes shape");
        let scales_shape = Shape::new(&[n, k / 128]).expect("scales shape");
        let out_shape = Shape::new(&[m, n]).expect("out shape");
        let input = values(41, (m * k) as usize);
        let weight = values(43, (n * k) as usize);
        let (codes, scales) = pack_weights(&weight, n as usize, k as usize);

        let gpu_in = backend.upload_f32(input_shape, &input).expect("gpu in");
        let gpu_codes = backend
            .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
            .expect("gpu codes");
        let gpu_scales = backend
            .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
            .expect("gpu scales");
        let cpu_in = reference.upload_f32(input_shape, &input).expect("cpu in");
        let cpu_codes = reference
            .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
            .expect("cpu codes");
        let cpu_scales = reference
            .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
            .expect("cpu scales");
        let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
        backend
            .packed_linear(&mut gpu_out, &gpu_in, &gpu_codes, &gpu_scales)
            .expect("gpu packed linear");
        reference
            .packed_linear(&mut cpu_out, &cpu_in, &cpu_codes, &cpu_scales)
            .expect("cpu packed linear");
        assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-4, 1e-4);
    }
}

#[test]
fn packed_gather_rows_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let (n, k) = (64_u64, 256_u64);
    let codes_shape = Shape::new(&[n, k / 4]).expect("codes shape");
    let scales_shape = Shape::new(&[n, k / 128]).expect("scales shape");
    let weight = values(53, (n * k) as usize);
    let (codes, scales) = pack_weights(&weight, n as usize, k as usize);
    let ids = [63_u32, 0, 17, 17, 2, 41];
    let out_shape = Shape::new(&[6, k]).expect("out shape");

    let gpu_codes = backend
        .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
        .expect("gpu codes");
    let gpu_scales = backend
        .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
        .expect("gpu scales");
    let cpu_codes = reference
        .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
        .expect("cpu codes");
    let cpu_scales = reference
        .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
        .expect("cpu scales");
    let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
    let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
    backend
        .packed_gather_rows(&mut gpu_out, &gpu_codes, &gpu_scales, TokenIds::Host(&ids))
        .expect("gpu packed gather");
    reference
        .packed_gather_rows(&mut cpu_out, &cpu_codes, &cpu_scales, TokenIds::Host(&ids))
        .expect("cpu packed gather");
    // Dequantized weights are exact products, so parity is bitwise.
    assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
}

#[test]
fn packed_ops_reject_invalid_operands() {
    let Some(mut backend) = gpu() else { return };
    let input = backend
        .upload_f32(Shape::new(&[1, 128]).expect("shape"), &[0.0; 128])
        .expect("input");
    let codes = backend
        .upload_u8_classified(
            Shape::new(&[2, 32]).expect("shape"),
            &[0x55_u8; 64],
            AllocationClass::Weight,
        )
        .expect("codes");
    let scales = backend
        .upload_f32(Shape::new(&[2, 1]).expect("shape"), &[1.0, 1.0])
        .expect("scales");
    let f32_codes = backend
        .upload_f32(Shape::new(&[2, 32]).expect("shape"), &[0.0; 64])
        .expect("f32 codes");
    let bad_scales = backend
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[1.0; 4])
        .expect("bad scales");
    let mut out = backend
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("out");

    // f32 where u8 codes belong.
    assert!(
        backend
            .packed_linear(&mut out, &input, &f32_codes, &scales)
            .is_err()
    );
    // wrong group count.
    assert!(
        backend
            .packed_linear(&mut out, &input, &codes, &bad_scales)
            .is_err()
    );
    // input width mismatch.
    let wide = backend
        .upload_f32(Shape::new(&[1, 256]).expect("shape"), &[0.0; 256])
        .expect("wide input");
    assert!(
        backend
            .packed_linear(&mut out, &wide, &codes, &scales)
            .is_err()
    );
    // gather id past the row count.
    assert!(
        backend
            .packed_gather_rows(&mut out, &codes, &scales, TokenIds::Host(&[7]))
            .is_err()
    );
}

#[test]
fn argmax_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    for (rows, columns) in [(1_u64, 1024_u64), (3, 511)] {
        let shape = Shape::new(&[rows, columns]).expect("logits shape");
        // Exact ties (first wins), a negative-row max, and a late maximum.
        let mut logits = values(97, (rows * columns) as usize);
        if columns >= 8 {
            logits[2] = 4.0;
            logits[5] = 4.0;
        }
        let gpu_in = backend.upload_f32(shape, &logits).expect("gpu logits");
        let cpu_in = reference.upload_f32(shape, &logits).expect("cpu logits");
        let mut gpu_out = backend
            .allocate_f32(Shape::new(&[rows]).expect("out"))
            .expect("gpu argmax out");
        let mut cpu_out = reference
            .allocate_f32(Shape::new(&[rows]).expect("out"))
            .expect("cpu argmax out");
        backend.argmax(&mut gpu_out, &gpu_in).expect("gpu argmax");
        reference.argmax(&mut cpu_out, &cpu_in).expect("cpu argmax");
        assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
    }
}

#[test]
fn argmax_marks_nonfinite_rows_nan() {
    let Some(mut backend) = gpu() else { return };
    // Uploads reject non-finite values, so produce +inf on-device: 3e38 * 2
    // overflows f32. Rows 0 and 1 get one non-finite element each; row 2 stays
    // clean with its max at index 0.
    let base = [
        1.0, 3.0e38, 3.0, //
        0.5, 3.0e38, 2.0, //
        4.0, -1.0, 0.25,
    ];
    let two = [2.0_f32; 9];
    let shape = Shape::new(&[3, 3]).expect("shape");
    // Only the GPU half is checked here: CPU ops reject non-finite results
    // eagerly, so the poison path is a device-compute concern. CPU argmax's
    // NaN marking is covered by backend-cpu's unit tests.
    let gpu_base = backend.upload_f32(shape, &base).expect("gpu base");
    let gpu_two = backend.upload_f32(shape, &two).expect("gpu two");
    let mut gpu_logits = backend.allocate_f32(shape).expect("gpu logits");
    backend
        .multiply(&mut gpu_logits, &gpu_base, &gpu_two)
        .expect("gpu multiply");

    let mut gpu_out = backend
        .allocate_f32(Shape::new(&[3]).expect("out"))
        .expect("gpu argmax out");
    backend
        .argmax(&mut gpu_out, &gpu_logits)
        .expect("gpu argmax");
    let gpu_result = read(&backend, &gpu_out);
    for (row, value) in gpu_result.iter().enumerate().take(2) {
        assert!(value.is_nan(), "row {row} should be NaN");
    }
    assert_eq!(gpu_result[2], 0.0);
    // A clean row keeps a finite index.
    let clean = backend
        .upload_f32(Shape::new(&[1, 4]).expect("shape"), &[0.0, 9.0, -1.0, 9.0])
        .expect("clean logits");
    let mut out = backend
        .allocate_f32(Shape::new(&[1]).expect("out"))
        .expect("clean out");
    backend.argmax(&mut out, &clean).expect("clean argmax");
    assert_exact(&read(&backend, &out), &[1.0]);
}

#[test]
fn argmax_masked_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let mask_of = |width: usize, allowed: &[usize]| {
        let mut words = vec![0_u64; width.div_ceil(64)];
        for &index in allowed {
            words[index / 64] |= 1_u64 << (index % 64);
        }
        words
    };
    // Narrow single-workgroup and wide two-stage paths: the true argmax
    // winner is masked out, so the runner-up must win on both backends.
    for (rows, columns, winner, runner_up) in [
        (1_u64, 1024_u64, 700_usize, 42_usize),
        (3, 511, 500, 10),
        (2, 8192, 8000, 65),
    ] {
        let shape = Shape::new(&[rows, columns]).expect("logits shape");
        let width = columns as usize;
        let mut logits = vec![-1.0_f32; rows as usize * width];
        for row in 0..rows as usize {
            logits[row * width + winner] = 9.0;
            logits[row * width + runner_up] = 5.0;
        }
        let mask = mask_of(width, &[runner_up, 0]);
        let gpu_in = backend.upload_f32(shape, &logits).expect("gpu logits");
        let cpu_in = reference.upload_f32(shape, &logits).expect("cpu logits");
        let mut gpu_out = backend
            .allocate_f32(Shape::new(&[rows]).expect("out"))
            .expect("gpu masked argmax out");
        let mut cpu_out = reference
            .allocate_f32(Shape::new(&[rows]).expect("out"))
            .expect("cpu masked argmax out");
        backend
            .argmax_masked(&mut gpu_out, &gpu_in, &mask)
            .expect("gpu masked argmax");
        reference
            .argmax_masked(&mut cpu_out, &cpu_in, &mask)
            .expect("cpu masked argmax");
        assert_exact(&read(&backend, &gpu_out), cpu_out.as_slice());
        let expected = vec![runner_up as f32; rows as usize];
        assert_eq!(cpu_out.as_slice(), expected.as_slice());
    }
}

#[test]
fn argmax_masked_nan_and_empty_rows() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    // A fully masked-out row yields NaN on both backends.
    let shape = Shape::new(&[1, 512]).expect("shape");
    let logits = values(53, 512);
    let gpu_in = backend.upload_f32(shape, &logits).expect("gpu logits");
    let cpu_in = reference.upload_f32(shape, &logits).expect("cpu logits");
    let empty = vec![0_u64; 8];
    let mut gpu_out = backend
        .allocate_f32(Shape::new(&[1]).expect("out"))
        .expect("gpu out");
    let mut cpu_out = reference
        .allocate_f32(Shape::new(&[1]).expect("out"))
        .expect("cpu out");
    backend
        .argmax_masked(&mut gpu_out, &gpu_in, &empty)
        .expect("gpu masked argmax");
    reference
        .argmax_masked(&mut cpu_out, &cpu_in, &empty)
        .expect("cpu masked argmax");
    assert!(read(&backend, &gpu_out)[0].is_nan());
    assert!(cpu_out.as_slice()[0].is_nan());

    // An inf produced on-device at a masked position is skipped; the same inf
    // at an allowed position poisons the row to NaN.
    let base = [1.0, 3.0e38, 0.5, -1.0];
    let two = [2.0_f32; 4];
    let gpu_base = backend
        .upload_f32(Shape::new(&[1, 4]).expect("shape"), &base)
        .expect("gpu base");
    let gpu_two = backend
        .upload_f32(Shape::new(&[1, 4]).expect("shape"), &two)
        .expect("gpu two");
    let mut gpu_inf = backend
        .allocate_f32(Shape::new(&[1, 4]).expect("shape"))
        .expect("gpu inf");
    backend
        .multiply(&mut gpu_inf, &gpu_base, &gpu_two)
        .expect("gpu multiply");
    let mut out = backend
        .allocate_f32(Shape::new(&[1]).expect("out"))
        .expect("out");
    // Mask off index 1 (the +inf): index 0 wins at 2.0.
    backend
        .argmax_masked(&mut out, &gpu_inf, &[0b0101])
        .expect("masked argmax skips masked inf");
    assert_eq!(read(&backend, &out)[0], 0.0);
    // Allow the inf position: the row is poisoned.
    backend
        .argmax_masked(&mut out, &gpu_inf, &[0b0111])
        .expect("masked argmax poisons on allowed inf");
    assert!(read(&backend, &out)[0].is_nan());
}

#[test]
fn device_argmax_ids_feed_gathers() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    // Vocabulary-sized logits whose winners point at embedding rows.
    let (table_rows, width) = (8_u64, 16_u64);
    let table_shape = Shape::new(&[table_rows, width]).expect("table shape");
    let table = values(31, (table_rows * width) as usize);
    // Three rows of "logits" over the 8-row table: winners 3, 0, 7.
    let mut logits = vec![-1.0_f32; 3 * 8];
    logits[3] = 9.0;
    logits[8] = 9.0;
    logits[2 * 8 + 7] = 9.0;
    let logits_shape = Shape::new(&[3, 8]).expect("logits shape");

    let gpu_table = backend.upload_f32(table_shape, &table).expect("gpu table");
    let cpu_table = reference
        .upload_f32(table_shape, &table)
        .expect("cpu table");
    let gpu_logits = backend
        .upload_f32(logits_shape, &logits)
        .expect("gpu logits");
    let cpu_logits = reference
        .upload_f32(logits_shape, &logits)
        .expect("cpu logits");

    // The device argmax output feeds the embedding gather without a readback.
    let mut gpu_ids = backend
        .allocate_f32(Shape::new(&[3]).expect("ids"))
        .expect("gpu ids");
    backend
        .argmax(&mut gpu_ids, &gpu_logits)
        .expect("gpu argmax");
    let mut gpu_gathered = backend
        .allocate_f32(Shape::new(&[3, width]).expect("out"))
        .expect("gpu gathered");
    backend
        .gather_rows(&mut gpu_gathered, &gpu_table, TokenIds::Device(&gpu_ids))
        .expect("gpu device gather");

    // CPU reference resolves ids through its own argmax buffer.
    let mut cpu_ids = reference
        .allocate_f32(Shape::new(&[3]).expect("ids"))
        .expect("cpu ids");
    reference
        .argmax(&mut cpu_ids, &cpu_logits)
        .expect("cpu argmax");
    let mut cpu_gathered = reference
        .allocate_f32(Shape::new(&[3, width]).expect("out"))
        .expect("cpu gathered");
    reference
        .gather_rows(&mut cpu_gathered, &cpu_table, TokenIds::Device(&cpu_ids))
        .expect("cpu device gather");
    assert_exact(&read(&backend, &gpu_gathered), cpu_gathered.as_slice());
    // And matches a plain host gather of the expected rows.
    let mut host_gathered = reference
        .allocate_f32(Shape::new(&[3, width]).expect("out"))
        .expect("host gathered");
    reference
        .gather_rows(&mut host_gathered, &cpu_table, TokenIds::Host(&[3, 0, 7]))
        .expect("host gather");
    assert_exact(&read(&backend, &gpu_gathered), host_gathered.as_slice());
}

#[test]
fn argmax_rejects_invalid_operands() {
    let Some(mut backend) = gpu() else { return };
    let logits = backend
        .upload_f32(Shape::new(&[2, 4]).expect("shape"), &[0.0; 8])
        .expect("logits");
    let mut good_out = backend
        .allocate_f32(Shape::new(&[2]).expect("shape"))
        .expect("out");
    // wrong output shape
    let mut wide_out = backend
        .allocate_f32(Shape::new(&[3]).expect("shape"))
        .expect("wide out");
    assert!(backend.argmax(&mut wide_out, &logits).is_err());
    // rank-1 input rejected
    let flat = backend
        .upload_f32(Shape::new(&[8]).expect("shape"), &[0.0; 8])
        .expect("flat");
    assert!(backend.argmax(&mut good_out, &flat).is_err());
    // width beyond the exact f32 index range rejected
    let _ = (logits, good_out);
}

#[test]
fn packed_linear_pair_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    for (m, n, k) in [(1_u64, 37_u64, 256_u64), (5, 37, 384)] {
        let input_shape = Shape::new(&[m, k]).expect("input shape");
        let codes_shape = Shape::new(&[n, k / 4]).expect("codes shape");
        let scales_shape = Shape::new(&[n, k / 128]).expect("scales shape");
        let out_shape = Shape::new(&[m, n]).expect("out shape");
        let input = values(41, (m * k) as usize);
        let (codes_a, scales_a) =
            pack_weights(&values(43, (n * k) as usize), n as usize, k as usize);
        let (codes_b, scales_b) =
            pack_weights(&values(47, (n * k) as usize), n as usize, k as usize);

        let gpu_in = backend.upload_f32(input_shape, &input).expect("gpu in");
        let gpu_ca = backend
            .upload_u8_classified(codes_shape, &codes_a, AllocationClass::Weight)
            .expect("gpu codes a");
        let gpu_sa = backend
            .upload_f32_classified(scales_shape, &scales_a, AllocationClass::Weight)
            .expect("gpu scales a");
        let gpu_cb = backend
            .upload_u8_classified(codes_shape, &codes_b, AllocationClass::Weight)
            .expect("gpu codes b");
        let gpu_sb = backend
            .upload_f32_classified(scales_shape, &scales_b, AllocationClass::Weight)
            .expect("gpu scales b");
        let cpu_in = reference.upload_f32(input_shape, &input).expect("cpu in");
        let cpu_ca = reference
            .upload_u8_classified(codes_shape, &codes_a, AllocationClass::Weight)
            .expect("cpu codes a");
        let cpu_sa = reference
            .upload_f32_classified(scales_shape, &scales_a, AllocationClass::Weight)
            .expect("cpu scales a");
        let cpu_cb = reference
            .upload_u8_classified(codes_shape, &codes_b, AllocationClass::Weight)
            .expect("cpu codes b");
        let cpu_sb = reference
            .upload_f32_classified(scales_shape, &scales_b, AllocationClass::Weight)
            .expect("cpu scales b");
        let mut gpu_a = backend.allocate_f32(out_shape).expect("gpu a");
        let mut gpu_b = backend.allocate_f32(out_shape).expect("gpu b");
        let mut cpu_a = reference.allocate_f32(out_shape).expect("cpu a");
        let mut cpu_b = reference.allocate_f32(out_shape).expect("cpu b");
        backend
            .packed_linear_pair(
                &mut gpu_a, &mut gpu_b, &gpu_in, &gpu_ca, &gpu_sa, &gpu_cb, &gpu_sb,
            )
            .expect("gpu packed pair");
        reference
            .packed_linear_pair(
                &mut cpu_a, &mut cpu_b, &cpu_in, &cpu_ca, &cpu_sa, &cpu_cb, &cpu_sb,
            )
            .expect("cpu packed pair");
        assert_close(&read(&backend, &gpu_a), cpu_a.as_slice(), 1e-4, 1e-4);
        assert_close(&read(&backend, &gpu_b), cpu_b.as_slice(), 1e-4, 1e-4);
    }
}

#[test]
fn packed_swiglu_linear_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    for (m, n, k) in [(1_u64, 37_u64, 256_u64), (5, 37, 384)] {
        let input_shape = Shape::new(&[m, k]).expect("input shape");
        let codes_shape = Shape::new(&[n, k / 4]).expect("codes shape");
        let scales_shape = Shape::new(&[n, k / 128]).expect("scales shape");
        let out_shape = Shape::new(&[m, n]).expect("out shape");
        let gate = values(51, (m * k) as usize);
        let up = values(53, (m * k) as usize);
        let (codes, scales) = pack_weights(&values(43, (n * k) as usize), n as usize, k as usize);

        let gpu_g = backend.upload_f32(input_shape, &gate).expect("gpu gate");
        let gpu_u = backend.upload_f32(input_shape, &up).expect("gpu up");
        let gpu_c = backend
            .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
            .expect("gpu codes");
        let gpu_s = backend
            .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
            .expect("gpu scales");
        let cpu_g = reference.upload_f32(input_shape, &gate).expect("cpu gate");
        let cpu_u = reference.upload_f32(input_shape, &up).expect("cpu up");
        let cpu_c = reference
            .upload_u8_classified(codes_shape, &codes, AllocationClass::Weight)
            .expect("cpu codes");
        let cpu_s = reference
            .upload_f32_classified(scales_shape, &scales, AllocationClass::Weight)
            .expect("cpu scales");
        let mut gpu_out = backend.allocate_f32(out_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(out_shape).expect("cpu out");
        backend
            .packed_swiglu_linear(&mut gpu_out, &gpu_g, &gpu_u, &gpu_c, &gpu_s)
            .expect("gpu packed swiglu linear");
        reference
            .packed_swiglu_linear(&mut cpu_out, &cpu_g, &cpu_u, &cpu_c, &cpu_s)
            .expect("cpu packed swiglu linear");
        assert_close(&read(&backend, &gpu_out), cpu_out.as_slice(), 1e-4, 1e-4);
    }
}

#[test]
fn add_row_rms_norm_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let shape = Shape::new(&[3, 257]).expect("shape");
    let weight_shape = Shape::new(&[257]).expect("weight shape");
    let left = values(59, 771);
    let right = values(61, 771);
    let weight = values(67, 257);
    let gpu_l = backend.upload_f32(shape, &left).expect("gpu left");
    let gpu_r = backend.upload_f32(shape, &right).expect("gpu right");
    let gpu_w = backend
        .upload_f32(weight_shape, &weight)
        .expect("gpu weight");
    let cpu_l = reference.upload_f32(shape, &left).expect("cpu left");
    let cpu_r = reference.upload_f32(shape, &right).expect("cpu right");
    let cpu_w = reference
        .upload_f32(weight_shape, &weight)
        .expect("cpu weight");
    let mut gpu_sum = backend.allocate_f32(shape).expect("gpu sum");
    let mut gpu_normed = backend.allocate_f32(shape).expect("gpu normed");
    let mut cpu_sum = reference.allocate_f32(shape).expect("cpu sum");
    let mut cpu_normed = reference.allocate_f32(shape).expect("cpu normed");
    backend
        .add_row_rms_norm(&mut gpu_sum, &mut gpu_normed, &gpu_l, &gpu_r, &gpu_w, 1e-5)
        .expect("gpu add norm");
    reference
        .add_row_rms_norm(&mut cpu_sum, &mut cpu_normed, &cpu_l, &cpu_r, &cpu_w, 1e-5)
        .expect("cpu add norm");
    assert_exact(&read(&backend, &gpu_sum), cpu_sum.as_slice());
    assert_close(
        &read(&backend, &gpu_normed),
        cpu_normed.as_slice(),
        1e-5,
        1e-5,
    );
}

#[test]
fn qk_norm_rope_matches_cpu() {
    let Some(mut backend) = gpu() else { return };
    let mut reference = cpu();
    let (tokens, q_heads, kv_heads, head_dim) = (2_u64, 4_u32, 2_u32, 8_u32);
    let q_width = u64::from(q_heads * head_dim);
    let kv_width = u64::from(kv_heads * head_dim);
    let q_shape = Shape::new(&[tokens, q_width]).expect("q shape");
    let k_shape = Shape::new(&[tokens, kv_width]).expect("k shape");
    let w_shape = Shape::new(&[u64::from(head_dim)]).expect("weight shape");
    let query = values(71, (tokens * q_width) as usize);
    let key = values(73, (tokens * kv_width) as usize);
    let qw = values(79, head_dim as usize);
    let kw = values(83, head_dim as usize);
    let positions = [3_u64, 17];
    let rope = RotarySpec::new(
        PackedHeadSpec::new(q_heads, head_dim).expect("q heads"),
        10_000.0,
    )
    .expect("rope spec");
    let kv_spec = PackedHeadSpec::new(kv_heads, head_dim).expect("kv heads");

    let gpu_q = backend.upload_f32(q_shape, &query).expect("gpu q");
    let gpu_k = backend.upload_f32(k_shape, &key).expect("gpu k");
    let gpu_qw = backend.upload_f32(w_shape, &qw).expect("gpu qw");
    let gpu_kw = backend.upload_f32(w_shape, &kw).expect("gpu kw");
    let cpu_q = reference.upload_f32(q_shape, &query).expect("cpu q");
    let cpu_k = reference.upload_f32(k_shape, &key).expect("cpu k");
    let cpu_qw = reference.upload_f32(w_shape, &qw).expect("cpu qw");
    let cpu_kw = reference.upload_f32(w_shape, &kw).expect("cpu kw");
    let mut gpu_qo = backend.allocate_f32(q_shape).expect("gpu q out");
    let mut gpu_ko = backend.allocate_f32(k_shape).expect("gpu k out");
    let mut cpu_qo = reference.allocate_f32(q_shape).expect("cpu q out");
    let mut cpu_ko = reference.allocate_f32(k_shape).expect("cpu k out");
    backend
        .qk_norm_rope(
            &mut gpu_qo,
            &mut gpu_ko,
            &gpu_q,
            &gpu_k,
            &gpu_qw,
            &gpu_kw,
            &positions,
            rope,
            kv_spec,
            1e-5,
        )
        .expect("gpu qk norm rope");
    reference
        .qk_norm_rope(
            &mut cpu_qo,
            &mut cpu_ko,
            &cpu_q,
            &cpu_k,
            &cpu_qw,
            &cpu_kw,
            &positions,
            rope,
            kv_spec,
            1e-5,
        )
        .expect("cpu qk norm rope");
    assert_close(&read(&backend, &gpu_qo), cpu_qo.as_slice(), 1e-5, 1e-5);
    assert_close(&read(&backend, &gpu_ko), cpu_ko.as_slice(), 1e-5, 1e-5);
}

/// Scratch latency probe (dev tool, not a gate): measures submit->confirm wall
/// time for empty and dispatch-only batches to split sync latency from GPU
/// execution. Run with `--nocapture`.
#[test]
fn sync_floor_probe() {
    let Some(mut backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping probe");
        return;
    };
    let shape = Shape::new(&[1, 256]).expect("shape");
    let a = backend
        .upload_f32(shape, &values(11, 256))
        .expect("upload a");
    let b = backend
        .upload_f32(shape, &values(12, 256))
        .expect("upload b");
    let mut out = backend.allocate_f32(shape).expect("out");

    let fence_ready = |f: &mut minifield_backend_wgpu::WgpuFence| {
        let mut polls = 0_u64;
        loop {
            match f.poll_step() {
                CompletionPoll::Pending => polls += 1,
                CompletionPoll::Ready(r) => {
                    r.expect("fence");
                    return polls;
                }
            }
        }
    };

    // Warmup: pipelines, pools, first-submission paths.
    for _ in 0..4 {
        backend.multiply(&mut out, &a, &b).expect("warmup op");
        let mut f = backend.fence().expect("warmup fence");
        fence_ready(&mut f);
    }

    for n in [0_u32, 1, 16, 64, 128, 256] {
        let start = std::time::Instant::now();
        for _ in 0..n {
            backend.multiply(&mut out, &a, &b).expect("op");
        }
        let encoded = start.elapsed();
        let mut f = backend.fence().expect("fence");
        let polls = fence_ready(&mut f);
        let total = start.elapsed();
        eprintln!(
            "n={n:>4}: encode={:.3}ms confirm={:.3}ms polls={polls}",
            encoded.as_secs_f64() * 1e3,
            total.saturating_sub(encoded).as_secs_f64() * 1e3,
        );
    }

    // Readback latency floor: 4B copy + map in an otherwise-idle batch.
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let mut rb = backend.read_f32_async(&a).expect("readback");
        let mut polls = 0_u64;
        loop {
            match rb.poll_step() {
                CompletionPoll::Pending => polls += 1,
                CompletionPoll::Ready(r) => {
                    r.expect("readback");
                    break;
                }
            }
        }
        eprintln!(
            "1KB readback: {:.3}ms polls={polls}",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}
