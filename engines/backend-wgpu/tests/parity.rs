//! Op-level parity tests: `minifield-backend-wgpu` versus the scalar CPU
//! baseline on synthetic tensors.
//!
//! Every test skips cleanly when no wgpu adapter exists (headless CI without
//! a GPU driver), so the suite is safe everywhere but verifies on Metal,
//! Vulkan, DX12, and WebGPU-capable drivers.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]

use minifield_backend_cpu::CpuBackend;
use minifield_backend_wgpu::{WgpuBackend, WgpuBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, GatedShortConvSpec, GqaSpec, InferenceCompletion,
    PackedHeadSpec, RectCopy2d, ResourceLimits, RotarySpec, Shape,
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
        .gather_rows(&mut gpu_out, &gpu_table, &ids)
        .expect("gpu gather");
    reference
        .gather_rows(&mut cpu_out, &cpu_table, &ids)
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
        let kernel_shape = Shape::new(&[16, 3]).expect("kernel shape");
        let history_shape = Shape::new(&[2, 16]).expect("history shape");
        let count = (tokens * 16) as usize;
        let b = values(101, count);
        let c = values(103, count);
        let v = values(107, count);
        let kernel = values(109, 48);
        let history = values(113, 32);
        let gpu_b = backend.upload_f32(token_shape, &b).expect("gpu b");
        let gpu_c = backend.upload_f32(token_shape, &c).expect("gpu c");
        let gpu_v = backend.upload_f32(token_shape, &v).expect("gpu v");
        let gpu_k = backend.upload_f32(kernel_shape, &kernel).expect("gpu k");
        let mut gpu_h = backend
            .upload_f32_classified(history_shape, &history, AllocationClass::Cache)
            .expect("gpu history");
        let cpu_b = reference.upload_f32(token_shape, &b).expect("cpu b");
        let cpu_c = reference.upload_f32(token_shape, &c).expect("cpu c");
        let cpu_v = reference.upload_f32(token_shape, &v).expect("cpu v");
        let cpu_k = reference.upload_f32(kernel_shape, &kernel).expect("cpu k");
        let mut cpu_h = reference
            .upload_f32_classified(history_shape, &history, AllocationClass::Cache)
            .expect("cpu history");
        let mut gpu_out = backend.allocate_f32(token_shape).expect("gpu out");
        let mut cpu_out = reference.allocate_f32(token_shape).expect("cpu out");
        backend
            .gated_short_convolution(
                &mut gpu_out,
                &gpu_b,
                &gpu_c,
                &gpu_v,
                &gpu_k,
                &mut gpu_h,
                spec,
            )
            .expect("gpu conv");
        reference
            .gated_short_convolution(
                &mut cpu_out,
                &cpu_b,
                &cpu_c,
                &cpu_v,
                &cpu_k,
                &mut cpu_h,
                spec,
            )
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
