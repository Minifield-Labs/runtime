//! Independent scalar parity for complete-sequence encoder operations.
#![allow(clippy::expect_used, clippy::cast_precision_loss, clippy::float_cmp)]

mod common;
use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    EncoderOps, EncoderSegments, ExecutorError, GatedShortConvSpec, GqaSpec, ResourceLimits, Shape,
};

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: 1 << 28,
        max_total_bytes: 1 << 30,
        max_pending_operations: 2048,
    }
}

fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{index}: {actual}");
        assert!(
            (actual - expected).abs() <= 0.0001 + expected.abs() * 0.0001,
            "{index}: {actual} != {expected}"
        );
    }
}

#[test]
fn bidirectional_gqa_matches_scalar_across_head_groups_and_segment_boundaries() {
    let Some(mut gpu) = common::gpu(limits()) else {
        return;
    };
    let mut cpu = CpuBackend::new(13, limits());
    for (tokens, qheads, kvheads, dim) in
        [(1, 4, 2, 2), (7, 4, 2, 4), (65, 16, 8, 64), (257, 2, 1, 2)]
    {
        let spec = GqaSpec::new(qheads, kvheads, dim).expect("GQA");
        let qshape = Shape::new(&[tokens, u64::from(qheads) * u64::from(dim)]).expect("qshape");
        let kvshape = Shape::new(&[tokens, u64::from(kvheads) * u64::from(dim)]).expect("kvshape");
        let mut ids = vec![1; usize::try_from(tokens).expect("tokens")];
        if ids.len() > 2 {
            let middle = ids.len() / 2;
            ids[middle] = 0;
            ids[middle + 1..].fill(2);
        }
        let segments = EncoderSegments::new(ids).expect("segments");
        let query: Vec<f32> = (0..qshape.element_count().expect("elements"))
            .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.1)
            .collect();
        let key: Vec<f32> = (0..kvshape.element_count().expect("elements"))
            .map(|index| ((index * 11 % 31) as f32 - 15.0) * 0.1)
            .collect();
        let value: Vec<f32> = (0..kvshape.element_count().expect("elements"))
            .map(|index| ((index * 7 % 37) as f32 - 18.0) * 0.1)
            .collect();
        let cq = cpu.upload_f32(qshape, &query).expect("cpu q");
        let ck = cpu.upload_f32(kvshape, &key).expect("cpu k");
        let cv = cpu.upload_f32(kvshape, &value).expect("cpu v");
        let mut co = cpu.allocate_f32(qshape).expect("cpu output");
        cpu.bidirectional_gqa(&mut co, &cq, &ck, &cv, &segments, spec)
            .expect("CPU attention");
        let gq = gpu.upload_f32(qshape, &query).expect("GPU q");
        let gk = gpu.upload_f32(kvshape, &key).expect("GPU k");
        let gv = gpu.upload_f32(kvshape, &value).expect("GPU v");
        let mut go = gpu.allocate_f32(qshape).expect("GPU output");
        gpu.bidirectional_gqa(&mut go, &gq, &gk, &gv, &segments, spec)
            .expect("GPU attention");
        compare(&gpu.read_f32(&go).expect("readback"), co.as_slice());
    }
}

#[test]
fn bidirectional_gqa_guards_padded_flattened_workgroups() {
    let Some(mut gpu) = common::gpu(limits()) else {
        return;
    };
    let mut cpu = CpuBackend::new(13, limits());
    let tokens = 128_u64;
    let heads = 512_u32;
    let spec = GqaSpec::new(heads, 1, 2).expect("GQA");
    let query_shape = Shape::new(&[tokens, u64::from(heads) * 2]).expect("query shape");
    let kv_shape = Shape::new(&[tokens, 2]).expect("KV shape");
    let query = vec![
        0.0;
        usize::try_from(query_shape.element_count().expect("elements"))
            .expect("query size")
    ];
    let key =
        vec![0.0; usize::try_from(kv_shape.element_count().expect("elements")).expect("key size")];
    let value: Vec<_> = (0..tokens * 2)
        .map(|index| (index % 17) as f32 * 0.125 - 1.0)
        .collect();
    let segments =
        EncoderSegments::single(usize::try_from(tokens).expect("tokens")).expect("segments");
    // 65,536 logical workgroups flatten to (65,535, 2, 1). The extra
    // 65,534 groups must return before reading token or score storage.
    let cq = cpu.upload_f32(query_shape, &query).expect("CPU query");
    let ck = cpu.upload_f32(kv_shape, &key).expect("CPU key");
    let cv = cpu.upload_f32(kv_shape, &value).expect("CPU value");
    let mut co = cpu.allocate_f32(query_shape).expect("CPU output");
    cpu.bidirectional_gqa(&mut co, &cq, &ck, &cv, &segments, spec)
        .expect("CPU attention");
    let gq = gpu.upload_f32(query_shape, &query).expect("GPU query");
    let gk = gpu.upload_f32(kv_shape, &key).expect("GPU key");
    let gv = gpu.upload_f32(kv_shape, &value).expect("GPU value");
    let mut go = gpu.allocate_f32(query_shape).expect("GPU output");
    gpu.bidirectional_gqa(&mut go, &gq, &gk, &gv, &segments, spec)
        .expect("GPU attention");
    compare(&gpu.read_f32(&go).expect("GPU readback"), co.as_slice());
}

#[test]
fn attention_ignores_overflowed_masked_values_and_handles_extreme_active_scores() {
    let Some(mut gpu) = common::gpu(limits()) else {
        return;
    };
    let shape = Shape::new(&[3, 2]).expect("shape");
    let query = gpu.upload_f32(shape, &[1.0; 6]).expect("q");
    let key = gpu
        .upload_f32(shape, &[-1e30, -1e30, 1e30, 1e30, 1.0, 1.0])
        .expect("k");
    let base = gpu
        .upload_f32(shape, &[1.0, 2.0, 3.0, 4.0, 3e38, 3e38])
        .expect("base");
    let multiplier = gpu.upload_f32(shape, &[2.0; 6]).expect("multiplier");
    let mut value = gpu.allocate_f32(shape).expect("value");
    gpu.multiply(&mut value, &base, &multiplier)
        .expect("produce masked infinity");
    let mut output = gpu.allocate_f32(shape).expect("output");
    gpu.bidirectional_gqa(
        &mut output,
        &query,
        &key,
        &value,
        &EncoderSegments::new(vec![1, 1, 0]).expect("segments"),
        GqaSpec::new(1, 1, 2).expect("GQA"),
    )
    .expect("attention");
    compare(
        &gpu.read_f32(&output).expect("readback"),
        &[6.0, 8.0, 6.0, 8.0, 0.0, 0.0],
    );
}

#[test]
fn centered_convolution_matches_scalar_for_odd_even_and_segmented_shapes() {
    let Some(mut gpu) = common::gpu(limits()) else {
        return;
    };
    let mut cpu = CpuBackend::new(13, limits());
    for width in [1, 2, 3, 4, 7] {
        let tokens = 7_u64;
        let hidden = 3_u32;
        let pshape = Shape::new(&[tokens, u64::from(hidden) * 3]).expect("projection shape");
        let kshape = Shape::new(&[u64::from(hidden), u64::from(width)]).expect("kernel shape");
        let outshape = Shape::new(&[tokens, u64::from(hidden)]).expect("output shape");
        let projection: Vec<f32> = (0..pshape.element_count().expect("elements"))
            .map(|index| (index % 11) as f32 * 0.07 - 0.3)
            .collect();
        let kernel: Vec<f32> = (0..kshape.element_count().expect("elements"))
            .map(|index| (index % 7) as f32 * 0.1 - 0.2)
            .collect();
        let segments = EncoderSegments::new(vec![0, 1, 1, 2, 2, 2, 0]).expect("segments");
        let spec = GatedShortConvSpec::new(hidden, width).expect("spec");
        let cp = cpu.upload_f32(pshape, &projection).expect("CPU projection");
        let ck = cpu.upload_f32(kshape, &kernel).expect("CPU kernel");
        let mut co = cpu.allocate_f32(outshape).expect("CPU output");
        cpu.centered_gated_convolution(&mut co, &cp, &ck, &segments, spec)
            .expect("CPU convolution");
        let gp = gpu.upload_f32(pshape, &projection).expect("GPU projection");
        let gk = gpu.upload_f32(kshape, &kernel).expect("GPU kernel");
        let mut go = gpu.allocate_f32(outshape).expect("GPU output");
        gpu.centered_gated_convolution(&mut go, &gp, &gk, &segments, spec)
            .expect("GPU convolution");
        compare(&gpu.read_f32(&go).expect("readback"), co.as_slice());
    }
}

#[test]
fn encoder_rejects_invalid_geometry_before_recording_work() {
    let Some(mut gpu) = common::gpu(limits()) else {
        return;
    };
    let shape = Shape::new(&[2, 2]).expect("shape");
    let q = gpu.upload_f32(shape, &[0.0; 4]).expect("query");
    let k = gpu.upload_f32(shape, &[0.0; 4]).expect("key");
    let v = gpu.upload_f32(shape, &[0.0; 4]).expect("value");
    let mut output = gpu.allocate_f32(shape).expect("output");
    let result = gpu.bidirectional_gqa(
        &mut output,
        &q,
        &k,
        &v,
        &EncoderSegments::single(1).expect("segments"),
        GqaSpec::new(1, 1, 2).expect("GQA"),
    );
    assert!(matches!(result, Err(ExecutorError::InvalidShape(_))));
    gpu.bidirectional_gqa(
        &mut output,
        &q,
        &k,
        &v,
        &EncoderSegments::single(2).expect("segments"),
        GqaSpec::new(1, 1, 2).expect("GQA"),
    )
    .expect("recovery");
    compare(&gpu.read_f32(&output).expect("readback"), &[0.0; 4]);
}
