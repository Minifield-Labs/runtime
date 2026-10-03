//! Explicit hardware gate. Run with --ignored --test-threads=1 on real Metal.
#![cfg(target_os = "macos")]
#![allow(clippy::expect_used, clippy::float_cmp, clippy::cast_precision_loss)]
use minifield_backend_cpu::CpuBackend;
use minifield_backend_metal::MetalBackend;
use minifield_engine_api::{
    AllocationClass, CompletionPoll, EncoderOps, EncoderSegments, ExecutorError, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, PackedHeadSpec, RectCopy2d,
    ResourceLimits, RotarySpec, Shape, TokenIds,
};
use std::time::{Duration, Instant};
fn limits() -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: 64 << 20,
        max_total_bytes: 256 << 20,
        max_pending_operations: 64,
    }
}
fn metal() -> MetalBackend {
    MetalBackend::new(77, limits())
        .expect("real native Metal device and shader compilation required")
}
fn wait<C: InferenceCompletion>(mut completion: C) -> C::Output {
    let start = Instant::now();
    loop {
        match completion.poll_step() {
            CompletionPoll::Ready(result) => return result.expect("completion"),
            CompletionPoll::Pending => {
                assert!(
                    start.elapsed() < Duration::from_secs(20),
                    "native Metal completion deadline"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}
fn read<B: InferenceOps>(backend: &B, buffer: &B::Buffer) -> Vec<f32> {
    wait(backend.read_f32_async(buffer).expect("readback"))
}
fn close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "nonfinite {i}");
        assert!(
            (a - b).abs() <= 3e-4 + 3e-4 * a.abs(),
            "index {i}: CPU={a}, Metal={b}"
        );
    }
}
fn dense<B: InferenceOps>(b: &mut B) -> Vec<f32> {
    let input = b
        .upload_f32(
            Shape::new(&[3, 5]).expect("shape"),
            &[
                1., 2., -1., 0.5, 3., -2., 0., 1., 1.5, -1., 0.25, 0.5, 0.75, 1., 1.25,
            ],
        )
        .expect("input");
    let weight = b
        .upload_f32(
            Shape::new(&[2, 5]).expect("shape"),
            &[0.2, 0.3, -0.5, 1., 0.75, 1., -1., 0.25, 0.5, 0.],
        )
        .expect("weight");
    let mut out = b
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("output");
    b.linear(&mut out, &input, &weight).expect("linear");
    let mut result = read(b, &out);
    let norm = b
        .upload_f32(Shape::new(&[2]).expect("shape"), &[1.5, 0.5])
        .expect("norm weight");
    let mut normalized = b
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("normalized");
    b.row_rms_norm(&mut normalized, &out, &norm, 1e-5)
        .expect("rms");
    result.extend(read(b, &normalized));
    let mut sum = b
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("sum");
    let mut fused = b
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("fused");
    b.add_row_rms_norm(&mut sum, &mut fused, &out, &normalized, &norm, 1e-5)
        .expect("fused rms");
    result.extend(read(b, &sum));
    result.extend(read(b, &fused));
    let mut activation = b
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("act");
    b.swiglu(&mut activation, &out, &normalized)
        .expect("swiglu");
    result.extend(read(b, &activation));
    b.multiply(&mut activation, &out, &normalized)
        .expect("multiply");
    result.extend(read(b, &activation));
    let mut gathered = b
        .allocate_f32(Shape::new(&[2, 5]).expect("shape"))
        .expect("gathered");
    b.gather_rows(&mut gathered, &input, TokenIds::Host(&[2, 0]))
        .expect("gather");
    result.extend(read(b, &gathered));
    let mut columns = b
        .allocate_f32(Shape::new(&[3, 3]).expect("shape"))
        .expect("columns");
    b.gather_columns(&mut columns, &input, &[4, 0, 4])
        .expect("columns");
    result.extend(read(b, &columns));
    let mut copied = b
        .allocate_f32(Shape::new(&[3, 3]).expect("shape"))
        .expect("copy");
    b.copy(&mut copied, &columns).expect("copy");
    b.copy_rect_2d(&mut copied, &input, RectCopy2d::new(1, 1, 0, 0, 2, 2))
        .expect("rectangle");
    result.extend(read(b, &copied));
    let mut max = b
        .allocate_f32(Shape::new(&[3]).expect("shape"))
        .expect("max");
    b.argmax(&mut max, &input).expect("argmax");
    result.extend(read(b, &max));
    b.argmax_masked(&mut max, &input, &[0b00111])
        .expect("masked max");
    result.extend(read(b, &max));
    result
}
#[test]
#[ignore = "requires real native Metal"]
fn dense_norm_copy_selectors_and_fused_operations_match_cpu() {
    let mut cpu = CpuBackend::new(1, limits());
    let mut gpu = metal();
    close(&dense(&mut cpu), &dense(&mut gpu));
    assert!(
        gpu.dispatch_counts()
            .get("dense_linear")
            .copied()
            .unwrap_or(0)
            > 0
    );
    assert_eq!(gpu.device_info().api, "metal");
}
fn packed<B: InferenceOps>(b: &mut B, format: u32) -> Vec<f32> {
    let width = 256_usize;
    let rows = 3_usize;
    let bytes = match format {
        0 => width / 4,
        1 => width / 2,
        _ => width,
    };
    let stream: Vec<_> = (0..rows * bytes)
        .map(|i| match format {
            0 => 0x24_u8,
            1 => u8::try_from(i % 256).expect("byte"),
            _ => {
                if i % 2 == 0 {
                    3
                } else {
                    251
                }
            }
        })
        .collect();
    let codes = b
        .upload_u8_classified(
            Shape::new(&[rows as u64, bytes as u64]).expect("shape"),
            &stream,
            AllocationClass::Weight,
        )
        .expect("codes");
    let scales = b
        .upload_f32(
            Shape::new(&[3, 2]).expect("shape"),
            &[0.125, 0.25, 0.5, 0.125, 0.25, 0.5],
        )
        .expect("scales");
    let values: Vec<_> = (0..2 * width)
        .map(|i| ((i % 13) as f32 - 6.) * 0.03125)
        .collect();
    let input = b
        .upload_f32(Shape::new(&[2, 256]).expect("shape"), &values)
        .expect("input");
    let mut out = b
        .allocate_f32(Shape::new(&[2, 3]).expect("shape"))
        .expect("out");
    b.packed_linear(&mut out, &input, &codes, &scales)
        .expect("packed");
    let mut result = read(b, &out);
    let mut second = b
        .allocate_f32(Shape::new(&[2, 3]).expect("shape"))
        .expect("second");
    b.packed_linear_pair(
        &mut out,
        &mut second,
        &input,
        &codes,
        &scales,
        &codes,
        &scales,
    )
    .expect("pair");
    result.extend(read(b, &out));
    result.extend(read(b, &second));
    b.packed_swiglu_pair(&mut out, &input, &codes, &scales, &codes, &scales)
        .expect("packed swiglu pair");
    result.extend(read(b, &out));
    b.packed_swiglu_linear(&mut out, &input, &input, &codes, &scales)
        .expect("packed swiglu linear");
    result.extend(read(b, &out));
    let mut gather = b
        .allocate_f32(Shape::new(&[2, 256]).expect("shape"))
        .expect("gather");
    b.packed_gather_rows(&mut gather, &codes, &scales, TokenIds::Host(&[2, 0]))
        .expect("packed gather");
    result.extend(read(b, &gather));
    result
}
#[test]
#[ignore = "requires real native Metal"]
fn ternary_nf4_and_signed_int8_streams_match_cpu() {
    for format in 0..3 {
        let mut cpu = CpuBackend::new(1, limits());
        let mut gpu = metal();
        close(&packed(&mut cpu, format), &packed(&mut gpu, format));
    }
}
fn attention_conv<B: EncoderOps>(b: &mut B) -> Vec<f32> {
    let heads = PackedHeadSpec::new(2, 4).expect("heads");
    let kvheads = PackedHeadSpec::new(1, 4).expect("kv heads");
    let spec = GqaSpec::new(2, 1, 4).expect("GQA");
    let values: Vec<_> = (0..32).map(|i| (i as f32 - 15.) * 0.025).collect();
    let q = b
        .upload_f32(Shape::new(&[4, 8]).expect("shape"), &values)
        .expect("q");
    let kv: Vec<_> = (0..16).map(|i| (i as f32 - 7.) * 0.05).collect();
    let k = b
        .upload_f32(Shape::new(&[4, 4]).expect("shape"), &kv)
        .expect("k");
    let v = b
        .upload_f32(
            Shape::new(&[4, 4]).expect("shape"),
            &kv.iter().map(|x| x + 0.1).collect::<Vec<_>>(),
        )
        .expect("v");
    let qweight = b
        .upload_f32(Shape::new(&[4]).expect("shape"), &[1., 0.8, 1.2, 1.])
        .expect("qweight");
    let kweight = b
        .upload_f32(Shape::new(&[4]).expect("shape"), &[0.9, 1., 1., 1.1])
        .expect("kweight");
    let mut qo = b
        .allocate_f32(Shape::new(&[4, 8]).expect("shape"))
        .expect("qo");
    let mut ko = b
        .allocate_f32(Shape::new(&[4, 4]).expect("shape"))
        .expect("ko");
    b.qk_norm_rope(
        &mut qo,
        &mut ko,
        &q,
        &k,
        &qweight,
        &kweight,
        &[0, 1, 0, 0],
        RotarySpec::new(heads, 10000.).expect("rope"),
        kvheads,
        1e-5,
    )
    .expect("QK norm rope");
    let mut result = read(b, &qo);
    result.extend(read(b, &ko));
    let mut out = b
        .allocate_f32(Shape::new(&[4, 8]).expect("shape"))
        .expect("attention out");
    let segments = EncoderSegments::new(vec![1, 1, 2, 0]).expect("segments");
    b.bidirectional_gqa(&mut out, &qo, &ko, &v, &segments, spec)
        .expect("encoder GQA");
    result.extend(read(b, &out));
    let mut kc = b
        .allocate_f32(Shape::new(&[6, 4]).expect("shape"))
        .expect("key cache");
    let mut vc = b
        .allocate_f32(Shape::new(&[6, 4]).expect("shape"))
        .expect("value cache");
    let mut length = 0;
    b.causal_gqa(&mut out, &qo, &ko, &v, &mut kc, &mut vc, &mut length, spec)
        .expect("causal GQA");
    assert_eq!(length, 4);
    result.extend(read(b, &out));
    let conv = GatedShortConvSpec::new(2, 3).expect("conv");
    let pv: Vec<_> = (0..24).map(|i| (i as f32 - 10.) * 0.0625).collect();
    let projection = b
        .upload_f32(Shape::new(&[4, 6]).expect("shape"), &pv)
        .expect("projection");
    let kernel = b
        .upload_f32(
            Shape::new(&[2, 3]).expect("shape"),
            &[0.25, 0.5, -0.125, 0.75, -0.25, 0.125],
        )
        .expect("kernel");
    let mut history = b
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[0.1, 0.2, 0.3, 0.4])
        .expect("history");
    let mut co = b
        .allocate_f32(Shape::new(&[4, 2]).expect("shape"))
        .expect("conv output");
    b.gated_short_convolution(&mut co, &projection, &kernel, &mut history, conv)
        .expect("causal conv");
    result.extend(read(b, &co));
    result.extend(read(b, &history));
    b.centered_gated_convolution(&mut co, &projection, &kernel, &segments, conv)
        .expect("encoder conv");
    result.extend(read(b, &co));
    result
}
#[test]
#[ignore = "requires real native Metal"]
fn causal_and_segmented_attention_convolution_rope_match_cpu() {
    let mut cpu = CpuBackend::new(1, limits());
    let mut gpu = metal();
    close(&attention_conv(&mut cpu), &attention_conv(&mut gpu));
}
#[test]
#[ignore = "requires real native Metal"]
fn foreign_and_stale_buffers_reject_before_dispatch() {
    let mut a = metal();
    let mut b = metal();
    let source = a
        .upload_f32(Shape::new(&[1, 2]).expect("shape"), &[1., 2.])
        .expect("source");
    let mut output = b
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("output");
    assert_eq!(
        b.copy(&mut output, &source),
        Err(ExecutorError::WrongBackend)
    );
    let mut own = a
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("own");
    a.advance_generation().expect("generation");
    assert_eq!(a.copy(&mut own, &source), Err(ExecutorError::StaleBuffer));
    assert!(a.dispatch_counts().is_empty());
}
#[test]
#[ignore = "requires real native Metal"]
fn readback_is_snapshot_and_queued_buffers_survive_drop() {
    let mut gpu = metal();
    let source = gpu
        .upload_f32(Shape::new(&[4]).expect("shape"), &[1., 2., 3., 4.])
        .expect("source");
    let mut destination = gpu
        .allocate_f32(Shape::new(&[4]).expect("shape"))
        .expect("destination");
    gpu.copy(&mut destination, &source).expect("copy");
    let snapshot = gpu.read_f32_async(&destination).expect("snapshot");
    let replacement = gpu
        .upload_f32(Shape::new(&[4]).expect("shape"), &[5., 6., 7., 8.])
        .expect("replacement");
    gpu.copy(&mut destination, &replacement).expect("replace");
    drop(source);
    drop(replacement);
    assert_eq!(wait(snapshot), vec![1., 2., 3., 4.]);
    assert_eq!(read(&gpu, &destination), vec![5., 6., 7., 8.]);
    let peak = gpu.peak_accounted_bytes().expect("peak");
    assert!(peak >= 64);
    drop(destination);
    wait(gpu.fence().expect("fence"));
    assert_eq!(gpu.resource_report().total_owned_bytes().expect("total"), 0);
}
#[test]
#[ignore = "requires real native Metal"]
fn retirement_preserves_rejected_payload_and_unresolved_submission() {
    let mut a = metal();
    let b = metal();
    let buffer = a
        .upload_f32(Shape::new(&[4]).expect("shape"), &[1., 2., 3., 4.])
        .expect("buffer");
    let mut fence = a.fence().expect("fence");
    assert!(matches!(fence.cancel(), Err(ExecutorError::Unsupported(_))));
    let rejected = b
        .fence_retirement()
        .retire(fence, vec![buffer])
        .expect_err("foreign retirement rejected");
    assert_eq!(rejected.cause(), &ExecutorError::WrongBackend);
    let (_, fence, retained) = rejected.into_parts();
    assert!(
        a.fence_retirement().retire(fence, retained).is_ok(),
        "own retirement"
    );
    let queue = a.fence_retirement();
    let start = Instant::now();
    while queue.has_unresolved() {
        queue.poll_retired();
        assert!(start.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(a.resource_report().total_owned_bytes().expect("total"), 0);
}
#[test]
#[ignore = "requires real native Metal"]
fn empty_readback_and_nonfinite_device_selectors_are_defined() {
    let mut gpu = metal();
    let empty = gpu
        .allocate_f32(Shape::new(&[0, 3]).expect("shape"))
        .expect("empty");
    assert!(read(&gpu, &empty).is_empty());
    let table = gpu
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[1., 2., 3., 4.])
        .expect("table");
    let ids = gpu
        .upload_f32(
            Shape::new(&[3]).expect("shape"),
            &[f32::NAN, f32::INFINITY, -1.],
        )
        .expect("ids");
    let mut output = gpu
        .allocate_f32(Shape::new(&[3, 2]).expect("shape"))
        .expect("out");
    gpu.gather_rows(&mut output, &table, TokenIds::Device(&ids))
        .expect("gather");
    assert!(read(&gpu, &output).iter().all(|value| value.is_nan()));
}

#[test]
#[ignore = "requires real native Metal"]
fn masked_encoder_values_never_read_inactive_infinities() {
    let mut gpu = metal();
    let q = gpu
        .upload_f32(Shape::new(&[2, 2]).expect("shape"), &[1., 1., 1., 1.])
        .expect("query");
    let k = gpu
        .upload_f32(
            Shape::new(&[2, 2]).expect("shape"),
            &[0.5, 0.5, f32::INFINITY, f32::INFINITY],
        )
        .expect("key");
    let v = gpu
        .upload_f32(
            Shape::new(&[2, 2]).expect("shape"),
            &[2., 3., f32::INFINITY, f32::INFINITY],
        )
        .expect("value");
    let mut out = gpu
        .allocate_f32(Shape::new(&[2, 2]).expect("shape"))
        .expect("output");
    gpu.bidirectional_gqa(
        &mut out,
        &q,
        &k,
        &v,
        &EncoderSegments::new(vec![1, 0]).expect("segments"),
        GqaSpec::new(1, 1, 2).expect("spec"),
    )
    .expect("GQA");
    assert_eq!(read(&gpu, &out), vec![2., 3., 0., 0.]);
}

#[test]
#[ignore = "requires real native Metal"]
fn invalid_qk_head_dimensions_preserve_outputs_counts_and_resources() {
    let mut gpu = metal();
    let q = gpu
        .upload_f32(Shape::new(&[1, 2]).expect("query shape"), &[1., 2.])
        .expect("query");
    let qw = gpu
        .upload_f32(Shape::new(&[2]).expect("weight shape"), &[1., 1.])
        .expect("query weight");
    for dimension in [3_u32, 4] {
        let key_values = vec![1.; usize::try_from(dimension).expect("dimension")];
        let k = gpu
            .upload_f32(
                Shape::new(&[1, u64::from(dimension)]).expect("key shape"),
                &key_values,
            )
            .expect("key");
        let kw = gpu
            .upload_f32(
                Shape::new(&[u64::from(dimension)]).expect("key weight shape"),
                &key_values,
            )
            .expect("key weight");
        let mut qo = gpu
            .upload_f32(Shape::new(&[1, 2]).expect("query output shape"), &[8., 9.])
            .expect("query output");
        let sentinel = vec![7.; usize::try_from(dimension).expect("dimension")];
        let mut ko = gpu
            .upload_f32(
                Shape::new(&[1, u64::from(dimension)]).expect("key output shape"),
                &sentinel,
            )
            .expect("key output");
        let counts = gpu.dispatch_counts();
        let resources = gpu.resource_report();
        let peak = gpu.peak_accounted_bytes().expect("peak");
        assert!(
            gpu.qk_norm_rope(
                &mut qo,
                &mut ko,
                &q,
                &k,
                &qw,
                &kw,
                &[0],
                RotarySpec::new(PackedHeadSpec::new(1, 2).expect("query heads"), 10000.)
                    .expect("query rotary"),
                PackedHeadSpec::new(1, dimension).expect("key heads"),
                1e-5
            )
            .is_err()
        );
        assert_eq!(gpu.dispatch_counts(), counts);
        assert_eq!(gpu.resource_report(), resources);
        assert_eq!(gpu.peak_accounted_bytes().expect("peak"), peak);
        assert_eq!(read(&gpu, &qo), vec![8., 9.]);
        assert_eq!(read(&gpu, &ko), sentinel);
    }
}

#[test]
#[ignore = "requires real native Metal"]
fn readback_fits_four_independent_sixteen_byte_allocations() {
    let mut gpu = MetalBackend::new(
        89,
        ResourceLimits {
            max_allocation_bytes: 16,
            max_total_bytes: 64,
            max_pending_operations: 2,
        },
    )
    .expect("native Metal");
    let source = gpu
        .upload_f32(Shape::new(&[4]).expect("shape"), &[1., 2., 3., 4.])
        .expect("source");
    assert_eq!(read(&gpu, &source), vec![1., 2., 3., 4.]);
    assert_eq!(gpu.peak_accounted_bytes().expect("peak"), 64);
    assert_eq!(
        gpu.resource_report().total_owned_bytes().expect("total"),
        16
    );
}

#[test]
#[ignore = "requires real native Metal"]
fn dense_and_packed_gather_compare_large_row_bounds_as_integers() {
    // The last valid id is exactly representable; the table's odd row count
    // rounds down if converted to F32. This also exercises the uint cast guard.
    const ROWS: u64 = 16_777_217;
    let mut gpu = MetalBackend::new(
        90,
        ResourceLimits {
            max_allocation_bytes: 1 << 30,
            max_total_bytes: 2 << 30,
            max_pending_operations: 8,
        },
    )
    .expect("native Metal");
    let ids = gpu
        .upload_f32(
            Shape::new(&[3]).expect("ids shape"),
            &[16_777_216., 16_777_218., 4_294_967_296.],
        )
        .expect("device ids");
    let table = gpu
        .allocate_f32(Shape::new(&[ROWS, 1]).expect("dense table shape"))
        .expect("dense table");
    let mut selected = gpu
        .allocate_f32(Shape::new(&[3, 1]).expect("dense output shape"))
        .expect("dense output");
    gpu.gather_rows(&mut selected, &table, TokenIds::Device(&ids))
        .expect("dense gather");
    let values = read(&gpu, &selected);
    assert_eq!(values[0], 0.);
    assert!(values[1..].iter().all(|value| value.is_nan()));
    drop(table);
    drop(selected);

    let code_bytes = usize::try_from(ROWS * 32).expect("code bytes");
    let codes = gpu
        .upload_u8_classified(
            Shape::new(&[ROWS, 32]).expect("codes shape"),
            &vec![0; code_bytes],
            AllocationClass::Weight,
        )
        .expect("ternary codes");
    let scales = gpu
        .allocate_f32(Shape::new(&[ROWS, 1]).expect("scale shape"))
        .expect("zero scales");
    let mut selected = gpu
        .allocate_f32(Shape::new(&[3, 128]).expect("packed output shape"))
        .expect("packed output");
    gpu.packed_gather_rows(&mut selected, &codes, &scales, TokenIds::Device(&ids))
        .expect("packed gather");
    let values = read(&gpu, &selected);
    assert!(values[..128].iter().all(|&value| value == 0.));
    assert!(values[128..].iter().all(|value| value.is_nan()));
}

fn short_history_and_even_convolution<B: EncoderOps>(backend: &mut B) -> Vec<f32> {
    let projection = backend
        .upload_f32(
            Shape::new(&[1, 6]).expect("shape"),
            &[2., 3., 4., 5., 6., 7.],
        )
        .expect("projection");
    let kernel = backend
        .upload_f32(
            Shape::new(&[2, 4]).expect("shape"),
            &[0.5, 0.25, 0.125, 1., 0.5, 0.25, 0.125, 1.],
        )
        .expect("kernel");
    let mut history = backend
        .upload_f32(
            Shape::new(&[3, 2]).expect("shape"),
            &[1., 2., 3., 4., 5., 6.],
        )
        .expect("history");
    let mut output = backend
        .allocate_f32(Shape::new(&[1, 2]).expect("shape"))
        .expect("output");
    backend
        .gated_short_convolution(
            &mut output,
            &projection,
            &kernel,
            &mut history,
            GatedShortConvSpec::new(2, 4).expect("spec"),
        )
        .expect("short causal convolution");
    let mut values = read(backend, &output);
    values.extend(read(backend, &history));
    let projection = backend
        .upload_f32(
            Shape::new(&[3, 3]).expect("shape"),
            &[1., 1., 1., 1., 1., 2., 1., 1., 3.],
        )
        .expect("even projection");
    let kernel = backend
        .upload_f32(Shape::new(&[1, 4]).expect("shape"), &[1., 2., 3., 4.])
        .expect("even kernel");
    let mut output = backend
        .allocate_f32(Shape::new(&[3, 1]).expect("shape"))
        .expect("even output");
    backend
        .centered_gated_convolution(
            &mut output,
            &projection,
            &kernel,
            &EncoderSegments::single(3).expect("single segment"),
            GatedShortConvSpec::new(1, 4).expect("spec"),
        )
        .expect("even centered convolution");
    values.extend(read(backend, &output));
    backend
        .centered_gated_convolution(
            &mut output,
            &projection,
            &kernel,
            &EncoderSegments::new(vec![1, 1, 2]).expect("segments"),
            GatedShortConvSpec::new(1, 4).expect("spec"),
        )
        .expect("even segmented convolution");
    values.extend(read(backend, &output));
    values
}
#[test]
#[ignore = "requires real native Metal"]
fn short_history_updates_and_even_centered_crop_match_cpu() {
    let mut cpu = CpuBackend::new(1, limits());
    let mut gpu = metal();
    let expected = vec![
        55.5, 118.75, 3., 4., 5., 6., 12., 21., 11., 20., 14., 11., 8., 9.,
    ];
    assert_eq!(short_history_and_even_convolution(&mut cpu), expected);
    assert_eq!(short_history_and_even_convolution(&mut gpu), expected);
}

#[test]
#[ignore = "requires real native Metal"]
fn dependent_dispatches_survive_copy_and_submission_boundaries() {
    let mut gpu = metal();
    let shape = Shape::new(&[4]).expect("shape");
    let input = gpu.upload_f32(shape, &[1., 2., 3., 4.]).expect("input");
    let mut doubled = gpu.allocate_f32(shape).expect("doubled");
    let mut squared = gpu.allocate_f32(shape).expect("squared");
    let mut copied = gpu.allocate_f32(shape).expect("copied");
    let mut result = gpu.allocate_f32(shape).expect("result");
    gpu.add(&mut doubled, &input, &input)
        .expect("first compute");
    gpu.multiply(&mut squared, &doubled, &doubled)
        .expect("dependent compute");
    gpu.copy(&mut copied, &squared).expect("blit boundary");
    gpu.add(&mut result, &copied, &input)
        .expect("compute after blit");
    assert_eq!(read(&gpu, &result), [5., 18., 39., 68.]);
    gpu.add(&mut result, &copied, &doubled)
        .expect("new submission");
    assert_eq!(read(&gpu, &result), [6., 20., 42., 72.]);
}
