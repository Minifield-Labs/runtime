//! Hardware conformance SOURCE for the coherent signed-byte loader.
//! Run only after the frozen A/A audits and manual execution admission.
#![allow(clippy::expect_used)]

use minifield_backend_wgpu::{WgpuBackend, WgpuBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, ExecutorError, InferenceCompletion, ResourceLimits, Shape,
};
use std::time::{Duration, Instant};

mod common;

const LIMITS: ResourceLimits = ResourceLimits {
    max_allocation_bytes: 64 << 20,
    max_total_bytes: 512 << 20,
    max_pending_operations: 64,
};

fn shape(m: usize, n: usize) -> Shape {
    Shape::new(&[
        u64::try_from(m).expect("rows"),
        u64::try_from(n).expect("columns"),
    ])
    .expect("shape")
}

fn read(backend: &WgpuBackend, buffer: &WgpuBuffer) -> Vec<f32> {
    let mut task = backend.read_f32_async(buffer).expect("readback");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(Instant::now() < deadline, "INT8 readback deadline");
        match task.poll_step() {
            CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            CompletionPoll::Ready(result) => return result.expect("completion"),
        }
    }
}

fn settle(backend: &WgpuBackend) {
    let mut task = backend.fence().expect("fence");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(Instant::now() < deadline, "INT8 fence deadline");
        match task.poll_step() {
            CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            CompletionPoll::Ready(result) => return result.expect("completion"),
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "{label}[{index}] nonfinite"
        );
        assert!(
            (got - want).abs() <= 0.0001 + 0.0001 * want.abs(),
            "{label}[{index}]: {got} != {want}"
        );
    }
}

struct HostWeights {
    n: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
}
struct Weights {
    codes: WgpuBuffer,
    scales: WgpuBuffer,
}

impl HostWeights {
    fn new(n: usize, k: usize, stream: usize) -> Self {
        let codes = (0..n * k)
            .map(|index| {
                let row = index / k;
                let col = index % k;
                // Exhausts all bytes in every 256-column row, including 0x80.
                // The operation decoder supports that byte; model-bundle admission
                // still prohibits -128 in exported signed-INT8 model tensors.
                u8::try_from((row * 71 + col * 17 + stream * 113) % 256).expect("byte")
            })
            .collect();
        let groups = k / 128;
        let scales = (0..n * groups)
            .map(|index| {
                let mantissa = u16::try_from(
                    1 + 2 * ((index / groups * 37 + index % groups * 83 + stream * 19) % 511),
                )
                .expect("mantissa");
                let bits = (if stream == 0 { 0x2400 } else { 0x2800 }) | mantissa;
                normal_positive_half(bits)
            })
            .collect();
        Self {
            n,
            k,
            codes,
            scales,
        }
    }

    fn upload(&self, backend: &mut WgpuBackend) -> Weights {
        Weights {
            codes: backend
                .upload_u8_classified(shape(self.n, self.k), &self.codes, AllocationClass::Weight)
                .expect("codes"),
            scales: backend
                .upload_f32_classified(
                    shape(self.n, self.k / 128),
                    &self.scales,
                    AllocationClass::Weight,
                )
                .expect("scales"),
        }
    }

    fn coefficient(&self, row: usize, col: usize) -> f32 {
        f32::from(i8::from_ne_bytes([self.codes[row * self.k + col]]))
            * self.scales[row * (self.k / 128) + col / 128]
    }

    fn project(&self, input: &[f32], m: usize) -> Vec<f32> {
        let mut result = vec![0.0; m * self.n];
        for t in 0..m {
            for row in 0..self.n {
                let mut sum = 0.0_f32;
                for col in 0..self.k {
                    // Preserve baseline dequantize-before-product, ascending K.
                    let weight = self.coefficient(row, col);
                    sum += input[t * self.k + col] * weight;
                }
                result[t * self.n + row] = sum;
            }
        }
        result
    }
}

// Exact widening of a normal positive IEEE binary16. No exporter/library code
// is used by this independent fixture, and no rounding is needed for F32.
fn normal_positive_half(bits: u16) -> f32 {
    let exponent = u32::from((bits >> 10) & 31);
    assert!((1..31).contains(&exponent) && bits & 0x8000 == 0);
    f32::from_bits(((exponent + 112) << 23) | (u32::from(bits & 1023) << 13))
}

fn input_values(m: usize, k: usize, stream: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| {
            let value =
                i16::try_from((i / k * 7 + i % k * 11 + stream * 13) % 23).expect("input") - 11;
            f32::from(value) / 256.0
        })
        .collect()
}

fn swiglu(gate: f32, up: f32) -> f32 {
    (gate / (1.0 + (-gate).exp())) * up
}

fn exercise_forms(backend: &mut WgpuBackend, m: usize, a: &HostWeights, b: &HostWeights) {
    let wa = a.upload(backend);
    let wb = b.upload(backend);
    let x = input_values(m, a.k, 0);
    let gate = input_values(m, a.k, 1);
    let up = input_values(m, a.k, 2);
    let xb = backend.upload_f32(shape(m, a.k), &x).expect("input");
    let gb = backend.upload_f32(shape(m, a.k), &gate).expect("gate");
    let ub = backend.upload_f32(shape(m, a.k), &up).expect("up");
    let mut single = backend.allocate_f32(shape(m, a.n)).expect("single");
    let mut pair_a = backend.allocate_f32(shape(m, a.n)).expect("pair A");
    let mut pair_b = backend.allocate_f32(shape(m, a.n)).expect("pair B");
    let mut producer = backend.allocate_f32(shape(m, a.n)).expect("producer");
    let mut consumer = backend.allocate_f32(shape(m, a.n)).expect("consumer");
    backend
        .packed_linear(&mut single, &xb, &wa.codes, &wa.scales)
        .expect("single dispatch");
    backend
        .packed_linear_pair(
            &mut pair_a,
            &mut pair_b,
            &xb,
            &wa.codes,
            &wa.scales,
            &wb.codes,
            &wb.scales,
        )
        .expect("pair dispatch");
    backend
        .packed_swiglu_pair(
            &mut producer,
            &xb,
            &wa.codes,
            &wa.scales,
            &wb.codes,
            &wb.scales,
        )
        .expect("producer dispatch");
    backend
        .packed_swiglu_linear(&mut consumer, &gb, &ub, &wa.codes, &wa.scales)
        .expect("consumer dispatch");
    let expected_a = a.project(&x, m);
    let expected_b = b.project(&x, m);
    let expected_producer = expected_a
        .iter()
        .zip(&expected_b)
        .map(|(&g, &u)| swiglu(g, u))
        .collect::<Vec<_>>();
    let fused = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| swiglu(g, u))
        .collect::<Vec<_>>();
    let expected_consumer = a.project(&fused, m);
    let label = format!("M={m} N={} K={}", a.n, a.k);
    assert_close(
        &read(backend, &single),
        &expected_a,
        &format!("{label} single"),
    );
    assert_close(
        &read(backend, &pair_a),
        &expected_a,
        &format!("{label} pair A"),
    );
    assert_close(
        &read(backend, &pair_b),
        &expected_b,
        &format!("{label} pair B"),
    );
    assert_close(
        &read(backend, &producer),
        &expected_producer,
        &format!("{label} producer"),
    );
    assert_close(
        &read(backend, &consumer),
        &expected_consumer,
        &format!("{label} consumer"),
    );
    backend
        .resource_report()
        .validate(LIMITS)
        .expect("bounded resources");
}

#[test]
fn every_boundary_shape_matches_independent_math_in_all_four_forms() {
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    let before = backend.dispatch_counts();
    for n in [1_usize, 17, 31, 32, 33] {
        for k in [128_usize, 256, 384] {
            let a = HostWeights::new(n, k, 0);
            let b = HostWeights::new(n, k, 1);
            assert_ne!(a.codes, b.codes);
            assert_ne!(a.scales, b.scales);
            for m in [1_usize, 2, 4, 8, 16, 63, 64, 65] {
                exercise_forms(&mut backend, m, &a, &b);
            }
        }
    }
    let after = backend.dispatch_counts();
    for name in [
        "packed_gemm_int8",
        "packed_gemm_pair_int8",
        "packed_swiglu_gemm_int8",
        "packed_gemm_pair_swiglu_int8",
    ] {
        assert_eq!(
            after.get(name).copied().unwrap_or(0) - before.get(name).copied().unwrap_or(0),
            120,
            "{name}"
        );
    }
}

#[test]
fn word_and_group_boundary_impulses_select_exact_coefficients() {
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    for k in [128_usize, 256, 384] {
        let host = HostWeights::new(33, k, 0);
        let weights = host.upload(&mut backend);
        let positions = [0_usize, 3, 4, 15, 16, 127, 128, 255, 256, 383]
            .into_iter()
            .filter(|&p| p < k)
            .collect::<Vec<_>>();
        let m = positions.len();
        let mut x = vec![0.0; m * k];
        for (row, &col) in positions.iter().enumerate() {
            x[row * k + col] = 1.0;
        }
        let input = backend.upload_f32(shape(m, k), &x).expect("impulses");
        let mut output = backend.allocate_f32(shape(m, 33)).expect("output");
        backend
            .packed_linear(&mut output, &input, &weights.codes, &weights.scales)
            .expect("impulse dispatch");
        let expected = positions
            .iter()
            .flat_map(|&p| {
                let weights = &host;
                (0..33).map(move |row| weights.coefficient(row, p))
            })
            .collect::<Vec<_>>();
        // One coefficient per dot; exact equality isolates word/sign/scale addressing.
        assert_eq!(read(&backend, &output), expected);
    }
}

fn row_period_sentinel_weights(stream: usize) -> HostWeights {
    let mut host = HostWeights::new(513, 384, stream);
    let (scale_bits, codes): ([[u16; 3]; 3], [[i8; 3]; 3]) = if stream == 0 {
        (
            [
                [0x3001, 0x3003, 0x3005],
                [0x3101, 0x3103, 0x3105],
                [0x3201, 0x3203, 0x3205],
            ],
            [[1, -127, 17], [63, -7, 31], [-33, 65, -15]],
        )
    } else {
        (
            [
                [0x3401, 0x3403, 0x3405],
                [0x3501, 0x3503, 0x3505],
                [0x3601, 0x3603, 0x3605],
            ],
            [[-11, 29, -63], [17, -31, 43], [-65, 79, 9]],
        )
    };
    for (sentinel, row) in [0_usize, 511, 512].into_iter().enumerate() {
        for group in 0..3 {
            host.scales[row * 3 + group] = normal_positive_half(scale_bits[sentinel][group]);
            // Default row 512 has zero codes at columns 0 and 256. Explicit
            // nonzero codes make an incorrect scale observable at every impulse.
            host.codes[row * 384 + group * 128] = codes[sentinel][group].to_ne_bytes()[0];
        }
    }
    host
}

fn canonical_sentinel_impulse_outputs(host: &HostWeights) -> Vec<f32> {
    assert_eq!((host.n, host.k), (513, 384));
    let mut expected = Vec::with_capacity(3 * 513);
    for group in 0..3 {
        for row in 0..513 {
            // Direct canonical scalar indexing and an independent signed-byte
            // conversion. No candidate word/quartet or coefficient helper.
            let raw = host.codes[row * 384 + group * 128];
            let signed = i16::from(raw) - if raw >= 128 { 256 } else { 0 };
            expected.push(f32::from(signed) * host.scales[row * 3 + group]);
        }
    }
    expected
}

fn assert_sentinel_distinguishability(a: &HostWeights, b: &HostWeights) {
    let mut selected_scales = Vec::new();
    for host in [a, b] {
        let expected = canonical_sentinel_impulse_outputs(host);
        for row in [0_usize, 511, 512] {
            for group in 0..3 {
                let scale = host.scales[row * 3 + group];
                assert!(scale.is_finite() && scale > 0.0);
                assert!(
                    !selected_scales.contains(&scale.to_bits()),
                    "sentinel scales must distinguish rows, groups and streams"
                );
                selected_scales.push(scale.to_bits());
                let raw = host.codes[row * 384 + group * 128];
                assert_ne!(raw, 0, "nonzero sentinel code");
                let signed = i16::from(raw) - if raw >= 128 { 256 } else { 0 };
                let correct = expected[group * 513 + row];
                if row >= 511 {
                    let aliased = f32::from(signed) * host.scales[(row % 511) * 3 + group];
                    assert_ne!(
                        correct.to_bits(),
                        aliased.to_bits(),
                        "row%511 scale mutation"
                    );
                }
                if group == 2 {
                    let aliased = f32::from(signed) * host.scales[row * 3 + group % 2];
                    assert_ne!(
                        correct.to_bits(),
                        aliased.to_bits(),
                        "group%2 scale mutation"
                    );
                }
            }
        }
    }
}

#[test]
fn explicit_row511_row512_scales_survive_exact_impulses_and_all_forms() {
    // The general scale generator repeats after 511 rows. These explicit
    // sentinels extend the N<=33 boundary grid without inheriting that alias.
    let a = row_period_sentinel_weights(0);
    let b = row_period_sentinel_weights(1);
    assert_sentinel_distinguishability(&a, &b);
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    let mut x = vec![0.0; 3 * 384];
    for group in 0..3 {
        x[group * 384 + group * 128] = 1.0;
    }
    let input = backend
        .upload_f32(shape(3, 384), &x)
        .expect("sentinel impulses");
    for host in [&a, &b] {
        let weights = host.upload(&mut backend);
        let mut output = backend
            .allocate_f32(shape(3, 513))
            .expect("sentinel output");
        backend
            .packed_linear(&mut output, &input, &weights.codes, &weights.scales)
            .expect("sentinel impulse dispatch");
        assert_eq!(
            read(&backend, &output),
            canonical_sentinel_impulse_outputs(host)
        );
    }
    // Distinct matrices and gate/up inputs also expose row/scale aliasing in
    // paired and both fused headers. The original 120-shape counters stay local
    // to their separate test and are unchanged.
    exercise_forms(&mut backend, 3, &a, &b);
}

fn sparse_case(backend: &mut WgpuBackend, m: usize, n: usize, k: usize) {
    let host = HostWeights::new(n, k, 0);
    let weights = host.upload(backend);
    let positions = [0_usize, 3, 4, 127, 128, k / 2, k - 4, k - 1];
    let values = [
        1.0_f32,
        -0.5,
        0.25,
        -0.125,
        0.0625,
        -0.03125,
        0.015_625,
        -0.007_812_5,
    ];
    let mut x = vec![0.0; m * k];
    for row in 0..m {
        for (&col, &value) in positions.iter().zip(&values) {
            x[row * k + col] = value;
        }
    }
    let mut ordered = positions.into_iter().zip(values).collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|&(col, _)| col);
    let one_row = (0..n)
        .map(|row| {
            let mut sum = 0.0_f32;
            for &(col, value) in &ordered {
                sum += value * host.coefficient(row, col);
            }
            sum
        })
        .collect::<Vec<_>>();
    let expected = one_row.repeat(m);
    let input = backend.upload_f32(shape(m, k), &x).expect("sparse input");
    let mut output = backend.allocate_f32(shape(m, n)).expect("output");
    backend
        .packed_linear(&mut output, &input, &weights.codes, &weights.scales)
        .expect("actual-width dispatch");
    assert_close(
        &read(backend, &output),
        &expected,
        &format!("sparse M={m} N={n} K={k}"),
    );
}

#[test]
fn deployment_weight_widths_use_sparse_independently_auditable_outputs() {
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    // Union of small manifest metadata for actual classifier and pointer weights.
    // No models/assets are read by this test, and protected dense pointer heads
    // [256,1024] aren't part of the packed loader family.
    for (n, k) in [
        (512, 1024),
        (1024, 1024),
        (1024, 2560),
        (2560, 1024),
        (3072, 1024),
        (1024, 4608),
        (4608, 1024),
    ] {
        sparse_case(&mut backend, 65, n, k);
    }
    sparse_case(&mut backend, 346, 2560, 1024);
}

#[test]
fn invalid_admission_never_records_an_int8_dispatch() {
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    let host = HostWeights::new(17, 128, 0);
    let weights = host.upload(&mut backend);
    let input = backend
        .upload_f32(shape(2, 128), &[0.125; 256])
        .expect("input");
    let wrong_input = backend
        .upload_f32(shape(2, 127), &[0.125; 254])
        .expect("wrong input");
    let rank_one = backend
        .upload_f32(Shape::new(&[256]).expect("shape"), &[0.125; 256])
        .expect("rank one");
    let wrong_codes = backend
        .upload_u8_classified(shape(17, 127), &[0; 17 * 127], AllocationClass::Weight)
        .expect("wrong codes");
    let wrong_scales = backend
        .upload_f32(shape(16, 1), &[0.125; 16])
        .expect("wrong scales");
    let mut output = backend.allocate_f32(shape(2, 17)).expect("output");
    let mut wrong_output = backend.allocate_f32(shape(2, 16)).expect("wrong output");
    let mut pair_b = backend.allocate_f32(shape(2, 17)).expect("pair B");
    let before = backend.dispatch_counts();
    for operand in [&wrong_input, &rank_one] {
        assert!(matches!(
            backend.packed_linear(&mut output, operand, &weights.codes, &weights.scales),
            Err(ExecutorError::InvalidShape(_))
        ));
    }
    assert!(matches!(
        backend.packed_linear(&mut output, &input, &wrong_codes, &weights.scales),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.packed_linear(&mut output, &input, &weights.codes, &wrong_scales),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.packed_linear(&mut wrong_output, &input, &weights.codes, &weights.scales),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.packed_linear_pair(
            &mut output,
            &mut pair_b,
            &input,
            &weights.codes,
            &weights.scales,
            &wrong_codes,
            &weights.scales
        ),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.packed_swiglu_linear(
            &mut output,
            &input,
            &wrong_input,
            &weights.codes,
            &weights.scales
        ),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.packed_swiglu_pair(
            &mut output,
            &input,
            &weights.codes,
            &weights.scales,
            &weights.codes,
            &wrong_scales
        ),
        Err(ExecutorError::InvalidShape(_))
    ));
    assert!(matches!(
        backend.upload_f32(shape(1, 1), &[f32::NAN]),
        Err(ExecutorError::InvalidArgument(_))
    ));
    assert_eq!(backend.dispatch_counts(), before);
    backend
        .packed_linear(&mut output, &input, &weights.codes, &weights.scales)
        .expect("valid after rejection");
    assert_close(
        &read(&backend, &output),
        &host.project(&[0.125; 256], 2),
        "after rejection",
    );
}

#[test]
fn cancellation_keeps_readback_owned_and_next_int8_submission_correct() {
    let Some(mut backend) = common::gpu(LIMITS) else {
        return;
    };
    let host = HostWeights::new(33, 256, 0);
    let weights = host.upload(&mut backend);
    let x = input_values(65, 256, 0);
    let input = backend.upload_f32(shape(65, 256), &x).expect("input");
    let mut output = backend.allocate_f32(shape(65, 33)).expect("output");
    backend
        .packed_linear(&mut output, &input, &weights.codes, &weights.scales)
        .expect("dispatch");
    let before = backend
        .resource_report()
        .total_owned_bytes()
        .expect("accounting");
    let mut cancelled = backend.read_f32_async(&output).expect("readback");
    assert_eq!(backend.resource_report().pending_operations, 1);
    assert!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("accounting")
            > before
    );
    cancelled.cancel().expect("cancel");
    assert!(matches!(
        cancelled.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::Cancelled))
    ));
    drop(cancelled);
    // Dropped logical owners return to a serial-protected pool, not free memory.
    drop(output);
    assert!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("accounting")
            >= before
    );
    backend
        .resource_report()
        .validate(LIMITS)
        .expect("cancelled resources bounded");
    settle(&backend);
    assert_eq!(backend.resource_report().pending_operations, 0);
    let mut next = backend.allocate_f32(shape(65, 33)).expect("next output");
    backend.request_cancel();
    let dispatches = backend.dispatch_counts();
    assert!(matches!(
        backend.packed_linear(&mut next, &input, &weights.codes, &weights.scales),
        Err(ExecutorError::Cancelled)
    ));
    assert_eq!(backend.dispatch_counts(), dispatches);
    backend.clear_cancel();
    backend
        .packed_linear(&mut next, &input, &weights.codes, &weights.scales)
        .expect("next dispatch");
    assert_close(
        &read(&backend, &next),
        &host.project(&x, 65),
        "after cancellation",
    );
    backend
        .resource_report()
        .validate(LIMITS)
        .expect("next resources bounded");
}
