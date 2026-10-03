//! Signed-byte conformance across projection and gather boundaries.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    AllocationClass, CompletionPoll, InferenceCompletion, ResourceLimits, Shape, TokenIds,
};
mod common;

fn read(backend: &WgpuBackend, buffer: &minifield_backend_wgpu::WgpuBuffer) -> Vec<f32> {
    let mut task = backend.read_f32_async(buffer).unwrap();
    let started = std::time::Instant::now();
    loop {
        assert!(started.elapsed().as_secs() < 30, "INT8 readback timed out");
        match task.poll_step() {
            CompletionPoll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
            CompletionPoll::Ready(result) => return result.unwrap(),
        }
    }
}

#[test]
fn packed_gather_decodes_words_and_poisons_invalid_device_ids() {
    let Some(mut backend) = common::gpu(ResourceLimits {
        max_allocation_bytes: 64 << 20,
        max_total_bytes: 512 << 20,
        max_pending_operations: 64,
    }) else {
        return;
    };
    let rows = 3_usize;
    let inner = 384_usize;
    let codes = (0..rows * inner)
        .map(|i| ((i % 255) as i16 - 127).to_le_bytes()[0])
        .collect::<Vec<_>>();
    let scales = (0..rows * 3)
        .map(|i| (1 + i % 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let code_buffer = backend
        .upload_u8_classified(
            Shape::new(&[rows as u64, inner as u64]).unwrap(),
            &codes,
            AllocationClass::Weight,
        )
        .unwrap();
    let scale_buffer = backend
        .upload_f32(Shape::new(&[rows as u64, 3]).unwrap(), &scales)
        .unwrap();
    let id_shape = Shape::new(&[10]).unwrap();
    let base = backend
        .upload_f32(
            id_shape,
            &[
                0.0,
                2.0,
                -0.0,
                -1.0,
                0.5,
                3.0,
                16_777_216.0,
                3e38,
                3e38,
                -3e38,
            ],
        )
        .unwrap();
    let factors = backend
        .upload_f32(
            id_shape,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        )
        .unwrap();
    let keep = backend
        .upload_f32(
            id_shape,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
    let mut overflowed = backend.allocate_f32(id_shape).unwrap();
    let mut ids = backend.allocate_f32(id_shape).unwrap();
    backend.multiply(&mut overflowed, &base, &factors).unwrap();
    backend.multiply(&mut ids, &overflowed, &keep).unwrap();
    let produced_ids = read(&backend, &ids);
    assert!(produced_ids[7].is_infinite() && produced_ids[7].is_sign_positive());
    assert!(produced_ids[8].is_nan());
    assert!(produced_ids[9].is_infinite() && produced_ids[9].is_sign_negative());
    let mut output = backend
        .allocate_f32(Shape::new(&[10, inner as u64]).unwrap())
        .unwrap();
    // 960 code-word invocations leave 64 padded invocations in the last group.
    backend
        .packed_gather_rows(
            &mut output,
            &code_buffer,
            &scale_buffer,
            TokenIds::Device(&ids),
        )
        .unwrap();
    let actual = read(&backend, &output);
    for (destination, values) in actual.chunks_exact(inner).enumerate() {
        if destination < 3 {
            let source = [0, 2, 0][destination];
            for (column, &value) in values.iter().enumerate() {
                let expected = f32::from(i8::from_ne_bytes([codes[source * inner + column]]))
                    * scales[source * 3 + column / 128];
                assert_eq!(value.to_bits(), expected.to_bits());
            }
        } else {
            assert!(
                values.iter().all(|value| value.is_nan()),
                "row {destination}"
            );
        }
    }
    let mut empty = backend
        .allocate_f32(Shape::new(&[0, inner as u64]).unwrap())
        .unwrap();
    backend
        .packed_gather_rows(&mut empty, &code_buffer, &scale_buffer, TokenIds::Host(&[]))
        .unwrap();
    assert_eq!(
        backend.dispatch_counts().get("packed_gather_int8").copied(),
        Some(1)
    );
}
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn signed_bytes_match_independent_dense_math() {
    let Some(mut backend) = common::gpu(ResourceLimits {
        max_allocation_bytes: 64 << 20,
        max_total_bytes: 512 << 20,
        max_pending_operations: 64,
    }) else {
        return;
    };
    let output_width = 17_usize;
    let inner = 256_usize;
    let codes = (0..output_width * inner)
        .map(|i| ((i % 255) as i16 - 127).to_le_bytes()[0])
        .collect::<Vec<_>>();
    let scales = (0..output_width * 2)
        .map(|i| if i % 2 == 0 { 0.125 } else { 0.03125 })
        .collect::<Vec<_>>();
    let codes_buffer = backend
        .upload_u8_classified(
            Shape::new(&[output_width as u64, inner as u64]).unwrap(),
            &codes,
            AllocationClass::Weight,
        )
        .unwrap();
    let scales_buffer = backend
        .upload_f32(Shape::new(&[output_width as u64, 2]).unwrap(), &scales)
        .unwrap();
    for rows in [1_usize, 7, 95, 96, 97] {
        let activations = (0..rows * inner)
            .map(|i| (i % 19) as f32 * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let input = backend
            .upload_f32(
                Shape::new(&[rows as u64, inner as u64]).unwrap(),
                &activations,
            )
            .unwrap();
        let mut out = backend
            .allocate_f32(Shape::new(&[rows as u64, output_width as u64]).unwrap())
            .unwrap();
        backend
            .packed_linear(&mut out, &input, &codes_buffer, &scales_buffer)
            .unwrap();
        let actual = read(&backend, &out);
        for t in 0..rows {
            for row in 0..output_width {
                let expected = (0..inner)
                    .map(|i| {
                        activations[t * inner + i]
                            * f32::from(i8::from_ne_bytes([codes[row * inner + i]]))
                            * scales[row * 2 + i / 128]
                    })
                    .sum::<f32>();
                assert!(
                    (actual[t * output_width + row] - expected).abs() < 0.002,
                    "rows={rows} t={t} row={row}:{} != {expected}",
                    actual[t * output_width + row]
                );
            }
        }
    }
    let input = backend
        .upload_f32(
            Shape::new(&[2, inner as u64]).unwrap(),
            &vec![0.007_812_5; 2 * inner],
        )
        .unwrap();
    let mut pair_a = backend
        .allocate_f32(Shape::new(&[2, output_width as u64]).unwrap())
        .unwrap();
    let mut pair_b = backend
        .allocate_f32(Shape::new(&[2, output_width as u64]).unwrap())
        .unwrap();
    backend
        .packed_linear_pair(
            &mut pair_a,
            &mut pair_b,
            &input,
            &codes_buffer,
            &scales_buffer,
            &codes_buffer,
            &scales_buffer,
        )
        .unwrap();
    let plain_a = read(&backend, &pair_a);
    let plain_b = read(&backend, &pair_b);
    assert_eq!(plain_a, plain_b);
    let mut fused = backend
        .allocate_f32(Shape::new(&[2, output_width as u64]).unwrap())
        .unwrap();
    backend
        .packed_swiglu_pair(
            &mut fused,
            &input,
            &codes_buffer,
            &scales_buffer,
            &codes_buffer,
            &scales_buffer,
        )
        .unwrap();
    let fused_values = read(&backend, &fused);
    for (actual, gate_value) in fused_values.iter().zip(&plain_a) {
        assert!((actual - gate_value / (1.0 + (-gate_value).exp()) * gate_value).abs() < 0.0001);
    }
    let mut down = backend
        .allocate_f32(Shape::new(&[2, output_width as u64]).unwrap())
        .unwrap();
    backend
        .packed_swiglu_linear(&mut down, &input, &input, &codes_buffer, &scales_buffer)
        .unwrap();
    let down_values = read(&backend, &down);
    let gate_value = 0.007_812_5_f32;
    let scale = gate_value / (1.0 + (-gate_value).exp());
    for (actual, plain) in down_values.iter().zip(&plain_a) {
        assert!((actual - plain * scale).abs() < 0.0001);
    }
    assert!(
        backend
            .dispatch_counts()
            .get("packed_gemm_int8")
            .copied()
            .unwrap_or(0)
            >= 5
    );
}
