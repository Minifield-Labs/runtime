//! Signed-byte conformance at GEMM tile and dispatch boundaries.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    AllocationClass, CompletionPoll, InferenceCompletion, ResourceLimits, Shape,
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
