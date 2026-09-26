//! Allocation regressions run on a real adapter. Qualification requires it.
#![allow(clippy::expect_used)]

use super::*;
use std::time::{Duration, Instant};

fn limits(total: u64) -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: total,
        max_total_bytes: total,
        max_pending_operations: 16,
    }
}

fn gpu(limits: ResourceLimits) -> Option<WgpuBackend> {
    let required = match std::env::var("MINIFIELD_REQUIRE_GPU") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Err(std::env::VarError::NotPresent) => false,
        _ => panic!("MINIFIELD_REQUIRE_GPU must be 0 or 1"),
    };
    match WgpuBackend::new(0xB0D, limits) {
        Ok(backend) => {
            assert!(
                !required || backend.adapter_info().device_type != wgpu::DeviceType::Cpu,
                "required GPU qualification rejects a software CPU adapter"
            );
            Some(backend)
        }
        Err(ExecutorError::BackendFailure(message))
            if message.contains("no adapters found") && !required =>
        {
            eprintln!("SKIP GPU allocation tests: {message}");
            None
        }
        Err(error) => panic!("GPU initialization failed: {error}"),
    }
}

fn settle(backend: &WgpuBackend) {
    let mut fence = backend.fence().expect("fence");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match fence.poll_step() {
            CompletionPoll::Ready(result) => {
                result.expect("completion");
                return;
            }
            CompletionPoll::Pending => {
                assert!(Instant::now() < deadline, "GPU fence deadline expired");
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn dropped_and_pooled_allocations_stay_charged_until_safe_eviction() {
    let budget = limits(3 << 19);
    let Some(mut backend) = gpu(budget) else {
        return;
    };
    let baseline = backend
        .resource_report()
        .total_owned_bytes()
        .expect("report");
    let shape = Shape::new(&[262_144]).expect("shape");
    let buffer = backend.allocate_f32(shape).expect("first buffer");
    let charged = backend
        .resource_report()
        .total_owned_bytes()
        .expect("report");
    assert_eq!(charged, baseline + 1_048_576);
    drop(buffer);
    assert_eq!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("report"),
        charged
    );
    assert!(matches!(
        backend.allocate_f32(shape),
        Err(ExecutorError::ResourceLimit(_))
    ));
    settle(&backend);
    // A different class evicts the completed 1 MiB pool allocation.
    let replacement = backend
        .allocate_f32(Shape::new(&[131_072]).expect("shape"))
        .expect("safe pool eviction");
    assert_eq!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("report"),
        baseline + 524_288
    );
    backend
        .resource_report()
        .validate(budget)
        .expect("bounded report");
    drop(replacement);
}

#[test]
fn scratch_and_readback_staging_obey_physical_allocation_limits() {
    let budget = ResourceLimits {
        max_allocation_bytes: 1024,
        ..limits(8192)
    };
    let Some(mut backend) = gpu(budget) else {
        return;
    };
    let before = backend
        .resource_report()
        .total_owned_bytes()
        .expect("report");
    assert!(matches!(
        backend.device.alloc_storage(65_536),
        Err(ExecutorError::ResourceLimit(_))
    ));
    assert!(matches!(
        backend.device.alloc_staging(1025),
        Err(ExecutorError::ResourceLimit(_))
    ));
    assert_eq!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("report"),
        before
    );
    let source = backend
        .upload_f32(Shape::new(&[128]).expect("shape"), &[1.0; 128])
        .expect("source");
    let before_read = backend
        .resource_report()
        .total_owned_bytes()
        .expect("report");
    let read = backend.read_f32_async(&source).expect("readback");
    assert_eq!(
        backend
            .resource_report()
            .total_owned_bytes()
            .expect("report"),
        before_read + 1024
    );
    drop(read);
    backend
        .resource_report()
        .validate(budget)
        .expect("dropped readback stays bounded");
    settle(&backend);
    assert_eq!(
        backend.read_f32(&source).expect("read again"),
        vec![1.0; 128]
    );
}

#[test]
fn odd_u8_uploads_preserve_bytes_and_pad_the_physical_transfer() {
    let Some(mut backend) = gpu(limits(1 << 20)) else {
        return;
    };
    for len in 0..=5 {
        let bytes: Vec<u8> = (0..len)
            .map(|value| u8::try_from(value + 1).expect("byte"))
            .collect();
        let packed = backend
            .upload_u8_classified(
                Shape::new(&[len]).expect("shape"),
                &bytes,
                AllocationClass::Weight,
            )
            .expect("arbitrary byte length");
        assert_eq!(packed.byte_len(), len);
        if len == 0 {
            continue;
        }
        let words = len.div_ceil(4);
        let output = backend
            .allocate_f32(Shape::new(&[words]).expect("shape"))
            .expect("output");
        backend.device.record_copy(
            packed.wgpu_buffer().expect("packed"),
            0,
            output.wgpu_buffer().expect("output"),
            0,
            words * 4,
        );
        let got: Vec<u8> = backend
            .read_f32(&output)
            .expect("readback")
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        assert_eq!(&got[..bytes.len()], bytes.as_slice());
        assert!(got[bytes.len()..].iter().all(|&byte| byte == 0));
    }
}

#[test]
fn raw_and_repacked_streams_cannot_cross_kernel_interfaces() {
    let Some(mut backend) = gpu(limits(1 << 20)) else {
        return;
    };
    let raw = backend
        .upload_u8_classified(
            Shape::new(&[1, 32]).expect("shape"),
            &[0x55; 32],
            AllocationClass::Weight,
        )
        .expect("raw codes");
    let lut2 = backend.repack_ternary_lut2(&raw).expect("repack");
    let scales = backend
        .upload_f32(Shape::new(&[1, 1]).expect("shape"), &[1.0])
        .expect("scales");
    let input = backend
        .upload_f32(Shape::new(&[96, 128]).expect("shape"), &vec![1.0; 96 * 128])
        .expect("input");
    let mut output = backend
        .allocate_f32(Shape::new(&[96, 1]).expect("shape"))
        .expect("output");
    assert!(matches!(
        backend.packed_linear_lut2(&mut output, &input, &raw, &scales),
        Err(ExecutorError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.packed_linear(&mut output, &input, &lut2, &scales),
        Err(ExecutorError::InvalidArgument(_))
    ));
    assert!(matches!(
        backend.repack_ternary_lut2(&lut2),
        Err(ExecutorError::InvalidArgument(_))
    ));
    backend
        .packed_linear_lut2(&mut output, &input, &lut2, &scales)
        .expect("correct layout");
    assert_eq!(backend.read_f32(&output).expect("readback"), vec![0.0; 96]);
}
