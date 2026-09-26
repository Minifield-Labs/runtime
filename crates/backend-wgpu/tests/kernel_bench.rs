//! Kernel micro-benchmark at real LFM2.5-230M shapes (dev tool, not a gate).
//!
//! Dispatches each op N times inside one submission and reports per-dispatch
//! wall time — dominated by GPU execution plus per-dispatch launch overhead.
//! Run: `cargo test --release -p minifield-backend-wgpu --test kernel_bench
//! -- --ignored --nocapture`. Skips cleanly without an adapter.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::unchecked_duration_subtraction
)]

mod common;

use minifield_backend_wgpu::{WgpuBackend, WgpuBuffer};

use minifield_engine_api::{
    AllocationClass, CompletionPoll, GqaSpec, InferenceCompletion, ResourceLimits, Shape,
};

/// Capability probe for the native-only cooperative-matrix path the E9
/// experiment would need. Prints adapter info, experimental feature
/// support, and the cooperative matrix configurations the driver offers.
#[test]
fn cooperative_matrix_probe() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(a) => a,
            Err(e) => {
                assert!(!common::required(), "required GPU unavailable: {e}");
                eprintln!("SKIP no adapter: {e}");
                return;
            }
        };
    eprintln!("adapter: {:?}", adapter.get_info());
    let limits = adapter.limits();
    eprintln!(
        "limits: workgroup_storage={} invocations_per_wg={} wg_size={:?}",
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    );
    let features = adapter.features();
    eprintln!(
        "SHADER_F16={}",
        features.contains(wgpu::Features::SHADER_F16)
    );
    let props = adapter.cooperative_matrix_properties();
    eprintln!("cooperative matrix configs: {props:#?}");
    eprintln!("empty = unsupported on this adapter/driver");
}

fn gpu() -> Option<WgpuBackend> {
    common::gpu(ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 1 << 36,
        max_pending_operations: 1024,
    })
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

fn bench(backend: &WgpuBackend, label: &str, reps: u32, mut record: impl FnMut()) {
    for _ in 0..2 {
        record();
    }
    fence_wait(backend);
    let start = std::time::Instant::now();
    for _ in 0..reps {
        record();
    }
    let encode = start.elapsed();
    fence_wait(backend);
    let total = start.elapsed();
    let wait = total.saturating_sub(encode);
    eprintln!(
        "{label:<44} reps={reps:>4} encode={:>8.2}ms wait={:>8.2}ms  per-dispatch wait={:>7.1}us",
        encode.as_secs_f64() * 1e3,
        wait.as_secs_f64() * 1e3,
        wait.as_secs_f64() * 1e6 / f64::from(reps),
    );
}

fn packed(backend: &mut WgpuBackend, k: u64, n: u64) -> (WgpuBuffer, WgpuBuffer) {
    packed_fmt(backend, k, n, false)
}

fn packed_fmt(backend: &mut WgpuBackend, k: u64, n: u64, nf4: bool) -> (WgpuBuffer, WgpuBuffer) {
    let w = if nf4 { k / 2 } else { k / 4 };
    let codes = backend
        .upload_u8_classified(
            Shape::new(&[n, w]).expect("codes shape"),
            &vec![0xA5u8; (n * w) as usize],
            AllocationClass::Scratch,
        )
        .expect("codes");
    let scales = backend
        .upload_f32(
            Shape::new(&[n, k / 128]).expect("scales shape"),
            &vec![0.5f32; (n * k / 128) as usize],
        )
        .expect("scales");
    (codes, scales)
}

fn x_of(backend: &mut WgpuBackend, k: u64) -> WgpuBuffer {
    backend
        .upload_f32(
            Shape::new(&[1, k]).expect("x shape"),
            &vec![0.25f32; k as usize],
        )
        .expect("x")
}

fn dense_w(backend: &mut WgpuBackend, k: u64, n: u64) -> WgpuBuffer {
    backend
        .upload_f32(
            Shape::new(&[n, k]).expect("w shape"),
            &vec![0.01f32; (n * k) as usize],
        )
        .expect("dense w")
}

fn out_of(backend: &mut WgpuBackend, n: u64) -> WgpuBuffer {
    backend
        .allocate_f32_classified(Shape::new(&[1, n]).expect("out"), AllocationClass::Scratch)
        .expect("out")
}

#[test]
fn adapter_features_probe() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()).await {
            let features = adapter.features();
            eprintln!(
                "{}: SUBGROUP={} TIMESTAMP_QUERY={}",
                adapter.get_info().name,
                features.contains(wgpu::Features::SUBGROUP),
                features.contains(wgpu::Features::TIMESTAMP_QUERY),
            );
        }
    });
}

#[test]
#[ignore = "device timing experiment; run explicitly with --ignored --nocapture"]
fn kernel_bench_lfm25_shapes() {
    let Some(backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut backend = backend;

    for (k, n, reps) in [
        (1024_u64, 1024_u64, 400_u32),
        (1024, 3072, 400),
        (1024, 5120, 200),
        (2560, 1024, 400),
        (1024, 65536, 24),
    ] {
        let x = x_of(&mut backend, k);
        let (codes, scales) = packed(&mut backend, k, n);
        let mut out = out_of(&mut backend, n);
        bench(
            &backend,
            &format!("packed_linear k={k} n={n}"),
            reps,
            || {
                backend
                    .packed_linear(&mut out, &x, &codes, &scales)
                    .expect("op");
            },
        );
    }

    for (k, n, reps) in [(1024_u64, 1024_u64, 400_u32), (1024, 65536, 24)] {
        let x = x_of(&mut backend, k);
        let w = dense_w(&mut backend, k, n);
        let mut out = out_of(&mut backend, n);
        bench(&backend, &format!("linear k={k} n={n}"), reps, || {
            backend.linear(&mut out, &x, &w).expect("op");
        });
    }

    for (k, n, reps) in [(1024_u64, 512_u64, 400_u32), (1024, 2560, 200)] {
        let x = x_of(&mut backend, k);
        let (codes_a, scales_a) = packed(&mut backend, k, n);
        let (codes_b, scales_b) = packed(&mut backend, k, n);
        let mut out_a = out_of(&mut backend, n);
        let mut out_b = out_of(&mut backend, n);
        bench(
            &backend,
            &format!("packed_linear_pair k={k} n={n}"),
            reps,
            || {
                backend
                    .packed_linear_pair(
                        &mut out_a, &mut out_b, &x, &codes_a, &scales_a, &codes_b, &scales_b,
                    )
                    .expect("op");
            },
        );
    }

    {
        let (codes, scales) = packed(&mut backend, 2560, 1024);
        let gate = x_of(&mut backend, 2560);
        let up = x_of(&mut backend, 2560);
        let mut out = out_of(&mut backend, 1024);
        bench(&backend, "packed_swiglu_linear k=2560 n=1024", 400, || {
            backend
                .packed_swiglu_linear(&mut out, &gate, &up, &codes, &scales)
                .expect("op");
        });
    }

    {
        let logits = backend
            .upload_f32(
                Shape::new(&[1, 65536]).expect("logits"),
                &vec![0.1f32; 65536],
            )
            .expect("logits");
        let mut out = backend
            .allocate_f32_classified(Shape::new(&[1]).expect("out"), AllocationClass::Scratch)
            .expect("out");
        bench(&backend, "argmax [1,65536]", 400, || {
            backend.argmax(&mut out, &logits).expect("op");
        });
    }

    {
        let a = x_of(&mut backend, 1024);
        let b = x_of(&mut backend, 1024);
        let w = backend
            .upload_f32(Shape::new(&[1024]).expect("w shape"), &vec![0.5f32; 1024])
            .expect("w");
        let mut sum = out_of(&mut backend, 1024);
        let mut normed = out_of(&mut backend, 1024);
        bench(&backend, "add_row_rms_norm [1,1024]", 400, || {
            backend
                .add_row_rms_norm(&mut sum, &mut normed, &a, &b, &w, 1e-5)
                .expect("op");
        });
    }

    eprintln!("bench done");
}

/// Prefill probe: same packed matmul shapes the classifier runs at m=346.
#[test]
#[ignore = "device timing experiment; run explicitly with --ignored --nocapture"]
fn packed_prefill_bench() {
    let Some(backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut backend = backend;

    for (m, k, n, reps, nf4) in [
        (346_u64, 1024_u64, 1024_u64, 30_u32, false),
        (346, 1024, 3072, 30, false),
        (346, 2560, 1024, 30, false),
        (1, 1024, 1024, 200, false),
        (346, 1024, 1024, 30, true),
        (346, 1024, 3072, 30, true),
        (346, 2560, 1024, 30, true),
    ] {
        let x = backend
            .upload_f32(
                Shape::new(&[m, k]).expect("x shape"),
                &vec![0.25f32; (m * k) as usize],
            )
            .expect("x");
        let (codes, scales) = packed_fmt(&mut backend, k, n, nf4);
        let mut out = backend
            .allocate_f32_classified(Shape::new(&[m, n]).expect("out"), AllocationClass::Scratch)
            .expect("out");
        bench(
            &backend,
            &format!("packed_linear m={m} k={k} n={n} nf4={nf4}"),
            reps,
            || {
                backend
                    .packed_linear(&mut out, &x, &codes, &scales)
                    .expect("op");
            },
        );
    }
    eprintln!("prefill bench done");
}

/// E5 probe: scale m at the down-projection shape to separate per-row-tile
/// decode amplification from per-column tile work.
#[test]
#[ignore = "device timing experiment; run explicitly with --ignored --nocapture"]
fn packed_m_scaling_bench() {
    let Some(backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut backend = backend;
    for nf4 in [true, false] {
        let label = if nf4 { "nf4" } else { "ternary" };
        for (m, k, n) in [
            (80_u64, 2560_u64, 1024_u64),
            (96, 2560, 1024),
            (112, 2560, 1024),
            (192, 2560, 1024),
            (256, 2560, 1024),
            (299, 2560, 1024),
            (320, 2560, 1024),
            (384, 2560, 1024),
            (346, 2560, 1024),
            (80, 1024, 2560),
            (112, 1024, 2560),
            (299, 1024, 2560),
            (320, 1024, 2560),
            (384, 1024, 2560),
            (346, 1024, 2560),
        ] {
            let x = backend
                .upload_f32(
                    Shape::new(&[m, k]).expect("x shape"),
                    &vec![0.25f32; (m * k) as usize],
                )
                .expect("x");
            let (codes, scales) = packed_fmt(&mut backend, k, n, nf4);
            let mut out = backend
                .allocate_f32_classified(
                    Shape::new(&[m, n]).expect("out"),
                    AllocationClass::Scratch,
                )
                .expect("out");
            bench(
                &backend,
                &format!("packed_linear {label} m={m} k={k} n={n}"),
                30,
                || {
                    backend
                        .packed_linear(&mut out, &x, &codes, &scales)
                        .expect("op");
                },
            );
        }
    }
}

/// Dense GEMM reference at prefill width for the same projection shape.
#[test]
#[ignore = "device timing experiment; run explicitly with --ignored --nocapture"]
fn dense_prefill_bench() {
    let Some(backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut backend = backend;
    for (m, k, n, reps) in [
        (346_u64, 1024_u64, 1024_u64, 30_u32),
        (346, 1024, 3072, 15),
        (346, 2560, 1024, 15),
    ] {
        let x = backend
            .upload_f32(
                Shape::new(&[m, k]).expect("x shape"),
                &vec![0.25f32; (m * k) as usize],
            )
            .expect("x");
        let w = dense_w(&mut backend, k, n);
        let mut out = backend
            .allocate_f32_classified(Shape::new(&[m, n]).expect("out"), AllocationClass::Scratch)
            .expect("out");
        bench(
            &backend,
            &format!("dense linear m={m} k={k} n={n}"),
            reps,
            || {
                backend.linear(&mut out, &x, &w).expect("op");
            },
        );
    }
    eprintln!("dense bench done");
}

/// Isolate attention and FFN costs at the polyomino classifier's prefill shape.
#[test]
#[ignore = "device timing experiment; run explicitly with --ignored --nocapture"]
fn classifier_prefill_components() {
    let Some(mut backend) = gpu() else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let m = 346;
    let query = backend
        .upload_f32(
            Shape::new(&[m, 1024]).expect("q"),
            &vec![0.25; (m * 1024) as usize],
        )
        .expect("q upload");
    let key = backend
        .upload_f32(
            Shape::new(&[m, 512]).expect("k"),
            &vec![0.125; (m * 512) as usize],
        )
        .expect("k upload");
    let value = backend
        .upload_f32(
            Shape::new(&[m, 512]).expect("v"),
            &vec![0.5; (m * 512) as usize],
        )
        .expect("v upload");
    let mut kc = backend
        .allocate_f32(Shape::new(&[m, 512]).expect("kc"))
        .expect("kc alloc");
    let mut vc = backend
        .allocate_f32(Shape::new(&[m, 512]).expect("vc"))
        .expect("vc alloc");
    let mut output = backend
        .allocate_f32(Shape::new(&[m, 1024]).expect("output"))
        .expect("output alloc");
    let spec = GqaSpec::new(16, 8, 64).expect("heads");
    bench(&backend, "GQA m=346 q=16 kv=8 dim=64", 6, || {
        backend
            .causal_gqa(
                &mut output,
                &query,
                &key,
                &value,
                &mut kc,
                &mut vc,
                &mut 0,
                spec,
            )
            .expect("gqa");
    });
    let (ca, sa) = packed_fmt(&mut backend, 1024, 2560, true);
    let (cb, sb) = packed_fmt(&mut backend, 1024, 2560, true);
    let mut gate = backend
        .allocate_f32(Shape::new(&[m, 2560]).expect("gate"))
        .expect("gate alloc");
    let mut up = backend
        .allocate_f32(Shape::new(&[m, 2560]).expect("up"))
        .expect("up alloc");
    bench(&backend, "NF4 FFN pair m=346 k=1024 n=2560", 6, || {
        backend
            .packed_linear_pair(&mut gate, &mut up, &query, &ca, &sa, &cb, &sb)
            .expect("ffn pair");
    });
    let (cd, sd) = packed_fmt(&mut backend, 2560, 1024, true);
    bench(&backend, "NF4 FFN down m=346 k=2560 n=1024", 6, || {
        backend
            .packed_swiglu_linear(&mut output, &gate, &up, &cd, &sd)
            .expect("ffn down");
    });
    let mut hidden = backend
        .allocate_f32(Shape::new(&[m, 2560]).expect("hidden"))
        .expect("hidden alloc");
    bench(
        &backend,
        "E1 swiglu + NF4 down m=346 k=2560 n=1024",
        6,
        || {
            backend.swiglu(&mut hidden, &gate, &up).expect("ffn swiglu");
            backend
                .packed_linear(&mut output, &hidden, &cd, &sd)
                .expect("ffn down");
        },
    );
    bench(&backend, "E2 pair+swiglu m=346 k=1024 n=2560", 6, || {
        backend
            .packed_swiglu_pair(&mut hidden, &query, &ca, &sa, &cb, &sb)
            .expect("ffn pair swiglu");
    });
    bench(&backend, "E2 pair_swiglu + NF4 down m=346", 6, || {
        backend
            .packed_swiglu_pair(&mut hidden, &query, &ca, &sa, &cb, &sb)
            .expect("ffn pair swiglu");
        backend
            .packed_linear(&mut output, &hidden, &cd, &sd)
            .expect("ffn down");
    });
}
