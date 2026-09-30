//! Explicit backend construction and evidence. Unsupported selections fail.
use crate::{HostResult, request::Request, run};
use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{EncoderOps, ResourceLimits};
use serde_json::{Value, json};
use std::time::Instant;

pub(crate) trait HostBackend: EncoderOps {
    fn backend_evidence(&self) -> Value;
    fn dispatch_counts(&self) -> Value;
}

impl HostBackend for CpuBackend {
    fn backend_evidence(&self) -> Value {
        json!({"implementation":"cpu_reference","api":"cpu","device":std::env::consts::ARCH,"driver":"rust-cpu-reference"})
    }
    fn dispatch_counts(&self) -> Value {
        json!({})
    }
}

#[cfg(feature = "wgpu")]
impl HostBackend for minifield_backend_wgpu::WgpuBackend {
    fn backend_evidence(&self) -> Value {
        let info = self.adapter_info();
        json!({"implementation":"wgpu_metal","api":format!("{:?}",info.backend).to_lowercase(),"device":info.name,
            "driver":format!("wgpu=30.0.1;reported_driver={};reported_info={}", info.driver, info.driver_info)})
    }
    fn dispatch_counts(&self) -> Value {
        json!(self.dispatch_counts())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl HostBackend for minifield_backend_metal::MetalBackend {
    fn backend_evidence(&self) -> Value {
        let info = self.device_info();
        json!({"implementation":"native_metal","api":info.api,"device":info.name,
            "driver":format!("objc2-metal=0.3.2;reported_driver=unavailable;registry_id={:#x}", info.registry_id)})
    }
    fn dispatch_counts(&self) -> Value {
        json!(self.dispatch_counts())
    }
}

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 8 << 30,
        max_pending_operations: 4096,
    }
}

pub(crate) fn evaluate(request: &Request, started: Instant) -> HostResult<Value> {
    match request.backend.as_str() {
        "cpu_reference" => run::evaluate(CpuBackend::new(91, limits()), request, started),
        #[cfg(feature = "wgpu")]
        "wgpu_metal" => {
            let backend = minifield_backend_wgpu::WgpuBackend::new_with_options(
                91,
                limits(),
                minifield_backend_wgpu::WgpuOptions::default(),
            )?;
            let identity = backend.backend_evidence();
            if identity["api"] != "metal"
                || format!("{:?}", backend.adapter_info().device_type) == "Cpu"
            {
                return Err("WGPU evaluation requires a hardware Metal adapter".into());
            }
            run::evaluate(backend, request, started)
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "native_metal" => run::evaluate(
            minifield_backend_metal::MetalBackend::new(91, limits())?,
            request,
            started,
        ),
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        "native_metal" => {
            Err("dedicated native Metal backend wasn't compiled for this platform".into())
        }
        _ => Err("requested backend wasn't compiled into this executable".into()),
    }
}
