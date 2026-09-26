use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{ExecutorError, ResourceLimits};

pub fn required() -> bool {
    match std::env::var("MINIFIELD_REQUIRE_GPU") {
        Ok(value) if value == "1" => true,
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        _ => panic!("MINIFIELD_REQUIRE_GPU must be 0 or 1"),
    }
}

pub fn gpu(limits: ResourceLimits) -> Option<WgpuBackend> {
    match WgpuBackend::new(0x77, limits) {
        Ok(backend) => Some(backend),
        Err(ExecutorError::BackendFailure(message))
            if message.contains("no adapters found") && !required() =>
        {
            eprintln!("SKIP: {message}; use MINIFIELD_REQUIRE_GPU=1 for qualification");
            None
        }
        Err(error) => panic!("GPU qualification initialization failed: {error}"),
    }
}
