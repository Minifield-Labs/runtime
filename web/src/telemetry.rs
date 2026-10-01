//! Browser adapter for content-free inference records.
use minifield_backend_wgpu::WgpuBackend;
use minifield_executor_core::InferenceWork;
use minifield_runtime_telemetry::{Hardware, Measurement, Model};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

#[wasm_bindgen(module = "/telemetry.mjs")]
extern "C" {
    #[wasm_bindgen(catch, js_name = configureTelemetry)]
    fn configure(options: &JsValue) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = reportTelemetry)]
    fn report(encoded: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = telemetryEnabled)]
    fn enabled() -> Result<bool, JsValue>;
    #[wasm_bindgen(catch, js_name = flushTelemetry)]
    async fn flush() -> Result<(), JsValue>;
}

/// Configure reporting for this WASM instance/Worker. Call before the first inference.
#[wasm_bindgen]
pub fn configure_telemetry(options: &JsValue) -> Result<(), JsValue> {
    configure(options)
}

/// Flush the pending batch, for example before terminating an inference Worker.
#[wasm_bindgen]
pub async fn flush_telemetry() {
    let _ = flush().await;
}

pub struct BrowserTelemetry {
    pub model: Model,
    hardware: Hardware,
}

impl BrowserTelemetry {
    pub fn new(model: Model, backend: &WgpuBackend) -> Self {
        let info = backend.adapter_info();
        let vendor = match info.vendor {
            0x106b => Some("apple"),
            0x10de => Some("nvidia"),
            0x1002 => Some("amd"),
            0x8086 => Some("intel"),
            0x13b5 => Some("arm"),
            0x5143 => Some("qualcomm"),
            _ => None,
        };
        Self {
            model,
            hardware: Hardware {
                kind: if info.device_type == wgpu_types::DeviceType::Cpu {
                    "cpu"
                } else {
                    "gpu"
                },
                vendor,
                architecture: None,
            },
        }
    }

    pub fn finish(&self, measurement: Measurement, work: InferenceWork, succeeded: bool) {
        if !enabled().unwrap_or(false) {
            return;
        }
        let mut record = measurement.finish(&self.model, work, succeeded, "webgpu");
        record["hardware"] = serde_json::to_value(&self.hardware).unwrap_or_default();
        let _ = report(&record.to_string());
    }
}
