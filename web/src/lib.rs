#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
//! Browser demo harness for the packed LFM2 wgpu path.
//!
//! The host page fetches the bundle bytes and hands them to [`load`]; the
//! executor, tokenizer, and generation loop all live here in Rust. Every
//! completion is pumped cooperatively: when a poll is pending we yield one
//! macrotask so the browser can resolve WebGPU device-timeline promises.

use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::TokenId;
use minifield_executor_core::{Lfm2Classifier, Lfm2Executor, Lfm2Prefix};
use minifield_text_tokenizer::Tokenizer;
use wasm_bindgen::prelude::wasm_bindgen;

mod classifier;
mod constrained;
mod generation;
mod interop;
mod loading;
mod telemetry;
pub use telemetry::{configure_telemetry, flush_telemetry};

pub use loading::{load, load_classifier};

const MAX_LOGICAL_TOKENS: u64 = 512;
const STOP_TOKEN_IDS: [TokenId; 1] = [7];

/// Loaded executor plus tokenizer, ready for repeated `generate` calls.
#[wasm_bindgen]
pub struct WebDemo {
    telemetry: telemetry::BrowserTelemetry,
    executor: Lfm2Executor<WgpuBackend>,
    tokenizer: Tokenizer,
    /// Prefilled KV snapshot for the tool-call prompt's fixed system block,
    /// keyed by the exact system text. `generate_json` appends the short
    /// per-call tail onto this instead of re-prefilling the tool list.
    tool_prefix: Option<(String, Lfm2Prefix<WgpuBackend>)>,
}

/// Independent classifier; prompts already contain their exact chat template.
#[wasm_bindgen]
pub struct WebClassifier {
    telemetry: telemetry::BrowserTelemetry,
    classes: u32,
    classifier: Lfm2Classifier<WgpuBackend>,
    tokenizer: Tokenizer,
}
