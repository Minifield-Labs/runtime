//! Fresh and shared-prefix classifier bindings.

use minifield_engine_api::TokenChunk;
use minifield_runtime_telemetry::{Measurement, Mode};
use minifield_text_tokenizer::EncodeOptions;
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use super::{
    WebClassifier,
    interop::{js_error, pump},
};

#[wasm_bindgen]
impl WebClassifier {
    /// Fresh prompt state each call. The caller masks any reserved class.
    pub async fn classify(&mut self, prompt: String) -> Result<Vec<f32>, JsValue> {
        let mut measurement = Measurement::new(Mode::SingleStep, self.classifier.inference_work());
        measurement.alternatives = Some(self.classes as usize);
        let result = async {
            let ids = self
                .tokenizer
                .encode(
                    &prompt,
                    EncodeOptions {
                        add_special_tokens: false,
                    },
                )
                .map_err(js_error)?;
            measurement.tokenized(ids.len());
            let mut task = self
                .classifier
                .classify(TokenChunk::all(&ids))
                .map_err(js_error)?;
            pump(&mut task).await
        }
        .await;
        self.telemetry.finish(
            measurement,
            self.classifier.inference_work(),
            result.is_ok(),
        );
        result
    }

    /// Classify with shared-prefix reuse: the token head common to every
    /// prompt seen so far is prefilled once into a cached base, and each
    /// decision branches from that base by appending only its own tail.
    /// Falls back to a full prefill when a prompt does not share the head.
    pub async fn classify_cached(&mut self, prompt: String) -> Result<Vec<f32>, JsValue> {
        let mut measurement = Measurement::new(Mode::SingleStep, self.classifier.inference_work());
        measurement.alternatives = Some(self.classes as usize);
        let result = async {
            let ids = self
                .tokenizer
                .encode(
                    &prompt,
                    EncodeOptions {
                        add_special_tokens: false,
                    },
                )
                .map_err(js_error)?;

            measurement.tokenized(ids.len());
            let mut task = self
                .classifier
                .classify_cached(TokenChunk::all(&ids))
                .map_err(js_error)?;
            let result = pump(&mut task).await;
            let cache = task.cache_stats();
            measurement.cache_rebuilds += cache.rebuilds;
            measurement.cache_reused += cache.reused_tokens;
            measurement.fallback_used = cache.fallback_used;
            result
        }
        .await;
        self.telemetry.finish(
            measurement,
            self.classifier.inference_work(),
            result.is_ok(),
        );
        result
    }
}
