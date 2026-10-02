//! Cached tool prompts and grammar-constrained assistant generation.

use js_sys::Function;
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{DecodeConstraint, TokenChunk, TokenExecutor, TokenId};
use minifield_executor_core::Lfm2Prefix;
use minifield_json_grammar::AssistantCallEnforcer;
use minifield_runtime_telemetry::{Measurement, Mode};
use minifield_text_generation::{GenerationInput, GenerationRequest, StopReason};
use minifield_text_tokenizer::{EncodeOptions, MODEL_VOCAB_SIZE};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use super::{
    MAX_LOGICAL_TOKENS, STOP_TOKEN_IDS, WebDemo,
    interop::{js_error, pump, stats},
};

#[wasm_bindgen]
impl WebDemo {
    /// Prefill and cache the tool-call prompt's fixed system block so the
    /// first constrained run skips it too.
    pub async fn warm_tools(&mut self, system: String) -> Result<(), JsValue> {
        if system.is_empty() || matches!(&self.tool_prefix, Some((cached, _)) if *cached == system)
        {
            return Ok(());
        }
        let ids = self
            .tokenizer
            .encode(
                &system,
                EncodeOptions {
                    add_special_tokens: true,
                },
            )
            .map_err(js_error)?;
        let mut prefill = self
            .executor
            .prefill(TokenChunk::all(&ids))
            .map_err(js_error)?;
        let base = pump(&mut prefill).await?;
        self.tool_prefix = Some((system, base));
        Ok(())
    }

    /// Greedy-generate constrained to the serialized assistant body
    /// `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
    /// where `<name>` is one of the comma-separated `names`, matching the
    /// lfm2-chatml-tool-json training serializer. A byte-level grammar
    /// enforcer gates every argmax, so only tokens that keep the document
    /// acceptable can win. Stops when the document completes, on EOS, or at
    /// `max_tokens`. Resolves to `{ text, tokens, stopped }`.
    ///
    /// `system` is the fixed tool-list block and `rest` the per-call
    /// user/assistant turns; concatenating them yields the full serialized
    /// prompt. The system block's KV state is cached between calls so each
    /// run only processes the short tail. The executor applies the grammar
    /// mask before publishing the tail's first sample.
    pub async fn generate_json(
        &mut self,
        system: String,
        rest: String,
        names: String,
        max_tokens: u32,
        on_token: Function,
    ) -> Result<JsValue, JsValue> {
        let mut measurement =
            Measurement::new(Mode::Autoregressive, self.executor.inference_work());
        measurement.max_output_tokens = max_tokens as usize;
        measurement.constraint = "tool_call";
        let result = async {
            let mut input_ids = self
                .tokenizer
                .encode(
                    &system,
                    EncodeOptions {
                        add_special_tokens: true,
                    },
                )
                .map_err(js_error)?;
            let system_ids = input_ids.clone();
            input_ids.extend(
                self.tokenizer
                    .encode(
                        &rest,
                        EncodeOptions {
                            add_special_tokens: false,
                        },
                    )
                    .map_err(js_error)?,
            );
            measurement.tokenized(input_ids.len());
            let prompt = format!("{system}{rest}");
            let mut request = GenerationRequest::with_eos(
                &prompt,
                max_tokens as usize,
                usize::try_from(MAX_LOGICAL_TOKENS).unwrap_or(usize::MAX),
            );
            request.add_bos = true;
            request.skip_special_tokens = false;
            let mut enforcer = self.assistant_enforcer(&names)?;
            let input = if max_tokens == 0 {
                GenerationInput::Prompt
            } else {
                let prefix = self
                    .constrained_prefix(
                        &system,
                        &system_ids,
                        &input_ids,
                        &mut enforcer,
                        &mut measurement,
                    )
                    .await?;
                GenerationInput::Prefix(prefix)
            };
            let generated = self
                .run_generation(
                    &request,
                    input,
                    Some(&mut enforcer),
                    &on_token,
                    &mut measurement,
                )
                .await?;
            stats(
                &generated.text,
                generated.generated_ids.len(),
                generated.stop_reason != StopReason::MaxOutputTokens,
            )
        }
        .await;
        self.telemetry
            .finish(measurement, self.executor.inference_work(), result.is_ok());
        result
    }
}

impl WebDemo {
    fn assistant_enforcer(&self, names: &str) -> Result<AssistantCallEnforcer, JsValue> {
        let names: Vec<Vec<u8>> = names
            .split(',')
            .map(|name| name.trim().as_bytes().to_vec())
            .filter(|name| !name.is_empty())
            .collect();
        if names.is_empty()
            || names.iter().any(|name| {
                !name
                    .iter()
                    .all(|b| (0x20..=0x7e).contains(b) && *b != b'"' && *b != b'\\')
            })
        {
            return Err(js_error(
                "tool names must be nonempty printable ASCII without '\"' or '\\'",
            ));
        }
        let vocab: Vec<Vec<u8>> = (0..MODEL_VOCAB_SIZE)
            .map(|id| {
                self.tokenizer
                    .token_bytes(id)
                    .map(<[u8]>::to_vec)
                    .unwrap_or_default()
            })
            .collect();
        // Serialized assistant body, compact canonical JSON.
        Ok(AssistantCallEnforcer::new(vocab, STOP_TOKEN_IDS[0], names))
    }

    async fn constrained_prefix(
        &mut self,
        system: &str,
        system_ids: &[TokenId],
        input_ids: &[TokenId],
        enforcer: &mut AssistantCallEnforcer,
        measurement: &mut Measurement,
    ) -> Result<Lfm2Prefix<WgpuBackend>, JsValue> {
        let rest_ids = input_ids
            .get(system_ids.len()..)
            .ok_or_else(|| JsValue::from_str("tool system prefix exceeds the full prompt"))?;
        let prefix = if system_ids.is_empty() {
            let mut prefill = self
                .executor
                .prefill_masked(TokenChunk::all(input_ids), enforcer.allowed())
                .map_err(js_error)?;
            pump(&mut prefill).await?
        } else {
            let base = match &self.tool_prefix {
                Some((cached_system, prefix)) if *cached_system == system => prefix.clone(),
                _ => {
                    let mut prefill = self
                        .executor
                        .prefill(TokenChunk::all(system_ids))
                        .map_err(js_error)?;
                    let base = pump(&mut prefill).await?;
                    self.tool_prefix = Some((system.to_owned(), base.clone()));
                    measurement.cache_rebuilds += 1;
                    base
                }
            };
            measurement.cache_reused += base.logical_length();
            let mut append = self
                .executor
                .append_known_masked(&base, TokenChunk::all(rest_ids), enforcer.allowed())
                .map_err(js_error)?;
            pump(&mut append).await?
        };
        Ok(prefix)
    }
}
