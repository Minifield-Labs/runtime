//! Cached tool prompts and grammar-constrained assistant generation.

use js_sys::Function;
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{DecodeConstraint, TokenChunk, TokenExecutor, TokenId};
use minifield_executor_core::Lfm2Prefix;
use minifield_json_grammar::AssistantCallEnforcer;
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
    /// run only prefills the short tail. When the cache is cold or the tail's
    /// unmasked greedy sample is not grammar-legal, the whole prompt is
    /// prefilled with the mask applied directly.
    pub async fn generate_json(
        &mut self,
        system: String,
        rest: String,
        names: String,
        max_tokens: u32,
        on_token: Function,
    ) -> Result<JsValue, JsValue> {
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
        let max_tokens = usize::try_from(MAX_LOGICAL_TOKENS)
            .unwrap_or(usize::MAX)
            .saturating_sub(input_ids.len())
            .min(usize::try_from(max_tokens).unwrap_or(usize::MAX));
        if max_tokens == 0 {
            return Err(JsValue::from_str("prompt fills the context budget"));
        }
        let mut enforcer = self.assistant_enforcer(&names)?;

        let mut prefix = self
            .constrained_prefix(&system, &system_ids, &input_ids, &mut enforcer)
            .await?;

        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut generated = 0_usize;
        let mut stopped = false;

        loop {
            let next = self
                .executor
                .sampled_token(&prefix)
                .map_err(js_error)?
                .ok_or_else(|| JsValue::from_str("executor published no greedy sample"))?;
            if STOP_TOKEN_IDS.contains(&next) {
                stopped = true;
                break;
            }
            let fragment = decoder.push(&[next]).map_err(js_error)?;
            text.push_str(&fragment);
            generated += 1;
            enforcer.advance(next);
            let _ = on_token.call2(
                &JsValue::NULL,
                &JsValue::from_str(&fragment),
                &JsValue::from_f64(f64::from(next)),
            );
            if enforcer.complete() {
                stopped = true;
                break;
            }
            if generated >= max_tokens {
                break;
            }
            let mut append = self
                .executor
                .append_argmax_masked(prefix, enforcer.allowed())
                .map_err(js_error)?;
            prefix = pump(&mut append).await?;
        }
        text.push_str(&decoder.finish().map_err(js_error)?);
        stats(&text, generated, stopped)
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
                    base
                }
            };
            let mut append = self
                .executor
                .append_known(&base, TokenChunk::all(rest_ids))
                .map_err(js_error)?;
            let staged = pump(&mut append).await?;
            // The tail append resolved its pending sample unmasked: reuse it
            // only when it is grammar-legal, else prefill the full prompt
            // with the mask so the first emitted token is still constrained.
            let mask = enforcer.allowed();
            let legal = self
                .executor
                .sampled_token(&staged)
                .map_err(js_error)?
                .is_some_and(|id| {
                    mask.get(id as usize / 64)
                        .is_some_and(|word| word >> (id % 64) & 1 == 1)
                });
            if legal {
                staged
            } else {
                let mut prefill = self
                    .executor
                    .prefill_masked(TokenChunk::all(input_ids), mask)
                    .map_err(js_error)?;
                pump(&mut prefill).await?
            }
        };
        Ok(prefix)
    }
}
