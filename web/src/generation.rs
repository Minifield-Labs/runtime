//! Greedy text generation and shared-base structured choices.

use js_sys::{Function, Reflect};
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{TokenChoiceExecutor, TokenChunk, TokenExecutor};
use minifield_executor_core::Lfm2Prefix;
use minifield_runtime_telemetry::{Measurement, Mode};
use minifield_text_generation::{ChoiceCriterion, ChoiceRequest, finish_choice, prepare_choice};
use minifield_text_tokenizer::EncodeOptions;
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

use super::{
    MAX_LOGICAL_TOKENS, STOP_TOKEN_IDS, WebDemo,
    interop::{js_error, pump, stats, to_string_vec},
};

#[wasm_bindgen]
impl WebDemo {
    /// Greedy-generate up to `max_tokens`, invoking `on_token(fragment)` per
    /// decoded piece. Resolves to `{ text, tokens, stopped }`.
    pub async fn generate(
        &mut self,
        prompt: String,
        max_tokens: u32,
        on_token: Function,
    ) -> Result<JsValue, JsValue> {
        let mut measurement =
            Measurement::new(Mode::Autoregressive, self.executor.inference_work());
        measurement.max_output_tokens = max_tokens as usize;
        let result = async {
            let (input_ids, max_tokens) = self.prompt_budget(&prompt, max_tokens)?;
            measurement.tokenized(input_ids.len());
            if max_tokens < measurement.max_output_tokens {
                measurement.stop_reason = "context_limit";
            }

            let mut prefill = self
                .executor
                .prefill(TokenChunk::all(&input_ids))
                .map_err(js_error)?;
            let mut prefix: Lfm2Prefix<WgpuBackend> = pump(&mut prefill).await?;
            measurement.prefilled(self.executor.inference_work());
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
                    measurement.stop_reason = "end_token";
                    break;
                }
                let fragment = decoder.push(&[next]).map_err(js_error)?;
                text.push_str(&fragment);
                generated += 1;
                measurement.emitted();
                let _ = on_token.call2(
                    &JsValue::NULL,
                    &JsValue::from_str(&fragment),
                    &JsValue::from_f64(f64::from(next)),
                );
                if generated >= max_tokens {
                    break;
                }
                let mut append = self.executor.append_argmax(prefix).map_err(js_error)?;
                prefix = pump(&mut append).await?;
            }
            text.push_str(&decoder.finish().map_err(js_error)?);
            stats(&text, generated, stopped)
        }
        .await;
        self.telemetry
            .finish(measurement, self.executor.inference_work(), result.is_ok());
        result
    }

    /// Score named criteria serially against one shared base prompt: the base
    /// is prefilled once, then each criterion tail branches off it and its
    /// true/false selector logits are read back and normalized. Every array
    /// element must be a string. Resolves to
    /// `{ type: "choice", choice, confidence, probabilities }` where
    /// `probabilities` maps each criterion name to its relative score.
    pub async fn choose(
        &mut self,
        base_prompt: String,
        names: js_sys::Array,
        tails: js_sys::Array,
    ) -> Result<JsValue, JsValue> {
        let mut measurement = Measurement::new(Mode::SingleStep, self.executor.inference_work());
        let result = async {
            if names.length() != tails.length() {
                return Err(JsValue::from_str("names and tails must have equal lengths"));
            }
            let names = to_string_vec(&names)?;
            let tails = to_string_vec(&tails)?;
            let criteria: Vec<ChoiceCriterion<'_>> = names
                .iter()
                .zip(&tails)
                .map(|(name, tail)| ChoiceCriterion {
                    name: name.as_str(),
                    tail: tail.as_str(),
                })
                .collect();
            let prepared = prepare_choice(
                &self.tokenizer,
                &ChoiceRequest {
                    base_prompt: &base_prompt,
                    criteria: &criteria,
                    true_selector: "true",
                    false_selector: "false",
                    add_bos: true,
                    max_context_tokens: usize::try_from(MAX_LOGICAL_TOKENS).unwrap_or(usize::MAX),
                },
            )
            .map_err(js_error)?;
            measurement.tokenized(
                prepared.base_input_ids().len()
                    + prepared
                        .criteria()
                        .iter()
                        .map(|c| c.tail_ids().len())
                        .sum::<usize>(),
            );
            measurement.alternatives = Some(prepared.criteria().len());
            let mut base_task = self
                .executor
                .prefill_choice_base(TokenChunk::all(prepared.base_input_ids()))
                .map_err(js_error)?;
            let base: Lfm2Prefix<WgpuBackend> = pump(&mut base_task).await?;
            let mut logit_pairs = Vec::with_capacity(prepared.criteria().len());
            for criterion in prepared.criteria() {
                measurement.cache_reused += base.logical_length();
                let mut task = self
                    .executor
                    .append_choice_logits(
                        &base,
                        TokenChunk::all(criterion.tail_ids()),
                        prepared.token_ids(),
                    )
                    .map_err(js_error)?;
                logit_pairs.push(pump(&mut task).await?);
            }
            let result = finish_choice(prepared, logit_pairs).map_err(js_error)?;

            let probabilities =
                js_sys::Object::create(&JsValue::NULL.unchecked_into::<js_sys::Object>());
            for probability in &result.probabilities {
                Reflect::set(
                    &probabilities,
                    &JsValue::from_str(&probability.name),
                    &JsValue::from_f64(f64::from(probability.probability)),
                )?;
            }
            let out = js_sys::Object::new();
            Reflect::set(
                &out,
                &JsValue::from_str("type"),
                &JsValue::from_str("choice"),
            )?;
            Reflect::set(
                &out,
                &JsValue::from_str("choice"),
                &JsValue::from_str(&result.choice),
            )?;
            Reflect::set(
                &out,
                &JsValue::from_str("confidence"),
                &JsValue::from_f64(f64::from(result.confidence)),
            )?;
            Reflect::set(&out, &JsValue::from_str("probabilities"), &probabilities)?;
            Ok(out.into())
        }
        .await;
        self.telemetry
            .finish(measurement, self.executor.inference_work(), result.is_ok());
        result
    }
}

impl WebDemo {
    fn prompt_budget(&self, prompt: &str, max_tokens: u32) -> Result<(Vec<u32>, usize), JsValue> {
        let input_ids = self
            .tokenizer
            .encode(
                prompt,
                EncodeOptions {
                    add_special_tokens: true,
                },
            )
            .map_err(js_error)?;
        let budget = usize::try_from(MAX_LOGICAL_TOKENS)
            .unwrap_or(usize::MAX)
            .saturating_sub(input_ids.len());
        let max_tokens = usize::try_from(max_tokens)
            .unwrap_or(usize::MAX)
            .min(budget);
        if max_tokens == 0 {
            return Err(JsValue::from_str("prompt fills the context budget"));
        }
        Ok((input_ids, max_tokens))
    }
}
