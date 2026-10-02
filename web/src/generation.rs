//! Greedy text generation and shared-base structured choices.

use js_sys::{Function, Reflect};
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{DecodeConstraint, TokenChoiceExecutor, TokenChunk};
use minifield_executor_core::{Lfm2Executor, Lfm2Prefix};
use minifield_runtime_telemetry::{Measurement, Mode};
use minifield_text_generation::{
    ChoiceCriterion, ChoiceRequest, GenerationEvent, GenerationInput, GenerationRequest,
    GenerationResult, NeverCancel, StopReason, finish_choice, generate_task, prepare_choice,
};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

use super::{
    MAX_LOGICAL_TOKENS, WebDemo,
    interop::{js_error, pump, pump_future, stats, to_string_vec},
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
            let mut request = GenerationRequest::with_eos(
                &prompt,
                max_tokens as usize,
                usize::try_from(MAX_LOGICAL_TOKENS).unwrap_or(usize::MAX),
            );
            request.add_bos = true;
            request.skip_special_tokens = false;
            let generated = self
                .run_generation(
                    &request,
                    GenerationInput::Prompt,
                    None,
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
    pub(super) async fn run_generation(
        &mut self,
        request: &GenerationRequest<'_>,
        input: GenerationInput<Lfm2Prefix<WgpuBackend>>,
        constraint: Option<&mut dyn DecodeConstraint>,
        on_token: &Function,
        measurement: &mut Measurement,
    ) -> Result<GenerationResult, JsValue> {
        let mut observe =
            |event: GenerationEvent<'_>, executor: &Lfm2Executor<WgpuBackend>| match event {
                GenerationEvent::Tokenized(count) => {
                    if measurement.tokenization_ms.is_none() {
                        measurement.tokenized(count);
                    }
                }
                GenerationEvent::Prefilled => measurement.prefilled(executor.inference_work()),
                GenerationEvent::Token { id, fragment } => {
                    measurement.emitted();
                    let _ = on_token.call2(
                        &JsValue::NULL,
                        &JsValue::from_str(fragment),
                        &JsValue::from_f64(f64::from(id)),
                    );
                }
            };
        let result = pump_future(generate_task(
            &mut self.executor,
            &self.tokenizer,
            request,
            input,
            constraint,
            &mut NeverCancel,
            &mut observe,
        ))
        .await?;
        measurement.stop_reason = match result.stop_reason {
            StopReason::MaxOutputTokens => "output_limit",
            StopReason::StopToken(_) => "end_token",
            StopReason::ConstraintComplete => "constraint_complete",
        };
        Ok(result)
    }
}
