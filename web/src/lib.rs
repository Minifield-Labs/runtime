#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
//! Browser demo harness for the packed LFM2 wgpu path.
//!
//! The host page fetches the bundle bytes and hands them to [`load`]; the
//! executor, tokenizer, and generation loop all live here in Rust. Every
//! completion is pumped cooperatively: when a poll is pending we yield one
//! macrotask so the browser can resolve WebGPU device-timeline promises.

use core::fmt::Display;

use js_sys::{Function, Promise, Reflect};
use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{
    CompletionPoll, DecodeConstraint, InferenceCompletion, MemoryAssetProvider, ResourceLimits,
    TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenId,
};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2Prefix, Lfm2WeightFormat,
    Lfm2WeightLoadTask, LoaderLimits, LoaderPoll,
};
use minifield_json_grammar::AssistantCallEnforcer;
use minifield_text_generation::{ChoiceRequest, finish_choice, prepare_choice};
use minifield_text_tokenizer::{EncodeOptions, MODEL_VOCAB_SIZE, Tokenizer, TokenizerLimits};
use sha2::{Digest, Sha256};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

const MAX_LOGICAL_TOKENS: u64 = 512;
const STOP_TOKEN_IDS: [TokenId; 1] = [7];

/// Loaded executor plus tokenizer, ready for repeated `generate` calls.
#[wasm_bindgen]
pub struct WebDemo {
    executor: Lfm2Executor<WgpuBackend>,
    tokenizer: Tokenizer,
    /// Prefilled KV snapshot for the tool-call prompt's fixed system block,
    /// keyed by the exact system text. `generate_json` appends the short
    /// per-call tail onto this instead of re-prefilling the tool list.
    tool_prefix: Option<(String, Lfm2Prefix<WgpuBackend>)>,
}

fn js_error(error: impl Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn js_debug(error: impl core::fmt::Debug) -> JsValue {
    JsValue::from_str(&format!("{error:?}"))
}

/// Yield one macrotask so the browser can advance the WebGPU device timeline.
///
/// `map_async` and `on_submitted_work_done` resolve as JS promises only after
/// the event loop turns; a tight Rust poll loop would starve them forever.
/// The page provides `__minifieldYield` (a `MessageChannel` post, unclamped);
/// `setTimeout(0)` is the portable fallback.
async fn browser_yield() {
    let global = js_sys::global();
    let helper = Reflect::get(&global, &JsValue::from_str("__minifieldYield"))
        .ok()
        .and_then(|value| value.dyn_into::<Function>().ok());
    let promise = helper
        .and_then(|helper| helper.call0(&JsValue::NULL).ok())
        .and_then(|value| value.dyn_into::<Promise>().ok())
        .unwrap_or_else(timeout_promise);
    let _ = JsFuture::from(promise).await;
}

fn timeout_promise() -> Promise {
    Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        let set_timeout = Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|value| value.dyn_into::<Function>().ok());
        match set_timeout {
            Some(set_timeout) => {
                let _ = set_timeout.call1(&global, &resolve);
            }
            None => {
                let _ = resolve.call0(&JsValue::NULL);
            }
        }
    })
}

async fn pump<T: InferenceCompletion>(task: &mut T) -> Result<T::Output, JsValue> {
    loop {
        match task.poll_step() {
            CompletionPoll::Pending => browser_yield().await,
            CompletionPoll::Ready(result) => return result.map_err(js_error),
        }
    }
}

/// Load the packed ternary bundle from bytes the page fetched.
#[wasm_bindgen]
pub async fn load(
    config: Vec<u8>,
    weights: Vec<u8>,
    tokenizer: Vec<u8>,
) -> Result<WebDemo, JsValue> {
    let limits = ResourceLimits {
        max_allocation_bytes: 1 << 30,
        max_total_bytes: 2 << 30,
        max_pending_operations: 512,
    };
    let mut backend = WgpuBackend::new_async(0xE0_3C, limits)
        .await
        .map_err(js_error)?;

    let weights_len = weights.len() as u64;
    let request = Lfm2LoadRequest::new_with_format(
        config.clone(),
        Sha256::digest(&config).into(),
        weights_len,
        Sha256::digest(&weights).into(),
        LoaderLimits {
            max_asset_bytes: weights_len,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: weights_len,
            max_retained_host_bytes: weights_len * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        Lfm2WeightFormat::TernaryV1,
    )
    .map_err(js_debug)?;
    let mut provider = MemoryAssetProvider::new(weights, weights_len);
    let mut task = Lfm2WeightLoadTask::begin(request).map_err(js_debug)?;
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => browser_yield().await,
            LoaderPoll::Ready(result) => break result.map_err(js_debug)?,
        }
    };
    let executor = Lfm2Executor::new(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: MAX_LOGICAL_TOKENS,
        },
    )
    .map_err(js_error)?;
    let tokenizer =
        Tokenizer::from_json_bytes(&tokenizer, TokenizerLimits::default()).map_err(js_error)?;
    Ok(WebDemo {
        executor,
        tokenizer,
        tool_prefix: None,
    })
}

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
        let (input_ids, max_tokens) = self.prompt_budget(&prompt, max_tokens)?;

        let mut prefill = self
            .executor
            .prefill(TokenChunk::all(&input_ids))
            .map_err(js_error)?;
        let mut prefix: Lfm2Prefix<WgpuBackend> = pump(&mut prefill).await?;
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

    /// Score a prompt against one-token selectors in a single pass: prefill
    /// once, gather the selector logits, normalize, and return the winner.
    /// Only the selected logits cross to the host. Every array element must
    /// be a string. Resolves to `{ selectedIndex, selected, tokenId,
    /// scores }` where each score is `{ selector, tokenId, probability }`.
    pub async fn choose(
        &mut self,
        prompt: String,
        selectors: js_sys::Array,
    ) -> Result<JsValue, JsValue> {
        let mut owned = Vec::with_capacity(selectors.length() as usize);
        for value in selectors.iter() {
            owned.push(
                value
                    .as_string()
                    .ok_or_else(|| JsValue::from_str("selectors must be strings"))?,
            );
        }
        let selector_refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let prepared = prepare_choice(
            &self.tokenizer,
            &ChoiceRequest {
                prompt: &prompt,
                add_bos: true,
                selectors: &selector_refs,
                max_context_tokens: usize::try_from(MAX_LOGICAL_TOKENS).unwrap_or(usize::MAX),
            },
        )
        .map_err(js_error)?;
        let mut prefill = self
            .executor
            .prefill(TokenChunk::all(prepared.input_ids()))
            .map_err(js_error)?;
        let prefix: Lfm2Prefix<WgpuBackend> = pump(&mut prefill).await?;
        let mut selected = self
            .executor
            .choice_logits(&prefix, prepared.token_ids())
            .map_err(js_error)?;
        let logits = pump(&mut selected).await?;
        let result = finish_choice(prepared, logits).map_err(js_error)?;

        let scores = js_sys::Array::new();
        for score in &result.scores {
            let entry = js_sys::Object::new();
            Reflect::set(
                &entry,
                &JsValue::from_str("selector"),
                &JsValue::from_str(&score.selector),
            )?;
            Reflect::set(
                &entry,
                &JsValue::from_str("tokenId"),
                &JsValue::from_f64(f64::from(score.token_id)),
            )?;
            Reflect::set(
                &entry,
                &JsValue::from_str("probability"),
                &JsValue::from_f64(f64::from(score.probability)),
            )?;
            scores.push(&entry);
        }
        let winner = &result.scores[result.selected_index];
        let out = js_sys::Object::new();
        Reflect::set(
            &out,
            &JsValue::from_str("selectedIndex"),
            &JsValue::from_f64(
                u32::try_from(result.selected_index)
                    .unwrap_or(u32::MAX)
                    .into(),
            ),
        )?;
        Reflect::set(
            &out,
            &JsValue::from_str("selected"),
            &JsValue::from_str(&winner.selector),
        )?;
        Reflect::set(
            &out,
            &JsValue::from_str("tokenId"),
            &JsValue::from_f64(f64::from(winner.token_id)),
        )?;
        Reflect::set(&out, &JsValue::from_str("scores"), &scores)?;
        Ok(out.into())
    }

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
        let rest_ids = &input_ids[system_ids.len()..];
        let max_tokens = usize::try_from(MAX_LOGICAL_TOKENS)
            .unwrap_or(usize::MAX)
            .saturating_sub(input_ids.len())
            .min(usize::try_from(max_tokens).unwrap_or(usize::MAX));
        if max_tokens == 0 {
            return Err(JsValue::from_str("prompt fills the context budget"));
        }
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
        let mut enforcer = AssistantCallEnforcer::new(vocab, STOP_TOKEN_IDS[0], names);

        let mut prefix: Lfm2Prefix<WgpuBackend> = if system_ids.is_empty() {
            let mut prefill = self
                .executor
                .prefill_masked(TokenChunk::all(&input_ids), enforcer.allowed())
                .map_err(js_error)?;
            pump(&mut prefill).await?
        } else {
            let base = match &self.tool_prefix {
                Some((cached_system, prefix)) if *cached_system == system => prefix.clone(),
                _ => {
                    let mut prefill = self
                        .executor
                        .prefill(TokenChunk::all(&system_ids))
                        .map_err(js_error)?;
                    let base = pump(&mut prefill).await?;
                    self.tool_prefix = Some((system.clone(), base.clone()));
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
                    .prefill_masked(TokenChunk::all(&input_ids), mask)
                    .map_err(js_error)?;
                pump(&mut prefill).await?
            }
        };
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

fn stats(text: &str, generated: usize, stopped: bool) -> Result<JsValue, JsValue> {
    let stats = js_sys::Object::new();
    Reflect::set(&stats, &JsValue::from_str("text"), &JsValue::from_str(text))?;
    Reflect::set(
        &stats,
        &JsValue::from_str("tokens"),
        &JsValue::from_f64(u32::try_from(generated).unwrap_or(u32::MAX).into()),
    )?;
    Reflect::set(
        &stats,
        &JsValue::from_str("stopped"),
        &JsValue::from_bool(stopped),
    )?;
    Ok(stats.into())
}
