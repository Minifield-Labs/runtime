//! Complete-sequence pointer inference. Hosts supply joint layouts and wording.

use std::{cell::RefCell, rc::Rc};

use minifield_backend_wgpu::WgpuBackend;
use minifield_engine_api::{EncoderSegments, MemoryAssetProvider, ResourceLimits};
use minifield_executor_core::{
    EncoderInput, EncoderLimits, EncoderLoadRequest, EncoderWeightLoadTask, Lfm2PointerEncoder,
    LoaderLimits, LoaderPoll, PointerAnswer, PointerQuestion, PointerQuestionKind,
    detect_lfm2_weight_format, parse_lfm2_tensor_quantization,
};
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerLimits};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use super::interop::{browser_yield, js_debug, js_error, pump};

/// Joint tokens per request unless the host asks for another bound.
const DEFAULT_MAX_TOKENS: u32 = 2048;
const MAX_QUESTIONS: u32 = 32;
/// JSON bytes allowed per joint token: room for an extract `selectable`
/// entry per question plus IDs and segments.
const INPUT_BYTES_PER_TOKEN: usize = 256;

#[wasm_bindgen]
pub struct WebPointerEncoder {
    encoder: Lfm2PointerEncoder<WgpuBackend>,
    tokenizer: Tokenizer,
    max_tokens: usize,
}

/// A tokenizer on its own, so hosts can lay out requests before a model loads.
#[wasm_bindgen]
pub struct WebTokenizer {
    tokenizer: Tokenizer,
}

/// Load a bundle's `tokenizer.json` without a model or GPU.
#[wasm_bindgen]
pub fn load_tokenizer(tokenizer: &[u8]) -> Result<WebTokenizer, JsValue> {
    Ok(WebTokenizer {
        tokenizer: Tokenizer::from_json_bytes(tokenizer, TokenizerLimits::default())
            .map_err(js_error)?,
    })
}

#[wasm_bindgen]
impl WebTokenizer {
    /// Exact IDs and half-open UTF-8 byte offsets into `text`, as `WebPointerEncoder::tokenize`.
    pub fn tokenize(&self, text: &str, bos: bool) -> Result<JsValue, JsValue> {
        tokenize(&self.tokenizer, text, bos)
    }
}

/// Load a bidirectional pointer bundle, independently of causal generation.
///
/// `max_tokens` bounds joint tokens per request (default 2048); larger
/// bounds cost GPU memory and time.
#[wasm_bindgen]
pub async fn load_pointer_encoder(
    config: Vec<u8>,
    weights: Vec<u8>,
    tokenizer: Vec<u8>,
    max_tokens: Option<u32>,
) -> Result<WebPointerEncoder, JsValue> {
    let max_tokens = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(JsValue::from_str("max_tokens must be positive"));
    }
    let tokenizer =
        Tokenizer::from_json_bytes(&tokenizer, TokenizerLimits::default()).map_err(js_error)?;
    let mut backend = WgpuBackend::new_async(
        0xE0_3D,
        ResourceLimits {
            max_allocation_bytes: 1 << 30,
            max_total_bytes: 2 << 30,
            max_pending_operations: 512,
        },
    )
    .await
    .map_err(js_error)?;
    let len = weights.len() as u64;
    let format = detect_lfm2_weight_format(&weights).map_err(js_debug)?;
    let overrides = parse_lfm2_tensor_quantization(&weights).map_err(js_debug)?;
    let request = EncoderLoadRequest::new_with_quantization(
        config.clone(),
        Sha256::digest(&config).into(),
        len,
        Sha256::digest(&weights).into(),
        LoaderLimits {
            max_asset_bytes: len,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: len,
            max_retained_host_bytes: len * 6,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        format,
        &overrides,
    )
    .map_err(js_error)?;
    let mut provider = MemoryAssetProvider::new(weights, len);
    let mut task = EncoderWeightLoadTask::begin(request).map_err(js_debug)?;
    let typed = loop {
        match task.poll_step(&mut provider, &mut backend) {
            LoaderPoll::Pending => browser_yield().await,
            LoaderPoll::Ready(result) => break result.map_err(js_debug)?,
        }
    };
    let limits = EncoderLimits {
        max_tokens: u64::from(max_tokens),
        max_questions: MAX_QUESTIONS,
    };
    let encoder = Lfm2PointerEncoder::new(Rc::new(RefCell::new(backend)), Rc::new(typed), limits)
        .map_err(js_error)?;
    Ok(WebPointerEncoder {
        encoder,
        tokenizer,
        max_tokens: max_tokens as usize,
    })
}

#[wasm_bindgen]
impl WebPointerEncoder {
    /// Exact IDs and half-open UTF-8 byte offsets into `text`. Optional BOS has offset `[0,0]`.
    pub fn tokenize(&self, text: &str, bos: bool) -> Result<JsValue, JsValue> {
        tokenize(&self.tokenizer, text, bos)
    }

    /// Predict an explicit joint layout. Spans are half-open source-relative token offsets.
    pub async fn predict(&mut self, input: &str) -> Result<JsValue, JsValue> {
        let input = parse_input(input, self.max_tokens).map_err(js_error)?;
        let mut task = self.encoder.begin_predict(input).map_err(js_error)?;
        let output = pump(&mut task).await?;
        let answers: Vec<Value> = output.answers.into_iter().map(|answer| match answer {
            PointerAnswer::Choice { index, probabilities } =>
                json!({"type": "choice", "index": index, "probabilities": probabilities}),
            PointerAnswer::Ordinal { value, probabilities } =>
                json!({"type": "ordinal", "value": value, "probabilities": probabilities}),
            PointerAnswer::Binary { probability, probabilities } =>
                json!({"type": "binary", "probability": probability, "probabilities": probabilities}),
            PointerAnswer::Span { span, presence } =>
                json!({"type": "extract", "span": span, "presence": presence}),
        }).collect();
        js_sys::JSON::parse(&json!({
            "tokens": output.tokens, "start": output.start, "end": output.end, "answers": answers,
        }).to_string())
    }
}

/// IDs and original-text byte offsets. The tokenizer encodes normalized
/// text, so offsets map back through its normalizer; a span covers every
/// original byte its normalized bytes came from.
fn tokenize(tokenizer: &Tokenizer, text: &str, bos: bool) -> Result<JsValue, JsValue> {
    let ids = tokenizer
        .encode(
            text,
            EncodeOptions {
                add_special_tokens: bos,
            },
        )
        .map_err(js_error)?;
    let normalized = tokenizer.normalize(text).map_err(js_error)?;
    let mut cursor = 0;
    let mut offsets = Vec::with_capacity(ids.len());
    for (index, &id) in ids.iter().enumerate() {
        if bos && index == 0 {
            offsets.push([0, 0]);
            continue;
        }
        let bytes = tokenizer.token_bytes(id).map_err(js_error)?;
        let end = cursor + bytes.len();
        if normalized.text.as_bytes().get(cursor..end) != Some(bytes) {
            return Err(JsValue::from_str("token bytes differ from source text"));
        }
        let (start, stop) = normalized.original_span(cursor, end);
        offsets.push([start, stop]);
        cursor = end;
    }
    if cursor != normalized.text.len() {
        return Err(JsValue::from_str("token bytes don't cover source text"));
    }
    js_sys::JSON::parse(&json!({"ids": ids, "offsets": offsets}).to_string())
}

fn unsigned(value: &Value) -> Result<u32, &'static str> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or("expected a u32 integer")
}

fn integers(value: &Value, limit: usize) -> Result<Vec<u32>, &'static str> {
    let values = value
        .as_array()
        .filter(|values| values.len() <= limit)
        .ok_or("expected a bounded integer array")?;
    values.iter().map(unsigned).collect()
}

fn parse_input(input: &str, max_tokens: usize) -> Result<EncoderInput, &'static str> {
    if input.len() > max_tokens * INPUT_BYTES_PER_TOKEN {
        return Err("pointer input exceeds byte limit");
    }
    let value: Value = serde_json::from_str(input).map_err(|_| "invalid pointer input JSON")?;
    let token_ids = integers(&value["token_ids"], max_tokens)?;
    let segment_ids = match value.get("segment_ids") {
        Some(ids) => integers(ids, max_tokens)?,
        None => vec![1; token_ids.len()],
    };
    let segments = EncoderSegments::new(segment_ids).map_err(|_| "invalid encoder segments")?;
    let questions = value["questions"]
        .as_array()
        .filter(|questions| !questions.is_empty() && questions.len() <= MAX_QUESTIONS as usize)
        .ok_or("expected 1..=32 pointer questions")?;
    let questions = questions
        .iter()
        .map(|question| {
            let policy = &question["kind"];
            let kind = match policy["type"].as_str() {
                Some("choice") => PointerQuestionKind::Choice,
                Some("ordinal") => PointerQuestionKind::Ordinal,
                Some("binary") => PointerQuestionKind::Binary {
                    positive_option: unsigned(&policy["positive_option"])? as usize,
                },
                Some("extract") => {
                    let selectable = policy["selectable"]
                        .as_array()
                        .filter(|values| values.len() <= max_tokens)
                        .ok_or("expected a bounded selectable array")?
                        .iter()
                        .map(|value| value.as_bool().ok_or("selectable values must be boolean"))
                        .collect::<Result<Vec<_>, _>>()?;
                    PointerQuestionKind::Extract {
                        absent_index: unsigned(&policy["absent_index"])?,
                        source_start: unsigned(&policy["source_start"])?,
                        selectable,
                        presence_threshold: policy["presence_threshold"]
                            .as_f64()
                            .ok_or("expected a presence threshold")?,
                    }
                }
                _ => return Err("unsupported pointer question type"),
            };
            Ok(PointerQuestion {
                query_index: unsigned(&question["query_index"])?,
                option_indices: integers(&question["option_indices"], max_tokens)?,
                kind,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EncoderInput {
        token_ids,
        segments,
        questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_explicit_segments_and_extract_policy_without_coercion() {
        let input = json!({"token_ids": [1, 2, 3], "segment_ids": [1, 1, 1], "questions": [{
            "query_index": 0, "option_indices": [1], "kind": {
                "type": "extract", "absent_index": 1, "source_start": 2,
                "selectable": [true], "presence_threshold": 0.5,
            },
        }]});
        assert_eq!(
            parse_input(&input.to_string(), 512).map(|input| input.token_ids),
            Ok(vec![1, 2, 3])
        );
        let mut bad = input.clone();
        bad["questions"][0]["kind"]["selectable"][0] = json!(1);
        assert!(parse_input(&bad.to_string(), 512).is_err());
        let mut bad = input;
        bad["token_ids"][0] = json!(1.5);
        assert!(parse_input(&bad.to_string(), 512).is_err());
    }

    #[test]
    fn rejects_oversized_or_unknown_requests() {
        assert!(parse_input(&" ".repeat(131_073), 512).is_err());
        assert!(
            parse_input(
                &json!({"token_ids": vec![1; 513], "questions": []}).to_string(),
                512
            )
            .is_err()
        );
        assert!(parse_input(r#"{"token_ids":[1],"questions":[{"query_index":0,"option_indices":[0],"kind":{"type":"unknown"}}]}"#, 512).is_err());
    }
}
