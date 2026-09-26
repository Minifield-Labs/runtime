//! Fresh and shared-prefix classifier bindings.

use minifield_engine_api::{ExecutorError, TokenChunk, TokenId};
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
        let ids = self
            .tokenizer
            .encode(
                &prompt,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
            .map_err(js_error)?;
        let mut task = self
            .classifier
            .classify(TokenChunk::all(&ids))
            .map_err(js_error)?;
        pump(&mut task).await
    }

    /// Classify with shared-prefix reuse: the token head common to every
    /// prompt seen so far is prefilled once into a cached base, and each
    /// decision branches from that base by appending only its own tail.
    /// Falls back to a full prefill when a prompt does not share the head.
    pub async fn classify_cached(&mut self, prompt: String) -> Result<Vec<f32>, JsValue> {
        let ids = self
            .tokenizer
            .encode(
                &prompt,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
            .map_err(js_error)?;

        validate_cached_input(&ids, self.max_logical_tokens).map_err(js_error)?;

        match &self.anchor_ids {
            None => {
                self.anchor_ids = Some(ids.clone());
                self.shared_head = Some(ids.len());
            }
            Some(anchor) => {
                let mut head = self
                    .shared_head
                    .unwrap_or(0)
                    .min(ids.len())
                    .min(anchor.len());
                while head > 0 && ids[..head] != anchor[..head] {
                    head -= 1;
                }
                self.shared_head = Some(head);
            }
        }

        let head = self.shared_head.unwrap_or(0);
        let base_fits = match &self.base {
            Some(base) => checked_tail(&ids, base.token_history(), base.logical_length())
                .map_err(js_error)?
                .is_some(),
            None => false,
        };
        if !base_fits && head >= 16 {
            let anchor_head = self
                .anchor_ids
                .as_ref()
                .and_then(|anchor| anchor.get(..head))
                .ok_or_else(|| {
                    JsValue::from_str("classifier cache anchor is missing or too short")
                })?
                .to_vec();
            let mut task = self
                .classifier
                .prefill_base(TokenChunk::all(&anchor_head))
                .map_err(js_error)?;
            self.base = Some(pump(&mut task).await?);
        }

        if let Some(base) = self.base.clone()
            && let Some(tail) =
                checked_tail(&ids, base.token_history(), base.logical_length()).map_err(js_error)?
        {
            let mut task = self
                .classifier
                .classify_tail(&base, TokenChunk::all(tail))
                .map_err(js_error)?;
            return pump(&mut task).await;
        }

        let mut task = self
            .classifier
            .classify(TokenChunk::all(&ids))
            .map_err(js_error)?;
        pump(&mut task).await
    }
}

fn validate_cached_input(
    ids: &[TokenId],
    max_logical_tokens: u64,
) -> minifield_engine_api::Result<()> {
    if ids.is_empty() {
        return Err(ExecutorError::InvalidArgument(
            "classification requires a nonempty prompt",
        ));
    }
    let count = u64::try_from(ids.len())
        .map_err(|_| ExecutorError::Overflow("classification prompt length exceeds u64"))?;
    if count > max_logical_tokens {
        return Err(ExecutorError::ResourceLimit(
            "classification prompt exceeds prefix capacity",
        ));
    }
    Ok(())
}

fn checked_tail<'a>(
    ids: &'a [TokenId],
    history: &[TokenId],
    logical_length: u64,
) -> minifield_engine_api::Result<Option<&'a [TokenId]>> {
    let head = usize::try_from(logical_length)
        .map_err(|_| ExecutorError::Overflow("cached prefix length exceeds usize"))?;
    if history.len() != head {
        return Err(ExecutorError::InvalidArgument(
            "cached prefix length differs from token history",
        ));
    }
    if ids.len() <= head || ids.get(..head) != Some(history) {
        return Ok(None);
    }
    Ok(ids.get(head..))
}

#[cfg(test)]
mod tests {
    use super::{ExecutorError, checked_tail, validate_cached_input};

    #[test]
    fn cached_input_rejects_empty_and_over_capacity_before_admission() {
        assert_eq!(
            validate_cached_input(&[], 512),
            Err(ExecutorError::InvalidArgument(
                "classification requires a nonempty prompt"
            ))
        );
        assert_eq!(
            validate_cached_input(&[1; 513], 512),
            Err(ExecutorError::ResourceLimit(
                "classification prompt exceeds prefix capacity"
            ))
        );
        assert_eq!(validate_cached_input(&[1], 512), Ok(()));
        assert_eq!(validate_cached_input(&[1; 512], 512), Ok(()));
    }

    #[test]
    fn cached_tail_rejects_inconsistent_or_unrepresentable_lengths() {
        assert_eq!(
            checked_tail(&[1, 2, 3], &[1, 2], 3),
            Err(ExecutorError::InvalidArgument(
                "cached prefix length differs from token history"
            ))
        );
        assert!(checked_tail(&[1, 2, 3], &[1, 2], u64::MAX).is_err());
    }

    #[test]
    fn cached_tail_falls_back_for_short_or_changed_prompts() {
        assert_eq!(checked_tail(&[1], &[1, 2], 2), Ok(None));
        assert_eq!(checked_tail(&[1, 2], &[1, 2], 2), Ok(None));
        assert_eq!(checked_tail(&[1, 3, 4], &[1, 2], 2), Ok(None));
    }
}
