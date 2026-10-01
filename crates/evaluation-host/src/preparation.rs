//! Tokenization, model admission, and cached-base preparation before timing.
use crate::{
    HostResult,
    backend::HostBackend,
    completion::{check, deadline, wait},
    loading::{self, ModelAsset},
    measurement::{self, Snapshot},
    prediction::{self, Prediction},
    request::{Case, Request},
};
use minifield_engine_api::TokenChunk;
use minifield_executor_core::{
    EncoderInput, EncoderLimits, Lfm2Classifier, Lfm2ExecutionLimits, Lfm2ExecutionOptions,
    Lfm2Lut2Mode, Lfm2PointerEncoder, Lfm2Prefix,
};
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerLimits};
use std::{cell::RefCell, rc::Rc, time::Instant};

pub(crate) trait PreparedModel {
    fn predict(&mut self, index: usize, seconds: f64) -> HostResult<Prediction>;
    fn snapshot(&self) -> HostResult<Snapshot>;
}

fn lut2(mode: &str) -> HostResult<Lfm2Lut2Mode> {
    match mode {
        "raw" | "off" => Ok(Lfm2Lut2Mode::Off),
        "down" => Ok(Lfm2Lut2Mode::DownOnly),
        "auto" => Ok(Lfm2Lut2Mode::Auto),
        _ => Err("unknown LUT2 policy".into()),
    }
}

pub(crate) fn common_prefix(encoded: &[Vec<u32>]) -> usize {
    let Some(first) = encoded.first() else {
        return 0;
    };
    let mut length = encoded
        .iter()
        .map(|ids| ids.len().saturating_sub(1))
        .min()
        .unwrap_or(0);
    for ids in &encoded[1..] {
        while ids[..length] != first[..length] {
            length -= 1;
        }
    }
    length
}

fn classifier_tokens(cases: &[Case], tokenizer: &[u8], context: u64) -> HostResult<Vec<Vec<u32>>> {
    let tokenizer = if cases.iter().any(|case| case.text.is_some()) {
        Some(Tokenizer::from_json_bytes(
            tokenizer,
            TokenizerLimits::default(),
        )?)
    } else {
        None
    };
    let encoded = cases
        .iter()
        .map(|case| match (&case.text, &case.token_ids) {
            (Some(text), None)
                if !text.is_empty() && case.questions.is_empty() && case.segments.is_none() =>
            {
                Ok(tokenizer
                    .as_ref()
                    .ok_or("text case requires a tokenizer")?
                    .encode(
                        text,
                        EncodeOptions {
                            add_special_tokens: false,
                        },
                    )?)
            }
            (None, Some(ids)) if case.questions.is_empty() && case.segments.is_none() => {
                Ok(ids.clone())
            }
            _ => Err("classifier case requires exactly text or token_ids".into()),
        })
        .collect::<HostResult<Vec<_>>>()?;
    if encoded
        .iter()
        .any(|ids| ids.is_empty() || u64::try_from(ids.len()).map_or(true, |count| count > context))
    {
        return Err("classifier tokens must be nonempty and fit context".into());
    }
    Ok(encoded)
}

pub(crate) struct Classifier<B: HostBackend> {
    model: Lfm2Classifier<B>,
    encoded: Vec<Vec<u32>>,
    common: usize,
    base: Option<Lfm2Prefix<B>>,
    classes: u32,
}

impl<B: HostBackend> Classifier<B> {
    pub(crate) fn prepare(
        mut backend: B,
        asset: ModelAsset,
        cases: &[Case],
        tokenizer: &[u8],
        request: &Request,
        limit: Instant,
    ) -> HostResult<Self> {
        let encoded = classifier_tokens(cases, tokenizer, request.context)?;
        let common = if request.mode == "cached" {
            common_prefix(&encoded)
        } else {
            0
        };
        if request.mode == "cached" && common == 0 {
            return Err("cached mode requires a nonempty common prefix and tails".into());
        }
        let classes = request.classes.ok_or("classes missing")?;
        let weights = loading::classifier(asset, &mut backend, classes, limit)?;
        let mut model = Lfm2Classifier::new_with_options(
            backend,
            weights,
            Lfm2ExecutionLimits {
                max_logical_tokens: request.context,
            },
            Lfm2ExecutionOptions {
                lut2_mode: lut2(&request.lut2_mode)?,
                max_lut2_bytes: request.max_lut2_bytes,
            },
        )?;
        check(limit)?;
        let base = if common > 0 {
            Some(wait(
                &mut model.prefill_base(TokenChunk::all(&encoded[0][..common]))?,
                limit,
            )?)
        } else {
            None
        };
        Ok(Self {
            model,
            encoded,
            common,
            base,
            classes,
        })
    }
}

impl<B: HostBackend> PreparedModel for Classifier<B> {
    fn predict(&mut self, index: usize, seconds: f64) -> HostResult<Prediction> {
        let limit = deadline(seconds)?;
        let ids = self
            .encoded
            .get(index)
            .ok_or("classifier case index exceeds prepared inputs")?;
        let output = if let Some(base) = &self.base {
            wait(
                &mut self
                    .model
                    .classify_tail(base, TokenChunk::all(&ids[self.common..]))?,
                limit,
            )?
        } else {
            wait(&mut self.model.classify(TokenChunk::all(ids))?, limit)?
        };
        if output.len() != usize::try_from(self.classes)? {
            return Err("classifier output width differs from classes".into());
        }
        let decision = prediction::argmax(&output)?;
        check(limit)?;
        Ok((
            output.into_iter().map(f64::from).collect(),
            serde_json::json!(decision),
        ))
    }
    fn snapshot(&self) -> HostResult<Snapshot> {
        self.model.inspect_backend(measurement::snapshot)?
    }
}

pub(crate) struct Pointer<B: HostBackend> {
    model: Lfm2PointerEncoder<B>,
    shared: Rc<RefCell<B>>,
    inputs: Vec<EncoderInput>,
}
impl<B: HostBackend> Pointer<B> {
    pub(crate) fn prepare(
        mut backend: B,
        asset: ModelAsset,
        cases: &[Case],
        request: &Request,
        limit: Instant,
    ) -> HostResult<Self> {
        let inputs = cases
            .iter()
            .map(Case::pointer_input)
            .collect::<HostResult<Vec<_>>>()?;
        let weights = loading::pointer(asset, &mut backend, limit)?;
        let shared = Rc::new(RefCell::new(backend));
        let model = Lfm2PointerEncoder::new(
            Rc::clone(&shared),
            Rc::new(weights),
            EncoderLimits {
                max_tokens: request.context,
                max_questions: 64,
            },
        )?;
        // Validate every frozen case without recording work, before any timed cycle.
        for input in &inputs {
            drop(model.begin_predict(input.clone())?);
        }
        check(limit)?;
        Ok(Self {
            model,
            shared,
            inputs,
        })
    }
}
impl<B: HostBackend> PreparedModel for Pointer<B> {
    fn predict(&mut self, index: usize, seconds: f64) -> HostResult<Prediction> {
        let limit = deadline(seconds)?;
        let input = self
            .inputs
            .get(index)
            .ok_or("pointer case index exceeds prepared inputs")?;
        let output =
            prediction::pointer(wait(&mut self.model.begin_predict(input.clone())?, limit)?)?;
        check(limit)?;
        Ok(output)
    }
    fn snapshot(&self) -> HostResult<Snapshot> {
        measurement::snapshot(
            &*self
                .shared
                .try_borrow()
                .map_err(|_| "pointer backend is busy")?,
        )
    }
}
