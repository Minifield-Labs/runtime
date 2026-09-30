//! Versioned host input. Explicit settings keep process-global state out of the score.
use crate::HostResult;
use minifield_engine_api::EncoderSegments;
use minifield_executor_core::{EncoderInput, PointerQuestion, PointerQuestionKind};
use serde::Deserialize;
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub backend: String,
    pub arithmetic: String,
    pub task: String,
    pub bundle: PathBuf,
    pub tokenizer: PathBuf,
    pub inputs: PathBuf,
    pub classes: Option<u32>,
    pub context: u64,
    pub mode: String,
    pub warmups: usize,
    pub measured_cycles: usize,
    pub phase: String,
    pub deadline_seconds: f64,
    pub lut2_mode: String,
    pub max_lut2_bytes: u64,
}
impl Request {
    pub fn validate(&self) -> HostResult<()> {
        if self.schema_version != 1
            || self.arithmetic != "f32"
            || !matches!(self.task.as_str(), "classifier" | "pointer")
            || !matches!(
                self.backend.as_str(),
                "cpu_reference" | "wgpu_metal" | "native_metal"
            )
            || !matches!(self.mode.as_str(), "full" | "cached")
            || !matches!(self.phase.as_str(), "correctness" | "measure")
        {
            return Err("unsupported evaluation request contract".into());
        }
        if self.context == 0
            || self.measured_cycles == 0
            || !self.deadline_seconds.is_finite()
            || self.deadline_seconds <= 0.0
        {
            return Err("evaluation bounds must be finite and positive".into());
        }
        if self.phase == "correctness" && self.measured_cycles != 1 {
            return Err("correctness phase requires one completed prediction per case".into());
        }
        if self.task == "pointer" && self.mode != "full" {
            return Err("bidirectional pointer evaluation requires full mode".into());
        }
        if !matches!(self.lut2_mode.as_str(), "raw" | "off" | "down" | "auto") {
            return Err("unknown LUT2 policy".into());
        }
        if self.task == "pointer" && self.classes.is_some() {
            return Err("pointer task has no classification class count".into());
        }
        if self.task == "classifier" && !matches!(self.classes, Some(1..=65_536)) {
            return Err("classifier requires explicit classes".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub schema_version: u32,
    pub task: String,
    pub cases: Vec<Case>,
}
impl Inputs {
    pub fn validate(&self, request: &Request) -> HostResult<()> {
        if self.schema_version != 1 || self.task != request.task || self.cases.is_empty() {
            return Err("input task/schema must match and contain cases".into());
        }
        let mut ids = BTreeSet::new();
        if self
            .cases
            .iter()
            .any(|case| case.id.is_empty() || !ids.insert(&case.id))
        {
            return Err("input IDs must be nonempty and distinct".into());
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub text: Option<String>,
    pub token_ids: Option<Vec<u32>>,
    pub segments: Option<Vec<u32>>,
    #[serde(default)]
    pub questions: Vec<Question>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub query_index: u32,
    pub option_indices: Vec<u32>,
    pub kind: QuestionKind,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuestionKind {
    Choice,
    Ordinal,
    Binary {
        positive_option: usize,
    },
    Extract {
        absent_index: u32,
        source_start: u32,
        selectable: Vec<bool>,
        presence_threshold: f64,
    },
}
impl Case {
    pub fn pointer_input(&self) -> HostResult<EncoderInput> {
        let token_ids = self
            .token_ids
            .clone()
            .ok_or("pointer case requires frozen token IDs")?;
        if self.text.is_some() {
            return Err("pointer text must be tokenized before freezing".into());
        }
        let segments = if let Some(ids) = &self.segments {
            EncoderSegments::new(ids.clone())?
        } else {
            EncoderSegments::single(token_ids.len())?
        };
        let questions = self
            .questions
            .iter()
            .map(|q| PointerQuestion {
                query_index: q.query_index,
                option_indices: q.option_indices.clone(),
                kind: match &q.kind {
                    QuestionKind::Choice => PointerQuestionKind::Choice,
                    QuestionKind::Ordinal => PointerQuestionKind::Ordinal,
                    QuestionKind::Binary { positive_option } => PointerQuestionKind::Binary {
                        positive_option: *positive_option,
                    },
                    QuestionKind::Extract {
                        absent_index,
                        source_start,
                        selectable,
                        presence_threshold,
                    } => PointerQuestionKind::Extract {
                        absent_index: *absent_index,
                        source_start: *source_start,
                        selectable: selectable.clone(),
                        presence_threshold: *presence_threshold,
                    },
                },
            })
            .collect();
        Ok(EncoderInput {
            token_ids,
            segments,
            questions,
        })
    }
}
