//! Content-free inference measurement. Hosts own configuration and delivery.
#![forbid(unsafe_code)]

use minifield_executor_core::{FLOPS_ESTIMATOR_VERSION, InferenceWork};
use serde::Serialize;
use serde_json::{Value, json};
use web_time::Instant;

mod model;
pub use model::Model;

pub const ENDPOINT: &str = "https://telemetry.minifieldlabs.com/insert/jsonline";
pub const SCHEMA_VERSION: &str = "minifield.runtime-inference/1";

/// One prediction or one complete autoregressive sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    SingleStep,
    Autoregressive,
}

/// Host measurements contain counts and timings only. Never pass inference content here.
pub struct Measurement {
    started: Instant,
    before: InferenceWork,
    prefill_end: Option<(Instant, InferenceWork)>,
    mode: Mode,
    pub input_tokens: Option<usize>,
    pub output_tokens: usize,
    pub tokenization_ms: Option<f64>,
    pub time_to_first_token_ms: Option<f64>,
    pub cache_reused: u64,
    pub cache_rebuilds: u64,
    pub fallback_used: bool,
    pub alternatives: Option<usize>,
    pub max_output_tokens: usize,
    pub constraint: &'static str,
    pub stop_reason: &'static str,
    pub error_code: &'static str,
}

impl Measurement {
    #[must_use]
    pub fn new(mode: Mode, before: InferenceWork) -> Self {
        Self {
            started: Instant::now(),
            before,
            prefill_end: None,
            mode,
            input_tokens: None,
            output_tokens: 0,
            tokenization_ms: None,
            time_to_first_token_ms: None,
            cache_reused: 0,
            cache_rebuilds: 0,
            fallback_used: false,
            alternatives: None,
            max_output_tokens: 0,
            constraint: "none",
            stop_reason: "output_limit",
            error_code: "invalid_input",
        }
    }

    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    pub fn tokenized(&mut self, input_tokens: usize) {
        self.input_tokens = Some(input_tokens);
        self.tokenization_ms = Some(self.elapsed_ms());
        self.error_code = "execution_failed";
    }

    pub fn prefilled(&mut self, work: InferenceWork) {
        self.prefill_end = Some((Instant::now(), work));
    }

    pub fn emitted(&mut self) {
        if self.output_tokens == 0 {
            self.time_to_first_token_ms = Some(self.elapsed_ms());
        }
        self.output_tokens += 1;
    }

    /// Freeze the terminal record before scheduling any network work.
    #[must_use]
    pub fn finish(
        self,
        model: &Model,
        after: InferenceWork,
        succeeded: bool,
        backend: &str,
    ) -> Value {
        let ended = Instant::now();
        let elapsed = (ended - self.started).as_secs_f64() * 1000.0;
        let total = after.since(self.before);
        let prefill = self
            .prefill_end
            .map_or(total, |(_, end)| end.since(self.before));
        let prefill_ms = self.prefill_end.map_or(elapsed, |(end, _)| {
            (end - self.started).as_secs_f64() * 1000.0
        }) - self.tokenization_ms.unwrap_or(0.0);
        let decode = (self.mode == Mode::Autoregressive).then(|| {
            self.prefill_end.map_or_else(
                || {
                    if succeeded && total.forward_passes == 0 {
                        phase(Some(0.0), Some(InferenceWork::default()))
                    } else {
                        phase(None, None)
                    }
                },
                |(end, work)| {
                    phase(
                        Some((ended - end).as_secs_f64() * 1000.0),
                        Some(after.since(work)),
                    )
                },
            )
        });
        let commit = env!("MINIFIELD_GIT_COMMIT");
        json!({
            "schema_version": SCHEMA_VERSION,
            "inference_id": inference_id(uuid::Uuid::now_v7()),
            "mode": if self.mode == Mode::SingleStep { "single_step" } else { "autoregressive" },
            "status": if succeeded { "completed" } else { "failed" },
            "error_code": (!succeeded).then_some(self.error_code),
            "runtime": {
                "version": env!("CARGO_PKG_VERSION"), "build_id": env!("MINIFIELD_BUILD_ID"),
                "git_commit": (!commit.is_empty()).then_some(commit),
                "target": if cfg!(target_arch = "wasm32") { "wasm" } else { "native" }
            },
            "model": model,
            "execution": {
                "backend": backend, "compute_precisions": ["fp32"], "elapsed_ms": elapsed,
                "tokenization_ms": self.tokenization_ms, "time_to_first_token_ms": self.time_to_first_token_ms,
                "tokens": { "input": self.input_tokens, "output": self.output_tokens },
                "cache": { "token_positions_reused": self.cache_reused, "rebuilds": self.cache_rebuilds },
                "prefill": phase(Some(prefill_ms.max(0.0)), Some(prefill)), "decode": decode,
                "estimated_flops": total.estimated_flops.to_string(),
                "flops_estimator_version": FLOPS_ESTIMATOR_VERSION,
                "flops_estimate_coverage": if succeeded { "complete" } else { "partial" },
                "fallback_used": self.fallback_used
            },
            "single_step": (self.mode == Mode::SingleStep).then(|| json!({
                "predictions_produced": usize::from(succeeded), "alternatives_evaluated": self.alternatives
            })),
            "autoregressive": (self.mode == Mode::Autoregressive).then(|| json!({
                "decoding_mode": "greedy", "constraint": self.constraint,
                "max_output_tokens": self.max_output_tokens,
                "stop_reason": if succeeded { self.stop_reason } else { "error" }
            }))
        })
    }
}

fn phase(elapsed_ms: Option<f64>, work: Option<InferenceWork>) -> Value {
    json!({ "elapsed_ms": elapsed_ms,
        "forward_passes": work.map(|w| w.forward_passes),
        "token_positions_processed": work.map(|w| w.token_positions_processed),
        "estimated_flops": work.map(|w| w.estimated_flops.to_string()) })
}

fn inference_id(uuid: uuid::Uuid) -> String {
    const ALPHABET: &[u8; 32] = b"0123456789abcdefghjkmnpqrstvwxyz";
    let mut suffix = [b'0'; 26];
    let mut value = uuid.as_u128();
    for ch in suffix.iter_mut().rev() {
        *ch = ALPHABET[(value & 31) as usize];
        value >>= 5;
    }
    let mut id = String::from("inf_");
    id.extend(suffix.map(char::from));
    id
}

/// Coarse device description. Unknown architecture stays absent; backend names aren't ISAs.
#[derive(Clone, Debug, Serialize)]
pub struct Hardware {
    pub kind: &'static str,
    pub vendor: Option<&'static str>,
    pub architecture: Option<&'static str>,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn typeid_matches_jetify_reference_and_encodes_uuid7() {
        let reference =
            uuid::Uuid::parse_str("0188bac7-4afa-78aa-bc3b-bd1eef28d881").expect("uuid");
        assert_eq!(inference_id(reference), "inf_01h2xcejqtf2nbrexx3vqjhp41");
        let a = uuid::Uuid::now_v7();
        assert_eq!(a.get_version_num(), 7);
        let id = inference_id(a);
        assert_eq!(id.len(), 30);
        assert_ne!(id, inference_id(uuid::Uuid::now_v7()));
    }

    #[test]
    fn one_sequence_record_keeps_counts_and_phase_work_separate() {
        let model = Model {
            id: "synthetic".into(),
            name: None,
            revision: "test".into(),
            bundle_sha256: "a".repeat(64),
            architecture: "lfm2",
            parameter_count: Some(100),
            weight_formats: vec!["bf16", "nf4"],
        };
        let before = InferenceWork {
            forward_passes: 10,
            token_positions_processed: 100,
            estimated_flops: 1000,
        };
        let prefill = InferenceWork {
            forward_passes: 12,
            token_positions_processed: 108,
            estimated_flops: 1200,
        };
        let after = InferenceWork {
            forward_passes: 14,
            token_positions_processed: 110,
            estimated_flops: 1500,
        };
        let mut measured = Measurement::new(Mode::Autoregressive, before);
        measured.tokenized(12);
        measured.prefilled(prefill);
        measured.cache_reused = 4;
        measured.fallback_used = true;
        for _ in 0..3 {
            measured.emitted();
        }
        let value = measured.finish(&model, after, true, "cpu");
        assert_eq!(value["execution"]["estimated_flops"], "500");
        assert_eq!(value["execution"]["prefill"]["forward_passes"], 2);
        assert_eq!(value["execution"]["decode"]["token_positions_processed"], 2);
        assert_eq!(value["execution"]["tokens"]["output"], 3);
        assert_eq!(value["execution"]["tokens"]["input"], 12);
        assert_eq!(value["execution"]["cache"]["token_positions_reused"], 4);
        assert!(value["single_step"].is_null());
        assert!(value["execution"]["time_to_first_token_ms"].is_number());
        assert!(value.get("start").is_none() && value.get("end").is_none());
        let failed =
            Measurement::new(Mode::SingleStep, before).finish(&model, before, false, "cpu");
        assert_eq!(failed["single_step"]["predictions_produced"], 0);
        assert_eq!(failed["execution"]["flops_estimate_coverage"], "partial");
        assert_eq!(failed["status"], "failed");
    }
}
