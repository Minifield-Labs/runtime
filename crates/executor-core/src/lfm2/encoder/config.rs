use crate::lfm2::{Lfm2Config, parse_lfm2_config};
use minifield_engine_api::{ExecutorError, Result};
use serde_json::Value;

/// Backbone geometry and the four pointer projection widths.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderConfig {
    pub backbone: Lfm2Config,
    pub pointer_width: u32,
}

pub fn parse_encoder_config(bytes: &[u8]) -> Result<EncoderConfig> {
    let mut value: Value = serde_json::from_slice(bytes)
        .map_err(|_| ExecutorError::InvalidArgument("encoder config is not valid JSON"))?;
    let object = value.as_object_mut().ok_or(ExecutorError::InvalidArgument(
        "encoder config must be an object",
    ))?;
    if object.get("architectures") != Some(&serde_json::json!(["Lfm2BidirectionalForMaskedLM"])) {
        return Err(ExecutorError::Unsupported(
            "encoder requires the bidirectional LFM2 architecture",
        ));
    }
    if object.get("use_cache").and_then(Value::as_bool) != Some(false) {
        return Err(ExecutorError::Unsupported(
            "bidirectional encoder must disable causal cache",
        ));
    }
    let pointer = object
        .get("minifield_pointer")
        .and_then(Value::as_object)
        .ok_or(ExecutorError::InvalidArgument(
            "encoder pointer metadata is missing",
        ))?;
    if pointer.get("format").and_then(Value::as_str) != Some("minifield.magicbox-joint-pointer/1") {
        return Err(ExecutorError::Unsupported(
            "encoder pointer metadata format is unsupported",
        ));
    }
    let pointer_width = pointer
        .get("projection_dim")
        .and_then(Value::as_u64)
        .and_then(|width| u32::try_from(width).ok())
        .filter(|&width| width > 0 && width <= 4096)
        .ok_or(ExecutorError::InvalidArgument(
            "encoder pointer width must be in 1..=4096",
        ))?;
    match object.get("model_type").and_then(Value::as_str) {
        Some("lfm2" | "lfm2_bidirectional") => {}
        _ => {
            return Err(ExecutorError::Unsupported(
                "encoder model_type is unsupported",
            ));
        }
    }
    object.insert("model_type".to_owned(), Value::String("lfm2".to_owned()));
    let normalized = serde_json::to_vec(&value)
        .map_err(|_| ExecutorError::InvalidArgument("encoder config serialization failed"))?;
    let backbone = parse_lfm2_config(&normalized)?;
    Ok(EncoderConfig {
        backbone,
        pointer_width,
    })
}
