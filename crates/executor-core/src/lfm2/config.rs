//! Strict supported LFM2 configuration parsing.
//!
//! The parser resolves published aliases before any model allocation. Unknown metadata remains
//! format-owned, but every accepted field that changes the CR02 math is validated or rejected.

use minifield_engine_api::{ExecutorError, Result};
use serde_json::{Map, Value};

/// The two layer operators supported by the initial LFM2 math contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Conv,
    FullAttention,
}

/// The scalar rounding sequence model code must use for LFM2 RMS normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericalMode {
    /// f32 sum, f32 mean, reciprocal-square-root, then multiply input and gamma.
    F32ReciprocalSqrtMultiply,
}

/// Checked storage dtype expected for every physical LFM2 checkpoint tensor in this loader slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lfm2StorageDType {
    F32,
    BF16,
    F16,
}

/// Checked LFM2 configuration normalized across published alias spellings.
#[derive(Clone, Debug, PartialEq)]
pub struct Lfm2Config {
    pub hidden_size: u32,
    pub raw_intermediate_size: u32,
    pub effective_intermediate_size: u32,
    pub attention_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub conv_width: u32,
    pub vocab_size: u32,
    pub layers: Vec<LayerKind>,
    pub rope_theta: f32,
    pub tie_embedding: bool,
    pub norm_epsilon: f32,
    pub block_norm_epsilon: f32,
    pub max_position_embeddings: Option<u64>,
    pub numerical_mode: NumericalMode,
    /// Source storage is expanded to the executor's F32 compute buffers by the loader.
    pub weight_storage_dtype: Lfm2StorageDType,
}

impl Lfm2Config {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.raw_intermediate_size == 0
            || self.effective_intermediate_size == 0
            || self.attention_heads == 0
            || self.key_value_heads == 0
            || self.conv_width == 0
            || self.vocab_size == 0
            || self.layers.is_empty()
        {
            return Err(ExecutorError::InvalidArgument(
                "LFM2 dimensions and layer list must be nonzero",
            ));
        }
        if self.hidden_size % self.attention_heads != 0
            || self.head_dim != self.hidden_size / self.attention_heads
        {
            return Err(ExecutorError::InvalidShape(
                "hidden size must divide evenly into query attention heads",
            ));
        }
        if self.attention_heads % self.key_value_heads != 0 {
            return Err(ExecutorError::InvalidShape(
                "query attention heads must be a multiple of key/value heads",
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(ExecutorError::InvalidShape(
                "LFM2 split-half RoPE requires an even head dimension",
            ));
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.block_norm_epsilon.is_finite()
            || self.block_norm_epsilon <= 0.0
        {
            return Err(ExecutorError::InvalidArgument(
                "LFM2 theta and normalization epsilons must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Parse a JSON header into the strictly supported LFM2 numerical configuration.
pub fn parse_lfm2_config(bytes: &[u8]) -> Result<Lfm2Config> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| ExecutorError::InvalidArgument("LFM2 config is not valid JSON"))?;
    let object = object(&value)?;
    if required_text(object, "model_type")? != "lfm2" {
        return Err(ExecutorError::Unsupported("config model_type is not lfm2"));
    }
    reject_rope_scaling(object)?;
    reject_false_bool(
        object,
        "block_use_swiglu",
        "only SwiGLU LFM2 blocks are supported",
    )?;
    reject_true_bool(object, "conv_bias", "convolution bias is unsupported")?;

    let hidden_size = required_u32(object, "hidden_size")?;
    require_matching_u32_alias(object, "hidden_size", "block_dim", hidden_size)?;
    require_matching_u32_alias(object, "hidden_size", "conv_dim", hidden_size)?;
    let raw_intermediate_size = required_u32(object, "intermediate_size")?;
    require_matching_u32_alias(
        object,
        "intermediate_size",
        "block_ff_dim",
        raw_intermediate_size,
    )?;
    let auto_adjust = optional_bool(object, "block_auto_adjust_ff_dim")?.unwrap_or(false);
    let effective_intermediate_size = effective_ff_dim(object, raw_intermediate_size, auto_adjust)?;

    let attention_heads = resolve_u32_alias(object, "num_attention_heads", "num_heads")?.ok_or(
        ExecutorError::InvalidArgument("attention head count is missing"),
    )?;
    let key_value_heads = required_u32(object, "num_key_value_heads")?;
    let conv_width = required_u32(object, "conv_L_cache")?;
    let vocab_size = required_u32(object, "vocab_size")?;
    let num_hidden_layers = required_u32(object, "num_hidden_layers")?;
    let layers = parse_layers(object, num_hidden_layers)?;
    let rope_theta = resolve_rope_theta(object)?;
    let tie_embedding = resolve_bool_alias(object, "tie_embedding", "tie_word_embeddings")?.ok_or(
        ExecutorError::InvalidArgument("embedding tie declaration is missing"),
    )?;
    if !tie_embedding {
        return Err(ExecutorError::Unsupported(
            "untied embedding and language-model head are unsupported",
        ));
    }
    let norm_epsilon = required_f32(object, "norm_eps")?;
    let block_norm_epsilon = optional_f32(object, "block_norm_eps")?.unwrap_or(norm_epsilon);
    let max_position_embeddings = optional_u64(object, "max_position_embeddings")?;
    let weight_storage_dtype = parse_weight_storage_dtype(object)?;

    let config = Lfm2Config {
        hidden_size,
        raw_intermediate_size,
        effective_intermediate_size,
        attention_heads,
        key_value_heads,
        head_dim: hidden_size / attention_heads.max(1),
        conv_width,
        vocab_size,
        layers,
        rope_theta,
        tie_embedding,
        norm_epsilon,
        block_norm_epsilon,
        max_position_embeddings,
        numerical_mode: NumericalMode::F32ReciprocalSqrtMultiply,
        weight_storage_dtype,
    };
    config.validate()?;
    Ok(config)
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(ExecutorError::InvalidArgument(
        "LFM2 config root must be a JSON object",
    ))
}

fn required_value<'a>(object: &'a Map<String, Value>, name: &'static str) -> Result<&'a Value> {
    object.get(name).ok_or(ExecutorError::InvalidArgument(name))
}

fn required_text<'a>(object: &'a Map<String, Value>, name: &'static str) -> Result<&'a str> {
    required_value(object, name)?
        .as_str()
        .ok_or(ExecutorError::InvalidArgument(name))
}

fn optional_bool(object: &Map<String, Value>, name: &'static str) -> Result<Option<bool>> {
    object.get(name).map_or(Ok(None), |value| {
        value
            .as_bool()
            .map(Some)
            .ok_or(ExecutorError::InvalidArgument(name))
    })
}

fn parse_weight_storage_dtype(object: &Map<String, Value>) -> Result<Lfm2StorageDType> {
    match object.get("dtype") {
        None => Ok(Lfm2StorageDType::F32),
        Some(Value::String(text)) => match text.as_str() {
            "float32" | "f32" => Ok(Lfm2StorageDType::F32),
            "bfloat16" | "bf16" => Ok(Lfm2StorageDType::BF16),
            "float16" | "f16" => Ok(Lfm2StorageDType::F16),
            _ => Err(ExecutorError::Unsupported(
                "configured LFM2 source storage dtype is unsupported",
            )),
        },
        Some(_) => Err(ExecutorError::InvalidArgument(
            "LFM2 config dtype must be a supported string",
        )),
    }
}

fn required_u32(object: &Map<String, Value>, name: &'static str) -> Result<u32> {
    let value = required_value(object, name)?;
    let parsed = value.as_u64().ok_or(ExecutorError::InvalidArgument(name))?;
    u32::try_from(parsed).map_err(|_| ExecutorError::Overflow(name))
}

fn optional_u64(object: &Map<String, Value>, name: &'static str) -> Result<Option<u64>> {
    object.get(name).map_or(Ok(None), |value| {
        value
            .as_u64()
            .map(Some)
            .ok_or(ExecutorError::InvalidArgument(name))
    })
}

fn optional_u32(object: &Map<String, Value>, name: &'static str) -> Result<Option<u32>> {
    object.get(name).map_or(Ok(None), |value| {
        value
            .as_u64()
            .ok_or(ExecutorError::InvalidArgument(name))
            .and_then(|raw| {
                u32::try_from(raw)
                    .map(Some)
                    .map_err(|_| ExecutorError::Overflow(name))
            })
    })
}

fn required_f32(object: &Map<String, Value>, name: &'static str) -> Result<f32> {
    optional_f32(object, name)?.ok_or(ExecutorError::InvalidArgument(name))
}

#[allow(clippy::cast_possible_truncation)]
fn optional_f32(object: &Map<String, Value>, name: &'static str) -> Result<Option<f32>> {
    object.get(name).map_or(Ok(None), |value| {
        let number = value.as_f64().ok_or(ExecutorError::InvalidArgument(name))?;
        let cast = number as f32;
        if !number.is_finite() || !cast.is_finite() {
            return Err(ExecutorError::InvalidArgument(name));
        }
        Ok(Some(cast))
    })
}

fn require_matching_u32_alias(
    object: &Map<String, Value>,
    primary: &'static str,
    alias: &'static str,
    expected: u32,
) -> Result<()> {
    if let Some(value) = optional_u32(object, alias)?
        && value != expected
    {
        return Err(ExecutorError::InvalidArgument(
            "conflicting published LFM2 dimension aliases",
        ));
    }
    let _ = primary;
    Ok(())
}

fn resolve_u32_alias(
    object: &Map<String, Value>,
    primary: &'static str,
    alias: &'static str,
) -> Result<Option<u32>> {
    let primary_value = optional_u32(object, primary)?;
    let alias_value = optional_u32(object, alias)?;
    if let (Some(primary_value), Some(alias_value)) = (primary_value, alias_value)
        && primary_value != alias_value
    {
        return Err(ExecutorError::InvalidArgument(
            "conflicting published LFM2 dimension aliases",
        ));
    }
    Ok(primary_value.or(alias_value))
}

fn resolve_bool_alias(
    object: &Map<String, Value>,
    primary: &'static str,
    alias: &'static str,
) -> Result<Option<bool>> {
    let primary_value = optional_bool(object, primary)?;
    let alias_value = optional_bool(object, alias)?;
    if let (Some(primary_value), Some(alias_value)) = (primary_value, alias_value)
        && primary_value != alias_value
    {
        return Err(ExecutorError::InvalidArgument(
            "conflicting published LFM2 boolean aliases",
        ));
    }
    Ok(primary_value.or(alias_value))
}

fn reject_false_bool(
    object: &Map<String, Value>,
    name: &'static str,
    message: &'static str,
) -> Result<()> {
    if optional_bool(object, name)? == Some(false) {
        return Err(ExecutorError::Unsupported(message));
    }
    Ok(())
}

fn reject_true_bool(
    object: &Map<String, Value>,
    name: &'static str,
    message: &'static str,
) -> Result<()> {
    if optional_bool(object, name)? == Some(true) {
        return Err(ExecutorError::Unsupported(message));
    }
    Ok(())
}

fn effective_ff_dim(object: &Map<String, Value>, raw: u32, auto_adjust: bool) -> Result<u32> {
    if !auto_adjust {
        return Ok(raw);
    }
    let multiplier = object
        .get("block_ffn_dim_multiplier")
        .ok_or(ExecutorError::InvalidArgument(
            "block_ffn_dim_multiplier is required when ff auto adjustment is enabled",
        ))?
        .as_f64()
        .ok_or(ExecutorError::InvalidArgument("block_ffn_dim_multiplier"))?;
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(ExecutorError::InvalidArgument(
            "block_ffn_dim_multiplier must be finite and positive",
        ));
    }
    let multiple_of = required_u32(object, "block_multiple_of")?;
    if multiple_of == 0 {
        return Err(ExecutorError::InvalidArgument(
            "block_multiple_of must be nonzero",
        ));
    }
    let adjusted = raw.checked_mul(2).ok_or(ExecutorError::Overflow(
        "FF dimension adjustment overflows u32",
    ))? / 3;
    let scaled_ff = f64::from(adjusted) * multiplier;
    if !scaled_ff.is_finite() || scaled_ff > f64::from(u32::MAX) {
        return Err(ExecutorError::Overflow(
            "FF dimension multiplier exceeds u32",
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let truncated = scaled_ff.trunc() as u64;
    let multiple = u64::from(multiple_of);
    let rounded = truncated
        .checked_add(
            multiple
                .checked_sub(1)
                .ok_or(ExecutorError::Overflow("FF dimension multiple underflows"))?,
        )
        .ok_or(ExecutorError::Overflow(
            "FF dimension rounding overflows u64",
        ))?
        / multiple
        * multiple;
    u32::try_from(rounded)
        .map_err(|_| ExecutorError::Overflow("effective FF dimension exceeds u32"))
}

fn parse_layers(object: &Map<String, Value>, count: u32) -> Result<Vec<LayerKind>> {
    let values = required_value(object, "layer_types")?
        .as_array()
        .ok_or(ExecutorError::InvalidArgument("layer_types"))?;
    if values.len()
        != usize::try_from(count)
            .map_err(|_| ExecutorError::Overflow("layer count exceeds usize"))?
    {
        return Err(ExecutorError::InvalidShape(
            "layer_types length differs from num_hidden_layers",
        ));
    }
    values
        .iter()
        .map(|value| match value.as_str() {
            Some("conv") => Ok(LayerKind::Conv),
            Some("full_attention") => Ok(LayerKind::FullAttention),
            _ => Err(ExecutorError::Unsupported("unsupported LFM2 layer type")),
        })
        .collect()
}

fn resolve_rope_theta(object: &Map<String, Value>) -> Result<f32> {
    let legacy = optional_f32(object, "rope_theta")?;
    let nested = match object.get("rope_parameters") {
        None => None,
        Some(value) => {
            let parameters = value.as_object().ok_or(ExecutorError::InvalidArgument(
                "rope_parameters must be an object",
            ))?;
            if let Some(kind) = parameters.get("rope_type")
                && kind.as_str() != Some("default")
            {
                return Err(ExecutorError::Unsupported("only default RoPE is supported"));
            }
            parameters
                .get("rope_theta")
                .ok_or(ExecutorError::InvalidArgument(
                    "rope_parameters.rope_theta is missing",
                ))
                .and_then(value_f32)?
                .into()
        }
    };
    if let (Some(nested), Some(legacy)) = (nested, legacy)
        && nested.to_bits() != legacy.to_bits()
    {
        return Err(ExecutorError::InvalidArgument(
            "conflicting published LFM2 RoPE theta aliases",
        ));
    }
    nested
        .or(legacy)
        .ok_or(ExecutorError::InvalidArgument("RoPE theta is missing"))
}

#[allow(clippy::cast_possible_truncation)]
fn value_f32(value: &Value) -> Result<f32> {
    let number = value
        .as_f64()
        .ok_or(ExecutorError::InvalidArgument("RoPE theta must be numeric"))?;
    let cast = number as f32;
    if !number.is_finite() || !cast.is_finite() {
        return Err(ExecutorError::InvalidArgument(
            "RoPE theta must be a finite f32",
        ));
    }
    Ok(cast)
}

fn reject_rope_scaling(object: &Map<String, Value>) -> Result<()> {
    if let Some(value) = object.get("rope_scaling")
        && !value.is_null()
    {
        return Err(ExecutorError::Unsupported("scaled RoPE is unsupported"));
    }
    if let Some(parameters) = object.get("rope_parameters").and_then(Value::as_object)
        && (parameters.contains_key("factor") || parameters.contains_key("rope_scaling"))
    {
        return Err(ExecutorError::Unsupported("scaled RoPE is unsupported"));
    }
    Ok(())
}
