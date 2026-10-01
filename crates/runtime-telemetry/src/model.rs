use minifield_executor_core::{Lfm2WeightFormat, Lfm2WeightPlan, StorageDType};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

/// Artifact identity and admitted storage formats. No paths or model content.
#[derive(Clone, Debug, Serialize)]
pub struct Model {
    pub id: String,
    pub name: Option<String>,
    pub revision: String,
    pub bundle_sha256: String,
    pub architecture: &'static str,
    pub parameter_count: Option<u64>,
    pub weight_formats: Vec<&'static str>,
}

impl Model {
    /// The bundle identity hashes a domain separator then config, weights, and tokenizer SHA256s.
    #[must_use]
    pub fn from_plan(
        plan: &Lfm2WeightPlan,
        quantization: &HashMap<String, Lfm2WeightFormat>,
        hashes: [[u8; 32]; 3],
    ) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"minifield.runtime-bundle/1\0");
        for hash in hashes {
            digest.update(hash);
        }
        let hash = format!("{:x}", digest.finalize());
        let mut formats = BTreeSet::new();
        for item in &plan.generic_plan().requirements {
            match item.storage_dtype {
                StorageDType::F32 => {
                    formats.insert("fp32");
                }
                StorageDType::BF16 => {
                    formats.insert("bf16");
                }
                StorageDType::F16 => {
                    formats.insert("fp16");
                }
                StorageDType::U8 => {
                    let role = item
                        .tensor_name
                        .strip_suffix(".codes")
                        .unwrap_or(&item.tensor_name);
                    let format = quantization.get(role).copied().unwrap_or(plan.format());
                    formats.insert(match format {
                        Lfm2WeightFormat::TernaryV1 => "ternary",
                        Lfm2WeightFormat::Nf4V1 => "nf4",
                        Lfm2WeightFormat::Int8V1 => "int8",
                        _ => "other",
                    });
                }
            }
        }
        // Count logical parameters using the equivalent dense plan, excluding tied aliases.
        let dense = match plan.classes() {
            Some(classes) => Lfm2WeightPlan::from_config_classifier(plan.config().clone(), classes),
            None => Lfm2WeightPlan::from_config(plan.config().clone()),
        };
        Self {
            id: format!("sha256:{hash}"),
            name: None,
            revision: hash.clone(),
            bundle_sha256: hash,
            architecture: "lfm2",
            parameter_count: dense.and_then(|p| p.physical_parameter_count()).ok(),
            weight_formats: formats.into_iter().collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use minifield_executor_core::parse_lfm2_config;

    #[test]
    fn mixed_formats_and_logical_parameters_come_from_the_admitted_inventory() {
        let config = parse_lfm2_config(br#"{"model_type":"lfm2","hidden_size":128,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"conv_L_cache":3,"vocab_size":32,"num_hidden_layers":2,"layer_types":["conv","full_attention"],"rope_theta":1000000.0,"tie_word_embeddings":true,"norm_eps":0.00001,"block_auto_adjust_ff_dim":false,"conv_bias":false,"dtype":"bfloat16"}"#).expect("config");
        let quantization = HashMap::from([
            (
                "model.layers.0.feed_forward.w1.weight".into(),
                Lfm2WeightFormat::TernaryV1,
            ),
            (
                "model.layers.0.feed_forward.w2.weight".into(),
                Lfm2WeightFormat::Nf4V1,
            ),
            (
                "model.layers.0.feed_forward.w3.weight".into(),
                Lfm2WeightFormat::Int8V1,
            ),
        ]);
        let packed = Lfm2WeightPlan::from_config_with_quantization(
            config.clone(),
            Lfm2WeightFormat::MixedV1,
            &quantization,
        )
        .expect("mixed plan");
        let dense = Lfm2WeightPlan::from_config(config).expect("dense plan");
        let model = Model::from_plan(&packed, &quantization, [[1; 32], [2; 32], [3; 32]]);
        assert_eq!(
            model.weight_formats,
            vec!["bf16", "fp16", "int8", "nf4", "ternary"]
        );
        assert_eq!(
            model.parameter_count,
            Some(dense.physical_parameter_count().expect("parameters"))
        );
        let changed = Model::from_plan(&packed, &quantization, [[1; 32], [2; 32], [4; 32]]);
        assert_ne!(model.bundle_sha256, changed.bundle_sha256);
        assert_eq!(model.id, format!("sha256:{}", model.bundle_sha256));
    }
}
