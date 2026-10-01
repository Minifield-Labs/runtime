//! Shared model configuration and state contracts for the custom Rust executor.
//!
//! This crate has no tensor engine, filesystem, device SDK, tokenizer, or model importer.
//! Format adapters hand it immutable JSON header bytes; model execution and cache ownership
//! are introduced only after the `CR02a` primitives have independent numerical coverage.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

pub mod lfm2;
pub mod loader;
pub use lfm2::{
    AppendChoiceTask, ChoiceLogitsTask, EncoderConfig, EncoderInput, EncoderLimits,
    EncoderLoadRequest, EncoderTypedWeights, EncoderWeightLoadTask, EncoderWeightPlan, LayerKind,
    Lfm2Classifier, Lfm2Config, Lfm2ExecutionLimits, Lfm2ExecutionOptions, Lfm2Executor,
    Lfm2LayerWeightRole, Lfm2LoadRequest, Lfm2Lut2Mode, Lfm2PointerEncoder, Lfm2Prefix,
    Lfm2ResolvedWeight, Lfm2StorageDType, Lfm2TypedWeights, Lfm2WeightFormat, Lfm2WeightLoadTask,
    Lfm2WeightPlan, Lfm2WeightRole, LogitsTask, NumericalMode, PointerAnswer, PointerOutput,
    PointerQuestion, PointerQuestionKind, PointerTask, PointerWeightRole, PrefillChoiceTask,
    PrefixTask, ScoreTask, decode_pointer, detect_lfm2_weight_format, parse_encoder_config,
    parse_lfm2_config, parse_lfm2_tensor_quantization,
};
pub use lfm2::{FLOPS_ESTIMATOR_VERSION, InferenceWork};
pub use loader::{
    LoadRequest, LoaderError, LoaderLimits, LoaderPoll, LoaderResourceReport, LoaderStage,
    ParsedAsset, ParsedTensor, StorageDType, TypedWeights, WeightLayout, WeightLoadTask,
    WeightPlan, WeightRequirement, parse_safetensors_header,
};
