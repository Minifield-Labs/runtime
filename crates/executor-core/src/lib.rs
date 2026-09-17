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
    LayerKind, Lfm2Config, Lfm2LayerWeightRole, Lfm2LoadRequest, Lfm2StorageDType,
    Lfm2TypedWeights, Lfm2WeightLoadTask, Lfm2WeightPlan, Lfm2WeightRole, NumericalMode,
    parse_lfm2_config,
};
pub use loader::{
    LoadRequest, LoaderError, LoaderLimits, LoaderPoll, LoaderResourceReport, LoaderStage,
    ParsedAsset, ParsedTensor, StorageDType, TypedWeights, WeightLayout, WeightLoadTask,
    WeightPlan, WeightRequirement, parse_safetensors_header,
};
