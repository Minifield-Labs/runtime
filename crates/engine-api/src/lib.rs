//! Portable, backend-neutral contracts for the Minifield inference executor.
//!
//! This crate owns descriptors, bounded asset access, lifecycle/completion semantics,
//! and token-level interfaces. It deliberately has no tensor engine, model equation,
//! filesystem, thread, network, or GPU dependency.

#![forbid(unsafe_code)]
// CR01 uses compact checked descriptors whose Result errors are documented by the public
// ExecutorError taxonomy. Model-specific APIs add operation-level error details later.
#![allow(clippy::missing_errors_doc)]

mod assets;
mod backend;
mod error;
mod inference;
mod resources;
mod tensor;
mod tokens;

pub use assets::{
    AssetBytes, AssetLimits, AssetManifest, AssetProvider, MAX_TENSOR_REQUIREMENTS,
    MemoryAssetProvider, MemoryAssetRead, TensorBinding, TensorRecord, TensorRequirement,
    validate_asset_manifest,
};
pub use backend::{
    BackendCapabilities, BackendIdentity, BackendKind, BackendLease, GatedShortConvSpec, GqaSpec,
    OperationKind, OperationSet, PackedHeadSpec, PrecisionPolicy, RectCopy2d, RotarySpec,
};
pub use error::{ExecutorError, Result};
pub use inference::{InferenceOps, TokenIds};
pub use resources::{
    AllocationClass, CompletionPoll, FenceRetirement, InferenceCompletion, ReadyCompletion,
    ResourceLimits, ResourceReport, RetirementRejection,
};
pub use tensor::{
    BufferAccess, BufferDescriptor, ByteRange, DType, DTypeSet, MAX_RANK, Shape, TensorLayout,
};
pub use tokens::{
    CandidateScore, DecodeConstraint, TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenId,
};
