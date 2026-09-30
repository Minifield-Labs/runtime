//! Bidirectional LFM2 pointer encoding over complete, isolated segments.
//!
//! Equations follow training revision 96fd486d1601bb910b390652bb21d53463e8237a;
//! packed-segment isolation follows 736fe88. No training code is imported.

mod config;
mod execution;
mod prediction;
mod task;
mod weights;

pub use config::{EncoderConfig, parse_encoder_config};
pub use prediction::{
    EncoderInput, EncoderLimits, PointerAnswer, PointerOutput, PointerQuestion,
    PointerQuestionKind, decode_pointer,
};
pub use task::{Lfm2PointerEncoder, PointerTask};
pub use weights::{
    EncoderLoadRequest, EncoderTypedWeights, EncoderWeightLoadTask, EncoderWeightPlan,
    PointerWeightRole,
};

#[cfg(test)]
mod tests;
