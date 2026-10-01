pub mod config;
pub mod encoder;
pub mod executor;
pub mod weights;
pub mod work;
pub use work::{FLOPS_ESTIMATOR_VERSION, InferenceWork};

pub use config::{LayerKind, Lfm2Config, Lfm2StorageDType, NumericalMode, parse_lfm2_config};
pub use encoder::{
    EncoderConfig, EncoderInput, EncoderLimits, EncoderLoadRequest, EncoderTypedWeights,
    EncoderWeightLoadTask, EncoderWeightPlan, Lfm2PointerEncoder, PointerAnswer, PointerOutput,
    PointerQuestion, PointerQuestionKind, PointerTask, PointerWeightRole, decode_pointer,
    parse_encoder_config,
};
pub use executor::{
    AppendChoiceTask, ChoiceLogitsTask, Lfm2Classifier, Lfm2ExecutionLimits, Lfm2ExecutionOptions,
    Lfm2Executor, Lfm2Lut2Mode, Lfm2Prefix, LogitsTask, PrefillChoiceTask, PrefixTask, ScoreTask,
};
pub use weights::{
    Lfm2LayerWeightRole, Lfm2LoadRequest, Lfm2ResolvedWeight, Lfm2TypedWeights, Lfm2WeightFormat,
    Lfm2WeightLoadTask, Lfm2WeightPlan, Lfm2WeightRole, detect_lfm2_weight_format,
    parse_lfm2_tensor_quantization,
};
