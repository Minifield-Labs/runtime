use core::fmt;

/// Errors produced while loading or running a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A tensor or input had the wrong number of elements.
    DimensionMismatch {
        /// Name of the offending value.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// A model configuration value was invalid.
    InvalidConfig(&'static str),
    /// A binary model was truncated.
    Truncated,
    /// A binary model used an unsupported magic value or version.
    UnsupportedFormat,
    /// A binary model section did not match its declared shape.
    InvalidModel(&'static str),
    /// An arithmetic overflow was detected while validating a shape.
    SizeOverflow,
    /// A token ID was outside the model vocabulary.
    TokenOutOfRange(u32),
    /// Every token was masked, so mean pooling could not produce an embedding.
    EmptyInput,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                name,
                expected,
                actual,
            } => write!(f, "{name} has {actual} elements; expected {expected}"),
            Self::InvalidConfig(message) => write!(f, "invalid model configuration: {message}"),
            Self::Truncated => f.write_str("model data is truncated"),
            Self::UnsupportedFormat => f.write_str("unsupported model format"),
            Self::InvalidModel(message) => write!(f, "invalid model: {message}"),
            Self::SizeOverflow => f.write_str("tensor shape overflows addressable memory"),
            Self::TokenOutOfRange(token) => write!(f, "token ID {token} is outside the vocabulary"),
            Self::EmptyInput => f.write_str("input contains no unmasked tokens"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;
