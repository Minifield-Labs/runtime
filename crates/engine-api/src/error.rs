//! Typed failures shared by portable contracts.

use core::fmt;

/// Executor result type.
pub type Result<T> = std::result::Result<T, ExecutorError>;

/// Typed failures exposed by portable executor and backend APIs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorError {
    InvalidArgument(&'static str),
    InvalidShape(&'static str),
    InvalidLayout(&'static str),
    InvalidDType(&'static str),
    Overflow(&'static str),
    OutOfBounds(&'static str),
    Unsupported(&'static str),
    ResourceLimit(&'static str),
    WrongBackend,
    StaleBuffer,
    DuplicateName,
    MissingRequiredTensor,
    UnexpectedTensor,
    InvalidTie,
    Cancelled,
    CompletionConsumed,
    BackendFailure(&'static str),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidArgument(message)
            | Self::InvalidShape(message)
            | Self::InvalidLayout(message)
            | Self::InvalidDType(message)
            | Self::Overflow(message)
            | Self::OutOfBounds(message)
            | Self::Unsupported(message)
            | Self::ResourceLimit(message)
            | Self::BackendFailure(message) => message,
            Self::WrongBackend => "buffer belongs to another backend",
            Self::StaleBuffer => "buffer belongs to an earlier backend generation",
            Self::DuplicateName => "duplicate name",
            Self::MissingRequiredTensor => "required tensor is missing",
            Self::UnexpectedTensor => "undeclared tensor is present",
            Self::InvalidTie => "invalid tied tensor declaration",
            Self::Cancelled => "operation cancelled",
            Self::CompletionConsumed => "completion result was already consumed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ExecutorError {}
