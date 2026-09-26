//! Owned scalar Rust CPU operations for the portable inference executor.
//!
//! This crate implements a finite inference operation surface. It is not a
//! tensor expression engine and contains no GPU, filesystem, network, or thread dependency.

#![forbid(unsafe_code)]
// The shared ExecutorError taxonomy documents CR01 operation failures; model-specific APIs
// add narrower error details as their equation and loader layers are introduced.
#![allow(clippy::missing_errors_doc)]

mod attention_convolution;
mod completion;
mod dense;
mod dispatch;
mod normalization_rotary;
mod packed;
mod storage;

pub use completion::{CpuCompletion, CpuFenceRetirement};
pub use storage::{CpuBackend, CpuBuffer};

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
