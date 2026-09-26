//! Scalar quantized matrix oracles. No decoder or GPU backend is included.
mod error;
pub mod q4;
pub mod q8;
pub use error::{Error, Result};
