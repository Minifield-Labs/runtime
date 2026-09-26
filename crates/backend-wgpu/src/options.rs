//! Explicit device policy. Hosts parse command-line or environment settings.

/// Workgroup staging precision for explicitly enabled research kernels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Nf4Staging {
    #[default]
    F32,
    F16Weights,
    F16Activations,
    F16Both,
}

impl Nf4Staging {
    pub(crate) const fn types(self) -> (&'static str, &'static str) {
        match self {
            Self::F32 => ("f32", "f32"),
            Self::F16Weights => ("f32", "f16"),
            Self::F16Activations => ("f16", "f32"),
            Self::F16Both => ("f16", "f16"),
        }
    }
}

/// Immutable policy used when the device and shader pipelines are created.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WgpuOptions {
    /// Collect host timing and print a diagnostic summary when the device drops.
    /// Dispatch counters remain available programmatically regardless of this flag.
    pub diagnostics: bool,
    /// F16 modes require the `experimental-kernels` feature and device support.
    pub nf4_staging: Nf4Staging,
}
