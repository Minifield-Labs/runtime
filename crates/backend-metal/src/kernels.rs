//! Finite native entry-point registry. Names are also raw dispatch-counter keys.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Kernel {
    Elementwise,
    RectCopy,
    Gather,
    Columns,
    ArgmaxRows,
    DenseLinear,
    PackedGather,
    PackedLinear,
    PackedPair,
    RmsNorm,
    Rotary,
    Attention,
    CausalConv,
    ConvHistory,
    CenteredConv,
    PackedLinearTile8,
    PackedPairTile8,
}

impl Kernel {
    #[cfg(any(target_os = "macos", test))]
    pub(crate) const ALL: [Self; 17] = [
        Self::Elementwise,
        Self::RectCopy,
        Self::Gather,
        Self::Columns,
        Self::ArgmaxRows,
        Self::DenseLinear,
        Self::PackedGather,
        Self::PackedLinear,
        Self::PackedPair,
        Self::RmsNorm,
        Self::Rotary,
        Self::Attention,
        Self::CausalConv,
        Self::ConvHistory,
        Self::CenteredConv,
        Self::PackedLinearTile8,
        Self::PackedPairTile8,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::RectCopy => "rect_copy",
            Self::Gather => "gather",
            Self::Columns => "columns",
            Self::ArgmaxRows => "argmax_rows",
            Self::DenseLinear => "dense_linear",
            Self::PackedGather => "packed_gather",
            Self::PackedLinear => "packed_linear",
            Self::PackedPair => "packed_pair",
            Self::RmsNorm => "rms_norm",
            Self::Rotary => "rotary",
            Self::Attention => "attention",
            Self::CausalConv => "causal_conv",
            Self::ConvHistory => "conv_history",
            Self::CenteredConv => "centered_conv",
            Self::PackedLinearTile8 => "packed_linear_tile8",
            Self::PackedPairTile8 => "packed_pair_tile8",
        }
    }

    pub(crate) const fn is_tile8(self) -> bool {
        matches!(self, Self::PackedLinearTile8 | Self::PackedPairTile8)
    }
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn library_source() -> String {
    // One library keeps the canonical NF4 table and SiLU expression shared by
    // the scalar and cooperative native bodies. Neither source imports WGPU.
    let baseline = include_str!("kernels.metal");
    let tiled = include_str!("shaders/packed_tile8.metal");
    let mut source = String::with_capacity(baseline.len() + tiled.len() + 1);
    source.push_str(baseline);
    source.push('\n');
    source.push_str(tiled);
    source
}

#[cfg(test)]
mod tests {
    use super::{Kernel, library_source};
    use std::collections::BTreeSet;

    #[test]
    fn every_registered_entry_point_has_one_source_definition_and_unique_name() {
        let source = library_source();
        let names: BTreeSet<_> = Kernel::ALL.iter().map(|kernel| kernel.name()).collect();
        assert_eq!(names.len(), Kernel::ALL.len());
        for kernel in Kernel::ALL {
            let definition = format!("kernel void {}(", kernel.name());
            assert_eq!(source.matches(&definition).count(), 1, "{}", kernel.name());
        }
    }

    #[test]
    fn baseline_counter_names_are_preserved_and_only_two_tile_names_are_added() {
        let legacy = [
            "elementwise",
            "rect_copy",
            "gather",
            "columns",
            "argmax_rows",
            "dense_linear",
            "packed_gather",
            "packed_linear",
            "packed_pair",
            "rms_norm",
            "rotary",
            "attention",
            "causal_conv",
            "conv_history",
            "centered_conv",
        ];
        let names: BTreeSet<_> = Kernel::ALL.iter().map(|kernel| kernel.name()).collect();
        for name in legacy {
            assert!(names.contains(name), "{name}");
        }
        let tiled: Vec<_> = Kernel::ALL
            .into_iter()
            .filter(|kernel| kernel.is_tile8())
            .collect();
        assert_eq!(
            tiled.as_slice(),
            &[Kernel::PackedLinearTile8, Kernel::PackedPairTile8]
        );
        assert_eq!(names.len(), legacy.len() + tiled.len());
    }
}
