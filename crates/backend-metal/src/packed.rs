//! Backend-private canonical packing and checked dispatch geometry.

use crate::{Device, MetalBuffer, kernels::Kernel, product};
use minifield_engine_api::{ExecutorError, Result};

pub(crate) const TILE_T: u64 = 8;
pub(crate) const TILE_N: u64 = 32;
#[cfg(test)]
const TILE_K: usize = 32;
pub(crate) const THREADS: usize = 256;
pub(crate) const SIMD_WIDTH: usize = 32;
pub(crate) const SCALE_GROUP: u64 = 128;
#[cfg(test)]
const WEIGHT_PITCH: usize = 33;
#[cfg(test)]
const SINGLE_SHARED_BYTES: usize = 5_248;
#[cfg(test)]
const PAIR_SHARED_BYTES: usize = 9_472;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Ternary,
    Nf4,
    Int8,
}

impl Format {
    pub(crate) fn from_code_width(bytes: u64, width: u64) -> Result<Self> {
        if width == 0 || width % SCALE_GROUP != 0 {
            return Err(ExecutorError::InvalidShape(
                "Metal packed input width is invalid",
            ));
        }
        if bytes == width / 4 {
            Ok(Self::Ternary)
        } else if bytes == width / 2 {
            Ok(Self::Nf4)
        } else if bytes == width {
            Ok(Self::Int8)
        } else {
            Err(ExecutorError::InvalidShape(
                "Metal packed code width is invalid",
            ))
        }
    }

    pub(crate) const fn id(self) -> u32 {
        match self {
            Self::Ternary => 0,
            Self::Nf4 => 1,
            Self::Int8 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Single,
    InputSwiGlu,
    Pair,
    PairSwiGlu,
}

impl Mode {
    const fn scalar_kernel(self) -> Kernel {
        match self {
            Self::Single | Self::InputSwiGlu => Kernel::PackedLinear,
            Self::Pair | Self::PairSwiGlu => Kernel::PackedPair,
        }
    }

    pub(crate) const fn tiled_kernel(self) -> Kernel {
        match self {
            Self::Single | Self::InputSwiGlu => Kernel::PackedLinearTile8,
            Self::Pair | Self::PairSwiGlu => Kernel::PackedPairTile8,
        }
    }
}

/// Observed compiled limits. Source-array totals aren't an allocation lower bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PipelineCaps {
    pub(crate) execution_width: usize,
    pub(crate) max_threads: usize,
    pub(crate) static_bytes: usize,
    pub(crate) device_threadgroup_bytes: usize,
}

impl PipelineCaps {
    pub(crate) const fn admits_rms_norm(self) -> bool {
        self.execution_width == SIMD_WIDTH
            && self.max_threads >= THREADS
            && self.static_bytes <= self.device_threadgroup_bytes
    }

    pub(crate) const fn admits_tile8(self) -> bool {
        self.execution_width == SIMD_WIDTH
            && self.max_threads >= THREADS
            && self.static_bytes <= self.device_threadgroup_bytes
    }
}

/// Only checked selection constructs this complete, positive 2D grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Tile8Grid {
    groups: [usize; 3],
}

impl Tile8Grid {
    fn for_shape(tokens: u64, rows: u64, width: u64) -> Option<Self> {
        if !(TILE_T..=512).contains(&tokens)
            || !(TILE_N..=8192).contains(&rows)
            || !(SCALE_GROUP..=8192).contains(&width)
            || width % SCALE_GROUP != 0
        {
            return None;
        }
        for (left, right) in [(tokens, width), (rows, width), (tokens, rows)] {
            u32::try_from(left.checked_mul(right)?).ok()?;
        }
        let columns = usize::try_from(rows.checked_add(TILE_N - 1)? / TILE_N).ok()?;
        let token_rows = usize::try_from(tokens.checked_add(TILE_T - 1)? / TILE_T).ok()?;
        Some(Self {
            groups: [columns, token_rows, 1],
        })
    }

    pub(crate) const fn groups(self) -> [usize; 3] {
        self.groups
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Geometry {
    Linear(usize),
    Tile8(Tile8Grid),
    RmsNorm(usize),
}

impl Geometry {
    pub(crate) const fn matches_kernel(self, kernel: Kernel) -> bool {
        match self {
            Self::Tile8(_) => kernel.is_tile8(),
            Self::RmsNorm(_) => matches!(kernel, Kernel::RmsNormSimd),
            Self::Linear(_) => !kernel.is_tile8() && !matches!(kernel, Kernel::RmsNormSimd),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Plan {
    Scalar(Kernel),
    Tile8(Kernel, Tile8Grid),
}

impl Plan {
    fn select(mode: Mode, tokens: u64, rows: u64, width: u64, caps: PipelineCaps) -> Self {
        if caps.admits_tile8()
            && let Some(grid) = Tile8Grid::for_shape(tokens, rows, width)
        {
            return Self::Tile8(mode.tiled_kernel(), grid);
        }
        Self::Scalar(mode.scalar_kernel())
    }
}

impl Device {
    pub(crate) fn dispatch_packed(
        &self,
        mode: Mode,
        shape: [u64; 3],
        buffers: &[&MetalBuffer],
        words: &[u32],
    ) -> Result<()> {
        let [tokens, rows, width] = shape;
        // Validation of every tensor, alias and parameter precedes this helper.
        // Limits were queried from these exact pipeline objects at construction.
        let caps = self.raw.pipeline_caps(mode.tiled_kernel())?;
        match Plan::select(mode, tokens, rows, width, caps) {
            Plan::Scalar(kernel) => self.dispatch(kernel, buffers, words, product(tokens, rows)?),
            Plan::Tile8(kernel, grid) => self.encode(kernel, buffers, words, Geometry::Tile8(grid)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires actual Metal shader compilation and pipeline capability queries"]
    fn actual_tile8_pipeline_capabilities_are_retained() {
        let device = crate::bridge::Device::new().expect("actual Metal candidate pipelines");
        let (_, registry_id, _) = device.info();
        for kernel in [Kernel::PackedLinearTile8, Kernel::PackedPairTile8] {
            let observed = device
                .pipeline_caps(kernel)
                .expect("compiled tile8 pipeline caps");
            // Keep observed compiler allocation separate from declared array totals.
            // Static names and numeric fields make each record valid JSON.
            println!(
                "{{\"record\":\"native_tile8_pipeline_caps\",\"kernel\":\"{}\",\"registry_id\":{registry_id},\"execution_width\":{},\"max_threads\":{},\"static_bytes\":{},\"device_threadgroup_bytes\":{}}}",
                kernel.name(),
                observed.execution_width,
                observed.max_threads,
                observed.static_bytes,
                observed.device_threadgroup_bytes,
            );
            assert!(
                observed.admits_tile8(),
                "{} incompatible: {observed:?}",
                kernel.name()
            );
        }
    }

    fn caps() -> PipelineCaps {
        PipelineCaps {
            execution_width: 32,
            max_threads: 256,
            static_bytes: PAIR_SHARED_BYTES,
            device_threadgroup_bytes: PAIR_SHARED_BYTES,
        }
    }

    #[test]
    fn selection_preserves_all_four_scalar_routes_and_admits_t8_t9() {
        for mode in [
            Mode::Single,
            Mode::InputSwiGlu,
            Mode::Pair,
            Mode::PairSwiGlu,
        ] {
            for tokens in 0..8 {
                assert_eq!(
                    Plan::select(mode, tokens, 33, 128, caps()),
                    Plan::Scalar(mode.scalar_kernel())
                );
            }
            for tokens in [8, 9] {
                assert!(
                    matches!(Plan::select(mode, tokens, 33, 128, caps()), Plan::Tile8(kernel, _) if kernel == mode.tiled_kernel())
                );
            }
        }
    }

    #[test]
    fn compiled_limits_use_actual_allocation_and_exact_thread_boundary() {
        assert!(caps().admits_tile8());
        assert!(caps().admits_rms_norm());
        for incompatible in [
            PipelineCaps {
                execution_width: 16,
                ..caps()
            },
            PipelineCaps {
                execution_width: 64,
                ..caps()
            },
            PipelineCaps {
                max_threads: 255,
                ..caps()
            },
            PipelineCaps {
                device_threadgroup_bytes: PAIR_SHARED_BYTES - 1,
                ..caps()
            },
        ] {
            assert!(!incompatible.admits_rms_norm());
            assert_eq!(
                Plan::select(Mode::Pair, 9, 33, 128, incompatible),
                Plan::Scalar(Kernel::PackedPair)
            );
        }
        for declared in [SINGLE_SHARED_BYTES, PAIR_SHARED_BYTES] {
            let reported = declared - 128;
            let smaller = PipelineCaps {
                static_bytes: reported,
                device_threadgroup_bytes: reported,
                ..caps()
            };
            assert!(smaller.admits_tile8());
            assert!(
                !PipelineCaps {
                    device_threadgroup_bytes: reported - 1,
                    ..smaller
                }
                .admits_tile8()
            );
        }
    }

    #[test]
    fn shape_limits_and_complete_group_geometry_are_bounded() {
        for shape in [
            [7, 32, 128],
            [513, 32, 128],
            [8, 31, 128],
            [8, 8193, 128],
            [8, 32, 0],
            [8, 32, 127],
            [8, 32, 129],
            [8, 32, 8320],
            [u64::MAX, 32, 128],
            [8, u64::MAX, 128],
        ] {
            assert!(Tile8Grid::for_shape(shape[0], shape[1], shape[2]).is_none());
        }
        assert_eq!(
            Tile8Grid::for_shape(9, 33, 384).map(Tile8Grid::groups),
            Some([2, 2, 1])
        );
        assert_eq!(
            Tile8Grid::for_shape(512, 8192, 8192).map(Tile8Grid::groups),
            Some([256, 64, 1])
        );
        for (tokens, groups, tail) in [
            (111, 14, 1),
            (220, 28, 4),
            (308, 39, 4),
            (345, 44, 7),
            (346, 44, 6),
            (347, 44, 5),
            (298, 38, 6),
            (299, 38, 5),
            (300, 38, 4),
            (47, 6, 1),
        ] {
            let grid = Tile8Grid::for_shape(tokens, 33, 128);
            assert_eq!(grid.map(Tile8Grid::groups), Some([2, groups, 1]));
            assert_eq!(
                u64::try_from(groups).expect("bounded group count") * TILE_T - tokens,
                tail
            );
        }
    }

    #[test]
    fn code_width_inference_preserves_canonical_format_ids() {
        for (format, p) in [(Format::Ternary, 4), (Format::Nf4, 2), (Format::Int8, 1)] {
            assert_eq!(Format::from_code_width(384 / p, 384), Ok(format));
        }
        assert_eq!(
            [Format::Ternary.id(), Format::Nf4.id(), Format::Int8.id()],
            [0, 1, 2]
        );
        for (bytes, width) in [(0, 0), (32, 127), (33, 128), (129, 128)] {
            assert!(Format::from_code_width(bytes, width).is_err());
        }
    }

    #[test]
    fn declared_shared_ranges_and_weight_producer_ownership_are_disjoint() {
        use std::collections::BTreeSet;
        let input_words = 8 * TILE_K;
        let weight_words = TILE_K * WEIGHT_PITCH;
        assert_eq!(4 * (input_words + weight_words), SINGLE_SHARED_BYTES);
        assert_eq!(4 * (input_words + 2 * weight_words), PAIR_SHARED_BYTES);
        for p in [1, 2, 4] {
            let mut occupied = BTreeSet::new();
            let mut coefficients = vec![None; weight_words];
            for q in 0..THREADS {
                for j in (q..32 * (32 / p)).step_by(THREADS) {
                    let output = j / (32 / p);
                    let byte = j % (32 / p);
                    for component in 0..p {
                        let column = p * byte + component;
                        let position = column * WEIGHT_PITCH + output;
                        assert!(position < weight_words && occupied.insert(position));
                        coefficients[position] = Some((output, column));
                    }
                }
            }
            assert_eq!(occupied.len(), 32 * 32);
            for canonical_output in 0..32 {
                for canonical_column in 0..32 {
                    assert_eq!(
                        coefficients[canonical_column * WEIGHT_PITCH + canonical_output],
                        Some((canonical_output, canonical_column))
                    );
                }
            }
            for column in 0..32 {
                assert_eq!(coefficients[column * WEIGHT_PITCH + 32], None);
            }
        }
    }

    #[test]
    fn fixed_geometry_cannot_be_used_with_a_scalar_entry_point() {
        let grid = Tile8Grid::for_shape(9, 33, 128).expect("admitted synthetic shape");
        assert!(Geometry::Tile8(grid).matches_kernel(Kernel::PackedLinearTile8));
        assert!(!Geometry::Tile8(grid).matches_kernel(Kernel::PackedLinear));
        assert!(!Geometry::Linear(1).matches_kernel(Kernel::PackedPairTile8));
        assert!(Geometry::RmsNorm(3).matches_kernel(Kernel::RmsNormSimd));
        assert!(!Geometry::RmsNorm(3).matches_kernel(Kernel::RmsNorm));
        assert!(!Geometry::Linear(3).matches_kernel(Kernel::RmsNormSimd));
    }

    #[test]
    fn masked_shared_indices_match_independent_canonical_rows_and_scale_groups() {
        use std::collections::BTreeSet;
        // This integer model writes flat shared storage before consuming it.
        // Expected input/weight coordinates are reconstructed separately from
        // the output coordinate, never from a producer's row/store index.
        let (tokens, rows) = (9_usize, 33_usize);
        for width in [128, 256, 384] {
            for p in [1, 2, 4] {
                let mut outputs = BTreeSet::new();
                for group_y in 0..2 {
                    for group_x in 0..2 {
                        let token_base = group_y * 8;
                        let output_base = group_x * 32;
                        for q in 0..THREADS {
                            let token = token_base + q / 32;
                            let output = output_base + q % 32;
                            if token < tokens && output < rows {
                                assert!(outputs.insert(token * rows + output));
                            }
                        }
                        for k_base in (0..width).step_by(TILE_K) {
                            let mut staged_input = vec![None; 8 * TILE_K];
                            let mut staged_weights = vec![None; TILE_K * WEIGHT_PITCH];
                            let mut input_owners = BTreeSet::new();
                            let mut weight_owners = BTreeSet::new();
                            for q in 0..THREADS {
                                let input_token = token_base + q / 32;
                                let input_column = k_base + q % 32;
                                let position = (q / 32) * TILE_K + q % 32;
                                assert!(input_owners.insert(position));
                                if input_token < tokens {
                                    let address = input_token * width + input_column;
                                    assert!(address < tokens * width);
                                    staged_input[position] = Some(address);
                                }
                                for j in (q..32 * (TILE_K / p)).step_by(THREADS) {
                                    let local_output = j / (TILE_K / p);
                                    let local_byte = j % (TILE_K / p);
                                    let matrix_row = output_base + local_output;
                                    for component in 0..p {
                                        let local_k = p * local_byte + component;
                                        let shared = local_k * WEIGHT_PITCH + local_output;
                                        assert!(weight_owners.insert(shared));
                                        if matrix_row < rows {
                                            let code_byte =
                                                matrix_row * (width / p) + k_base / p + local_byte;
                                            let scale = matrix_row * (width / 128) + k_base / 128;
                                            assert!(code_byte < rows * (width / p));
                                            assert!(scale < rows * (width / 128));
                                            staged_weights[shared] =
                                                Some((code_byte, scale, component));
                                        }
                                    }
                                }
                            }
                            assert_eq!(input_owners.len(), 8 * TILE_K);
                            assert_eq!(weight_owners.len(), 32 * TILE_K);
                            for canonical_slot in 0..8 {
                                let canonical_token = token_base + canonical_slot;
                                for canonical_k in 0..TILE_K {
                                    let expected = (canonical_token < tokens)
                                        .then_some(canonical_token * width + k_base + canonical_k);
                                    assert_eq!(
                                        staged_input[canonical_slot * TILE_K + canonical_k],
                                        expected
                                    );
                                }
                            }
                            for canonical_output in 0..32 {
                                let matrix_row = output_base + canonical_output;
                                for canonical_k in 0..TILE_K {
                                    let column = k_base + canonical_k;
                                    let expected = (matrix_row < rows).then_some((
                                        matrix_row * (width / p) + column / p,
                                        matrix_row * (width / 128) + column / 128,
                                        column % p,
                                    ));
                                    assert_eq!(
                                        staged_weights
                                            [canonical_k * WEIGHT_PITCH + canonical_output],
                                        expected
                                    );
                                }
                            }
                        }
                    }
                }
                assert_eq!(outputs, (0..tokens * rows).collect());
            }
        }
    }
}
