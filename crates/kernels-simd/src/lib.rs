//! Platform SIMD kernels for the `minifield.ternary.v1` packed weight path.
//!
//! Each kernel has a scalar fallback so the crate compiles and stays correct
//! on every target; accelerated implementations live behind `cfg` modules.
//! `unsafe` is denied crate-wide and explicitly allowed per intrinsics module.

#![deny(unsafe_code)]

// NEON intrinsics are unsafe; the allow is scoped to this one module, and
// its loads are statically bounded by the fixed-size array arguments.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;

/// Dot product of one packed ternary weight row against an f32 activation row.
///
/// Computes `Σ_g scale_g · Σ_j∈g (code_j − 1) · x_j` where `codes` is the
/// row's `K/4` byte stream (128 weights per 32-byte group, weight `j` at byte
/// `j/4`, bits `2*(j%4)`), `scales` is the row's `K/128` group scales, and
/// `input` is the `K` f32 activations. `K` must be a multiple of 128.
///
/// The products `(code − 1) · x` are exact in f32 (codes are −1, 0, +1, +2);
/// only the accumulation order varies between implementations, so results
/// differ from the per-element reference by at most small reorder error.
///
/// # Panics
///
/// Panics if `input` isn't a multiple of 128 wide or `codes`/`scales` don't
/// match the derived widths. Callers validate operand shapes first.
#[must_use]
pub fn ternary_row_dot(codes: &[u8], scales: &[f32], input: &[f32]) -> f32 {
    assert!(
        input.len().is_multiple_of(128),
        "ternary row input width must be a multiple of 128"
    );
    assert_eq!(
        codes.len(),
        input.len() / 4,
        "ternary row code bytes must be input width / 4"
    );
    assert_eq!(
        scales.len(),
        input.len() / 128,
        "ternary row scales must be input width / 128"
    );
    let mut row = 0.0_f32;
    for (group, (codes, input)) in codes
        .as_chunks::<32>()
        .0
        .iter()
        .zip(input.as_chunks::<128>().0.iter())
        .enumerate()
    {
        row += scales[group] * group_dot(codes, input);
    }
    row
}

/// Unscaled dot of one 128-weight group: `Σ_j (code_j − 1) · x_j`.
#[cfg(target_arch = "aarch64")]
fn group_dot(codes: &[u8; 32], input: &[f32; 128]) -> f32 {
    neon::group_dot(codes, input)
}

/// Unscaled dot of one 128-weight group: `Σ_j (code_j − 1) · x_j`.
#[cfg(not(target_arch = "aarch64"))]
fn group_dot(codes: &[u8; 32], input: &[f32; 128]) -> f32 {
    scalar_group_dot(codes, input)
}

/// Scalar reference for one group. The shared correctness oracle; SIMD
/// implementations must match it within reorder tolerance.
#[cfg(any(test, not(target_arch = "aarch64")))]
fn scalar_group_dot(codes: &[u8; 32], input: &[f32; 128]) -> f32 {
    let mut acc = 0.0_f32;
    for (index, x) in input.iter().enumerate() {
        let code = (codes[index / 4] >> (2 * (index % 4))) & 0x3;
        acc += (f32::from(code) - 1.0) * x;
    }
    acc
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    /// Hand-packed group: weights [1, -1, 0, 2] repeating in scale units.
    /// Codes {2, 0, 1, 3} pack to byte `0b11_01_00_10` = `0xD2`.
    #[test]
    fn group_dot_matches_scalar_reference() {
        let codes = [0xD2_u8; 32];
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.03125).collect();
        let input: &[f32; 128] = input.as_slice().try_into().expect("input width");
        let expected = scalar_group_dot(&codes, input);
        let actual = group_dot(&codes, input);
        assert!(
            (actual - expected).abs() <= 1e-4,
            "simd {actual} vs scalar {expected}"
        );
    }

    #[test]
    fn row_dot_accumulates_groups() {
        let codes = vec![0xD2_u8; 64];
        let scales = [0.5_f32, 2.0];
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let result = ternary_row_dot(&codes, &scales, &input);
        let mut expected = 0.0_f32;
        for (group, scale) in scales.iter().enumerate() {
            let codes: &[u8; 32] = codes[group * 32..group * 32 + 32]
                .try_into()
                .expect("group codes");
            let x: &[f32; 128] = input[group * 128..group * 128 + 128]
                .try_into()
                .expect("group input");
            expected += scale * scalar_group_dot(codes, x);
        }
        assert!(
            (result - expected).abs() <= 1e-4,
            "row {result} vs expected {expected}"
        );
    }

    #[test]
    fn zero_scale_and_zero_codes() {
        let codes = vec![0x55_u8; 32]; // all codes 1 → weight 0
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        assert_eq!(ternary_row_dot(&codes, &[1.5], &input), 0.0);
        let codes = vec![0x00_u8; 32]; // all codes 0 → weight -1
        let sum: f32 = input.iter().sum();
        let result = ternary_row_dot(&codes, &[2.0], &input);
        assert!((result - (-2.0 * sum)).abs() <= 1e-3);
    }

    #[test]
    #[should_panic(expected = "input width / 4")]
    fn rejects_mismatched_code_width() {
        let _ = ternary_row_dot(&[0_u8; 16], &[1.0], &[0.0; 128]);
    }
}
