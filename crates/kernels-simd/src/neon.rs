//! NEON implementation of the ternary group dot for aarch64.
//!
//! Weight order: byte `i` of the 32-byte group holds weights `4i..4i+3`, two
//! bits each, so extracting sub-position `s` of every byte yields the weights
//! `{4i + s}` — a stride-4 gather over the activations. `vld4q_f32` already
//! deinterleaves four stride-4 vectors, so each extracted code lane pairs
//! directly with one `vld4` field and no gather or shuffle is needed.

use core::arch::aarch64::{
    float32x4_t, uint8x16_t, vaddq_f32, vaddvq_f32, vandq_u8, vcvtq_f32_s32, vdupq_n_f32,
    vdupq_n_s16, vdupq_n_u8, vfmaq_f32, vget_high_s16, vget_high_u8, vget_low_s16, vget_low_u8,
    vld1q_u8, vld4q_f32, vmovl_s16, vmovl_u8, vreinterpretq_s16_u16, vshrq_n_u8, vsubq_s16,
};

/// Extract the `s`-th 2-bit code of each byte, widened to four f32x4
/// vectors of `code - 1` values.
#[target_feature(enable = "neon")]
#[inline]
fn codes_minus_one(bytes: uint8x16_t, shift_index: usize) -> [float32x4_t; 4] {
    let mask = vdupq_n_u8(0x03);
    let c16 = match shift_index {
        0 => vandq_u8(bytes, mask),
        1 => vandq_u8(vshrq_n_u8::<2>(bytes), mask),
        2 => vandq_u8(vshrq_n_u8::<4>(bytes), mask),
        _ => vandq_u8(vshrq_n_u8::<6>(bytes), mask),
    };
    let one = vdupq_n_s16(1);
    let lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(c16))), one);
    let hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(c16))), one);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
    ]
}

/// Unscaled dot of one 128-weight group: `Σ_j (code_j − 1) · x_j`.
pub(super) fn group_dot(codes: &[u8; 32], input: &[f32; 128]) -> f32 {
    // SAFETY: aarch64 always has NEON. `codes` has 32 bytes and `input` has
    // 128 f32; every load inside stays inside those bounds.
    unsafe { group_dot_neon(codes, input) }
}

/// Unscaled dot of one 128-weight group: `Σ_j (code_j − 1) · x_j`.
#[target_feature(enable = "neon")]
// The shift loop indexes four parallel lane arrays; iterating by index is
// clearer than zipping here.
#[allow(clippy::needless_range_loop)]
fn group_dot_neon(codes: &[u8; 32], input: &[f32; 128]) -> f32 {
    let mut acc = [
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    ];
    for block in 0..2 {
        // SAFETY: the fixed-size arguments guarantee 32 code bytes and
        // 128 f32, and the slices bound each load before the raw read.
        let (bytes, x) = unsafe {
            (
                vld1q_u8(codes[block * 16..].as_ptr()),
                [
                    vld4q_f32(input[block * 64..].as_ptr()),
                    vld4q_f32(input[block * 64 + 16..].as_ptr()),
                    vld4q_f32(input[block * 64 + 32..].as_ptr()),
                    vld4q_f32(input[block * 64 + 48..].as_ptr()),
                ],
            )
        };
        // vld4 field `s` of vector k holds activations
        // `{block*64 + k*16 + s + 4i | i in 0..4}`, matching code lanes
        // `4k..4k+4` of shift `s`.
        let xs = [
            [x[0].0, x[0].1, x[0].2, x[0].3],
            [x[1].0, x[1].1, x[1].2, x[1].3],
            [x[2].0, x[2].1, x[2].2, x[2].3],
            [x[3].0, x[3].1, x[3].2, x[3].3],
        ];
        for s in 0..4 {
            let f = codes_minus_one(bytes, s);
            acc[0] = vfmaq_f32(acc[0], f[0], xs[0][s]);
            acc[1] = vfmaq_f32(acc[1], f[1], xs[1][s]);
            acc[2] = vfmaq_f32(acc[2], f[2], xs[2][s]);
            acc[3] = vfmaq_f32(acc[3], f[3], xs[3][s]);
        }
    }
    let total = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
    vaddvq_f32(total)
}
