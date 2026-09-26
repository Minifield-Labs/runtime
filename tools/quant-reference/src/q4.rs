// Adapted from unrvl-embdb/src/quant.rs. See ../../../docs/wires-audit.md.
use crate::{Error, Result};

/// A row-major matrix with signed, packed INT4 weights and per-group scales.
///
/// Two weights occupy each byte. The even linear index uses the low nibble and
/// the odd index uses the high nibble. Nibbles are signed two's-complement
/// values in `-8..=7`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMatrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
}

impl QuantizedMatrix {
    /// Builds a matrix from already packed data.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        group_size: usize,
        packed: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidModel("matrix dimensions must be non-zero"));
        }
        if group_size == 0 {
            return Err(Error::InvalidModel("matrix group size must be non-zero"));
        }

        let elements = rows.checked_mul(cols).ok_or(Error::SizeOverflow)?;
        let packed_len = elements.checked_add(1).ok_or(Error::SizeOverflow)? / 2;
        let groups_per_row = cols.div_ceil(group_size);
        let scales_len = rows
            .checked_mul(groups_per_row)
            .ok_or(Error::SizeOverflow)?;

        if packed.len() != packed_len {
            return Err(Error::DimensionMismatch {
                name: "packed weights",
                expected: packed_len,
                actual: packed.len(),
            });
        }
        if scales.len() != scales_len {
            return Err(Error::DimensionMismatch {
                name: "weight scales",
                expected: scales_len,
                actual: scales.len(),
            });
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(Error::InvalidModel(
                "matrix scales must be finite and positive",
            ));
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            packed,
            scales,
        })
    }

    /// Symmetrically quantizes a row-major FP32 matrix.
    ///
    /// The quantizer uses `max(abs(group)) / 7` and reserves the `-8` code for
    /// imported models that use the full signed nibble range.
    pub fn quantize(rows: usize, cols: usize, group_size: usize, weights: &[f32]) -> Result<Self> {
        let elements = rows.checked_mul(cols).ok_or(Error::SizeOverflow)?;
        if weights.len() != elements {
            return Err(Error::DimensionMismatch {
                name: "source weights",
                expected: elements,
                actual: weights.len(),
            });
        }
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(Error::InvalidModel(
                "matrix dimensions and group size must be non-zero",
            ));
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(Error::InvalidModel("source weights must be finite"));
        }

        let groups_per_row = cols.div_ceil(group_size);
        let mut packed = vec![0_u8; elements.div_ceil(2)];
        let mut scales = Vec::with_capacity(rows * groups_per_row);

        for row in 0..rows {
            let row_start = row * cols;
            for group in 0..groups_per_row {
                let start = group * group_size;
                let end = (start + group_size).min(cols);
                let max_abs = weights[row_start + start..row_start + end]
                    .iter()
                    .fold(0.0_f32, |current, value| current.max(value.abs()));
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales.push(scale);

                for col in start..end {
                    let quantized =
                        (weights[row_start + col] / scale).round().clamp(-7.0, 7.0) as i8;
                    set_nibble(&mut packed, row_start + col, quantized);
                }
            }
        }

        Self::from_parts(rows, cols, group_size, packed, scales)
    }

    /// Number of matrix rows.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of matrix columns.
    #[inline]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Number of columns sharing each scale.
    #[inline]
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Packed weight bytes.
    #[inline]
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// Per-row, per-group FP32 scales.
    #[inline]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns a single dequantized weight.
    pub fn value(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let index = row * self.cols + col;
        let group = row * self.cols.div_ceil(self.group_size) + col / self.group_size;
        Some(f32::from(get_nibble(&self.packed, index)) * self.scales[group])
    }

    /// Computes `output = self * input`.
    ///
    /// Accumulation stays in FP32. Group-local sums are multiplied by their
    /// scale once, which keeps the inner loop compact for WASM auto-vectorizers.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.cols {
            return Err(Error::DimensionMismatch {
                name: "matrix input",
                expected: self.cols,
                actual: input.len(),
            });
        }
        if output.len() != self.rows {
            return Err(Error::DimensionMismatch {
                name: "matrix output",
                expected: self.rows,
                actual: output.len(),
            });
        }

        let groups_per_row = self.cols.div_ceil(self.group_size);
        for (row, destination) in output.iter_mut().enumerate() {
            let row_start = row * self.cols;
            let mut total = 0.0_f32;
            for group in 0..groups_per_row {
                let start = group * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut group_sum = 0.0_f32;
                for (col, input_value) in input.iter().copied().enumerate().take(end).skip(start) {
                    let weight = f32::from(get_nibble(&self.packed, row_start + col));
                    group_sum = input_value.mul_add(weight, group_sum);
                }
                total = self.scales[row * groups_per_row + group].mul_add(group_sum, total);
            }
            *destination = total;
        }
        Ok(())
    }

    /// Computes a batch of row-major vectors against the same matrix.
    ///
    /// `input` has shape `batch × cols` and `output` has shape `batch × rows`.
    /// Packed weights are unpacked once per group and reused across the batch,
    /// matching the way all tokens in an encoder sequence share projections.
    pub fn matmul(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<()> {
        if batch == 0 {
            return Err(Error::InvalidConfig("matrix batch must be non-zero"));
        }
        let expected_input = batch.checked_mul(self.cols).ok_or(Error::SizeOverflow)?;
        let expected_output = batch.checked_mul(self.rows).ok_or(Error::SizeOverflow)?;
        if input.len() != expected_input {
            return Err(Error::DimensionMismatch {
                name: "batched matrix input",
                expected: expected_input,
                actual: input.len(),
            });
        }
        if output.len() != expected_output {
            return Err(Error::DimensionMismatch {
                name: "batched matrix output",
                expected: expected_output,
                actual: output.len(),
            });
        }
        if batch == 1 {
            return self.matvec(input, output);
        }

        output.fill(0.0);
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let mut unpacked = vec![0_i8; self.group_size.min(self.cols)];

        for row in 0..self.rows {
            let row_start = row * self.cols;
            for group in 0..groups_per_row {
                let start = group * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let group_length = end - start;
                for (offset, value) in unpacked.iter_mut().take(group_length).enumerate() {
                    *value = get_nibble(&self.packed, row_start + start + offset);
                }
                let scale = self.scales[row * groups_per_row + group];

                for batch_index in 0..batch {
                    let input_start = batch_index * self.cols + start;
                    let input_group = &input[input_start..input_start + group_length];
                    let group_sum = input_group
                        .iter()
                        .zip(&unpacked)
                        .fold(0.0_f32, |sum, (input, weight)| {
                            input.mul_add(f32::from(*weight), sum)
                        });
                    output[batch_index * self.rows + row] =
                        scale.mul_add(group_sum, output[batch_index * self.rows + row]);
                }
            }
        }
        Ok(())
    }

    pub fn add_row_to(&self, row: usize, output: &mut [f32]) -> Result<()> {
        if row >= self.rows {
            return Err(Error::TokenOutOfRange(row as u32));
        }
        if output.len() != self.cols {
            return Err(Error::DimensionMismatch {
                name: "embedding output",
                expected: self.cols,
                actual: output.len(),
            });
        }

        let groups_per_row = self.cols.div_ceil(self.group_size);
        let row_start = row * self.cols;
        for group in 0..groups_per_row {
            let start = group * self.group_size;
            let end = (start + self.group_size).min(self.cols);
            let scale = self.scales[row * groups_per_row + group];
            for (col, output_value) in output.iter_mut().enumerate().take(end).skip(start) {
                *output_value += f32::from(get_nibble(&self.packed, row_start + col)) * scale;
            }
        }
        Ok(())
    }
}

#[inline]
fn get_nibble(packed: &[u8], index: usize) -> i8 {
    let byte = packed[index / 2];
    let nibble = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    ((nibble as i8) << 4) >> 4
}

#[inline]
fn set_nibble(packed: &mut [u8], index: usize, value: i8) {
    let nibble = (value as u8) & 0x0f;
    let byte = &mut packed[index / 2];
    if index & 1 == 0 {
        *byte = (*byte & 0xf0) | nibble;
    } else {
        *byte = (*byte & 0x0f) | (nibble << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_nibbles_round_trip() {
        let mut packed = [0_u8; 8];
        for (index, value) in (-8_i8..=7).enumerate() {
            set_nibble(&mut packed, index, value);
        }
        for (index, expected) in (-8_i8..=7).enumerate() {
            assert_eq!(get_nibble(&packed, index), expected);
        }
    }

    #[test]
    fn quantized_matvec_tracks_float_reference() {
        let weights = [
            -1.0, -0.5, 0.25, 1.0, // row 0
            0.75, 0.0, -0.25, 0.5, // row 1
        ];
        let matrix = QuantizedMatrix::quantize(2, 4, 4, &weights).unwrap();
        let input = [0.5, 1.0, -2.0, 0.25];
        let mut output = [0.0; 2];
        matrix.matvec(&input, &mut output).unwrap();

        let expected = [-1.25, 1.0];
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.15, "{actual} != {expected}");
        }
    }

    #[test]
    fn batched_matmul_matches_repeated_matvec() {
        let weights: Vec<f32> = (0..35).map(|index| index as f32 / 9.0 - 2.0).collect();
        let matrix = QuantizedMatrix::quantize(5, 7, 4, &weights).unwrap();
        let input: Vec<f32> = (0..21).map(|index| index as f32 / 13.0 - 0.5).collect();
        let mut batched = vec![0.0; 15];
        matrix.matmul(&input, 3, &mut batched).unwrap();

        for batch in 0..3 {
            let mut expected = vec![0.0; 5];
            matrix
                .matvec(&input[batch * 7..(batch + 1) * 7], &mut expected)
                .unwrap();
            assert_eq!(&batched[batch * 5..(batch + 1) * 5], expected);
        }
    }
}
