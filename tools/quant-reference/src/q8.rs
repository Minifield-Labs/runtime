//! Audited row-local INT8 storage and CPU oracle, adapted from Wires' w8.rs.
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Int8Matrix {
    rows: usize,
    cols: usize,
    group_size: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

impl Int8Matrix {
    pub fn from_parts(
        rows: usize,
        cols: usize,
        group_size: usize,
        codes: Vec<i8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(Error::InvalidModel(
                "matrix dimensions and group size must be positive",
            ));
        }
        let count = rows.checked_mul(cols).ok_or(Error::SizeOverflow)?;
        let scale_count = rows
            .checked_mul(cols.div_ceil(group_size))
            .ok_or(Error::SizeOverflow)?;
        if codes.len() != count || scales.len() != scale_count {
            return Err(Error::InvalidModel(
                "INT8 matrix dimensions don't match storage",
            ));
        }
        if scales
            .iter()
            .any(|x| !x.is_finite() || *x < f32::MIN_POSITIVE)
        {
            return Err(Error::InvalidModel(
                "scales must be finite, normal, and positive",
            ));
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            codes,
            scales,
        })
    }

    /// Read an MFQ8 v1 matrix within a caller-specified element budget.
    pub fn from_bytes(bytes: &[u8], max_elements: usize) -> Result<Self> {
        if bytes.len() < 28 {
            return Err(Error::Truncated);
        }
        let u32_at = |offset: usize| -> usize {
            u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("bounded header"),
            ) as usize
        };
        if &bytes[..4] != b"MFQ8" || u32_at(4) != 1 {
            return Err(Error::UnsupportedFormat);
        }
        let (rows, cols, group_size, count, scale_count) =
            (u32_at(8), u32_at(12), u32_at(16), u32_at(20), u32_at(24));
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(Error::InvalidModel("zero matrix dimension"));
        }
        let expected_count = rows.checked_mul(cols).ok_or(Error::SizeOverflow)?;
        let expected_scales = rows
            .checked_mul(cols.div_ceil(group_size))
            .ok_or(Error::SizeOverflow)?;
        if count != expected_count || scale_count != expected_scales || count > max_elements {
            return Err(Error::InvalidModel(
                "invalid matrix shape or element budget exceeded",
            ));
        }
        let codes_end = 28usize.checked_add(count).ok_or(Error::SizeOverflow)?;
        let scales_start = codes_end.checked_add(3).ok_or(Error::SizeOverflow)? & !3;
        let expected_bytes = scales_start
            .checked_add(scale_count.checked_mul(4).ok_or(Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        if bytes.len() < expected_bytes {
            return Err(Error::Truncated);
        }
        if bytes.len() != expected_bytes {
            return Err(Error::InvalidModel("trailing matrix bytes"));
        }
        if bytes[codes_end..scales_start].iter().any(|x| *x != 0) {
            return Err(Error::InvalidModel("nonzero matrix padding"));
        }
        // Validate all counts and bounds before allocating decoded storage.
        let codes = bytes[28..codes_end].iter().map(|x| *x as i8).collect();
        let scales = bytes[scales_start..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect();
        Self::from_parts(rows, cols, group_size, codes, scales)
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Compute input[batch, cols] times weights[rows, cols]^T in FP32.
    /// Packed values stay compact; each scale applies once per group sum.
    pub fn matmul(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<()> {
        if batch == 0 {
            return Err(Error::InvalidConfig("matrix batch must be positive"));
        }
        let input_count = batch.checked_mul(self.cols).ok_or(Error::SizeOverflow)?;
        let output_count = batch.checked_mul(self.rows).ok_or(Error::SizeOverflow)?;
        if input.len() != input_count || output.len() != output_count {
            return Err(Error::InvalidConfig("matrix input/output length mismatch"));
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidConfig("input must be finite"));
        }
        let groups = self.cols.div_ceil(self.group_size);
        for b in 0..batch {
            for row in 0..self.rows {
                let mut total = 0.0f32;
                for group in 0..groups {
                    let start = group * self.group_size;
                    let end = start + self.group_size.min(self.cols - start);
                    let mut sum = 0.0f32;
                    for column in start..end {
                        sum += input[b * self.cols + column]
                            * f32::from(self.codes[row * self.cols + column]);
                    }
                    total += sum * self.scales[row * groups + group];
                }
                if !total.is_finite() {
                    return Err(Error::InvalidConfig("non-finite matrix result"));
                }
                output[b * self.rows + row] = total;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const FIXTURE: &[u8] = include_bytes!("../../../examples/quant-matrix-v0.1.0/fixture.mfq8");

    #[test]
    fn loads_python_matrix_with_partial_groups() {
        let matrix = Int8Matrix::from_bytes(FIXTURE, 100).unwrap();
        assert_eq!(
            (matrix.rows(), matrix.cols(), matrix.group_size()),
            (2, 7, 4)
        );
        assert_eq!(&matrix.codes()[..4], &[-127, -1, 1, 127]);
        assert_eq!(matrix.scales()[1], 1.0);
        let mut output = [0.0; 4];
        matrix.matmul(&[1.0; 14], 2, &mut output).unwrap();
        assert_eq!(output[0], 0.0);
        assert!((output[1] + 2.1377953).abs() < 0.00001);
        assert_eq!(&output[..2], &output[2..]);
    }

    #[test]
    fn rejects_malformed_headers_before_allocation() {
        for length in 0..FIXTURE.len() {
            assert!(Int8Matrix::from_bytes(&FIXTURE[..length], 100).is_err());
        }
        for offset in [8, 12, 16, 20, 24] {
            let mut malformed = FIXTURE.to_vec();
            malformed[offset..offset + 4].fill(0);
            assert!(Int8Matrix::from_bytes(&malformed, 100).is_err());
        }
        assert!(Int8Matrix::from_bytes(FIXTURE, 13).is_err());
        let mut trailing = FIXTURE.to_vec();
        trailing.push(0);
        assert!(Int8Matrix::from_bytes(&trailing, 100).is_err());
    }

    #[test]
    fn rejects_bad_scales_and_call_shapes_without_panicking() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(Int8Matrix::from_parts(1, 1, 1, vec![1], vec![scale]).is_err());
        }
        assert!(Int8Matrix::from_parts(1, 1, 0, vec![1], vec![1.0]).is_err());
        let matrix = Int8Matrix::from_parts(1, 2, 8, vec![-128, 127], vec![0.5]).unwrap();
        let mut out = [0.0];
        matrix.matmul(&[1.0, 1.0], 1, &mut out).unwrap();
        assert_eq!(out[0], -0.5);
        assert!(matrix.matmul(&[1.0], 1, &mut out).is_err());
        assert!(matrix.matmul(&[f32::NAN, 1.0], 1, &mut out).is_err());
    }
}
