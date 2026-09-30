//! Finite operations for complete-sequence, segment-isolated encoders.

use crate::{ExecutorError, GatedShortConvSpec, GqaSpec, InferenceOps, Result};

/// A packed sequence's nonzero segment labels. Zero marks padding.
///
/// Each active label occupies one contiguous run. This makes restarting rotary
/// positions and independent-sequence convolution padding unambiguous.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderSegments {
    ids: Vec<u32>,
    positions: Vec<u64>,
}

impl EncoderSegments {
    pub fn new(ids: Vec<u32>) -> Result<Self> {
        let mut seen = std::collections::BTreeSet::new();
        let mut previous = 0;
        let mut position = 0_u64;
        let mut positions = Vec::new();
        positions.try_reserve_exact(ids.len()).map_err(|_| {
            ExecutorError::ResourceLimit("encoder segment positions allocation failed")
        })?;
        for &id in &ids {
            if id != previous {
                if id != 0 && !seen.insert(id) {
                    return Err(ExecutorError::InvalidArgument(
                        "encoder segment labels must occupy contiguous runs",
                    ));
                }
                position = 0;
            }
            positions.push(position);
            position = if id == 0 {
                0
            } else {
                position.checked_add(1).ok_or(ExecutorError::Overflow(
                    "encoder segment position overflows u64",
                ))?
            };
            previous = id;
        }
        Ok(Self { ids, positions })
    }

    pub fn single(tokens: usize) -> Result<Self> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(tokens).map_err(|_| {
            ExecutorError::ResourceLimit("encoder segment labels allocation failed")
        })?;
        ids.resize(tokens, 1);
        Self::new(ids)
    }

    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    #[must_use]
    pub fn positions(&self) -> &[u64] {
        &self.positions
    }

    pub fn validate_tokens(&self, tokens: u64) -> Result<()> {
        if u64::try_from(self.ids.len())
            .map_err(|_| ExecutorError::Overflow("encoder segment count exceeds u64"))?
            != tokens
        {
            return Err(ExecutorError::InvalidShape(
                "encoder segments must match the token count",
            ));
        }
        Ok(())
    }
}

/// Complete-sequence operations, separate from causal cache mutation.
pub trait EncoderOps: InferenceOps {
    /// Noncausal GQA, attending only within the query's active segment.
    /// Inputs use the existing packed head layouts; inactive query rows are zero.
    fn bidirectional_gqa(
        &self,
        output: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        segments: &EncoderSegments,
        spec: GqaSpec,
    ) -> Result<()>;

    /// `C[t] * sum_j kernel[j] * B[t+j-width/2] * V[t+j-width/2]`.
    /// Padding and other segments contribute zero. Even-width kernels use
    /// floor(width/2) left padding and crop the extra output row on the right.
    fn centered_gated_convolution(
        &self,
        output: &mut Self::Buffer,
        projection: &Self::Buffer,
        kernel: &Self::Buffer,
        segments: &EncoderSegments,
        spec: GatedShortConvSpec,
    ) -> Result<()>;
}
