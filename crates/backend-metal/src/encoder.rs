//! Complete-sequence attention and centered convolution on native Metal.
use crate::{MetalBackend, MetalBuffer, kernels::Kernel, p, product};
use minifield_engine_api::{
    DType, EncoderOps, EncoderSegments, ExecutorError, GatedShortConvSpec, GqaSpec, Result, Shape,
};
impl EncoderOps for MetalBackend {
    fn bidirectional_gqa(
        &self,
        output: &mut MetalBuffer,
        query: &MetalBuffer,
        key: &MetalBuffer,
        value: &MetalBuffer,
        segments: &EncoderSegments,
        spec: GqaSpec,
    ) -> Result<()> {
        for b in [query, key, value] {
            self.check(b, DType::F32)?;
        }
        let tokens = spec
            .query_heads()
            .validate_packed(query.descriptor.layout.shape())?;
        for b in [key, value] {
            if spec
                .key_value_heads()
                .validate_packed(b.descriptor.layout.shape())?
                != tokens
            {
                return Err(ExecutorError::InvalidShape(
                    "Metal encoder GQA token counts differ",
                ));
            }
        }
        segments.validate_tokens(tokens)?;
        self.output(output, query.descriptor.layout.shape())?;
        Self::distinct(output, &[query, key, value])?;
        let staged = self.stage_u32(segments.ids())?;
        self.attention(
            output, query, key, value, &staged, tokens, tokens, 0, spec, true,
        )
    }
    fn centered_gated_convolution(
        &self,
        output: &mut MetalBuffer,
        projection: &MetalBuffer,
        kernel: &MetalBuffer,
        segments: &EncoderSegments,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check(projection, DType::F32)?;
        self.check(kernel, DType::F32)?;
        let s = projection.descriptor.layout.shape();
        let h = u64::from(spec.hidden());
        let w = u64::from(spec.width());
        if s.rank() != 2
            || s.dim(1)? != product(h, 3)?
            || kernel.descriptor.layout.shape() != Shape::new(&[h, w])?
        {
            return Err(ExecutorError::InvalidShape(
                "Metal centered convolution geometry differs",
            ));
        }
        if s.dim(0)?.checked_add(w).is_none_or(|n| n > i32::MAX as u64) {
            return Err(ExecutorError::ResourceLimit(
                "Metal convolution signed tap range exceeded",
            ));
        }
        segments.validate_tokens(s.dim(0)?)?;
        self.output(output, Shape::new(&[s.dim(0)?, h])?)?;
        Self::distinct(output, &[projection, kernel])?;
        let staged = self.stage_u32(segments.ids())?;
        self.device.dispatch(
            Kernel::CenteredConv,
            &[output, projection, kernel, &staged],
            &[p(s.dim(0)?)?, spec.hidden(), spec.width()],
            product(s.dim(0)?, h)?,
        )
    }
}
