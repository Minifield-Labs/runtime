//! Research kernels, enabled with `experimental-kernels`. Each call declares
//! its implementation and uploads with the corresponding physical packing.
//! Production `InferenceOps` dispatch never reads this policy.

use super::{
    AllocationClass, CodeLayout, ExecutorError, Kernel, OperationKind, PackedStreamFormat, Result,
    Shape, WgpuBackend, WgpuBuffer, flat_grid, param32, params,
};

/// Explicit research implementation for a single projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LowbitsKernel {
    Baseline,
    Scale128,
    TernarySign,
    TernarySignSelect,
    TernaryLut2,
    TernaryLut2Alternate,
    TernaryPn4,
    Nf4Register,
    Nf4Product,
}

impl std::str::FromStr for LowbitsKernel {
    type Err = ExecutorError;

    fn from_str(value: &str) -> Result<Self> {
        Ok(match value {
            "baseline" => Self::Baseline,
            "scale128_control" => Self::Scale128,
            "ternary_sign" => Self::TernarySign,
            "ternary_sign_sel" => Self::TernarySignSelect,
            "ternary_lut2" => Self::TernaryLut2,
            "ternary_lut2_64x32" => Self::TernaryLut2Alternate,
            "ternary_pn4" => Self::TernaryPn4,
            "nf4_register" => Self::Nf4Register,
            "nf4_product" => Self::Nf4Product,
            _ => return Err(ExecutorError::InvalidArgument("unknown low-bit experiment")),
        })
    }
}

impl LowbitsKernel {
    fn layout(self) -> CodeLayout {
        match self {
            Self::TernaryLut2 | Self::TernaryLut2Alternate => CodeLayout::TernaryLut2,
            Self::TernaryPn4 => CodeLayout::TernaryPn4,
            _ => CodeLayout::Canonical,
        }
    }

    fn kernel(self, format: PackedStreamFormat) -> Result<(Kernel, u64, u64)> {
        use PackedStreamFormat::{Nf4V1, TernaryV1};
        Ok(match (self, format) {
            (Self::Scale128, TernaryV1) => (Kernel::PackedGemmTernaryScale128, 64, 32),
            (Self::Scale128, Nf4V1) => (Kernel::PackedGemmNf4Scale128, 64, 32),
            (Self::TernarySign, TernaryV1) => (Kernel::PackedGemmTernarySign, 64, 32),
            (Self::TernarySignSelect, TernaryV1) => (Kernel::PackedGemmTernarySignSel, 64, 32),
            (Self::TernaryLut2, TernaryV1) => (Kernel::PackedGemmTernaryLut2, 32, 64),
            (Self::TernaryLut2Alternate, TernaryV1) => (Kernel::PackedGemmTernaryLut2Alt, 64, 32),
            (Self::TernaryPn4, TernaryV1) => (Kernel::PackedGemmTernaryPn4, 32, 64),
            (Self::Nf4Register, Nf4V1) => (Kernel::PackedGemmNf4Register, 64, 32),
            (Self::Nf4Product, Nf4V1) => (Kernel::PackedGemmNf4Product, 16, 64),
            _ => {
                return Err(ExecutorError::InvalidArgument(
                    "experiment and weight encoding disagree",
                ));
            }
        })
    }
}

impl WgpuBackend {
    /// Upload explicitly encoded research weights. The caller supplies the
    /// bytes for the selected encoding; conformance tests compare independent
    /// CPU decoding of the original weights against these packed bytes.
    pub fn upload_experimental_codes(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        kernel: LowbitsKernel,
    ) -> Result<WgpuBuffer> {
        let mut buffer = self.upload_u8_classified(shape, bytes, AllocationClass::Weight)?;
        buffer.code_layout = kernel.layout();
        Ok(buffer)
    }

    /// Run one explicit experiment. Repacked encodings reject short rows;
    /// they can never silently fall back to a raw-code decoder.
    pub fn experimental_packed_linear(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
        experiment: LowbitsKernel,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinear)?;
        self.check_f32_buffer(input)?;
        if codes.code_layout != experiment.layout() {
            return Err(ExecutorError::InvalidArgument(
                "experiment and physical code layout disagree",
            ));
        }
        let (width, inner, format) = self.check_packed_layout(codes, scales)?;
        let shape = input.descriptor.layout.shape();
        if shape.rank() != 2 || shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "experimental projection input width differs",
            ));
        }
        let rows = shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, width])?)?;
        if experiment == LowbitsKernel::Baseline {
            return self.packed_linear(output, input, codes, scales);
        }
        let (kernel, tile_m, tile_n) = experiment.kernel(format)?;
        if rows < 96 {
            if codes.code_layout != CodeLayout::Canonical {
                return Err(ExecutorError::Unsupported(
                    "repacked research kernels require at least 96 rows",
                ));
            }
            return self.packed_linear(output, input, codes, scales);
        }
        let columns = width.div_ceil(tile_n);
        let groups = rows
            .div_ceil(tile_m)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow("experimental grid overflows u64"))?;
        let x = input.wgpu_buffer()?;
        self.device.dispatch(
            kernel,
            &[
                output.wgpu_buffer()?,
                x,
                codes.wgpu_buffer()?,
                scales.wgpu_buffer()?,
                x,
            ],
            &params(&[
                param32(rows)?,
                param32(width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }
}
