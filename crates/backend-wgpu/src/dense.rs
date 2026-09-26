//! Dense operation admission and dispatch.

use super::{
    BufferAccess, ExecutorError, Kernel, MAX_WGS_PER_DIM, OperationKind, RectCopy2d, Result, Shape,
    WgpuBackend, WgpuBuffer, element_groups, flat_grid, gemv_lanes, gemv_tiles, param32, params,
};

impl WgpuBackend {
    /// Full-buffer copy between distinct or identical contiguous f32 buffers.
    pub fn copy(&self, output: &mut WgpuBuffer, input: &WgpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::Copy)?;
        self.check_f32_buffer(input)?;
        self.check_output_shape(output, input.descriptor.layout.shape())?;
        if output.descriptor.allocation == input.descriptor.allocation {
            return Ok(());
        }
        let bytes = input.byte_len();
        let destination = output.wgpu_buffer()?.clone();
        let source = input.wgpu_buffer()?.clone();
        self.device.record_copy(&source, 0, &destination, 0, bytes);
        Ok(())
    }

    /// Copy one checked row-major rectangle between distinct packed rank-two
    /// buffers.
    pub fn copy_rect_2d(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        self.check_operation(OperationKind::RectCopy2d)?;
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(output)?;
        if input.descriptor.allocation == output.descriptor.allocation {
            return Err(ExecutorError::Unsupported(
                "rectangular copy does not permit overlapping source and destination allocation",
            ));
        }
        if output.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument(
                "rectangular copy destination is read-only",
            ));
        }
        let source_shape = input.descriptor.layout.shape();
        let destination_shape = output.descriptor.layout.shape();
        rectangle.validate(source_shape, destination_shape)?;
        if rectangle.rows() == 0 || rectangle.columns() == 0 {
            return Ok(());
        }
        let source_width = source_shape.dim(1)?;
        let destination_width = destination_shape.dim(1)?;
        let words = [
            param32(rectangle.source_row())?,
            param32(rectangle.source_column())?,
            param32(rectangle.rows())?,
            param32(rectangle.columns())?,
            param32(source_width)?,
            param32(rectangle.destination_row())?,
            param32(destination_width)?,
            param32(rectangle.destination_column())?,
        ];
        let groups = element_groups(rectangle.rows().checked_mul(rectangle.columns()).ok_or(
            ExecutorError::Overflow("rectangular copy count overflows u64"),
        )?);
        let source = input.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::Copy2d,
            &[&source, &destination],
            &params(&words),
            flat_grid(groups)?,
        )
    }

    /// Elementwise add over equal contiguous layouts.
    pub fn add(
        &self,
        output: &mut WgpuBuffer,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
    ) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Add)
    }

    /// Elementwise multiply over equal contiguous layouts.
    pub fn multiply(
        &self,
        output: &mut WgpuBuffer,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
    ) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Multiply)
    }

    pub(super) fn elementwise(
        &self,
        output: &mut WgpuBuffer,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
        operation: OperationKind,
    ) -> Result<()> {
        self.check_operation(operation)?;
        self.check_f32_buffer(left)?;
        self.check_f32_buffer(right)?;
        if left.descriptor.layout != right.descriptor.layout {
            return Err(ExecutorError::InvalidShape(
                "elementwise operands have different layouts",
            ));
        }
        self.check_output_shape(output, left.descriptor.layout.shape())?;
        let elements = left.descriptor.layout.shape().element_count()?;
        if elements == 0 {
            return Ok(());
        }
        let op = match operation {
            OperationKind::Add => 0_u32,
            OperationKind::Multiply => 1_u32,
            _ => {
                return Err(ExecutorError::Unsupported(
                    "operation is not an elementwise wgpu kernel",
                ));
            }
        };
        let lhs = left.wgpu_buffer()?.clone();
        let rhs = right.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::Binary,
            &[&lhs, &rhs, &destination],
            &params(&[param32(elements)?, op]),
            flat_grid(element_groups(elements))?,
        )
    }

    /// Row-major linear projection: input [m, k] times weight [n, k] yields
    /// output [m, n]. m == 1 uses the shared-memory-reduction GEMV; larger m
    /// uses the 16x16 tiled GEMM.
    pub fn linear(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::Linear)?;
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let weight_shape = weight.descriptor.layout.shape();
        if input_shape.rank() != 2 || weight_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "linear input and weight must be rank two",
            ));
        }
        let rows = input_shape.dim(0)?;
        let inner = input_shape.dim(1)?;
        let output_width = weight_shape.dim(0)?;
        if inner != weight_shape.dim(1)? {
            return Err(ExecutorError::InvalidShape(
                "linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[rows, output_width])?;
        self.check_output_shape(output, output_shape)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            // sum over an empty axis is zero.
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let x = input.wgpu_buffer()?.clone();
        let w = weight.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        if rows == 1 {
            // `w4` rebinds the weight as vec4 for 128-bit row loads.
            let lanes = gemv_lanes(output_width);
            self.device.dispatch(
                Kernel::Gemv,
                &[&destination, &x, &w, &w],
                &params(&[param32(output_width)?, param32(inner)?, param32(lanes)?]),
                flat_grid(gemv_tiles(output_width, lanes))?,
            )
        } else {
            let max = u64::from(MAX_WGS_PER_DIM);
            let column_tiles = output_width.div_ceil(16);
            let row_tiles = rows.div_ceil(16);
            if column_tiles > max || row_tiles > max * max {
                return Err(ExecutorError::ResourceLimit(
                    "linear dimensions exceed GEMM grid capacity",
                ));
            }
            let grid = (
                u32::try_from(column_tiles)
                    .map_err(|_| ExecutorError::ResourceLimit("linear width exceeds GEMM grid"))?,
                u32::try_from(row_tiles.min(max))
                    .map_err(|_| ExecutorError::Overflow("GEMM grid y overflows u32"))?,
                u32::try_from(row_tiles.div_ceil(max).max(1))
                    .map_err(|_| ExecutorError::Overflow("GEMM grid z overflows u32"))?,
            );
            self.device.dispatch(
                Kernel::Gemm,
                &[&destination, &x, &w],
                &params(&[param32(rows)?, param32(output_width)?, param32(inner)?]),
                grid,
            )
        }
    }

    /// `SiLU`(gate) * up over matching contiguous f32 layouts.
    pub fn swiglu(
        &self,
        output: &mut WgpuBuffer,
        gate: &WgpuBuffer,
        up: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::SwiGlu)?;
        self.check_f32_buffer(gate)?;
        self.check_f32_buffer(up)?;
        if gate.descriptor.layout != up.descriptor.layout {
            return Err(ExecutorError::InvalidShape(
                "SwiGLU gate and up layouts differ",
            ));
        }
        self.check_output_shape(output, gate.descriptor.layout.shape())?;
        let elements = gate.descriptor.layout.shape().element_count()?;
        if elements == 0 {
            return Ok(());
        }
        let g = gate.wgpu_buffer()?.clone();
        let u = up.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::SwiGlu,
            &[&g, &u, &destination],
            &params(&[param32(elements)?]),
            flat_grid(element_groups(elements))?,
        )
    }
}
