//! Sampling operation admission and dispatch.

use super::{
    ExecutorError, Kernel, OperationKind, PooledBuf, Result, Shape, TokenId, TokenIds, WgpuBackend,
    WgpuBuffer, element_groups, flat_grid, param32, params,
};

impl WgpuBackend {
    /// Gather selected rows from a contiguous [rows, columns] f32 table.
    /// Resolve gather row selectors into a device f32 id buffer.
    ///
    /// Host ids are bounds-checked, staged as exact f32 integers into pooled
    /// scratch, and returned with the scratch allocation to defer. Device ids
    /// bind the caller's buffer directly (typically an `argmax` output), so a
    /// sampled token feeds embedding without a host roundtrip; the kernel
    /// writes NaN rows for invalid selectors.
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn stage_token_ids(
        &self,
        ids: &TokenIds<'_, Self>,
        rows: u64,
    ) -> Result<(wgpu::Buffer, u64, Option<PooledBuf>)> {
        match ids {
            TokenIds::Host(ids) => {
                for id in *ids {
                    if u64::from(*id) >= rows || *id >= (1 << 24) {
                        return Err(ExecutorError::OutOfBounds(
                            "gather identifier exceeds row count",
                        ));
                    }
                }
                // ids is host data: stage it into a pooled scratch buffer that
                // stays alive until this batch's submission is confirmed.
                let id_bytes = u64::try_from(ids.len())
                    .map_err(|_| ExecutorError::Overflow("gather ids overflow u64"))?
                    .checked_mul(4)
                    .ok_or(ExecutorError::Overflow(
                        "gather ids byte count overflows u64",
                    ))?;
                let scratch = self.device.alloc_storage(id_bytes)?;
                let mut bytes = Vec::with_capacity(ids.len() * 4);
                for id in *ids {
                    bytes.extend_from_slice(&(*id as f32).to_le_bytes());
                }
                self.device.queue.write_buffer(&scratch.buffer, 0, &bytes);
                let count = u64::try_from(ids.len())
                    .map_err(|_| ExecutorError::Overflow("id count overflows u64"))?;
                Ok((scratch.buffer.clone(), count, Some(scratch)))
            }
            TokenIds::Device(buffer) => {
                self.check_f32_buffer(buffer)?;
                let shape = buffer.descriptor.layout.shape();
                if shape.rank() != 1 {
                    return Err(ExecutorError::InvalidShape(
                        "device token-id buffer must be rank one",
                    ));
                }
                Ok((buffer.wgpu_buffer()?.clone(), shape.dim(0)?, None))
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn gather_rows(
        &self,
        output: &mut WgpuBuffer,
        table: &WgpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherRows)?;
        self.check_f32_buffer(table)?;
        let table_shape = table.descriptor.layout.shape();
        if table_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("gather table must be rank two"));
        }
        let rows = table_shape.dim(0)?;
        let columns = table_shape.dim(1)?;
        let (id_buffer, id_count, staged) = self.stage_token_ids(&ids, rows)?;
        let output_shape = Shape::new(&[id_count, columns])?;
        self.check_output_shape(output, output_shape)?;
        if id_count == 0 || columns == 0 {
            return Ok(());
        }
        let groups = element_groups(id_count.checked_mul(columns).ok_or(
            ExecutorError::Overflow("gather element count overflows u64"),
        )?);
        let source = table.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::Gather,
            &[&source, &id_buffer, &destination],
            &params(&[
                param32(id_count)?,
                param32(columns)?,
                param32(rows)?,
                f32::NAN.to_bits(),
            ]),
            flat_grid(groups)?,
        );
        if let Some(scratch) = staged {
            drop(scratch);
        }
        result
    }

    /// Gather selected columns of a contiguous [rows, width] f32 input:
    /// `output[r, k] = input[r, columns[k]]`. The host u32 selector list is
    /// range-checked, staged once into pooled scratch, and consumed by a
    /// single element-parallel dispatch covering `rows * columns.len()`.
    pub fn gather_columns(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        columns: &[TokenId],
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherColumns)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "column gather input must be rank two",
            ));
        }
        let rows = input_shape.dim(0)?;
        let width = input_shape.dim(1)?;
        for id in columns {
            if u64::from(*id) >= width {
                return Err(ExecutorError::OutOfBounds(
                    "column gather identifier exceeds input width",
                ));
            }
        }
        let count = u64::try_from(columns.len())
            .map_err(|_| ExecutorError::Overflow("column count overflows u64"))?;
        self.check_output_shape(output, Shape::new(&[rows, count])?)?;
        if rows == 0 || count == 0 {
            return Ok(());
        }
        let id_bytes = count.checked_mul(4).ok_or(ExecutorError::Overflow(
            "gather ids byte count overflows u64",
        ))?;
        let scratch = self.device.alloc_storage(id_bytes)?;
        let mut bytes = Vec::with_capacity(columns.len() * 4);
        for id in columns {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        self.device.queue.write_buffer(&scratch.buffer, 0, &bytes);
        let groups = element_groups(rows.checked_mul(count).ok_or(ExecutorError::Overflow(
            "column gather element count overflows u64",
        ))?);
        let source = input.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::GatherColumns,
            &[&source, &scratch.buffer, &destination],
            &params(&[param32(rows)?, param32(count)?, param32(width)?]),
            flat_grid(groups)?,
        );
        drop(scratch);
        result
    }

    /// Row-wise argmax over f32 `[T, V]` logits: `output[t]` is the index of
    /// the first strict maximum in row `t` as an exact f32 integer, or NaN
    /// when the row contains any non-finite element. The output feeds the
    /// f32-id gather path directly, so greedy decode keeps token selection on
    /// device. `V` must be at most `1 << 24` so indices stay exactly
    /// representable.
    pub fn argmax(&self, output: &mut WgpuBuffer, input: &WgpuBuffer) -> Result<()> {
        self.argmax_impl(output, input, None)
    }

    pub(super) fn argmax_impl(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        mask: Option<&PooledBuf>,
    ) -> Result<()> {
        self.check_operation(OperationKind::Argmax)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("argmax input must be rank two"));
        }
        let rows = input_shape.dim(0)?;
        let columns = input_shape.dim(1)?;
        if columns == 0 || columns > (1 << 24) {
            return Err(ExecutorError::Unsupported(
                "argmax width must be in [1, 2^24] for exact f32 indices",
            ));
        }
        self.check_output_shape(output, Shape::new(&[rows])?)?;
        if rows == 0 {
            return Ok(());
        }
        let source = input.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        // The unmasked path rebinds `source` at the `allow` slot (both are
        // read-only, so the alias is legal) and clears the mask flag.
        let (mask_binding, use_mask) = match mask {
            Some(scratch) => (scratch.buffer.clone(), 1_u32),
            None => (source.clone(), 0_u32),
        };
        if columns <= 2048 {
            return self.device.dispatch(
                Kernel::Argmax,
                &[&destination, &source, &mask_binding],
                &params(&[
                    param32(rows)?,
                    param32(columns)?,
                    use_mask,
                    f32::NAN.to_bits(),
                ]),
                flat_grid(rows)?,
            );
        }
        // Wide rows split into 2048-element blocks so a single workgroup does
        // not serialize the whole scan: stage 1 writes one (value, index)
        // partial per block, stage 2 reduces them per row.
        let blocks = columns.div_ceil(2048);
        let partials = self.device.alloc_storage(
            rows.checked_mul(blocks)
                .and_then(|count| count.checked_mul(8))
                .ok_or(ExecutorError::Overflow(
                    "argmax partials size overflows u64",
                ))?,
        )?;
        let result = self.device.dispatch(
            Kernel::ArgmaxBlocks,
            &[&partials.buffer, &source, &mask_binding],
            &params(&[
                param32(rows)?,
                param32(columns)?,
                param32(blocks)?,
                f32::NAN.to_bits(),
                use_mask,
            ]),
            flat_grid(
                rows.checked_mul(blocks)
                    .ok_or(ExecutorError::Overflow("argmax block count overflows u64"))?,
            )?,
        );
        if result.is_err() {
            drop(partials);
            return result;
        }
        let result = self.device.dispatch(
            Kernel::ArgmaxFinal,
            &[&destination, &partials.buffer],
            &params(&[
                param32(rows)?,
                param32(columns)?,
                param32(blocks)?,
                f32::NAN.to_bits(),
            ]),
            flat_grid(rows)?,
        );
        drop(partials);
        result
    }

    /// Masked variant of [`Self::argmax`]: only positions whose bit is set in
    /// `mask` (LSB-first u64 words, `ceil(columns / 64)` long) are candidates.
    /// Masked-out values are skipped entirely, so their NaN or infinity cannot
    /// poison the row; a row with no allowed candidate yields NaN.
    pub fn argmax_masked(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        mask: &[u64],
    ) -> Result<()> {
        let columns = input.descriptor.layout.shape().dim(1).unwrap_or(0);
        let words = usize::try_from(columns)
            .map_err(|_| ExecutorError::Overflow("argmax width exceeds usize"))?
            .div_ceil(64);
        if mask.len() != words {
            return Err(ExecutorError::InvalidArgument(
                "argmax mask length must be ceil(width / 64)",
            ));
        }
        let bytes = u64::try_from(mask.len() * 8)
            .map_err(|_| ExecutorError::Overflow("argmax mask bytes overflow u64"))?;
        let scratch = self.device.alloc_storage(bytes)?;
        let mut raw = Vec::with_capacity(mask.len() * 8);
        for word in mask {
            raw.extend_from_slice(&word.to_le_bytes());
        }
        self.device.queue.write_buffer(&scratch.buffer, 0, &raw);
        let result = self.argmax_impl(output, input, Some(&scratch));
        drop(scratch);
        result
    }
}
