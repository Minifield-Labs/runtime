//! Checked finite operations. Shader indexing follows the admitted geometry.
use crate::{
    Bucket, MetalBackend, MetalBuffer, MetalFence, MetalFenceRetirement, MetalReadback,
    encode_words,
    kernels::Kernel,
    p,
    packed::{Format, Mode},
    product,
};
use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendLease, DType, ExecutorError,
    FenceRetirement, GatedShortConvSpec, GqaSpec, InferenceOps, PackedHeadSpec, RectCopy2d,
    ResourceReport, Result, RotarySpec, Shape, TokenId, TokenIds,
};
use std::rc::Rc;

impl MetalBackend {
    fn shape(b: &MetalBuffer) -> Shape {
        b.descriptor.layout.shape()
    }
    fn equal(&self, output: &MetalBuffer, a: &MetalBuffer, b: &MetalBuffer) -> Result<u64> {
        self.check(a, DType::F32)?;
        self.check(b, DType::F32)?;
        if Self::shape(a) != Self::shape(b) {
            return Err(ExecutorError::InvalidShape(
                "Metal elementwise shapes differ",
            ));
        }
        self.output(output, Self::shape(a))?;
        Self::distinct(output, &[a, b])?;
        Self::shape(a).element_count()
    }
    fn packed(&self, codes: &MetalBuffer, scales: &MetalBuffer) -> Result<(u64, u64, Format)> {
        self.check(codes, DType::U8)?;
        self.check(scales, DType::F32)?;
        let c = Self::shape(codes);
        let s = Self::shape(scales);
        if c.rank() != 2 || s.rank() != 2 || c.dim(0)? != s.dim(0)? || s.dim(1)? == 0 {
            return Err(ExecutorError::InvalidShape(
                "Metal packed codes/scales shapes differ",
            ));
        }
        let width = product(s.dim(1)?, 128)?;
        let bytes = c.dim(1)?;
        let format = Format::from_code_width(bytes, width)?;
        Ok((c.dim(0)?, width, format))
    }
    fn packed_input(
        &self,
        output: &MetalBuffer,
        input: &MetalBuffer,
        codes: &MetalBuffer,
        scales: &MetalBuffer,
    ) -> Result<(u64, u64, u64, Format)> {
        let (rows, width, format) = self.packed(codes, scales)?;
        self.check(input, DType::F32)?;
        let shape = Self::shape(input);
        if shape.rank() != 2 || shape.dim(1)? != width {
            return Err(ExecutorError::InvalidShape(
                "Metal packed input width differs",
            ));
        }
        let tokens = shape.dim(0)?;
        self.output(output, Shape::new(&[tokens, rows])?)?;
        Self::distinct(output, &[input, codes, scales])?;
        Ok((tokens, rows, width, format))
    }
    #[allow(clippy::float_cmp)] // Integer selectors must round-trip exactly.
    fn ids<'a>(&self, ids: &TokenIds<'a, Self>, count: u64, rows: u64) -> Result<MetalIds<'a>> {
        match ids {
            TokenIds::Host(ids) => {
                if u64::try_from(ids.len())
                    .map_err(|_| ExecutorError::Overflow("Metal selector count overflows"))?
                    != count
                {
                    return Err(ExecutorError::InvalidShape(
                        "Metal row selector count differs",
                    ));
                }
                if ids.iter().any(|&id| u64::from(id) >= rows) {
                    return Err(ExecutorError::OutOfBounds(
                        "Metal row selector exceeds table",
                    ));
                }
                #[allow(clippy::cast_precision_loss)]
                let values: Vec<_> = ids.iter().map(|&id| id as f32).collect();
                if ids
                    .iter()
                    .zip(&values)
                    .any(|(&id, &value)| f64::from(id) != f64::from(value))
                {
                    return Err(ExecutorError::ResourceLimit(
                        "Metal row selector exceeds exact F32 index range",
                    ));
                }
                Ok(MetalIds::Staged(self.stage_f32(&values)?))
            }
            TokenIds::Device(buffer) => {
                self.check(buffer, DType::F32)?;
                if Self::shape(buffer) != Shape::new(&[count])? {
                    return Err(ExecutorError::InvalidShape(
                        "Metal device selectors have wrong shape",
                    ));
                }
                Ok(MetalIds::Borrowed(buffer))
            }
        }
    }
    fn rms(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        rows: u64,
        width: u64,
        epsilon: f32,
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        self.check(weight, DType::F32)?;
        self.output(output, Self::shape(input))?;
        Self::distinct(output, &[input, weight])?;
        if Self::shape(weight) != Shape::new(&[width])? {
            return Err(ExecutorError::InvalidShape(
                "Metal RMS weight has wrong shape",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "Metal RMS epsilon must be finite and positive",
            ));
        }
        self.device.dispatch(
            Kernel::RmsNorm,
            &[output, input, weight],
            &[p(rows)?, p(width)?, epsilon.to_bits()],
            rows,
        )
    }
    fn new_scratch(&self, shape: Shape) -> Result<MetalBuffer> {
        self.allocate(shape, DType::F32, AllocationClass::Scratch, None)
    }
    #[allow(clippy::too_many_arguments)] // Finite attention operands and geometry.
    pub(crate) fn attention(
        &self,
        output: &mut MetalBuffer,
        query: &MetalBuffer,
        key: &MetalBuffer,
        value: &MetalBuffer,
        segments: &MetalBuffer,
        tokens: u64,
        key_tokens: u64,
        offset: u64,
        spec: GqaSpec,
        bidirectional: bool,
    ) -> Result<()> {
        let heads = u64::from(spec.query_heads().heads());
        let scores = self.new_scratch(Shape::new(&[product(tokens, heads)?, key_tokens])?)?;
        self.device.dispatch(
            Kernel::Attention,
            &[output, query, key, value, &scores, segments],
            &[
                p(tokens)?,
                p(heads)?,
                spec.query_heads().head_dim(),
                spec.key_value_heads().heads(),
                p(key_tokens)?,
                p(offset)?,
                u32::from(bidirectional),
            ],
            product(tokens, heads)?,
        )
    }
}
enum MetalIds<'a> {
    Staged(MetalBuffer),
    Borrowed(&'a MetalBuffer),
}
impl MetalIds<'_> {
    fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Staged(b) => b,
            Self::Borrowed(b) => b,
        }
    }
}

impl InferenceOps for MetalBackend {
    type Buffer = MetalBuffer;
    type Fence = MetalFence;
    type Readback = MetalReadback;
    type FenceRetirement = MetalFenceRetirement;
    fn identity(&self) -> BackendIdentity {
        self.device.identity.get()
    }
    fn lease(&self) -> BackendLease {
        self.device.lease.with_identity(self.identity())
    }
    fn fence_retirement(&self) -> Rc<MetalFenceRetirement> {
        Rc::clone(&self.retirement)
    }
    fn poll_retired_fences(&self) -> Result<()> {
        self.device.reap();
        self.retirement.poll_retired();
        Ok(())
    }
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }
    fn resource_report(&self) -> ResourceReport {
        self.device.reap();
        self.device.accounting.borrow().report
    }
    fn peak_accounted_bytes(&self) -> Result<u64> {
        Ok(self.device.accounting.borrow().peak)
    }
    fn advance_generation(&mut self) -> Result<()> {
        let mut identity = self.identity();
        identity.generation = identity
            .generation
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("Metal generation exhausted"))?;
        self.device.identity.set(identity);
        Ok(())
    }
    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<MetalBuffer> {
        self.allocate(shape, DType::F32, class, None)
    }
    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<MetalBuffer> {
        if shape.element_count()?
            != u64::try_from(values.len())
                .map_err(|_| ExecutorError::Overflow("Metal upload count overflows"))?
        {
            return Err(ExecutorError::InvalidShape(
                "Metal F32 upload count differs",
            ));
        }
        self.preflight_allocation(shape, DType::F32)?;
        let bytes = encode_words(values.iter().map(|value| value.to_bits()))?;
        self.allocate(shape, DType::F32, class, Some(&bytes))
    }
    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<MetalBuffer> {
        if shape.element_count()?
            != u64::try_from(bytes.len())
                .map_err(|_| ExecutorError::Overflow("Metal upload count overflows"))?
        {
            return Err(ExecutorError::InvalidShape("Metal U8 upload count differs"));
        }
        self.allocate(shape, DType::U8, class, Some(bytes))
    }
    fn fence(&self) -> Result<MetalFence> {
        Ok(MetalFence::new(
            Rc::clone(&self.device),
            self.device.submit()?,
        ))
    }
    fn read_f32_async(&self, buffer: &MetalBuffer) -> Result<MetalReadback> {
        self.check(buffer, DType::F32)?;
        let bytes = buffer.descriptor.layout.byte_extent();
        let staging = self.device.alloc(bytes, Bucket::Pending, None)?;
        let reserved = self.device.accounting.borrow_mut().reserve_buffers(
            Bucket::Pending,
            &[bytes, bytes],
            self.device.limits,
        )?;
        let recorded = self.device.record(
            |command| {
                self.device.raw.copy(
                    command,
                    &staging.raw,
                    &buffer.allocation.raw,
                    usize::try_from(bytes).map_err(|_| {
                        ExecutorError::ResourceLimit("Metal copy exceeds address space")
                    })?,
                )
            },
            &[Rc::clone(&staging), Rc::clone(&buffer.allocation)],
        );
        let submission = match recorded.and_then(|()| self.device.submit()) {
            Ok(s) => s,
            Err(error) => {
                self.device
                    .accounting
                    .borrow_mut()
                    .subtract(Bucket::Pending, reserved);
                return Err(error);
            }
        };
        Ok(MetalReadback::new(
            Rc::clone(&self.device),
            submission,
            staging,
            bytes,
            reserved,
        ))
    }
    fn copy(&self, output: &mut MetalBuffer, input: &MetalBuffer) -> Result<()> {
        self.check(input, input.descriptor.layout.dtype())?;
        self.check(output, input.descriptor.layout.dtype())?;
        if Self::shape(input) != Self::shape(output) {
            return Err(ExecutorError::InvalidShape("Metal copy shapes differ"));
        }
        Self::distinct(output, &[input])?;
        let bytes = input.descriptor.layout.byte_extent();
        if bytes == 0 {
            return Ok(());
        }
        self.device.record(
            |command| {
                self.device.raw.copy(
                    command,
                    &output.allocation.raw,
                    &input.allocation.raw,
                    usize::try_from(bytes).map_err(|_| {
                        ExecutorError::ResourceLimit("Metal copy exceeds address space")
                    })?,
                )
            },
            &[Rc::clone(&output.allocation), Rc::clone(&input.allocation)],
        )
    }
    fn copy_rect_2d(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        r: RectCopy2d,
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        self.check(output, DType::F32)?;
        Self::distinct(output, &[input])?;
        r.validate(Self::shape(input), Self::shape(output))?;
        self.device.dispatch(
            Kernel::RectCopy,
            &[output, input],
            &[
                p(r.rows())?,
                p(r.columns())?,
                p(r.destination_row())?,
                p(Self::shape(output).dim(1)?)?,
                p(r.destination_column())?,
                p(r.source_row())?,
                p(Self::shape(input).dim(1)?)?,
                p(r.source_column())?,
            ],
            product(r.rows(), r.columns())?,
        )
    }
    fn argmax(&self, output: &mut MetalBuffer, input: &MetalBuffer) -> Result<()> {
        self.argmax_impl(output, input, None)
    }
    fn argmax_masked(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        mask: &[u64],
    ) -> Result<()> {
        self.argmax_impl(output, input, Some(mask))
    }
    fn gather_rows(
        &self,
        output: &mut MetalBuffer,
        table: &MetalBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check(table, DType::F32)?;
        let s = Self::shape(table);
        let o = Self::shape(output);
        if s.rank() != 2 || o.rank() != 2 || o.dim(1)? != s.dim(1)? {
            return Err(ExecutorError::InvalidShape("Metal gather shape mismatch"));
        }
        self.output(output, o)?;
        Self::distinct(output, &[table])?;
        let selectors = self.ids(&ids, o.dim(0)?, s.dim(0)?)?;
        Self::distinct(output, &[selectors.buffer()])?;
        self.device.dispatch(
            Kernel::Gather,
            &[output, table, selectors.buffer()],
            &[p(o.dim(0)?)?, p(s.dim(1)?)?, p(s.dim(0)?)?],
            o.element_count()?,
        )
    }
    fn gather_columns(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        columns: &[TokenId],
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        let s = Self::shape(input);
        if s.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "Metal column gather requires rank two",
            ));
        }
        let count = u64::try_from(columns.len())
            .map_err(|_| ExecutorError::Overflow("Metal column count overflows"))?;
        self.output(output, Shape::new(&[s.dim(0)?, count])?)?;
        Self::distinct(output, &[input])?;
        let width = s.dim(1)?;
        if columns.iter().any(|&id| u64::from(id) >= width) {
            return Err(ExecutorError::OutOfBounds(
                "Metal column selector exceeds width",
            ));
        }
        let ids = self.stage_u32(columns)?;
        self.device.dispatch(
            Kernel::Columns,
            &[output, input, &ids],
            &[p(s.dim(0)?)?, p(count)?, p(s.dim(1)?)?],
            product(s.dim(0)?, count)?,
        )
    }
    fn packed_gather_rows(
        &self,
        output: &mut MetalBuffer,
        codes: &MetalBuffer,
        scales: &MetalBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        let (rows, width, format) = self.packed(codes, scales)?;
        let o = Self::shape(output);
        if o.rank() != 2 || o.dim(1)? != width {
            return Err(ExecutorError::InvalidShape(
                "Metal packed gather output shape differs",
            ));
        }
        self.output(output, o)?;
        Self::distinct(output, &[codes, scales])?;
        let selectors = self.ids(&ids, o.dim(0)?, rows)?;
        Self::distinct(output, &[selectors.buffer()])?;
        self.device.dispatch(
            Kernel::PackedGather,
            &[output, codes, scales, selectors.buffer()],
            &[p(o.dim(0)?)?, p(width)?, p(rows)?, format.id()],
            o.element_count()?,
        )
    }
    fn packed_linear(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        codes: &MetalBuffer,
        scales: &MetalBuffer,
    ) -> Result<()> {
        let (tokens, rows, width, format) = self.packed_input(output, input, codes, scales)?;
        self.device.dispatch_packed(
            Mode::Single,
            [tokens, rows, width],
            &[output, input, codes, scales, input],
            &[p(tokens)?, p(rows)?, p(width)?, format.id(), 0],
        )
    }
    fn packed_linear_pair(
        &self,
        out_a: &mut MetalBuffer,
        out_b: &mut MetalBuffer,
        input: &MetalBuffer,
        ca: &MetalBuffer,
        sa: &MetalBuffer,
        cb: &MetalBuffer,
        sb: &MetalBuffer,
    ) -> Result<()> {
        let (tokens, rows, width, format_a) = self.packed_input(out_a, input, ca, sa)?;
        let other = self.packed_input(out_b, input, cb, sb)?;
        if (tokens, rows, width) != (other.0, other.1, other.2) {
            return Err(ExecutorError::InvalidShape(
                "Metal packed pair geometry differs",
            ));
        }
        Self::distinct(out_a, &[out_b, cb, sb])?;
        Self::distinct(out_b, &[ca, sa])?;
        self.device.dispatch_packed(
            Mode::Pair,
            [tokens, rows, width],
            &[out_a, out_b, input, ca, sa, cb, sb],
            &[
                p(tokens)?,
                p(rows)?,
                p(width)?,
                format_a.id(),
                other.3.id(),
                0,
            ],
        )
    }
    fn packed_swiglu_linear(
        &self,
        output: &mut MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        codes: &MetalBuffer,
        scales: &MetalBuffer,
    ) -> Result<()> {
        let (tokens, rows, width, format) = self.packed_input(output, gate, codes, scales)?;
        self.check(up, DType::F32)?;
        if Self::shape(gate) != Self::shape(up) {
            return Err(ExecutorError::InvalidShape("Metal SwiGLU inputs differ"));
        }
        Self::distinct(output, &[up])?;
        self.device.dispatch_packed(
            Mode::InputSwiGlu,
            [tokens, rows, width],
            &[output, gate, codes, scales, up],
            &[p(tokens)?, p(rows)?, p(width)?, format.id(), 1],
        )
    }
    fn packed_swiglu_pair(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        ca: &MetalBuffer,
        sa: &MetalBuffer,
        cb: &MetalBuffer,
        sb: &MetalBuffer,
    ) -> Result<()> {
        let (tokens, rows, width, format_a) = self.packed_input(output, input, ca, sa)?;
        let (rows_b, width_b, format_b) = self.packed(cb, sb)?;
        if rows != rows_b || width != width_b {
            return Err(ExecutorError::InvalidShape(
                "Metal packed pair geometry differs",
            ));
        }
        Self::distinct(output, &[cb, sb])?;
        self.device.dispatch_packed(
            Mode::PairSwiGlu,
            [tokens, rows, width],
            &[output, output, input, ca, sa, cb, sb],
            &[
                p(tokens)?,
                p(rows)?,
                p(width)?,
                format_a.id(),
                format_b.id(),
                1,
            ],
        )
    }
    fn add_row_rms_norm(
        &self,
        sum: &mut MetalBuffer,
        normed: &mut MetalBuffer,
        left: &MetalBuffer,
        right: &MetalBuffer,
        weight: &MetalBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.equal(sum, left, right)?;
        self.output(normed, Self::shape(sum))?;
        Self::distinct(sum, &[normed, weight])?;
        Self::distinct(normed, &[sum, left, right, weight])?;
        // Validate the norm before recording the add, preserving rejected-input state.
        self.check(weight, DType::F32)?;
        let shape = Self::shape(sum);
        if shape.rank() != 2
            || Self::shape(weight) != Shape::new(&[shape.dim(1)?])?
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(ExecutorError::InvalidArgument(
                "Metal fused RMS geometry/epsilon invalid",
            ));
        }
        self.add(sum, left, right)?;
        self.row_rms_norm(normed, sum, weight, epsilon)
    }
    fn qk_norm_rope(
        &self,
        query_out: &mut MetalBuffer,
        key_out: &mut MetalBuffer,
        query: &MetalBuffer,
        key: &MetalBuffer,
        query_weight: &MetalBuffer,
        key_weight: &MetalBuffer,
        positions: &[u64],
        rope: RotarySpec,
        key_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        let key_rope = validate_qk_rotary(rope.heads(), key_heads, rope.theta())?;
        let tokens = rope.heads().validate_packed(Self::shape(query))?;
        if key_heads.validate_packed(Self::shape(key))? != tokens
            || positions.len()
                != usize::try_from(tokens).map_err(|_| {
                    ExecutorError::ResourceLimit("Metal token count exceeds address space")
                })?
        {
            return Err(ExecutorError::InvalidShape(
                "Metal QK positions/count differ",
            ));
        }
        self.check(query, DType::F32)?;
        self.check(key, DType::F32)?;
        self.check(query_weight, DType::F32)?;
        self.check(key_weight, DType::F32)?;
        self.output(query_out, Self::shape(query))?;
        self.output(key_out, Self::shape(key))?;
        if Self::shape(query_weight) != Shape::new(&[u64::from(rope.heads().head_dim())])?
            || Self::shape(key_weight) != Shape::new(&[u64::from(key_heads.head_dim())])?
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(ExecutorError::InvalidArgument(
                "Metal QK norm geometry/epsilon invalid",
            ));
        }
        Self::distinct(query_out, &[key_out, query, key, query_weight, key_weight])?;
        Self::distinct(key_out, &[query, key, query_weight, key_weight])?;
        let mut q = self.new_scratch(Self::shape(query))?;
        let mut k = self.new_scratch(Self::shape(key))?;
        self.head_rms_norm(&mut q, query, query_weight, rope.heads(), epsilon)?;
        self.head_rms_norm(&mut k, key, key_weight, key_heads, epsilon)?;
        self.split_half_rotary(query_out, &q, positions, rope)?;
        self.split_half_rotary(key_out, &k, positions, key_rope)
    }
    fn add(&self, output: &mut MetalBuffer, left: &MetalBuffer, right: &MetalBuffer) -> Result<()> {
        let n = self.equal(output, left, right)?;
        self.device
            .dispatch(Kernel::Elementwise, &[output, left, right], &[p(n)?, 0], n)
    }
    fn multiply(
        &self,
        output: &mut MetalBuffer,
        left: &MetalBuffer,
        right: &MetalBuffer,
    ) -> Result<()> {
        let n = self.equal(output, left, right)?;
        self.device
            .dispatch(Kernel::Elementwise, &[output, left, right], &[p(n)?, 1], n)
    }
    fn linear(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        self.check(weight, DType::F32)?;
        let i = Self::shape(input);
        let w = Self::shape(weight);
        if i.rank() != 2 || w.rank() != 2 || i.dim(1)? != w.dim(1)? {
            return Err(ExecutorError::InvalidShape(
                "Metal linear input/weight geometry differs",
            ));
        }
        self.output(output, Shape::new(&[i.dim(0)?, w.dim(0)?])?)?;
        Self::distinct(output, &[input, weight])?;
        self.device.dispatch(
            Kernel::DenseLinear,
            &[output, input, weight],
            &[p(i.dim(0)?)?, p(w.dim(0)?)?, p(i.dim(1)?)?],
            product(i.dim(0)?, w.dim(0)?)?,
        )
    }
    fn row_rms_norm(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        epsilon: f32,
    ) -> Result<()> {
        let s = Self::shape(input);
        if s.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "Metal row RMS requires rank two",
            ));
        }
        self.rms(output, input, weight, s.dim(0)?, s.dim(1)?, epsilon)
    }
    fn head_rms_norm(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        let tokens = heads.validate_packed(Self::shape(input))?;
        self.rms(
            output,
            input,
            weight,
            product(tokens, u64::from(heads.heads()))?,
            u64::from(heads.head_dim()),
            epsilon,
        )
    }
    fn split_half_rotary(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        let tokens = spec.heads().validate_packed(Self::shape(input))?;
        self.output(output, Self::shape(input))?;
        Self::distinct(output, &[input])?;
        if positions.len()
            != usize::try_from(tokens).map_err(|_| {
                ExecutorError::ResourceLimit("Metal positions count exceeds address space")
            })?
        {
            return Err(ExecutorError::InvalidShape(
                "Metal rotary positions count differs",
            ));
        }
        let half = u64::from(spec.heads().head_dim() / 2);
        let table_len = usize::try_from(product(tokens, half)?).map_err(|_| {
            ExecutorError::ResourceLimit("Metal rotary table exceeds address space")
        })?;
        let size = table_len
            .checked_mul(2)
            .ok_or(ExecutorError::Overflow("Metal rotary table size overflows"))?;
        let mut table = Vec::new();
        table
            .try_reserve_exact(size)
            .map_err(|_| ExecutorError::ResourceLimit("Metal rotary table allocation failed"))?;
        table.resize(size, 0.0_f32);
        // The portable contract rounds frequency and trig at explicit F64-to-F32
        // boundaries. This is parameter preparation; rotation runs on Metal.
        for (token, &position) in positions.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_f32 = position as f32;
            for column in 0..spec.heads().head_dim() / 2 {
                let exponent = -2.0_f64 * f64::from(column) / f64::from(spec.heads().head_dim());
                #[allow(clippy::cast_possible_truncation)]
                let frequency = f64::from(spec.theta()).powf(exponent) as f32;
                let angle = position_f32 * frequency;
                let index = token
                    * usize::try_from(half).map_err(|_| {
                        ExecutorError::ResourceLimit("Metal rotary half exceeds address space")
                    })?
                    + usize::try_from(column).map_err(|_| {
                        ExecutorError::ResourceLimit("Metal rotary column exceeds address space")
                    })?;
                #[allow(clippy::cast_possible_truncation)]
                {
                    table[index] = f64::from(angle).cos() as f32;
                    table[table_len + index] = f64::from(angle).sin() as f32;
                }
            }
        }
        let staged = self.stage_f32(&table)?;
        self.device.dispatch(
            Kernel::Rotary,
            &[output, input, &staged],
            &[
                p(tokens)?,
                spec.heads().heads(),
                spec.heads().head_dim(),
                spec.theta().to_bits(),
            ],
            product(
                product(tokens, u64::from(spec.heads().heads()))?,
                u64::from(spec.heads().head_dim() / 2),
            )?,
        )
    }
    fn causal_gqa(
        &self,
        output: &mut MetalBuffer,
        query: &MetalBuffer,
        key: &MetalBuffer,
        value: &MetalBuffer,
        key_cache: &mut MetalBuffer,
        value_cache: &mut MetalBuffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()> {
        for buffer in [query, key, value, &*key_cache, &*value_cache] {
            self.check(buffer, DType::F32)?;
        }
        let tokens = spec.query_heads().validate_packed(Self::shape(query))?;
        let width = spec.key_value_heads().packed_width()?;
        for buffer in [key, value] {
            if spec
                .key_value_heads()
                .validate_packed(Self::shape(buffer))?
                != tokens
            {
                return Err(ExecutorError::InvalidShape("Metal GQA token counts differ"));
            }
        }
        if Self::shape(key_cache) != Self::shape(value_cache) {
            return Err(ExecutorError::InvalidShape("Metal KV cache shapes differ"));
        }
        let capacity = spec
            .key_value_heads()
            .validate_packed(Self::shape(key_cache))?;
        let total = cache_len
            .checked_add(tokens)
            .ok_or(ExecutorError::Overflow("Metal KV length overflows"))?;
        if total > capacity {
            return Err(ExecutorError::OutOfBounds(
                "Metal KV cache capacity exceeded",
            ));
        }
        self.output(output, Self::shape(query))?;
        Self::distinct(output, &[query, key, value, key_cache, value_cache])?;
        Self::distinct(key_cache, &[query, key, value, value_cache])?;
        Self::distinct(value_cache, &[query, key, value])?;
        let segments = self.stage_u32(&[1])?;
        // Scores must be admitted before either cache is changed.
        let scores = self.new_scratch(Shape::new(&[
            product(tokens, u64::from(spec.query_heads().heads()))?,
            total,
        ])?)?;
        self.copy_rect_2d(
            key_cache,
            key,
            RectCopy2d::new(0, 0, *cache_len, 0, tokens, width),
        )?;
        self.copy_rect_2d(
            value_cache,
            value,
            RectCopy2d::new(0, 0, *cache_len, 0, tokens, width),
        )?;
        self.device.dispatch(
            Kernel::Attention,
            &[output, query, key_cache, value_cache, &scores, &segments],
            &[
                p(tokens)?,
                spec.query_heads().heads(),
                spec.query_heads().head_dim(),
                spec.key_value_heads().heads(),
                p(total)?,
                p(*cache_len)?,
                0,
            ],
            product(tokens, u64::from(spec.query_heads().heads()))?,
        )?;
        *cache_len = total;
        Ok(())
    }
    fn gated_short_convolution(
        &self,
        output: &mut MetalBuffer,
        projection: &MetalBuffer,
        kernel: &MetalBuffer,
        history: &mut MetalBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check(projection, DType::F32)?;
        self.check(kernel, DType::F32)?;
        self.check(history, DType::F32)?;
        let hidden = u64::from(spec.hidden());
        let width = u64::from(spec.width());
        let s = Self::shape(projection);
        if s.rank() != 2
            || s.dim(1)? != product(hidden, 3)?
            || Self::shape(kernel) != Shape::new(&[hidden, width])?
            || Self::shape(history) != Shape::new(&[width - 1, hidden])?
        {
            return Err(ExecutorError::InvalidShape(
                "Metal short convolution geometry differs",
            ));
        }
        if s.dim(0)?
            .checked_add(width)
            .is_none_or(|n| n > i32::MAX as u64)
        {
            return Err(ExecutorError::ResourceLimit(
                "Metal convolution signed tap range exceeded",
            ));
        }
        self.output(output, Shape::new(&[s.dim(0)?, hidden])?)?;
        Self::distinct(output, &[projection, kernel, history])?;
        Self::distinct(history, &[projection, kernel])?;
        let mut snapshot = self.new_scratch(Self::shape(history))?;
        self.copy(&mut snapshot, history)?;
        self.device.dispatch(
            Kernel::CausalConv,
            &[output, projection, kernel, &snapshot],
            &[p(s.dim(0)?)?, spec.hidden(), spec.width()],
            product(s.dim(0)?, hidden)?,
        )?;
        self.device.dispatch(
            Kernel::ConvHistory,
            &[history, projection, &snapshot],
            &[p(s.dim(0)?)?, spec.hidden(), spec.width() - 1],
            product(width - 1, hidden)?,
        )
    }
    fn swiglu(&self, output: &mut MetalBuffer, gate: &MetalBuffer, up: &MetalBuffer) -> Result<()> {
        let n = self.equal(output, gate, up)?;
        self.device
            .dispatch(Kernel::Elementwise, &[output, gate, up], &[p(n)?, 2], n)
    }
}
impl MetalBackend {
    fn argmax_impl(
        &self,
        output: &mut MetalBuffer,
        input: &MetalBuffer,
        mask: Option<&[u64]>,
    ) -> Result<()> {
        self.check(input, DType::F32)?;
        let s = Self::shape(input);
        if s.rank() != 2 || s.dim(1)? > (1 << 24) {
            return Err(ExecutorError::InvalidShape(
                "Metal argmax requires rank two and exact F32 width",
            ));
        }
        self.output(output, Shape::new(&[s.dim(0)?])?)?;
        Self::distinct(output, &[input])?;
        let words = if let Some(mask) = mask {
            if u64::try_from(mask.len())
                .map_err(|_| ExecutorError::Overflow("Metal mask count overflows"))?
                != s.dim(1)?.div_ceil(64)
            {
                return Err(ExecutorError::InvalidShape(
                    "Metal argmax mask length differs",
                ));
            }
            mask.iter()
                .flat_map(|v| {
                    [
                        u32::try_from(v & 0xffff_ffff).unwrap_or(0),
                        u32::try_from(v >> 32).unwrap_or(0),
                    ]
                })
                .collect::<Vec<_>>()
        } else {
            vec![0]
        };
        let staged = self.stage_u32(&words)?;
        self.device.dispatch(
            Kernel::ArgmaxRows,
            &[output, input, &staged],
            &[p(s.dim(0)?)?, p(s.dim(1)?)?, u32::from(mask.is_some())],
            s.dim(0)?,
        )
    }
}

/// Both operands share rotary dimensions; reject the complete fused operation
/// before it creates scratch or records any command.
pub(crate) fn validate_qk_rotary(
    query: PackedHeadSpec,
    key: PackedHeadSpec,
    theta: f32,
) -> Result<RotarySpec> {
    if query.head_dim() != key.head_dim() {
        return Err(ExecutorError::InvalidShape(
            "Metal query/key head dimensions differ",
        ));
    }
    RotarySpec::new(key, theta)
}
