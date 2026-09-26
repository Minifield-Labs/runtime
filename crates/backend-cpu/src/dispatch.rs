// Portable inference contract forwarding.

use super::{CpuBackend, CpuBuffer, CpuCompletion, CpuFenceRetirement};
use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendLease, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceOps, PackedHeadSpec, RectCopy2d, ResourceReport, Result,
    RotarySpec, Shape, TokenId, TokenIds,
};
use std::rc::Rc;

impl InferenceOps for CpuBackend {
    type Buffer = CpuBuffer;
    type Fence = CpuCompletion<()>;
    type Readback = CpuCompletion<Vec<f32>>;
    type FenceRetirement = CpuFenceRetirement;

    fn identity(&self) -> BackendIdentity {
        CpuBackend::identity(self)
    }

    fn lease(&self) -> BackendLease {
        CpuBackend::lease(self)
    }

    fn fence_retirement(&self) -> Rc<Self::FenceRetirement> {
        Rc::clone(&self.retirement)
    }

    fn poll_retired_fences(&self) -> Result<()> {
        self.retirement.poll_retired();
        Ok(())
    }

    fn capabilities(&self) -> BackendCapabilities {
        CpuBackend::capabilities(self)
    }

    fn resource_report(&self) -> ResourceReport {
        CpuBackend::resource_report(self)
    }

    fn advance_generation(&mut self) -> Result<()> {
        CpuBackend::advance_generation(self)
    }

    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        CpuBackend::allocate_f32_classified(self, shape, class)
    }

    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        CpuBackend::upload_f32_classified(self, shape, values, class)
    }

    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        CpuBackend::upload_u8_classified(self, shape, bytes, class)
    }

    fn fence(&self) -> Result<Self::Fence> {
        CpuBackend::fence(self)
    }

    fn read_f32_async(&self, buffer: &Self::Buffer) -> Result<Self::Readback> {
        CpuBackend::read_f32_async(self, buffer)
    }

    fn copy(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        CpuBackend::copy(self, output, input)
    }

    fn copy_rect_2d(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        CpuBackend::copy_rect_2d(self, output, input, rectangle)
    }

    fn gather_rows(
        &self,
        output: &mut Self::Buffer,
        table: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        CpuBackend::gather_rows(self, output, table, ids)
    }

    fn gather_columns(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        columns: &[TokenId],
    ) -> Result<()> {
        CpuBackend::gather_columns(self, output, input, columns)
    }

    fn packed_gather_rows(
        &self,
        output: &mut Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        CpuBackend::packed_gather_rows(self, output, codes, scales, ids)
    }

    fn argmax(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        CpuBackend::argmax(self, output, input)
    }

    fn argmax_masked(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        mask: &[u64],
    ) -> Result<()> {
        CpuBackend::argmax_masked(self, output, input, mask)
    }

    fn packed_linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::packed_linear(self, output, input, codes, scales)
    }

    fn add(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::add(self, output, left, right)
    }

    fn multiply(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::multiply(self, output, left, right)
    }

    fn linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::linear(self, output, input, weight)
    }

    fn row_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()> {
        CpuBackend::row_rms_norm(self, output, input, weight, epsilon)
    }

    fn head_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        CpuBackend::head_rms_norm(self, output, input, weight, heads, epsilon)
    }

    fn split_half_rotary(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        CpuBackend::split_half_rotary(self, output, input, positions, spec)
    }

    fn causal_gqa(
        &self,
        output: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()> {
        CpuBackend::causal_gqa(
            self,
            output,
            query,
            key,
            value,
            key_cache,
            value_cache,
            cache_len,
            spec,
        )
    }

    fn gated_short_convolution(
        &self,
        output: &mut Self::Buffer,
        projection: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        CpuBackend::gated_short_convolution(self, output, projection, kernel, history, spec)
    }

    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::swiglu(self, output, gate, up)
    }

    fn packed_linear_pair(
        &self,
        out_a: &mut Self::Buffer,
        out_b: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::packed_linear_pair(
            self, out_a, out_b, input, codes_a, scales_a, codes_b, scales_b,
        )
    }

    fn packed_swiglu_linear(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::packed_swiglu_linear(self, output, gate, up, codes, scales)
    }

    fn packed_swiglu_pair(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::packed_swiglu_pair(self, output, input, codes_a, scales_a, codes_b, scales_b)
    }

    fn add_row_rms_norm(
        &self,
        sum: &mut Self::Buffer,
        normed: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()> {
        CpuBackend::add_row_rms_norm(self, sum, normed, left, right, weight, epsilon)
    }

    fn qk_norm_rope(
        &self,
        query_out: &mut Self::Buffer,
        key_out: &mut Self::Buffer,
        query: &Self::Buffer,
        key: &Self::Buffer,
        query_weight: &Self::Buffer,
        key_weight: &Self::Buffer,
        positions: &[u64],
        rope: RotarySpec,
        key_value_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        CpuBackend::qk_norm_rope(
            self,
            query_out,
            key_out,
            query,
            key,
            query_weight,
            key_weight,
            positions,
            rope,
            key_value_heads,
            epsilon,
        )
    }
}
