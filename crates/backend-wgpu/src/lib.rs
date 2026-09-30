//! Nonblocking wgpu F32 inference operations for the portable executor.
//!
//! `WgpuBackend` implements `InferenceOps` on top of wgpu 30. Portable calls
//! record compute dispatches, copies, and deferred staging maps into one
//! shared `CommandEncoder`; `fence()` and `read_f32_async()` are the only
//! submission boundaries. Completion never blocks: `poll_step` pumps
//! `device.poll(PollType::Poll)` and observes `on_submitted_work_done` and
//! map callbacks.
//!
//! Buffers come from a size-classed pool. A dropped buffer whose batch may
//! still reference it is quarantined until the submission serial that could
//! touch it is confirmed complete, which keeps pool reuse sound without a
//! device-wide wait. Kernel parameters travel through a shared 256-byte-slot
//! uniform ring with dynamic offsets; a ring wrap mid-batch forces a
//! submission because `Queue::write_buffer` ordering is defined only across
//! submission boundaries.
//!
//! Dense F32, packed ternary, and packed NF4 kernels share the same checked API.

#![forbid(unsafe_code)]
// The shared ExecutorError taxonomy documents operation failures.
#![allow(clippy::missing_errors_doc)]

mod attention_convolution;
mod completion;
mod dense;
mod dispatch;
mod normalization;
mod packed;
mod sampling;

mod device;
#[cfg(feature = "experimental-kernels")]
pub mod experimental;
mod kernels;
mod options;

pub use device::DeviceStats;
pub use options::{Nf4Staging, WgpuOptions};

use std::{rc::Rc, sync::Arc};

use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendLease, BufferAccess,
    BufferDescriptor, CompletionPoll, DType, DTypeSet, ExecutorError, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, OperationKind, OperationSet,
    PackedHeadSpec, PrecisionPolicy, RectCopy2d, ResourceLimits, ResourceReport, Result,
    RotarySpec, Shape, TensorLayout, TokenId, TokenIds,
};

pub use completion::{WgpuFence, WgpuFenceRetirement, WgpuReadback};
use device::{DeviceInner, PooledBuf, tracker_owned_bytes};

#[cfg(test)]
mod tests;
use kernels::Kernel;

/// Maximum workgroups a dispatch may use on the required
/// `max_compute_workgroups_per_dimension` axis.
const MAX_WGS_PER_DIM: u32 = 65_535;

/// Workgroups needed for an element-parallel kernel at 256 threads.
fn element_groups(elements: u64) -> u64 {
    elements.div_ceil(u64::from(device::WORKGROUP_SIZE))
}

/// Split a flat workgroup count into (x, y, z) within per-dimension limits.
/// Kernels recover the flat index `wid.z * numw.y * numw.x + wid.y * numw.x +
/// wid.x`, so each dimension's dispatched count participates in the flatten.
fn flat_grid(groups: u64) -> Result<(u32, u32, u32)> {
    let max = u64::from(MAX_WGS_PER_DIM);
    let x = groups.min(max);
    let yz = groups.div_ceil(x.max(1));
    let y = yz.min(max);
    let z = yz.div_ceil(max).max(1);
    if z > max {
        return Err(ExecutorError::ResourceLimit(
            "wgpu dispatch exceeds flattened workgroup capacity",
        ));
    }
    Ok((
        u32::try_from(x).map_err(|_| ExecutorError::Overflow("wgpu grid x overflows u32"))?,
        u32::try_from(y).map_err(|_| ExecutorError::Overflow("wgpu grid y overflows u32"))?,
        u32::try_from(z).map_err(|_| ExecutorError::Overflow("wgpu grid z overflows u32"))?,
    ))
}

/// Lanes per output row for the GEMV-family kernels. A workgroup covers
/// `256 / lanes` rows, so narrow outputs get more lanes per row to keep the
/// workgroup count high enough to fill the GPU; wide outputs use 32-lane
/// groups, which already saturate through row count alone.
fn gemv_lanes(output_width: u64) -> u64 {
    // Measured on M1 Max: narrow outputs need more lanes per row for enough
    // workgroups; wide outputs benefit from more per-lane work (fewer lanes).
    if output_width < 4096 { 16 } else { 8 }
}

/// Workgroup tile count for a GEMV-family dispatch: each workgroup covers
/// `256 / lanes` consecutive output rows.
fn gemv_tiles(output_width: u64, lanes: u64) -> u64 {
    output_width.div_ceil(256 / lanes)
}

/// A u64 dimension as a u32 kernel parameter.
fn param32(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        ExecutorError::ResourceLimit("wgpu kernel dimension exceeds u32 parameter width")
    })
}

/// Serialize kernel parameters as little-endian u32 words.
fn params(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Packed weight stream decode, inferred from the codes/scale widths in
/// `check_packed_operands`. Both formats share one scales layout
/// (`[rows, k/128]` f32) and differ only in code packing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedStreamFormat {
    /// `minifield.ternary.v1`: four 2-bit codes per byte, `k/4` code bytes
    /// per row, `w = (code - 1) * scale`.
    TernaryV1,
    /// `minifield.nf4.v1`: two 4-bit codebook indices per byte (low nibble
    /// first), `k/2` code bytes per row, `w = NF4[code] * scale`.
    Nf4V1,
    /// Signed byte codes, one weight per byte, group-128 scales.
    Int8V1,
}

/// Physical packing of a byte buffer. Repacked codes cannot enter raw kernels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodeLayout {
    #[default]
    Canonical,
    TernaryLut2,
    TernaryPn4,
}

/// An owned f32 device buffer. Its storage cannot be used by another backend
/// owner or generation, and dropping it quarantines the pooled allocation
/// until any in-flight submission that could reference it is complete.
pub struct WgpuBuffer {
    code_layout: CodeLayout,
    descriptor: BufferDescriptor,
    pub(crate) class: AllocationClass,
    storage: Option<PooledBuf>,
    pub(crate) device: Rc<DeviceInner>,
}

impl WgpuBuffer {
    /// Portable descriptor metadata for this buffer.
    #[must_use]
    pub fn descriptor(&self) -> BufferDescriptor {
        self.descriptor
    }

    /// Logical byte extent from the descriptor layout.
    pub(crate) fn byte_len(&self) -> u64 {
        self.descriptor.layout.byte_extent()
    }

    fn wgpu_buffer(&self) -> Result<&wgpu::Buffer> {
        let Some(storage) = &self.storage else {
            return Err(ExecutorError::BackendFailure(
                "wgpu buffer storage was already released",
            ));
        };
        Ok(&storage.buffer)
    }
}

impl Drop for WgpuBuffer {
    fn drop(&mut self) {
        if let Some(storage) = self.storage.take() {
            drop(storage);
        }
        let mut tracker = self.device.tracker.borrow_mut();
        tracker.live_bytes = tracker.live_bytes.saturating_sub(self.byte_len());
        tracker
            .live_by_class
            .saturating_sub(self.class, self.byte_len());
    }
}

/// wgpu 30 backend implementing every `InferenceOps` primitive for F32.
///
/// Construct with [`WgpuBackend::new`]; adapter selection prefers discrete,
/// then integrated, virtual, and CPU adapters.
pub struct WgpuBackend {
    device: Rc<DeviceInner>,
    retirement: Rc<WgpuFenceRetirement>,
    capabilities: BackendCapabilities,
}

impl WgpuBackend {
    /// Create a backend on the best-ranked adapter. `ordinal` selects among
    /// adapters ranked discrete > integrated > virtual > CPU.
    pub fn new(owner: u64, limits: ResourceLimits) -> Result<Self> {
        Self::new_with_options(owner, limits, WgpuOptions::default())
    }

    /// Create a backend with an explicit immutable device policy.
    pub fn new_with_options(
        owner: u64,
        limits: ResourceLimits,
        options: WgpuOptions,
    ) -> Result<Self> {
        pollster::block_on(Self::new_async_with_options(owner, limits, options))
    }

    /// Browser-compatible constructor with the same explicit policy.
    pub async fn new_async_with_options(
        owner: u64,
        limits: ResourceLimits,
        options: WgpuOptions,
    ) -> Result<Self> {
        let device = DeviceInner::new_async(owner, 0, limits, options).await?;
        Ok(Self::with_device(device, limits))
    }

    /// Dispatch counts identify the kernels actually recorded by this backend.
    #[must_use]
    pub fn dispatch_counts(&self) -> std::collections::BTreeMap<&'static str, u64> {
        self.device.dispatch_counts()
    }

    #[must_use]
    pub fn stats(&self) -> DeviceStats {
        *self.device.stats.borrow()
    }

    /// Identity of the adapter actually selected for this backend.
    #[must_use]
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.device.adapter_info.clone()
    }

    /// Create a backend asynchronously on the best-ranked adapter.
    ///
    /// Browser hosts must use this path: WebGPU adapter and device requests
    /// resolve through the JS event loop, so `new`'s blocking executor would
    /// never make progress on wasm32.
    pub async fn new_async(owner: u64, limits: ResourceLimits) -> Result<Self> {
        Self::on_adapter_async(owner, 0, limits).await
    }

    /// Create a backend asynchronously on a specific ranked adapter position.
    pub async fn on_adapter_async(
        owner: u64,
        ordinal: u32,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let device = DeviceInner::new_async(owner, ordinal, limits, WgpuOptions::default()).await?;
        Ok(Self::with_device(device, limits))
    }

    /// Create a backend on a specific ranked adapter position.
    pub fn on_adapter(owner: u64, ordinal: u32, limits: ResourceLimits) -> Result<Self> {
        let device = DeviceInner::new(owner, ordinal, limits)?;
        Ok(Self::with_device(device, limits))
    }

    fn with_device(device: Rc<DeviceInner>, limits: ResourceLimits) -> Self {
        let operations = OperationSet::empty()
            .with(OperationKind::Copy)
            .with(OperationKind::RectCopy2d)
            .with(OperationKind::GatherRows)
            .with(OperationKind::GatherColumns)
            .with(OperationKind::Add)
            .with(OperationKind::Multiply)
            .with(OperationKind::Linear)
            .with(OperationKind::RowRmsNorm)
            .with(OperationKind::Rotary)
            .with(OperationKind::GroupedQueryAttention)
            .with(OperationKind::GatedShortConvolution)
            .with(OperationKind::SwiGlu)
            .with(OperationKind::PackedGatherRows)
            .with(OperationKind::PackedLinear)
            .with(OperationKind::PackedLinearPair)
            .with(OperationKind::PackedSwigluLinear)
            .with(OperationKind::PackedSwigluPair)
            .with(OperationKind::AddRowRmsNorm)
            .with(OperationKind::QkNormRope)
            .with(OperationKind::Argmax);
        let capabilities = BackendCapabilities {
            dtypes: DTypeSet::only(DType::F32).with(DType::U8),
            operations,
            precision: PrecisionPolicy {
                weights: DType::F32,
                activations: DType::F32,
                cache: DType::F32,
                accumulation: DType::F32,
            },
            max_rank: 4,
            max_elements: limits.max_total_bytes / DType::F32.byte_width(),
            // Storage buffers bind entire, so a single allocation must also
            // fit one storage binding.
            max_allocation_bytes: limits
                .max_allocation_bytes
                .min(device.device_limits.max_buffer_size)
                .min(device.device_limits.max_storage_buffer_binding_size),
            supports_nonblocking_completion: true,
        };
        let retirement = Rc::new(WgpuFenceRetirement::new(Rc::clone(&device)));
        Self {
            device,
            retirement,
            capabilities,
        }
    }

    #[must_use]
    pub fn identity(&self) -> BackendIdentity {
        self.device.tracker.borrow().identity
    }

    #[must_use]
    pub fn lease(&self) -> BackendLease {
        let tracker = self.device.tracker.borrow();
        tracker.lease.with_identity(tracker.identity)
    }

    #[must_use]
    pub const fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    #[must_use]
    pub fn peak_accounted_bytes(&self) -> u64 {
        self.device.tracker.borrow().peak_accounted_bytes.get()
    }

    #[must_use]
    pub fn resource_report(&self) -> ResourceReport {
        let tracker = self.device.tracker.borrow();
        ResourceReport {
            resident_weight_bytes: tracker
                .live_by_class
                .weight
                .saturating_sub(tracker.pending_retained_by_class.weight),
            cache_bytes: tracker
                .live_by_class
                .cache
                .saturating_sub(tracker.pending_retained_by_class.cache),
            scratch_bytes: tracker
                .live_by_class
                .scratch
                .saturating_sub(tracker.pending_retained_by_class.scratch)
                .saturating_add(
                    tracker
                        .owned_buffer_bytes
                        .get()
                        .saturating_sub(tracker.live_bytes),
                ),
            staged_branch_bytes: tracker
                .live_by_class
                .branch
                .saturating_sub(tracker.pending_retained_by_class.branch),
            pending_operation_bytes: tracker
                .pending_retained_bytes
                .saturating_add(tracker.pending_result_bytes),
            pending_operations: tracker.pending_operations,
        }
    }

    /// Invalidate old buffers after a backend reload or terminal quarantine.
    pub fn advance_generation(&mut self) -> Result<()> {
        let mut tracker = self.device.tracker.borrow_mut();
        tracker.identity.generation = tracker
            .identity
            .generation
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("backend generation overflows u64"))?;
        tracker.cancellation_requested = false;
        Ok(())
    }

    /// Flag subsequent submissions as cancelled, matching the CPU contract.
    pub fn request_cancel(&mut self) {
        self.device.tracker.borrow_mut().cancellation_requested = true;
    }

    /// Clear a previously requested cancellation.
    pub fn clear_cancel(&mut self) {
        self.device.tracker.borrow_mut().cancellation_requested = false;
    }

    fn check_submit(&self) -> Result<()> {
        if self.device.tracker.borrow().cancellation_requested {
            return Err(ExecutorError::Cancelled);
        }
        Ok(())
    }

    fn check_operation(&self, operation: OperationKind) -> Result<()> {
        self.check_submit()?;
        if !self.capabilities.operations.contains(operation) {
            return Err(ExecutorError::Unsupported(
                "operation is unsupported by wgpu backend",
            ));
        }
        Ok(())
    }

    fn check_buffer(&self, buffer: &WgpuBuffer) -> Result<()> {
        if !Rc::ptr_eq(&buffer.device, &self.device) {
            return Err(ExecutorError::WrongBackend);
        }
        buffer.descriptor.validate_for(self.identity())?;
        if !buffer.descriptor.layout.is_contiguous()? {
            return Err(ExecutorError::InvalidLayout(
                "wgpu backend requires contiguous zero-offset buffers",
            ));
        }
        Ok(())
    }

    /// Structural checks plus an f32 dtype requirement. Ops that read or
    /// write f32 elements must use this so a u8 buffer cannot reach an f32
    /// kernel binding.
    fn check_f32_buffer(&self, buffer: &WgpuBuffer) -> Result<()> {
        self.check_buffer(buffer)?;
        if buffer.descriptor.layout.dtype() != DType::F32 {
            return Err(ExecutorError::InvalidDType("expected an f32 operand"));
        }
        Ok(())
    }

    /// Structural checks plus a u8 dtype requirement, for packed byte
    /// streams.
    fn check_u8_buffer(&self, buffer: &WgpuBuffer) -> Result<()> {
        self.check_buffer(buffer)?;
        if buffer.descriptor.layout.dtype() != DType::U8 {
            return Err(ExecutorError::InvalidDType("expected a u8 operand"));
        }
        Ok(())
    }

    fn check_output_shape(&self, output: &WgpuBuffer, shape: Shape) -> Result<()> {
        self.check_f32_buffer(output)?;
        let expected = TensorLayout::contiguous(DType::F32, shape)?;
        if output.descriptor.layout != expected {
            return Err(ExecutorError::InvalidShape(
                "output layout differs from required contiguous shape",
            ));
        }
        if output.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument("output buffer is read-only"));
        }
        Ok(())
    }

    fn charge_completion(&self, result_bytes: u64) -> Result<()> {
        let mut tracker = self.device.tracker.borrow_mut();
        if tracker.pending_operations >= tracker.limits.max_pending_operations {
            return Err(ExecutorError::ResourceLimit(
                "pending operation count exceeds configured limit",
            ));
        }
        tracker
            .limits
            .validate_allocation(result_bytes, tracker_owned_bytes(&tracker)?)?;
        tracker.pending_operations =
            tracker
                .pending_operations
                .checked_add(1)
                .ok_or(ExecutorError::Overflow(
                    "pending operation count overflows u32",
                ))?;
        tracker.pending_result_bytes = tracker
            .pending_result_bytes
            .checked_add(result_bytes)
            .ok_or(ExecutorError::Overflow(
                "completion result bytes overflow u64",
            ))?;
        tracker.peak_accounted_bytes.set(
            tracker
                .peak_accounted_bytes
                .get()
                .max(tracker_owned_bytes(&tracker)?),
        );
        Ok(())
    }

    /// Allocate a zeroed, writable f32 buffer. The fill is recorded into the
    /// pending batch, so the zeroes exist before any subsequent kernel runs.
    pub fn allocate_f32(&mut self, shape: Shape) -> Result<WgpuBuffer> {
        self.allocate_f32_classified(shape, AllocationClass::Scratch)
    }

    /// Allocate zeroed f32 storage with an explicit logical resource class.
    pub fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<WgpuBuffer> {
        self.allocate_inner(shape, DType::F32, class, true)
    }

    /// Shared allocation path. `zero_fill` records a Fill dispatch; uploads
    /// skip it because `Queue::write_buffer` lands before the next submission,
    /// so a same-batch fill would erase the uploaded data.
    fn allocate_inner(
        &mut self,
        shape: Shape,
        dtype: DType,
        class: AllocationClass,
        zero_fill: bool,
    ) -> Result<WgpuBuffer> {
        self.check_submit()?;
        let layout = TensorLayout::contiguous(dtype, shape)?;
        self.capabilities.validate(
            dtype,
            OperationKind::Copy,
            shape.rank(),
            shape.element_count()?,
            layout.byte_extent(),
        )?;
        let (identity, allocation) = {
            let tracker = self.device.tracker.borrow();
            (tracker.identity, tracker.next_allocation)
        };
        let storage = self.device.alloc_storage(layout.byte_extent())?;
        {
            let mut tracker = self.device.tracker.borrow_mut();
            if tracker.next_allocation != allocation || tracker.identity != identity {
                return Err(ExecutorError::BackendFailure(
                    "wgpu allocator changed during allocation",
                ));
            }
            tracker.next_allocation =
                tracker
                    .next_allocation
                    .checked_add(1)
                    .ok_or(ExecutorError::Overflow(
                        "allocation identifier overflows u64",
                    ))?;
            tracker.live_bytes = tracker
                .live_bytes
                .checked_add(layout.byte_extent())
                .ok_or(ExecutorError::Overflow("wgpu live bytes overflow u64"))?;
            tracker
                .live_by_class
                .checked_add(class, layout.byte_extent())?;
        }
        let output = WgpuBuffer {
            code_layout: CodeLayout::Canonical,
            descriptor: BufferDescriptor {
                backend: identity,
                allocation,
                layout,
                access: BufferAccess::ReadWrite,
            },
            class,
            storage: Some(storage),
            device: Rc::clone(&self.device),
        };
        let elements = shape.element_count()?;
        if zero_fill && elements > 0 {
            self.device.dispatch(
                Kernel::Fill,
                &[output.wgpu_buffer()?],
                &params(&[param32(elements)?, 0.0_f32.to_bits()]),
                flat_grid(element_groups(elements))?,
            )?;
        }
        Ok(output)
    }

    /// Upload finite f32 values as Scratch-class device storage.
    pub fn upload_f32(&mut self, shape: Shape, values: &[f32]) -> Result<WgpuBuffer> {
        self.upload_f32_classified(shape, values, AllocationClass::Scratch)
    }

    /// Upload finite f32 values with an explicit logical resource class.
    pub fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<WgpuBuffer> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err(ExecutorError::InvalidArgument(
                "wgpu upload contains a non-finite value",
            ));
        }
        let expected = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;
        if expected != values.len() {
            return Err(ExecutorError::InvalidShape(
                "upload values differ from shape element count",
            ));
        }
        // No fill: the queue-ordered write below must not race a same-batch
        // zero dispatch.
        let output = self.allocate_inner(shape, DType::F32, class, false)?;
        if !values.is_empty() {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let Some(storage) = &output.storage else {
                return Err(ExecutorError::BackendFailure("wgpu upload storage missing"));
            };
            self.device.queue.write_buffer(&storage.buffer, 0, &bytes);
        }
        Ok(output)
    }

    /// Copy host bytes into an owned device buffer with an explicit resource
    /// class. Packed payload bytes land unmodified; no f32 finite-value check
    /// applies.
    pub fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<WgpuBuffer> {
        let expected = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;
        if expected != bytes.len() {
            return Err(ExecutorError::InvalidShape(
                "upload bytes differ from shape element count",
            ));
        }
        // No fill: the queue-ordered write below must not race a same-batch
        // zero dispatch.
        let output = self.allocate_inner(shape, DType::U8, class, false)?;
        if !bytes.is_empty() {
            let Some(storage) = &output.storage else {
                return Err(ExecutorError::BackendFailure("wgpu upload storage missing"));
            };
            // WebGPU copies are 4-byte aligned. Preserve the logical U8
            // extent and pad only the final physical transfer.
            let aligned = bytes.len() / 4 * 4;
            if aligned > 0 {
                self.device
                    .queue
                    .write_buffer(&storage.buffer, 0, &bytes[..aligned]);
            }
            if aligned < bytes.len() {
                let mut tail = [0_u8; 4];
                tail[..bytes.len() - aligned].copy_from_slice(&bytes[aligned..]);
                self.device
                    .queue
                    .write_buffer(&storage.buffer, aligned as u64, &tail);
            }
        }
        Ok(output)
    }

    /// Submit a completion boundary. Records no work itself; it finishes and
    /// submits the pending batch and returns a serial-keyed fence.
    pub fn fence(&self) -> Result<WgpuFence> {
        self.check_submit()?;
        self.charge_completion(0)?;
        let serial = self.device.submit_pending()?;
        Ok(WgpuFence::new(Rc::clone(&self.device), serial))
    }

    /// Submit a host-visible f32 readback. The staging copy and its deferred
    /// map ride the same submission as every prior recorded operation, so the
    /// result observes the full pending batch.
    pub fn read_f32_async(&self, buffer: &WgpuBuffer) -> Result<WgpuReadback> {
        self.check_submit()?;
        self.check_f32_buffer(buffer)?;
        let bytes = buffer.byte_len();
        let staging = self.device.alloc_staging(bytes.max(4))?;
        self.charge_completion(bytes)?;
        let source = buffer.wgpu_buffer()?.clone();
        self.device
            .record_copy(&source, 0, &staging.buffer, 0, bytes);
        let flag = Arc::new(portable_atomic::AtomicBool::new(false));
        let outcome = Arc::new(std::sync::Mutex::new(None::<bool>));
        {
            let flag = Arc::clone(&flag);
            let outcome = Arc::clone(&outcome);
            self.device
                .map_staging_on_submit(&staging.buffer, move |result| {
                    *outcome
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result.is_ok());
                    flag.store(true, portable_atomic::Ordering::Release);
                });
        }
        self.device.submit_pending()?;
        Ok(WgpuReadback::new(
            Rc::clone(&self.device),
            staging,
            flag,
            outcome,
            bytes,
        ))
    }

    /// Blocking convenience readback for tests and tools; still goes through
    /// the nonblocking completion path.
    pub fn read_f32(&self, buffer: &WgpuBuffer) -> Result<Vec<f32>> {
        let mut readback = self.read_f32_async(buffer)?;
        for _ in 0..600_000_u32 {
            match readback.poll_step() {
                CompletionPoll::Ready(result) => return result,
                CompletionPoll::Pending => std::hint::spin_loop(),
            }
        }
        Err(ExecutorError::BackendFailure(
            "wgpu readback did not complete within the poll budget",
        ))
    }
}
