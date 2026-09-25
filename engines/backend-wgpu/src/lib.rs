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
//! F32 only. Ternary kernels land separately.

#![forbid(unsafe_code)]
// The shared ExecutorError taxonomy documents operation failures.
#![allow(clippy::missing_errors_doc)]

mod completion;
mod device;
mod kernels;

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
}

/// An owned f32 device buffer. Its storage cannot be used by another backend
/// owner or generation, and dropping it quarantines the pooled allocation
/// until any in-flight submission that could reference it is complete.
pub struct WgpuBuffer {
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
            self.device.defer_free(storage);
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
    lowbits_experiment: std::cell::RefCell<Option<String>>,
}

impl WgpuBackend {
    /// Create a backend on the best-ranked adapter. `ordinal` selects among
    /// adapters ranked discrete > integrated > virtual > CPU.
    pub fn new(owner: u64, limits: ResourceLimits) -> Result<Self> {
        Self::on_adapter(owner, 0, limits)
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
        let device = DeviceInner::new_async(owner, ordinal, limits).await?;
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
            lowbits_experiment: std::cell::RefCell::new(None),
        }
    }

    /// Select a low-bit grouped-arithmetic experiment for `packed_linear` at
    /// m >= 96, replacing the `MINI_LOWBITS_EXPERIMENT` env knob for this
    /// backend. `None` clears the override; returns false for unknown names.
    /// Lookup variants reinterpret the ternary code stream, so callers must
    /// upload weights packed in the matching encoding.
    pub fn set_lowbits_experiment(&self, name: Option<&str>) -> bool {
        match name {
            None => {
                self.lowbits_experiment.replace(None);
                true
            }
            Some(n) if kernels::LOWBITS_EXPERIMENTS.contains(&n) => {
                self.lowbits_experiment.replace(Some(n.to_string()));
                true
            }
            Some(_) => false,
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
                .saturating_sub(tracker.pending_retained_by_class.scratch),
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
            tracker
                .limits
                .validate_allocation(layout.byte_extent(), tracker_owned_bytes(&tracker)?)?;
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
        let elements = shape.element_count()?;
        if zero_fill && elements > 0 {
            self.device.dispatch(
                Kernel::Fill,
                &[&storage.buffer],
                &params(&[param32(elements)?, 0.0_f32.to_bits()]),
                flat_grid(element_groups(elements))?,
            )?;
        }
        Ok(WgpuBuffer {
            descriptor: BufferDescriptor {
                backend: identity,
                allocation,
                layout,
                access: BufferAccess::ReadWrite,
            },
            class,
            storage: Some(storage),
            device: Rc::clone(&self.device),
        })
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
            self.device.queue.write_buffer(&storage.buffer, 0, bytes);
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
        self.charge_completion(bytes)?;
        let staging = self.device.alloc_staging(bytes.max(4));
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

    /// Gather selected rows from a contiguous [rows, columns] f32 table.
    /// Resolve gather row selectors into a device f32 id buffer.
    ///
    /// Host ids are bounds-checked, staged as exact f32 integers into pooled
    /// scratch, and returned with the scratch allocation to defer. Device ids
    /// bind the caller's buffer directly (typically an `argmax` output), so a
    /// sampled token feeds embedding without a host roundtrip; the kernel
    /// writes NaN rows for invalid selectors.
    #[allow(clippy::cast_precision_loss)]
    fn stage_token_ids(
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
            self.device.defer_free(scratch);
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
        self.device.defer_free(scratch);
        result
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

    fn elementwise(
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

    /// Validate packed weight operands and return (rows, inner weight width,
    /// stream format). `scales` is always `[rows, k/128]`, so `k` comes from
    /// the scales width; the codes width then selects the decode
    /// unambiguously: `k/4` bytes is `minifield.ternary.v1`, `k/2` bytes is
    /// `minifield.nf4.v1`.
    fn check_packed_operands(
        &self,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<(u64, u64, PackedStreamFormat)> {
        self.check_u8_buffer(codes)?;
        self.check_f32_buffer(scales)?;
        let codes_shape = codes.descriptor.layout.shape();
        let scales_shape = scales.descriptor.layout.shape();
        if codes_shape.rank() != 2 || scales_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed weight streams must be rank two",
            ));
        }
        let code_width = codes_shape.dim(1)?;
        let inner = scales_shape
            .dim(1)?
            .checked_mul(128)
            .ok_or(ExecutorError::Overflow("packed inner width overflows u64"))?;
        let format = if code_width == inner / 4 {
            PackedStreamFormat::TernaryV1
        } else if code_width == inner / 2 {
            PackedStreamFormat::Nf4V1
        } else {
            return Err(ExecutorError::InvalidShape(
                "packed code width is neither inner/4 (ternary) nor inner/2 (nf4)",
            ));
        };
        if scales_shape.dim(0)? != codes_shape.dim(0)? {
            return Err(ExecutorError::InvalidShape(
                "packed codes and scales disagree on row count",
            ));
        }
        Ok((codes_shape.dim(0)?, inner, format))
    }

    /// Packed linear: input [m, k] times the dequantized weight [n, k]
    /// carried as packed codes and per-128-group scales (`minifield.ternary.v1`
    /// or `minifield.nf4.v1`, inferred from the codes width). One workgroup
    /// reduces each output element.
    pub fn packed_linear(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinear)?;
        self.check_f32_buffer(input)?;
        let (output_width, inner, format) = self.check_packed_operands(codes, scales)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed linear input must be rank two",
            ));
        }
        let rows = input_shape.dim(0)?;
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed linear input width differs from weight width",
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
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed linear output count overflows u64",
        ))?;
        // `x4` rebinds the activation as vec4 for 128-bit loads when the inner
        // dimension is 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        // Multi-token tiles amortize each decode across the input tile;
        // m == 1 keeps the single-token kernel. Ternary shares the same
        // 64x32 K16 GEMM template through its own decode header.
        if rows >= 96 {
            // MINI_LOWBITS_EXPERIMENT or set_lowbits_experiment swaps the
            // production tile for a grouped-arithmetic variant with its own
            // tile geometry; the caller is responsible for matching weight
            // encodings.
            let mode = self
                .lowbits_experiment
                .borrow()
                .clone()
                .or_else(|| std::env::var("MINI_LOWBITS_EXPERIMENT").ok());
            if let Some((kernel, tile_m, tile_n)) =
                mode.and_then(|m| kernels::lowbits_gemm_kernel(format, &m))
            {
                let columns = output_width.div_ceil(tile_n);
                let groups = rows
                    .div_ceil(tile_m)
                    .checked_mul(columns)
                    .ok_or(ExecutorError::Overflow("lowbits grid overflows u64"))?;
                return self.device.dispatch(
                    kernel,
                    &[&destination, &x, &c, &s, &x],
                    &params(&[
                        param32(rows)?,
                        param32(output_width)?,
                        param32(inner)?,
                        param32(columns)?,
                    ]),
                    flat_grid(groups)?,
                );
            }
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedGemmNf4,
                PackedStreamFormat::TernaryV1 => Kernel::PackedGemmTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow("packed prefill grid overflows u64"))?;
            return self.device.dispatch(
                kernel,
                &[&destination, &x, &c, &s, &x],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed linear workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedGemvMtNf4,
                &[&destination, &x, &c, &s, &x],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemv,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemvNf4,
        };
        self.device.dispatch(
            kernel,
            &[&destination, &x, &c, &s, &x],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Packed gather: dequantize the selected weight rows (ternary or NF4,
    /// inferred from the codes width) into an f32 [ids, k] output.
    #[allow(clippy::needless_pass_by_value)]
    pub fn packed_gather_rows(
        &self,
        output: &mut WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedGatherRows)?;
        let (rows, inner, format) = self.check_packed_operands(codes, scales)?;
        let (id_buffer, id_count, staged) = self.stage_token_ids(&ids, rows)?;
        let output_shape = Shape::new(&[id_count, inner])?;
        self.check_output_shape(output, output_shape)?;
        if id_count == 0 || inner == 0 {
            return Ok(());
        }
        let groups = element_groups(id_count.checked_mul(inner).ok_or(ExecutorError::Overflow(
            "gather element count overflows u64",
        ))?);
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGather,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGatherNf4,
        };
        let result = self.device.dispatch(
            kernel,
            &[&destination, &id_buffer, &c, &s],
            &params(&[
                param32(id_count)?,
                param32(inner)?,
                param32(rows)?,
                f32::NAN.to_bits(),
            ]),
            flat_grid(groups)?,
        );
        if let Some(scratch) = staged {
            self.device.defer_free(scratch);
        }
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

    fn argmax_impl(
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
            self.device.defer_free(partials);
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
        self.device.defer_free(partials);
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
        self.device.defer_free(scratch);
        result
    }

    /// Paired packed ternary linear over one shared input: workgroup (i, j)
    /// computes both output elements, halving dispatches for projection pairs
    /// that share an activation (K/V, gate/up).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn packed_linear_pair(
        &self,
        out_a: &mut WgpuBuffer,
        out_b: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinearPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_operands(codes_a, scales_a)?;
        // Equal code and scale widths imply the same packed stream format.
        self.check_packed_operands(codes_b, scales_b)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair input must be rank two",
            ));
        }
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed linear pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        let output_shape = Shape::new(&[rows, output_width])?;
        self.check_output_shape(out_a, output_shape)?;
        self.check_output_shape(out_b, output_shape)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(out_a.wgpu_buffer()?, out_a.byte_len());
            self.device
                .record_clear(out_b.wgpu_buffer()?, out_b.byte_len());
            return Ok(());
        }
        let da = out_a.wgpu_buffer()?.clone();
        let db = out_b.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed linear pair output count overflows u64",
        ))?;
        // `x4` rebinds the activation as vec4 for 128-bit loads when the inner
        // dimension is 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        if rows >= 96 {
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedGemmPairNf4,
                PackedStreamFormat::TernaryV1 => Kernel::PackedGemmPairTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "packed pair prefill grid overflows u64",
                ))?;
            return self.device.dispatch(
                kernel,
                &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed linear pair workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedGemvPairMtNf4,
                &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemvPair,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemvPairNf4,
        };
        self.device.dispatch(
            kernel,
            &[&da, &db, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Packed linear over an on-the-fly `SiLU(gate) * up` activation (ternary
    /// or NF4, inferred from the codes width): the down projection consumes
    /// the activation without a materialized intermediate tensor.
    pub fn packed_swiglu_linear(
        &self,
        output: &mut WgpuBuffer,
        gate: &WgpuBuffer,
        up: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedSwigluLinear)?;
        self.check_f32_buffer(gate)?;
        self.check_f32_buffer(up)?;
        let (output_width, inner, format) = self.check_packed_operands(codes, scales)?;
        let gate_shape = gate.descriptor.layout.shape();
        if gate_shape.rank() != 2 || up.descriptor.layout.shape() != gate_shape {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU gate and up layouts must match and be rank two",
            ));
        }
        if gate_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU input width differs from weight width",
            ));
        }
        let rows = gate_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let g = gate.wgpu_buffer()?.clone();
        let u = up.wgpu_buffer()?.clone();
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let lanes = gemv_lanes(output_width);
        let tiles = gemv_tiles(output_width, lanes);
        let workgroups = rows.checked_mul(tiles).ok_or(ExecutorError::Overflow(
            "packed SwiGLU output count overflows u64",
        ))?;
        // `gate4`/`up4` rebind the operands as vec4 when the inner dimension is
        // 4-aligned; the kernel flag falls back to scalar reads.
        let vec_ok = u32::from(inner % 4 == 0);
        if rows >= 96 {
            let kernel = match format {
                PackedStreamFormat::Nf4V1 => Kernel::PackedSwigluGemmNf4,
                PackedStreamFormat::TernaryV1 => Kernel::PackedSwigluGemmTernary,
            };
            let columns = output_width.div_ceil(32);
            let groups = rows
                .div_ceil(64)
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "packed SwiGLU prefill grid overflows u64",
                ))?;
            return self.device.dispatch(
                kernel,
                &[&destination, &g, &u, &c, &s, &g, &u],
                &params(&[
                    param32(rows)?,
                    param32(output_width)?,
                    param32(inner)?,
                    param32(columns)?,
                ]),
                flat_grid(groups)?,
            );
        }
        if format == PackedStreamFormat::Nf4V1 && rows > 1 {
            let m_tiles = rows.div_ceil(8);
            let mt_workgroups = m_tiles.checked_mul(tiles).ok_or(ExecutorError::Overflow(
                "packed SwiGLU workgroup count overflows u64",
            ))?;
            return self.device.dispatch(
                Kernel::PackedSwigluGemvMtNf4,
                &[&destination, &g, &u, &c, &s, &g, &u],
                &params(&[
                    param32(output_width)?,
                    param32(inner)?,
                    param32(lanes)?,
                    param32(tiles)?,
                    param32(rows)?,
                    param32(m_tiles)?,
                    vec_ok,
                    0,
                ]),
                flat_grid(mt_workgroups)?,
            );
        }
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedSwigluGemv,
            PackedStreamFormat::Nf4V1 => Kernel::PackedSwigluGemvNf4,
        };
        self.device.dispatch(
            kernel,
            &[&destination, &g, &u, &c, &s, &g, &u],
            &params(&[
                param32(output_width)?,
                param32(inner)?,
                param32(lanes)?,
                param32(tiles)?,
                vec_ok,
                0,
                0,
                0,
            ]),
            flat_grid(workgroups)?,
        )
    }

    /// Paired packed projection with the SwiGLU epilogue inside the GEMM:
    /// `output[t, r] = silu(a[t, r]) * b[t, r]`. Each invocation already owns
    /// the finished gate and up fragments, so one hidden buffer replaces the
    /// separate gate/up materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn packed_swiglu_pair(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedSwigluPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_operands(codes_a, scales_a)?;
        self.check_packed_operands(codes_b, scales_b)?;
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair input must be rank two",
            ));
        }
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed SwiGLU pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let kernel = match format {
            PackedStreamFormat::TernaryV1 => Kernel::PackedGemmPairSwiglu,
            PackedStreamFormat::Nf4V1 => Kernel::PackedGemmPairSwigluNf4,
        };
        let columns = output_width.div_ceil(32);
        let groups = rows
            .div_ceil(64)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow("SwiGLU pair grid overflows u64"))?;
        self.device.dispatch(
            kernel,
            &[&destination, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }

    /// Whether this backend consumes LUT2-repacked ternary code streams via
    /// `packed_linear_lut2` and `packed_swiglu_pair_lut2`.
    pub fn supports_ternary_lut2(&self) -> bool {
        true
    }

    /// Rearrange a `minifield.ternary.v1` code stream into the LUT2 pair
    /// nibble layout, once at load. The returned buffer is only meaningful
    /// to the `*_lut2` ops; the raw stream stays resident for the short-row
    /// fallbacks that decode it directly.
    pub fn repack_ternary_lut2(&mut self, codes: &WgpuBuffer) -> Result<WgpuBuffer> {
        self.check_u8_buffer(codes)?;
        let shape = codes.descriptor.layout.shape();
        if shape.rank() != 2 || shape.dim(1)? % 4 != 0 {
            return Err(ExecutorError::InvalidShape(
                "lut2 repack expects rank-two u8 codes with a word-aligned width",
            ));
        }
        let out = self.allocate_inner(shape, DType::U8, codes.class, false)?;
        let source = codes.wgpu_buffer()?.clone();
        let destination = out.wgpu_buffer()?.clone();
        let words = shape
            .element_count()?
            .checked_div(4)
            .ok_or(ExecutorError::Overflow("lut2 repack word count overflows"))?;
        self.device.dispatch(
            Kernel::RepackTernaryLut2,
            &[&source, &destination],
            &params(&[param32(words)?, 0, 0, 0]),
            flat_grid(element_groups(words))?,
        )?;
        Ok(out)
    }

    /// Packed linear over LUT2-repacked ternary codes through the 32x64
    /// grouped-lookup tile. Same operand contract as `packed_linear`; the
    /// codes buffer must be the `repack_ternary_lut2` image of a ternary
    /// stream, not the raw stream.
    pub fn packed_linear_lut2(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes: &WgpuBuffer,
        scales: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedLinear)?;
        self.check_f32_buffer(input)?;
        let (output_width, inner, format) = self.check_packed_operands(codes, scales)?;
        if format != PackedStreamFormat::TernaryV1 {
            return Err(ExecutorError::InvalidShape(
                "lut2 linear requires a ternary-width code stream",
            ));
        }
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 linear input must be rank two",
            ));
        }
        let rows = input_shape.dim(0)?;
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 linear input width differs from weight width",
            ));
        }
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let x = input.wgpu_buffer()?.clone();
        let c = codes.wgpu_buffer()?.clone();
        let s = scales.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let columns = output_width.div_ceil(64);
        let groups = rows
            .div_ceil(32)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow("lut2 grid overflows u64"))?;
        self.device.dispatch(
            Kernel::PackedGemmTernaryLut2,
            &[&destination, &x, &c, &s, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }

    /// Paired LUT2 projection with the SwiGLU epilogue: one activation table
    /// per K16 tile feeds both the gate and up streams. Codes carry LUT2
    /// pair nibbles; scales remain per-128-group f32.
    #[allow(clippy::too_many_arguments)]
    pub fn packed_swiglu_pair_lut2(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        codes_a: &WgpuBuffer,
        scales_a: &WgpuBuffer,
        codes_b: &WgpuBuffer,
        scales_b: &WgpuBuffer,
    ) -> Result<()> {
        self.check_operation(OperationKind::PackedSwigluPair)?;
        self.check_f32_buffer(input)?;
        if codes_a.descriptor.layout.shape() != codes_b.descriptor.layout.shape()
            || scales_a.descriptor.layout.shape() != scales_b.descriptor.layout.shape()
        {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair requires equal weight shapes",
            ));
        }
        let (output_width, inner, format) = self.check_packed_operands(codes_a, scales_a)?;
        self.check_packed_operands(codes_b, scales_b)?;
        if format != PackedStreamFormat::TernaryV1 {
            return Err(ExecutorError::InvalidShape(
                "lut2 swiglu pair requires ternary-width code streams",
            ));
        }
        let input_shape = input.descriptor.layout.shape();
        if input_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair input must be rank two",
            ));
        }
        if input_shape.dim(1)? != inner {
            return Err(ExecutorError::InvalidShape(
                "packed lut2 SwiGLU pair input width differs from weight width",
            ));
        }
        let rows = input_shape.dim(0)?;
        self.check_output_shape(output, Shape::new(&[rows, output_width])?)?;
        if rows == 0 || output_width == 0 {
            return Ok(());
        }
        if inner == 0 {
            self.device
                .record_clear(output.wgpu_buffer()?, output.byte_len());
            return Ok(());
        }
        let destination = output.wgpu_buffer()?.clone();
        let x = input.wgpu_buffer()?.clone();
        let ca = codes_a.wgpu_buffer()?.clone();
        let sa = scales_a.wgpu_buffer()?.clone();
        let cb = codes_b.wgpu_buffer()?.clone();
        let sb = scales_b.wgpu_buffer()?.clone();
        let columns = output_width.div_ceil(64);
        let groups = rows
            .div_ceil(32)
            .checked_mul(columns)
            .ok_or(ExecutorError::Overflow(
                "lut2 SwiGLU pair grid overflows u64",
            ))?;
        self.device.dispatch(
            Kernel::PackedGemmPairSwigluLut2,
            &[&destination, &x, &ca, &sa, &cb, &sb, &x],
            &params(&[
                param32(rows)?,
                param32(output_width)?,
                param32(inner)?,
                param32(columns)?,
            ]),
            flat_grid(groups)?,
        )
    }

    /// Fused residual add plus row RMS norm: `sum = left + right` written
    /// beside `normed = rmsnorm(sum) * weight` in one workgroup per row.
    pub fn add_row_rms_norm(
        &self,
        sum: &mut WgpuBuffer,
        normed: &mut WgpuBuffer,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
        weight: &WgpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::AddRowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(left)?;
        self.check_f32_buffer(right)?;
        self.check_f32_buffer(weight)?;
        let input_shape = left.descriptor.layout.shape();
        if input_shape.rank() != 2 || right.descriptor.layout.shape() != input_shape {
            return Err(ExecutorError::InvalidShape(
                "add-norm inputs must share a rank-two layout",
            ));
        }
        let ncols = input_shape.dim(1)?;
        if weight.descriptor.layout.shape() != Shape::new(&[ncols])? {
            return Err(ExecutorError::InvalidShape(
                "add-norm weight must have one entry per column",
            ));
        }
        self.check_output_shape(sum, input_shape)?;
        self.check_output_shape(normed, input_shape)?;
        let nrows = input_shape.dim(0)?;
        if nrows == 0 || ncols == 0 {
            return Ok(());
        }
        let sum_buf = sum.wgpu_buffer()?.clone();
        let normed_buf = normed.wgpu_buffer()?.clone();
        let left_buf = left.wgpu_buffer()?.clone();
        let right_buf = right.wgpu_buffer()?.clone();
        let weight_buf = weight.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::AddNorm,
            &[&sum_buf, &normed_buf, &left_buf, &right_buf, &weight_buf],
            &params(&[param32(ncols)?, param32(nrows)?, epsilon.to_bits()]),
            flat_grid(nrows)?,
        )
    }

    /// Fused per-head RMS norm plus split-half rotary for query and key rows:
    /// one workgroup per [token, head] slice. The cos/sin table is the same
    /// host-computed layout `split_half_rotary` stages.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        query_out: &mut WgpuBuffer,
        key_out: &mut WgpuBuffer,
        query: &WgpuBuffer,
        key: &WgpuBuffer,
        query_weight: &WgpuBuffer,
        key_weight: &WgpuBuffer,
        positions: &[u64],
        rope: RotarySpec,
        key_value_heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::QkNormRope)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(query)?;
        self.check_f32_buffer(key)?;
        self.check_f32_buffer(query_weight)?;
        self.check_f32_buffer(key_weight)?;
        if key_value_heads.head_dim() != rope.heads().head_dim() {
            return Err(ExecutorError::InvalidShape(
                "RoPE head dimensions must match for query and key",
            ));
        }
        let query_tokens = rope
            .heads()
            .validate_packed(query.descriptor.layout.shape())?;
        let key_tokens = key_value_heads.validate_packed(key.descriptor.layout.shape())?;
        if query_tokens != key_tokens {
            return Err(ExecutorError::InvalidShape(
                "RoPE query and key token counts differ",
            ));
        }
        let token_count = usize::try_from(query_tokens)
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds usize"))?;
        if positions.len() != token_count {
            return Err(ExecutorError::InvalidShape(
                "RoPE position count differs from token count",
            ));
        }
        self.check_output_shape(query_out, query.descriptor.layout.shape())?;
        self.check_output_shape(key_out, key.descriptor.layout.shape())?;
        let head_dim = u64::from(rope.heads().head_dim());
        if head_dim == 0 || head_dim > 512 {
            return Err(ExecutorError::Unsupported(
                "RoPE head dimension exceeds shared-memory bound",
            ));
        }
        let head_dim_u64 = u64::from(rope.heads().head_dim());
        for weight in [query_weight, key_weight] {
            if weight.descriptor.layout.shape() != Shape::new(&[head_dim_u64])? {
                return Err(ExecutorError::InvalidShape(
                    "RoPE norm weight must have head_dim entries",
                ));
            }
        }
        if query_tokens == 0 {
            return Ok(());
        }
        let table = self.upload_rope_table(positions, rope.heads().head_dim(), rope.theta())?;
        let q_heads = u64::from(rope.heads().heads());
        let kv_heads = u64::from(key_value_heads.heads());
        let heads_total = q_heads
            .checked_add(kv_heads)
            .ok_or(ExecutorError::Overflow("RoPE head count overflows u64"))?;
        let q_width = rope.heads().packed_width()?;
        let kv_width = key_value_heads.packed_width()?;
        let workgroups = query_tokens
            .checked_mul(heads_total)
            .ok_or(ExecutorError::Overflow(
                "RoPE workgroup count overflows u64",
            ))?;
        let t = table.buffer.clone();
        let qo = query_out.wgpu_buffer()?.clone();
        let ko = key_out.wgpu_buffer()?.clone();
        let q = query.wgpu_buffer()?.clone();
        let k = key.wgpu_buffer()?.clone();
        let qw = query_weight.wgpu_buffer()?.clone();
        let kw = key_weight.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::QkNormRope,
            &[&t, &qo, &ko, &q, &k, &qw, &kw],
            &params(&[
                param32(q_heads)?,
                param32(heads_total)?,
                param32(head_dim)?,
                param32(q_width)?,
                param32(kv_width)?,
                param32(query_tokens)?,
                epsilon.to_bits(),
            ]),
            flat_grid(workgroups)?,
        );
        self.device.defer_free(table);
        result
    }

    /// Host-computed split-half cos/sin table staged into a pooled scratch
    /// buffer. Layout: `[tokens * half]` cosines then `[tokens * half]` sines,
    /// evaluated at the same f64-to-f32 boundaries as the scalar reference.
    fn upload_rope_table(&self, positions: &[u64], head_dim: u32, theta: f32) -> Result<PooledBuf> {
        let tokens = u64::try_from(positions.len())
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds u64"))?;
        let half = u64::from(head_dim) / 2;
        let half_usize = usize::try_from(half)
            .map_err(|_| ExecutorError::Overflow("RoPE half width exceeds usize"))?;
        let table_len = usize::try_from(
            tokens
                .checked_mul(half)
                .ok_or(ExecutorError::Overflow("RoPE table size overflows u64"))?,
        )
        .map_err(|_| ExecutorError::Overflow("RoPE table size exceeds usize"))?;
        let mut table = Vec::new();
        table
            .try_reserve_exact(
                table_len
                    .checked_mul(2)
                    .ok_or(ExecutorError::Overflow("RoPE table length overflows usize"))?,
            )
            .map_err(|_| ExecutorError::ResourceLimit("RoPE table allocation failed"))?;
        table.resize(table_len * 2, 0.0_f32);
        for (token, position) in positions.iter().copied().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_f32 = position as f32;
            if !position_f32.is_finite() {
                return Err(ExecutorError::InvalidArgument(
                    "RoPE position is not representable as finite f32",
                ));
            }
            for column in 0..half_usize {
                #[allow(clippy::cast_precision_loss)]
                let exponent = -2.0_f64 * (column as f64) / f64::from(head_dim);
                #[allow(clippy::cast_possible_truncation)]
                let frequency = (f64::from(theta).powf(exponent)) as f32;
                let angle = position_f32 * frequency;
                #[allow(clippy::cast_possible_truncation)]
                let cos = f64::from(angle).cos() as f32;
                #[allow(clippy::cast_possible_truncation)]
                let sin = f64::from(angle).sin() as f32;
                table[token * half_usize + column] = cos;
                table[table_len + token * half_usize + column] = sin;
            }
        }
        let scratch = self.device.alloc_storage(
            u64::try_from(table.len())
                .map_err(|_| ExecutorError::Overflow("RoPE table length overflows u64"))?
                .checked_mul(4)
                .ok_or(ExecutorError::Overflow("RoPE table bytes overflow u64"))?,
        )?;
        let mut bytes = Vec::with_capacity(table.len() * 4);
        for value in &table {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.device.queue.write_buffer(&scratch.buffer, 0, &bytes);
        Ok(scratch)
    }

    /// Row RMS norm over a [rows, width] input and a [width] weight.
    pub fn row_rms_norm(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::RowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let weight_shape = weight.descriptor.layout.shape();
        if input_shape.rank() != 2 || weight_shape.rank() != 1 {
            return Err(ExecutorError::InvalidShape(
                "RMS input must be rank two and weight rank one",
            ));
        }
        let rows = input_shape.dim(0)?;
        let width = input_shape.dim(1)?;
        if width != weight_shape.dim(0)? {
            return Err(ExecutorError::InvalidShape(
                "RMS input width differs from weight width",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        self.rms_norm_dispatch(output, input, weight, rows, width, epsilon)
    }

    /// Head-local RMS normalization over packed `[tokens, heads * head_dim]`.
    pub fn head_rms_norm(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        self.check_operation(OperationKind::RowRmsNorm)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ExecutorError::InvalidArgument(
                "RMS epsilon must be finite and positive",
            ));
        }
        self.check_f32_buffer(input)?;
        self.check_f32_buffer(weight)?;
        let input_shape = input.descriptor.layout.shape();
        let tokens = heads.validate_packed(input_shape)?;
        if weight.descriptor.layout.shape() != Shape::new(&[u64::from(heads.head_dim())])? {
            return Err(ExecutorError::InvalidShape(
                "head RMS weight must have head_dim entries",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let rows = tokens
            .checked_mul(u64::from(heads.heads()))
            .ok_or(ExecutorError::Overflow("head RMS row count overflows u64"))?;
        self.rms_norm_dispatch(
            output,
            input,
            weight,
            rows,
            u64::from(heads.head_dim()),
            epsilon,
        )
    }

    fn rms_norm_dispatch(
        &self,
        output: &WgpuBuffer,
        input: &WgpuBuffer,
        weight: &WgpuBuffer,
        rows: u64,
        width: u64,
        epsilon: f32,
    ) -> Result<()> {
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let source = input.wgpu_buffer()?.clone();
        let alpha = weight.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        self.device.dispatch(
            Kernel::RmsNorm,
            &[&source, &destination, &alpha],
            &params(&[param32(width)?, param32(rows)?, epsilon.to_bits()]),
            flat_grid(rows)?,
        )
    }

    /// Split-half `RoPE` for packed `[tokens, heads * head_dim]` rows. The
    /// cos/sin table is computed on the host with the same f64-to-f32
    /// boundaries as the scalar reference and staged into a scratch buffer.
    pub fn split_half_rotary(
        &self,
        output: &mut WgpuBuffer,
        input: &WgpuBuffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::Rotary)?;
        self.check_f32_buffer(input)?;
        let input_shape = input.descriptor.layout.shape();
        let tokens = spec.heads().validate_packed(input_shape)?;
        let token_count = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("RoPE token count exceeds usize"))?;
        if positions.len() != token_count {
            return Err(ExecutorError::InvalidShape(
                "RoPE position count differs from token count",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let heads = spec.heads().heads();
        let head_dim = spec.heads().head_dim();
        let half = u64::from(head_dim) / 2;
        if tokens == 0 || half == 0 {
            return Ok(());
        }
        let scratch = self.upload_rope_table(positions, head_dim, spec.theta())?;
        let groups = element_groups(
            tokens
                .checked_mul(u64::from(heads))
                .and_then(|value| value.checked_mul(half))
                .ok_or(ExecutorError::Overflow("RoPE element count overflows u64"))?,
        );
        let source = input.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::Rotary,
            &[&scratch.buffer, &source, &destination],
            &params(&[heads, param32(half)?, param32(tokens)?]),
            flat_grid(groups)?,
        );
        self.device.defer_free(scratch);
        result
    }

    /// Append packed K/V rows to the caches and calculate causal GQA output.
    /// One workgroup handles one query head per appended token; the cache
    /// append is a recorded copy inside the same batch.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn causal_gqa(
        &self,
        output: &mut WgpuBuffer,
        query: &WgpuBuffer,
        key: &WgpuBuffer,
        value: &WgpuBuffer,
        key_cache: &mut WgpuBuffer,
        value_cache: &mut WgpuBuffer,
        cache_len: &mut u64,
        spec: GqaSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GroupedQueryAttention)?;
        self.check_f32_buffer(query)?;
        self.check_f32_buffer(key)?;
        self.check_f32_buffer(value)?;
        self.check_f32_buffer(key_cache)?;
        self.check_f32_buffer(value_cache)?;
        if key_cache.descriptor.allocation == value_cache.descriptor.allocation {
            return Err(ExecutorError::InvalidArgument(
                "key and value caches must have distinct storage",
            ));
        }
        if key_cache.descriptor.access != BufferAccess::ReadWrite
            || value_cache.descriptor.access != BufferAccess::ReadWrite
        {
            return Err(ExecutorError::InvalidArgument("GQA caches are read-only"));
        }
        let query_shape = query.descriptor.layout.shape();
        let tokens = spec.query_heads().validate_packed(query_shape)?;
        if spec
            .key_value_heads()
            .validate_packed(key.descriptor.layout.shape())?
            != tokens
            || spec
                .key_value_heads()
                .validate_packed(value.descriptor.layout.shape())?
                != tokens
        {
            return Err(ExecutorError::InvalidShape(
                "GQA query, key, and value token counts differ",
            ));
        }
        self.check_output_shape(output, query_shape)?;
        let cache_shape = key_cache.descriptor.layout.shape();
        if cache_shape.rank() != 2
            || cache_shape.dim(1)? != spec.key_value_heads().packed_width()?
            || value_cache.descriptor.layout.shape() != cache_shape
        {
            return Err(ExecutorError::InvalidShape(
                "GQA caches must have matching [capacity, kv_heads * head_dim] shape",
            ));
        }
        let capacity = cache_shape.dim(0)?;
        let new_cache_len = cache_len
            .checked_add(tokens)
            .ok_or(ExecutorError::Overflow("GQA cache length overflows u64"))?;
        if *cache_len > capacity || new_cache_len > capacity {
            return Err(ExecutorError::OutOfBounds(
                "GQA append exceeds cache capacity",
            ));
        }
        if tokens == 0 {
            *cache_len = new_cache_len;
            return Ok(());
        }
        let query_heads = u64::from(spec.query_heads().heads());
        let head_dim = u64::from(spec.query_heads().head_dim());
        let group_size = u64::from(spec.group_size());
        let query_width = spec.query_heads().packed_width()?;
        let kv_width = spec.key_value_heads().packed_width()?;
        if query_heads > u64::from(MAX_WGS_PER_DIM) {
            return Err(ExecutorError::ResourceLimit(
                "GQA query head count exceeds dispatch grid",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        if !scale.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA scale is non-finite"));
        }

        let q = query.wgpu_buffer()?.clone();
        let k = key.wgpu_buffer()?.clone();
        let v = value.wgpu_buffer()?.clone();
        let kc = key_cache.wgpu_buffer()?.clone();
        let vc = value_cache.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        if tokens == 1 {
            // Score scratch: one row per query head, wide enough for the
            // visible length this token sees.
            let scores = self.device.alloc_storage(
                query_heads
                    .checked_mul(new_cache_len)
                    .and_then(|e| e.checked_mul(4))
                    .ok_or(ExecutorError::Overflow("GQA scratch bytes overflow u64"))?,
            )?;
            self.device.dispatch(
                Kernel::Gqa,
                &[&q, &k, &v, &kc, &vc, &scores.buffer, &destination],
                &params(&[
                    param32(group_size)?,
                    param32(head_dim)?,
                    param32(kv_width)?,
                    param32(query_width)?,
                    param32(*cache_len)?,
                    param32(0)?,
                    param32(new_cache_len)?,
                    scale.to_bits(),
                ]),
                (
                    u32::try_from(query_heads).map_err(|_| {
                        ExecutorError::ResourceLimit("GQA heads exceed dispatch grid")
                    })?,
                    1,
                    1,
                ),
            )?;
            self.device.defer_free(scores);
        } else {
            // Batched path: one dispatch covers a block of tokens, one score
            // row per (head, token) pair. The block caps the scratch at
            // ~64 MiB so very long prompts dispatch a few blocks instead of
            // one huge temporary.
            let budget = 16 * 1024 * 1024_u64; // score elements
            let block = (budget / (query_heads * new_cache_len)).clamp(1, tokens);
            let scores = self.device.alloc_storage(
                block
                    .checked_mul(query_heads)
                    .and_then(|e| e.checked_mul(new_cache_len))
                    .and_then(|e| e.checked_mul(4))
                    .ok_or(ExecutorError::Overflow("GQA scratch bytes overflow u64"))?,
            )?;
            let mut base = 0_u64;
            while base < tokens {
                let block_tokens = (tokens - base).min(block);
                self.device.dispatch(
                    Kernel::GqaBatch,
                    &[&q, &k, &v, &kc, &vc, &scores.buffer, &destination],
                    &params(&[
                        param32(group_size)?,
                        param32(head_dim)?,
                        param32(kv_width)?,
                        param32(query_width)?,
                        param32(*cache_len)?,
                        0,
                        param32(new_cache_len)?,
                        scale.to_bits(),
                        param32(query_heads)?,
                        param32(block_tokens)?,
                        param32(base)?,
                        0,
                    ]),
                    flat_grid(
                        block_tokens
                            .checked_mul(query_heads)
                            .ok_or(ExecutorError::Overflow("GQA batch grid overflows u64"))?,
                    )?,
                )?;
                base += block_tokens;
            }
            self.device.defer_free(scores);
        }
        // Append the new K/V rows to the caches inside the same batch. The GQA
        // kernel reads appended rows from k/v, not the cache tail, so ordering
        // of these copies is unobservable.
        let append_bytes = tokens
            .checked_mul(kv_width)
            .and_then(|value| value.checked_mul(4))
            .ok_or(ExecutorError::Overflow("GQA append bytes overflow u64"))?;
        let cache_offset = cache_len
            .checked_mul(kv_width)
            .and_then(|value| value.checked_mul(4))
            .ok_or(ExecutorError::Overflow("GQA cache offset overflows u64"))?;
        self.device
            .record_copy(&k, 0, &kc, cache_offset, append_bytes);
        self.device
            .record_copy(&v, 0, &vc, cache_offset, append_bytes);
        *cache_len = new_cache_len;
        Ok(())
    }

    /// Apply B*V gated short convolution and update the [width-1, hidden]
    /// rolling U history. The kernel pair stages `history ∥ (B*V)` in scratch,
    /// convolves depthwise, then assembles the new history in the tail of the
    /// same scratch so every buffer copy stays non-overlapping.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn gated_short_convolution(
        &self,
        output: &mut WgpuBuffer,
        projection: &WgpuBuffer,
        kernel: &WgpuBuffer,
        history: &mut WgpuBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatedShortConvolution)?;
        self.check_f32_buffer(projection)?;
        self.check_f32_buffer(kernel)?;
        self.check_f32_buffer(history)?;
        let hidden = u64::from(spec.hidden());
        let projection_width = hidden.checked_mul(3).ok_or(ExecutorError::Overflow(
            "short convolution projection width overflows u64",
        ))?;
        let projection_shape = projection.descriptor.layout.shape();
        if projection_shape.rank() != 2 || projection_shape.dim(1)? != projection_width {
            return Err(ExecutorError::InvalidShape(
                "short convolution projection must be [tokens, 3 * hidden]",
            ));
        }
        let token_shape = Shape::new(&[projection_shape.dim(0)?, hidden])?;
        self.check_output_shape(output, token_shape)?;
        if kernel.descriptor.layout.shape()
            != Shape::new(&[u64::from(spec.hidden()), u64::from(spec.width())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution kernel must be [hidden, width]",
            ));
        }
        let history_rows = spec.history_rows()?;
        if history.descriptor.layout.shape()
            != Shape::new(&[history_rows, u64::from(spec.hidden())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution history must be [width - 1, hidden]",
            ));
        }
        if history.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument(
                "short convolution history is read-only",
            ));
        }
        let tokens = token_shape.dim(0)?;
        let width = u64::from(spec.width());
        if tokens == 0 || hidden == 0 {
            return Ok(());
        }
        let hist_elems = history_rows
            .checked_mul(hidden)
            .ok_or(ExecutorError::Overflow(
                "short conv history size overflows u64",
            ))?;
        if tokens == 1 && history_rows > 0 {
            // Decode step: one fused dispatch computes the gate, conv, and the
            // shifted history in a fresh buffer, then the history buffer swaps
            // storage. This avoids the staged same-buffer copies the general
            // path needs.
            let new_hist = self.device.alloc_storage(hist_elems.checked_mul(4).ok_or(
                ExecutorError::Overflow("short conv history bytes overflow u64"),
            )?)?;
            let hp = projection.wgpu_buffer()?.clone();
            let hk = kernel.wgpu_buffer()?.clone();
            let hh = history.wgpu_buffer()?.clone();
            let destination = output.wgpu_buffer()?.clone();
            if let Err(err) = self.device.dispatch(
                Kernel::ConvStep,
                &[&hh, &hp, &hk, &new_hist.buffer, &destination],
                &params(&[
                    param32(hidden)?,
                    param32(history_rows)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(hidden))?,
            ) {
                drop(new_hist);
                return Err(err);
            }
            if let Some(old) = history.storage.replace(new_hist) {
                self.device.defer_free(old);
            }
            return Ok(());
        }
        let token_elems = tokens.checked_mul(hidden).ok_or(ExecutorError::Overflow(
            "short conv element count overflows u64",
        ))?;
        let ext_elems = hist_elems
            .checked_add(token_elems)
            .ok_or(ExecutorError::Overflow("short conv extent overflows u64"))?;
        // u_ext layout: [history | u]. The assembled replacement history
        // stages in `tail` because wgpu forbids same-buffer copies.
        let u_ext = self.device.alloc_storage(ext_elems.checked_mul(4).ok_or(
            ExecutorError::Overflow("short conv scratch bytes overflow u64"),
        )?)?;
        let tail = if history_rows > 0 && tokens < history_rows {
            Some(self.device.alloc_storage(hist_elems.checked_mul(4).ok_or(
                ExecutorError::Overflow("short conv tail bytes overflow u64"),
            )?)?)
        } else {
            None
        };
        let hp = projection.wgpu_buffer()?.clone();
        let hk = kernel.wgpu_buffer()?.clone();
        let hh = history.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = (|| {
            self.device.dispatch(
                Kernel::ConvGate,
                &[&hh, &hp, &u_ext.buffer],
                &params(&[
                    param32(hist_elems)?,
                    param32(ext_elems)?,
                    param32(hidden)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(ext_elems))?,
            )?;
            self.device.dispatch(
                Kernel::Conv,
                &[&u_ext.buffer, &hk, &hp, &destination],
                &params(&[
                    param32(tokens)?,
                    param32(hidden)?,
                    param32(width)?,
                    param32(projection_width)?,
                ]),
                flat_grid(element_groups(token_elems))?,
            )
        })();
        if result.is_err() {
            drop(u_ext);
            return result;
        }
        // Assemble the new history. u row r lives at u_ext[(hist_rows + r) *
        // hidden + c]. The tokens < history_rows path stages [keep | u] in
        // `tail` because wgpu forbids same-buffer copies.
        if history_rows > 0 {
            let hist_bytes = hist_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                "short conv history bytes overflow u64",
            ))?;
            if tokens >= history_rows {
                self.device.record_copy(
                    &u_ext.buffer,
                    token_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                        "short conv history offset overflows u64",
                    ))?,
                    &hh,
                    0,
                    hist_bytes,
                );
            } else {
                let Some(tail) = &tail else {
                    return Err(ExecutorError::BackendFailure(
                        "short conv tail scratch missing",
                    ));
                };
                let keep = history_rows - tokens;
                let keep_bytes = keep
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(4))
                    .ok_or(ExecutorError::Overflow(
                        "short conv shift bytes overflow u64",
                    ))?;
                self.device.record_copy(
                    &hh,
                    token_elems.checked_mul(4).ok_or(ExecutorError::Overflow(
                        "short conv shift offset overflows u64",
                    ))?,
                    &tail.buffer,
                    0,
                    keep_bytes,
                );
                self.device.record_copy(
                    &u_ext.buffer,
                    hist_bytes,
                    &tail.buffer,
                    keep_bytes,
                    token_elems
                        .checked_mul(4)
                        .ok_or(ExecutorError::Overflow("short conv u bytes overflow u64"))?,
                );
                self.device.record_copy(&tail.buffer, 0, &hh, 0, hist_bytes);
            }
        }
        self.device.defer_free(u_ext);
        if let Some(tail) = tail {
            self.device.defer_free(tail);
        }
        Ok(())
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

impl InferenceOps for WgpuBackend {
    type Buffer = WgpuBuffer;
    type Fence = WgpuFence;
    type Readback = WgpuReadback;
    type FenceRetirement = WgpuFenceRetirement;

    fn identity(&self) -> BackendIdentity {
        WgpuBackend::identity(self)
    }

    fn lease(&self) -> BackendLease {
        WgpuBackend::lease(self)
    }

    fn fence_retirement(&self) -> Rc<Self::FenceRetirement> {
        Rc::clone(&self.retirement)
    }

    fn poll_retired_fences(&self) -> Result<()> {
        self.retirement.poll_retired();
        Ok(())
    }

    fn capabilities(&self) -> BackendCapabilities {
        WgpuBackend::capabilities(self)
    }

    fn resource_report(&self) -> ResourceReport {
        WgpuBackend::resource_report(self)
    }

    fn advance_generation(&mut self) -> Result<()> {
        WgpuBackend::advance_generation(self)
    }

    fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        WgpuBackend::allocate_f32_classified(self, shape, class)
    }

    fn allocate_f32_uninit(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        self.allocate_inner(shape, DType::F32, class, false)
    }

    fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        WgpuBackend::upload_f32_classified(self, shape, values, class)
    }

    fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<Self::Buffer> {
        WgpuBackend::upload_u8_classified(self, shape, bytes, class)
    }

    fn fence(&self) -> Result<Self::Fence> {
        WgpuBackend::fence(self)
    }

    fn read_f32_async(&self, buffer: &Self::Buffer) -> Result<Self::Readback> {
        WgpuBackend::read_f32_async(self, buffer)
    }

    fn copy(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        WgpuBackend::copy(self, output, input)
    }

    fn copy_rect_2d(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        rectangle: RectCopy2d,
    ) -> Result<()> {
        WgpuBackend::copy_rect_2d(self, output, input, rectangle)
    }

    fn gather_rows(
        &self,
        output: &mut Self::Buffer,
        table: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        WgpuBackend::gather_rows(self, output, table, ids)
    }

    fn gather_columns(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        columns: &[TokenId],
    ) -> Result<()> {
        WgpuBackend::gather_columns(self, output, input, columns)
    }

    fn argmax(&self, output: &mut Self::Buffer, input: &Self::Buffer) -> Result<()> {
        WgpuBackend::argmax(self, output, input)
    }

    fn argmax_masked(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        mask: &[u64],
    ) -> Result<()> {
        WgpuBackend::argmax_masked(self, output, input, mask)
    }

    fn add(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::add(self, output, left, right)
    }

    fn multiply(
        &self,
        output: &mut Self::Buffer,
        left: &Self::Buffer,
        right: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::multiply(self, output, left, right)
    }

    fn linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::linear(self, output, input, weight)
    }

    fn packed_linear(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::packed_linear(self, output, input, codes, scales)
    }

    fn packed_gather_rows(
        &self,
        output: &mut Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
        ids: TokenIds<'_, Self>,
    ) -> Result<()> {
        WgpuBackend::packed_gather_rows(self, output, codes, scales, ids)
    }

    fn row_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        epsilon: f32,
    ) -> Result<()> {
        WgpuBackend::row_rms_norm(self, output, input, weight, epsilon)
    }

    fn head_rms_norm(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        heads: PackedHeadSpec,
        epsilon: f32,
    ) -> Result<()> {
        WgpuBackend::head_rms_norm(self, output, input, weight, heads, epsilon)
    }

    fn split_half_rotary(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        positions: &[u64],
        spec: RotarySpec,
    ) -> Result<()> {
        WgpuBackend::split_half_rotary(self, output, input, positions, spec)
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
        WgpuBackend::causal_gqa(
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
        WgpuBackend::gated_short_convolution(self, output, projection, kernel, history, spec)
    }

    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::swiglu(self, output, gate, up)
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
        WgpuBackend::packed_linear_pair(
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
        WgpuBackend::packed_swiglu_linear(self, output, gate, up, codes, scales)
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
        WgpuBackend::packed_swiglu_pair(self, output, input, codes_a, scales_a, codes_b, scales_b)
    }

    fn supports_ternary_lut2(&self) -> bool {
        WgpuBackend::supports_ternary_lut2(self)
    }

    fn repack_ternary_lut2(&mut self, codes: &Self::Buffer) -> Result<Self::Buffer> {
        WgpuBackend::repack_ternary_lut2(self, codes)
    }

    fn packed_linear_lut2(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes: &Self::Buffer,
        scales: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::packed_linear_lut2(self, output, input, codes, scales)
    }

    fn packed_swiglu_pair_lut2(
        &self,
        output: &mut Self::Buffer,
        input: &Self::Buffer,
        codes_a: &Self::Buffer,
        scales_a: &Self::Buffer,
        codes_b: &Self::Buffer,
        scales_b: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::packed_swiglu_pair_lut2(
            self, output, input, codes_a, scales_a, codes_b, scales_b,
        )
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
        WgpuBackend::add_row_rms_norm(self, sum, normed, left, right, weight, epsilon)
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
        WgpuBackend::qk_norm_rope(
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
