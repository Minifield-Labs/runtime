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
    RotarySpec, Shape, TensorLayout,
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
}

impl WgpuBackend {
    /// Create a backend on the best-ranked adapter. `ordinal` selects among
    /// adapters ranked discrete > integrated > virtual > CPU.
    pub fn new(owner: u64, limits: ResourceLimits) -> Result<Self> {
        Self::on_adapter(owner, 0, limits)
    }

    /// Create a backend on a specific ranked adapter position.
    pub fn on_adapter(owner: u64, ordinal: u32, limits: ResourceLimits) -> Result<Self> {
        let device = DeviceInner::new(owner, ordinal, limits)?;
        let operations = OperationSet::empty()
            .with(OperationKind::Copy)
            .with(OperationKind::RectCopy2d)
            .with(OperationKind::GatherRows)
            .with(OperationKind::Add)
            .with(OperationKind::Multiply)
            .with(OperationKind::Linear)
            .with(OperationKind::RowRmsNorm)
            .with(OperationKind::Rotary)
            .with(OperationKind::GroupedQueryAttention)
            .with(OperationKind::GatedShortConvolution)
            .with(OperationKind::SwiGlu);
        let capabilities = BackendCapabilities {
            dtypes: DTypeSet::only(DType::F32),
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
        Ok(Self {
            device,
            retirement,
            capabilities,
        })
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
        if buffer.descriptor.layout.dtype() != DType::F32 {
            return Err(ExecutorError::InvalidDType(
                "wgpu backend supports only f32",
            ));
        }
        if !buffer.descriptor.layout.is_contiguous()? {
            return Err(ExecutorError::InvalidLayout(
                "wgpu backend requires contiguous zero-offset buffers",
            ));
        }
        Ok(())
    }

    fn check_output_shape(&self, output: &WgpuBuffer, shape: Shape) -> Result<()> {
        self.check_buffer(output)?;
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
        self.allocate_inner(shape, class, true)
    }

    /// Shared allocation path. `zero_fill` records a Fill dispatch; uploads
    /// skip it because `Queue::write_buffer` lands before the next submission,
    /// so a same-batch fill would erase the uploaded data.
    fn allocate_inner(
        &mut self,
        shape: Shape,
        class: AllocationClass,
        zero_fill: bool,
    ) -> Result<WgpuBuffer> {
        self.check_submit()?;
        let layout = TensorLayout::contiguous(DType::F32, shape)?;
        self.capabilities.validate(
            DType::F32,
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
        let output = self.allocate_inner(shape, class, false)?;
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
        self.check_buffer(buffer)?;
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
        self.check_buffer(input)?;
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
        self.check_buffer(input)?;
        self.check_buffer(output)?;
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
    pub fn gather_rows(
        &self,
        output: &mut WgpuBuffer,
        table: &WgpuBuffer,
        ids: &[u32],
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherRows)?;
        self.check_buffer(table)?;
        let table_shape = table.descriptor.layout.shape();
        if table_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("gather table must be rank two"));
        }
        let rows = table_shape.dim(0)?;
        let columns = table_shape.dim(1)?;
        for id in ids {
            if u64::from(*id) >= rows {
                return Err(ExecutorError::OutOfBounds(
                    "gather identifier exceeds row count",
                ));
            }
        }
        let output_shape = Shape::new(&[
            u64::try_from(ids.len())
                .map_err(|_| ExecutorError::Overflow("id count overflows u64"))?,
            columns,
        ])?;
        self.check_output_shape(output, output_shape)?;
        if ids.is_empty() || columns == 0 {
            return Ok(());
        }
        // ids is host data: stage it into a pooled scratch buffer that stays
        // alive until this batch's submission is confirmed.
        let id_bytes = u64::try_from(ids.len())
            .map_err(|_| ExecutorError::Overflow("gather ids overflow u64"))?
            .checked_mul(4)
            .ok_or(ExecutorError::Overflow(
                "gather ids byte count overflows u64",
            ))?;
        let scratch = self.device.alloc_storage(id_bytes)?;
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        self.device.queue.write_buffer(&scratch.buffer, 0, &bytes);
        let groups = element_groups(
            u64::try_from(ids.len())
                .map_err(|_| ExecutorError::Overflow("gather row count overflows u64"))?
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "gather element count overflows u64",
                ))?,
        );
        let source = table.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = self.device.dispatch(
            Kernel::Gather,
            &[&source, &scratch.buffer, &destination],
            &params(&[
                param32(
                    u64::try_from(ids.len())
                        .map_err(|_| ExecutorError::Overflow("gather row count overflows u64"))?,
                )?,
                param32(columns)?,
            ]),
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
        self.check_buffer(left)?;
        self.check_buffer(right)?;
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
        self.check_buffer(input)?;
        self.check_buffer(weight)?;
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
            self.device.dispatch(
                Kernel::Gemv,
                &[&destination, &x, &w, &w],
                &params(&[param32(output_width)?, param32(inner)?]),
                flat_grid(output_width)?,
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
        self.check_buffer(input)?;
        self.check_buffer(weight)?;
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
        self.check_buffer(input)?;
        self.check_buffer(weight)?;
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
        self.check_buffer(input)?;
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
                let frequency = (f64::from(spec.theta()).powf(exponent)) as f32;
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
        self.check_buffer(query)?;
        self.check_buffer(key)?;
        self.check_buffer(value)?;
        self.check_buffer(key_cache)?;
        self.check_buffer(value_cache)?;
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

        // Score scratch: one row per query head, wide enough for the largest
        // visible length this batch.
        let scratch_elems = query_heads
            .checked_mul(new_cache_len)
            .ok_or(ExecutorError::Overflow("GQA scratch size overflows u64"))?;
        let scores = self.device.alloc_storage(
            scratch_elems
                .checked_mul(4)
                .ok_or(ExecutorError::Overflow("GQA scratch bytes overflow u64"))?,
        )?;
        let q = query.wgpu_buffer()?.clone();
        let k = key.wgpu_buffer()?.clone();
        let v = value.wgpu_buffer()?.clone();
        let kc = key_cache.wgpu_buffer()?.clone();
        let vc = value_cache.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        for token in 0..tokens {
            self.device.dispatch(
                Kernel::Gqa,
                &[&q, &k, &v, &kc, &vc, &scores.buffer, &destination],
                &params(&[
                    param32(group_size)?,
                    param32(head_dim)?,
                    param32(kv_width)?,
                    param32(query_width)?,
                    param32(*cache_len)?,
                    param32(token)?,
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
        }
        self.device.defer_free(scores);
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
        b: &WgpuBuffer,
        c: &WgpuBuffer,
        v: &WgpuBuffer,
        kernel: &WgpuBuffer,
        history: &mut WgpuBuffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        self.check_operation(OperationKind::GatedShortConvolution)?;
        self.check_buffer(b)?;
        self.check_buffer(c)?;
        self.check_buffer(v)?;
        self.check_buffer(kernel)?;
        self.check_buffer(history)?;
        let token_shape = b.descriptor.layout.shape();
        if token_shape.rank() != 2 || token_shape.dim(1)? != u64::from(spec.hidden()) {
            return Err(ExecutorError::InvalidShape(
                "short convolution B must be [tokens, hidden]",
            ));
        }
        if c.descriptor.layout.shape() != token_shape || v.descriptor.layout.shape() != token_shape
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution B, C, and V layouts must match",
            ));
        }
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
        let hidden = u64::from(spec.hidden());
        let width = u64::from(spec.width());
        if tokens == 0 || hidden == 0 {
            return Ok(());
        }
        let hist_elems = history_rows
            .checked_mul(hidden)
            .ok_or(ExecutorError::Overflow(
                "short conv history size overflows u64",
            ))?;
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
        let hb = b.wgpu_buffer()?.clone();
        let hc = c.wgpu_buffer()?.clone();
        let hv = v.wgpu_buffer()?.clone();
        let hk = kernel.wgpu_buffer()?.clone();
        let hh = history.wgpu_buffer()?.clone();
        let destination = output.wgpu_buffer()?.clone();
        let result = (|| {
            self.device.dispatch(
                Kernel::ConvGate,
                &[&hh, &hb, &hv, &u_ext.buffer],
                &params(&[param32(hist_elems)?, param32(ext_elems)?]),
                flat_grid(element_groups(ext_elems))?,
            )?;
            self.device.dispatch(
                Kernel::Conv,
                &[&u_ext.buffer, &hk, &hc, &destination],
                &params(&[param32(tokens)?, param32(hidden)?, param32(width)?]),
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
        self.check_buffer(gate)?;
        self.check_buffer(up)?;
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
        _shape: Shape,
        _bytes: &[u8],
        _class: AllocationClass,
    ) -> Result<Self::Buffer> {
        Err(ExecutorError::Unsupported(
            "u8 packed payloads arrive with the T7b ternary kernels",
        ))
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
        ids: &[u32],
    ) -> Result<()> {
        WgpuBackend::gather_rows(self, output, table, ids)
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
        _output: &mut Self::Buffer,
        _input: &Self::Buffer,
        _codes: &Self::Buffer,
        _scales: &Self::Buffer,
    ) -> Result<()> {
        Err(ExecutorError::Unsupported(
            "packed ternary matvec lands with the T7b wgpu kernels",
        ))
    }

    fn packed_gather_rows(
        &self,
        _output: &mut Self::Buffer,
        _codes: &Self::Buffer,
        _scales: &Self::Buffer,
        _ids: &[u32],
    ) -> Result<()> {
        Err(ExecutorError::Unsupported(
            "packed ternary gather lands with the T7b wgpu kernels",
        ))
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
        b: &Self::Buffer,
        c: &Self::Buffer,
        v: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        WgpuBackend::gated_short_convolution(self, output, b, c, v, kernel, history, spec)
    }

    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()> {
        WgpuBackend::swiglu(self, output, gate, up)
    }
}
