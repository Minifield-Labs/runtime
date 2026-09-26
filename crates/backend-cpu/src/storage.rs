// Owned storage, resource accounting, and operation admission.

use super::CpuFenceRetirement;
use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendKind, BackendLease, BufferAccess,
    BufferDescriptor, CompletionPoll, DType, DTypeSet, ExecutorError, InferenceCompletion,
    OperationKind, OperationSet, PrecisionPolicy, ResourceLimits, ResourceReport, Result, Shape,
    TensorLayout,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ClassBytes {
    pub(super) weight: u64,
    pub(super) cache: u64,
    pub(super) scratch: u64,
    pub(super) branch: u64,
}

impl ClassBytes {
    pub(super) fn checked_add(&mut self, class: AllocationClass, bytes: u64) -> Result<()> {
        let slot = match class {
            AllocationClass::Weight => &mut self.weight,
            AllocationClass::Cache => &mut self.cache,
            AllocationClass::Scratch => &mut self.scratch,
            AllocationClass::Branch => &mut self.branch,
        };
        *slot = slot.checked_add(bytes).ok_or(ExecutorError::Overflow(
            "CPU allocation class bytes overflow u64",
        ))?;
        Ok(())
    }

    pub(super) fn saturating_sub(&mut self, class: AllocationClass, bytes: u64) {
        let slot = match class {
            AllocationClass::Weight => &mut self.weight,
            AllocationClass::Cache => &mut self.cache,
            AllocationClass::Scratch => &mut self.scratch,
            AllocationClass::Branch => &mut self.branch,
        };
        *slot = slot.saturating_sub(bytes);
    }

    pub(super) fn total(self) -> Result<u64> {
        self.weight
            .checked_add(self.cache)
            .and_then(|value| value.checked_add(self.scratch))
            .and_then(|value| value.checked_add(self.branch))
            .ok_or(ExecutorError::Overflow(
                "CPU allocation class total overflows u64",
            ))
    }
}

#[derive(Debug)]
pub(super) struct Tracker {
    pub(super) identity: BackendIdentity,
    pub(super) lease: BackendLease,
    pub(super) limits: ResourceLimits,
    pub(super) next_allocation: u64,
    pub(super) live_bytes: u64,
    pub(super) live_by_class: ClassBytes,
    pub(super) pending_retained_bytes: u64,
    pub(super) pending_retained_by_class: ClassBytes,
    pub(super) pending_result_bytes: u64,
    pub(super) pending_operations: u32,
    pub(super) cancellation_requested: bool,
}

/// Bytes currently owned by the backend: resident CPU buffers plus completion-owned host results.
/// Retained completion buffers are already included in `live_bytes` and are not counted twice.
fn tracker_owned_bytes(tracker: &Tracker) -> Result<u64> {
    tracker
        .live_bytes
        .checked_add(tracker.pending_result_bytes)
        .ok_or(ExecutorError::Overflow(
            "CPU owned resource total overflows u64",
        ))
}

/// An owned f32 or u8 buffer. Its storage cannot be used by another backend owner or generation.
#[derive(Debug)]
pub struct CpuBuffer {
    pub(super) descriptor: BufferDescriptor,
    pub(super) class: AllocationClass,
    pub(super) values: Vec<f32>,
    pub(super) bytes: Vec<u8>,
    pub(super) tracker: Rc<RefCell<Tracker>>,
}

impl CpuBuffer {
    #[must_use]
    pub fn descriptor(&self) -> BufferDescriptor {
        self.descriptor
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Raw byte storage for U8-typed buffers such as packed weight codes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn byte_len(&self) -> u64 {
        self.descriptor.layout.byte_extent()
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        let mut tracker = self.tracker.borrow_mut();
        tracker.live_bytes = tracker.live_bytes.saturating_sub(self.byte_len());
        tracker
            .live_by_class
            .saturating_sub(self.class, self.byte_len());
    }
}

/// CPU baseline whose submission result is immediately ready, but still exposes the same
/// one-step completion contract used by nonblocking device backends.
#[derive(Debug)]
pub struct CpuBackend {
    pub(super) tracker: Rc<RefCell<Tracker>>,
    pub(super) retirement: Rc<CpuFenceRetirement>,
    pub(super) capabilities: BackendCapabilities,
}

impl CpuBackend {
    #[must_use]
    pub fn new(owner: u64, limits: ResourceLimits) -> Self {
        let identity = BackendIdentity {
            kind: BackendKind::Cpu,
            ordinal: 0,
            owner,
            generation: 1,
        };
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
            max_allocation_bytes: limits.max_allocation_bytes,
            supports_nonblocking_completion: true,
        };
        let tracker = Rc::new(RefCell::new(Tracker {
            identity,
            lease: BackendLease::new(identity),
            limits,
            next_allocation: 1,
            live_bytes: 0,
            live_by_class: ClassBytes::default(),
            pending_retained_bytes: 0,
            pending_retained_by_class: ClassBytes::default(),
            pending_result_bytes: 0,
            pending_operations: 0,
            cancellation_requested: false,
        }));
        let retirement = Rc::new(CpuFenceRetirement::new(Rc::clone(&tracker)));
        Self {
            tracker,
            retirement,
            capabilities,
        }
    }

    #[must_use]
    pub fn identity(&self) -> BackendIdentity {
        self.tracker.borrow().identity
    }

    #[must_use]
    pub fn lease(&self) -> BackendLease {
        let tracker = self.tracker.borrow();
        tracker.lease.with_identity(tracker.identity)
    }

    #[must_use]
    pub const fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    #[must_use]
    pub fn resource_report(&self) -> ResourceReport {
        let tracker = self.tracker.borrow();
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
        let mut tracker = self.tracker.borrow_mut();
        tracker.identity.generation = tracker
            .identity
            .generation
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("backend generation overflows u64"))?;
        tracker.cancellation_requested = false;
        Ok(())
    }

    pub fn request_cancel(&mut self) {
        self.tracker.borrow_mut().cancellation_requested = true;
    }

    pub fn clear_cancel(&mut self) {
        self.tracker.borrow_mut().cancellation_requested = false;
    }

    pub(super) fn check_submit(&self) -> Result<()> {
        if self.tracker.borrow().cancellation_requested {
            return Err(ExecutorError::Cancelled);
        }
        Ok(())
    }

    pub(super) fn check_operation(&self, operation: OperationKind) -> Result<()> {
        self.check_submit()?;
        let capabilities = self.capabilities;
        if !capabilities.operations.contains(operation) {
            return Err(ExecutorError::Unsupported(
                "operation is unsupported by CPU backend",
            ));
        }
        Ok(())
    }

    pub(super) fn check_buffer(&self, buffer: &CpuBuffer) -> Result<()> {
        if !Rc::ptr_eq(&buffer.tracker, &self.tracker) {
            return Err(ExecutorError::WrongBackend);
        }
        buffer.descriptor.validate_for(self.identity())?;
        if buffer.descriptor.layout.dtype() != DType::F32
            && buffer.descriptor.layout.dtype() != DType::U8
        {
            return Err(ExecutorError::InvalidDType(
                "CPU foundation supports only f32 and u8",
            ));
        }
        if !buffer.descriptor.layout.is_contiguous()? {
            return Err(ExecutorError::InvalidLayout(
                "CPU foundation requires contiguous zero-offset buffers",
            ));
        }
        let elements = buffer.descriptor.layout.shape().element_count()?;
        let elements = usize::try_from(elements)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;
        match buffer.descriptor.layout.dtype() {
            DType::F32 => {
                if elements != buffer.values.len() {
                    return Err(ExecutorError::BackendFailure(
                        "CPU buffer length differs from descriptor",
                    ));
                }
                if !buffer.values.iter().all(|value| value.is_finite()) {
                    return Err(ExecutorError::BackendFailure(
                        "CPU buffer contains a non-finite value",
                    ));
                }
            }
            DType::U8 => {
                if elements != buffer.bytes.len() {
                    return Err(ExecutorError::BackendFailure(
                        "CPU byte buffer length differs from descriptor",
                    ));
                }
            }
            _ => {
                return Err(ExecutorError::InvalidDType(
                    "CPU foundation supports only f32 and u8",
                ));
            }
        }
        Ok(())
    }

    /// Structural checks plus an f32 dtype requirement. Ops that read or write
    /// `values` must use this so a u8 buffer cannot reach an f32 code path.
    pub(super) fn check_f32_buffer(&self, buffer: &CpuBuffer) -> Result<()> {
        self.check_buffer(buffer)?;
        if buffer.descriptor.layout.dtype() != DType::F32 {
            return Err(ExecutorError::InvalidDType("expected an f32 operand"));
        }
        Ok(())
    }

    /// Structural checks plus a u8 dtype requirement, for packed byte streams.
    pub(super) fn check_u8_buffer(&self, buffer: &CpuBuffer) -> Result<()> {
        self.check_buffer(buffer)?;
        if buffer.descriptor.layout.dtype() != DType::U8 {
            return Err(ExecutorError::InvalidDType("expected a u8 operand"));
        }
        Ok(())
    }

    pub(super) fn check_output_shape(&self, output: &CpuBuffer, shape: Shape) -> Result<()> {
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

    /// Allocate a zeroed, writable f32 buffer. Empty layouts are valid and allocate no bytes.
    ///
    /// All aggregate limits and identifier arithmetic are checked before Vec reserves or
    /// zeroes memory, so an over-limit request never creates a transient allocation.
    pub fn allocate_f32(&mut self, shape: Shape) -> Result<CpuBuffer> {
        self.allocate_f32_classified(shape, AllocationClass::Scratch)
    }

    /// Allocate zeroed f32 storage with an explicit logical resource class.
    pub fn allocate_f32_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        self.check_submit()?;
        let layout = TensorLayout::contiguous(DType::F32, shape)?;
        self.capabilities.validate(
            DType::F32,
            OperationKind::Copy,
            shape.rank(),
            shape.element_count()?,
            layout.byte_extent(),
        )?;
        let elements = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;

        let (identity, allocation) = {
            let tracker = self.tracker.borrow();
            tracker
                .limits
                .validate_allocation(layout.byte_extent(), tracker_owned_bytes(&tracker)?)?;
            tracker
                .next_allocation
                .checked_add(1)
                .ok_or(ExecutorError::Overflow(
                    "allocation identifier overflows u64",
                ))?;
            tracker
                .live_bytes
                .checked_add(layout.byte_extent())
                .ok_or(ExecutorError::Overflow("CPU live bytes overflow u64"))?;
            let mut classes = tracker.live_by_class;
            classes.checked_add(class, layout.byte_extent())?;
            (tracker.identity, tracker.next_allocation)
        };

        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| ExecutorError::ResourceLimit("CPU allocation failed"))?;
        values.resize(elements, 0.0);

        let mut tracker = self.tracker.borrow_mut();
        tracker
            .limits
            .validate_allocation(layout.byte_extent(), tracker_owned_bytes(&tracker)?)?;
        if tracker.next_allocation != allocation || tracker.identity != identity {
            return Err(ExecutorError::BackendFailure(
                "CPU allocator changed during allocation",
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
            .ok_or(ExecutorError::Overflow("CPU live bytes overflow u64"))?;
        tracker
            .live_by_class
            .checked_add(class, layout.byte_extent())?;
        let descriptor = BufferDescriptor {
            backend: tracker.identity,
            allocation,
            layout,
            access: BufferAccess::ReadWrite,
        };
        drop(tracker);
        Ok(CpuBuffer {
            descriptor,
            class,
            values,
            bytes: Vec::new(),
            tracker: Rc::clone(&self.tracker),
        })
    }

    /// Allocate a writable u8 buffer for opaque packed payloads such as ternary
    /// weight codes. Empty layouts are valid and allocate no bytes.
    pub fn allocate_u8_classified(
        &mut self,
        shape: Shape,
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        self.check_submit()?;
        let layout = TensorLayout::contiguous(DType::U8, shape)?;
        self.capabilities.validate(
            DType::U8,
            OperationKind::Copy,
            shape.rank(),
            shape.element_count()?,
            layout.byte_extent(),
        )?;
        let elements = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;

        let (identity, allocation) = {
            let tracker = self.tracker.borrow();
            tracker
                .limits
                .validate_allocation(layout.byte_extent(), tracker_owned_bytes(&tracker)?)?;
            tracker
                .next_allocation
                .checked_add(1)
                .ok_or(ExecutorError::Overflow(
                    "allocation identifier overflows u64",
                ))?;
            tracker
                .live_bytes
                .checked_add(layout.byte_extent())
                .ok_or(ExecutorError::Overflow("CPU live bytes overflow u64"))?;
            let mut classes = tracker.live_by_class;
            classes.checked_add(class, layout.byte_extent())?;
            (tracker.identity, tracker.next_allocation)
        };

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(elements)
            .map_err(|_| ExecutorError::ResourceLimit("CPU allocation failed"))?;
        bytes.resize(elements, 0);

        let mut tracker = self.tracker.borrow_mut();
        tracker
            .limits
            .validate_allocation(layout.byte_extent(), tracker_owned_bytes(&tracker)?)?;
        if tracker.next_allocation != allocation || tracker.identity != identity {
            return Err(ExecutorError::BackendFailure(
                "CPU allocator changed during allocation",
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
            .ok_or(ExecutorError::Overflow("CPU live bytes overflow u64"))?;
        tracker
            .live_by_class
            .checked_add(class, layout.byte_extent())?;
        let descriptor = BufferDescriptor {
            backend: tracker.identity,
            allocation,
            layout,
            access: BufferAccess::ReadWrite,
        };
        drop(tracker);
        Ok(CpuBuffer {
            descriptor,
            class,
            values: Vec::new(),
            bytes,
            tracker: Rc::clone(&self.tracker),
        })
    }

    /// Copy host bytes into an owned CPU buffer with an explicit resource class.
    pub fn upload_u8_classified(
        &mut self,
        shape: Shape,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        let expected = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;
        if expected != bytes.len() {
            return Err(ExecutorError::InvalidShape(
                "upload bytes differ from shape element count",
            ));
        }
        let mut output = self.allocate_u8_classified(shape, class)?;
        output.bytes.copy_from_slice(bytes);
        Ok(output)
    }

    /// Copy finite host f32 values into an owned CPU buffer.
    pub fn upload_f32(&mut self, shape: Shape, values: &[f32]) -> Result<CpuBuffer> {
        self.upload_f32_classified(shape, values, AllocationClass::Scratch)
    }

    /// Upload finite f32 values with an explicit logical resource class.
    pub fn upload_f32_classified(
        &mut self,
        shape: Shape,
        values: &[f32],
        class: AllocationClass,
    ) -> Result<CpuBuffer> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err(ExecutorError::InvalidArgument(
                "CPU upload contains a non-finite value",
            ));
        }
        let expected = usize::try_from(shape.element_count()?)
            .map_err(|_| ExecutorError::Overflow("element count exceeds usize"))?;
        if expected != values.len() {
            return Err(ExecutorError::InvalidShape(
                "upload values differ from shape element count",
            ));
        }
        let mut output = self.allocate_f32_classified(shape, class)?;
        output.values.copy_from_slice(values);
        Ok(output)
    }

    /// Drive the immediate CPU readback through the same checked completion path used by
    /// portable callers. Once the ready value transfers to this caller, it is no longer
    /// backend-owned resource accounting.
    pub fn read_f32(&self, buffer: &CpuBuffer) -> Result<Vec<f32>> {
        let mut readback = self.read_f32_async(buffer)?;
        match readback.poll_step() {
            CompletionPoll::Ready(result) => result,
            CompletionPoll::Pending => Err(ExecutorError::BackendFailure(
                "immediate CPU readback unexpectedly remained pending",
            )),
        }
    }

    pub(super) fn stage_f32(&self, elements: usize) -> Result<Vec<f32>> {
        let bytes = u64::try_from(elements)
            .map_err(|_| ExecutorError::Overflow("CPU staging element count exceeds u64"))?
            .checked_mul(DType::F32.byte_width())
            .ok_or(ExecutorError::Overflow(
                "CPU staging byte count overflows u64",
            ))?;
        let tracker = self.tracker.borrow();
        tracker
            .limits
            .validate_allocation(bytes, tracker_owned_bytes(&tracker)?)?;
        drop(tracker);
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| ExecutorError::ResourceLimit("CPU staging allocation failed"))?;
        Ok(values)
    }
}
