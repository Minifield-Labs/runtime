//! Owned scalar Rust CPU operations for the portable inference executor.
//!
//! This crate implements a finite inference operation surface. It is not a
//! tensor expression engine and contains no GPU, filesystem, network, or thread dependency.

#![forbid(unsafe_code)]
// The shared ExecutorError taxonomy documents CR01 operation failures; model-specific APIs
// add narrower error details as their equation and loader layers are introduced.
#![allow(clippy::missing_errors_doc)]

use std::{cell::RefCell, rc::Rc};

use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendKind, BackendLease, BufferAccess,
    BufferDescriptor, CompletionPoll, DType, DTypeSet, ExecutorError, FenceRetirement,
    GatedShortConvSpec, GqaSpec, InferenceCompletion, InferenceOps, OperationKind, OperationSet,
    PackedHeadSpec, PrecisionPolicy, RectCopy2d, ResourceLimits, ResourceReport, Result,
    RetirementRejection, RotarySpec, Shape, TensorLayout,
};

#[derive(Clone, Copy, Debug, Default)]
struct ClassBytes {
    weight: u64,
    cache: u64,
    scratch: u64,
    branch: u64,
}

impl ClassBytes {
    fn checked_add(&mut self, class: AllocationClass, bytes: u64) -> Result<()> {
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

    fn saturating_sub(&mut self, class: AllocationClass, bytes: u64) {
        let slot = match class {
            AllocationClass::Weight => &mut self.weight,
            AllocationClass::Cache => &mut self.cache,
            AllocationClass::Scratch => &mut self.scratch,
            AllocationClass::Branch => &mut self.branch,
        };
        *slot = slot.saturating_sub(bytes);
    }

    fn total(self) -> Result<u64> {
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
struct Tracker {
    identity: BackendIdentity,
    lease: BackendLease,
    limits: ResourceLimits,
    next_allocation: u64,
    live_bytes: u64,
    live_by_class: ClassBytes,
    pending_retained_bytes: u64,
    pending_retained_by_class: ClassBytes,
    pending_result_bytes: u64,
    pending_operations: u32,
    cancellation_requested: bool,
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

/// An owned f32 buffer. Its storage cannot be used by another backend owner or generation.
#[derive(Debug)]
pub struct CpuBuffer {
    descriptor: BufferDescriptor,
    class: AllocationClass,
    values: Vec<f32>,
    tracker: Rc<RefCell<Tracker>>,
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

    fn byte_len(&self) -> u64 {
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
    tracker: Rc<RefCell<Tracker>>,
    retirement: Rc<CpuFenceRetirement>,
    capabilities: BackendCapabilities,
}

#[derive(Clone, Copy, Debug)]
struct RetirementAccounting {
    bytes: u64,
    by_class: ClassBytes,
}

#[derive(Debug)]
struct RetiredCpuFence {
    fence: CpuCompletion<()>,
    _retained: Vec<CpuBuffer>,
    accounting: Option<RetirementAccounting>,
}

/// CPU-owned quarantine for fences whose original loader task was abandoned.
///
/// The queue retains buffers until the fence reports a terminal result. It is advanced explicitly
/// through the portable backend contract and does not create threads or synchronize globally.
#[derive(Debug)]
pub struct CpuFenceRetirement {
    tracker: Rc<RefCell<Tracker>>,
    retired: RefCell<Vec<RetiredCpuFence>>,
}

impl CpuFenceRetirement {
    fn actual_fence_owner(&self, fence: &CpuCompletion<()>) -> bool {
        fence
            .tracker
            .as_ref()
            .is_some_and(|tracker| Rc::ptr_eq(tracker, &self.tracker))
    }

    fn retained_accounting(&self, retained: &[CpuBuffer]) -> Result<RetirementAccounting> {
        let mut by_class = ClassBytes::default();
        for buffer in retained {
            if !Rc::ptr_eq(&buffer.tracker, &self.tracker) {
                return Err(ExecutorError::WrongBackend);
            }
            by_class.checked_add(buffer.class, buffer.byte_len())?;
        }
        Ok(RetirementAccounting {
            bytes: by_class.total()?,
            by_class,
        })
    }

    fn preflight_charge(&self, accounting: RetirementAccounting) -> Result<()> {
        let tracker = self.tracker.borrow();
        let next_total = tracker
            .pending_retained_bytes
            .checked_add(accounting.bytes)
            .ok_or(ExecutorError::Overflow(
                "retired CPU buffer bytes overflow pending accounting",
            ))?;
        let next_weight = tracker
            .pending_retained_by_class
            .weight
            .checked_add(accounting.by_class.weight)
            .ok_or(ExecutorError::Overflow(
                "retired CPU weight bytes overflow pending accounting",
            ))?;
        let next_cache = tracker
            .pending_retained_by_class
            .cache
            .checked_add(accounting.by_class.cache)
            .ok_or(ExecutorError::Overflow(
                "retired CPU cache bytes overflow pending accounting",
            ))?;
        let next_scratch = tracker
            .pending_retained_by_class
            .scratch
            .checked_add(accounting.by_class.scratch)
            .ok_or(ExecutorError::Overflow(
                "retired CPU scratch bytes overflow pending accounting",
            ))?;
        let next_branch = tracker
            .pending_retained_by_class
            .branch
            .checked_add(accounting.by_class.branch)
            .ok_or(ExecutorError::Overflow(
                "retired CPU branch bytes overflow pending accounting",
            ))?;
        if next_total > tracker.live_bytes
            || next_weight > tracker.live_by_class.weight
            || next_cache > tracker.live_by_class.cache
            || next_scratch > tracker.live_by_class.scratch
            || next_branch > tracker.live_by_class.branch
        {
            return Err(ExecutorError::BackendFailure(
                "retired CPU buffers exceed actual backend live ownership",
            ));
        }
        Ok(())
    }

    fn charge(&self, accounting: RetirementAccounting) {
        let mut tracker = self.tracker.borrow_mut();
        // preflight_charge established that every addition fits and that these buffers are a
        // subset of this actual backend's current live ownership.
        tracker.pending_retained_bytes = tracker
            .pending_retained_bytes
            .saturating_add(accounting.bytes);
        tracker.pending_retained_by_class.weight = tracker
            .pending_retained_by_class
            .weight
            .saturating_add(accounting.by_class.weight);
        tracker.pending_retained_by_class.cache = tracker
            .pending_retained_by_class
            .cache
            .saturating_add(accounting.by_class.cache);
        tracker.pending_retained_by_class.scratch = tracker
            .pending_retained_by_class
            .scratch
            .saturating_add(accounting.by_class.scratch);
        tracker.pending_retained_by_class.branch = tracker
            .pending_retained_by_class
            .branch
            .saturating_add(accounting.by_class.branch);
    }

    fn release_accounting(&self, accounting: RetirementAccounting) {
        let mut tracker = self.tracker.borrow_mut();
        tracker.pending_retained_bytes = tracker
            .pending_retained_bytes
            .saturating_sub(accounting.bytes);
        tracker
            .pending_retained_by_class
            .saturating_sub(AllocationClass::Weight, accounting.by_class.weight);
        tracker
            .pending_retained_by_class
            .saturating_sub(AllocationClass::Cache, accounting.by_class.cache);
        tracker
            .pending_retained_by_class
            .saturating_sub(AllocationClass::Scratch, accounting.by_class.scratch);
        tracker
            .pending_retained_by_class
            .saturating_sub(AllocationClass::Branch, accounting.by_class.branch);
    }
}

impl FenceRetirement<CpuCompletion<()>, CpuBuffer> for CpuFenceRetirement {
    fn retire(
        &self,
        fence: CpuCompletion<()>,
        retained: Vec<CpuBuffer>,
    ) -> core::result::Result<(), RetirementRejection<CpuCompletion<()>, CpuBuffer>> {
        if !self.actual_fence_owner(&fence) {
            return Err(RetirementRejection::new(
                ExecutorError::WrongBackend,
                fence,
                retained,
            ));
        }
        let accounting = match self.retained_accounting(&retained) {
            Ok(accounting) => accounting,
            Err(cause) => return Err(RetirementRejection::new(cause, fence, retained)),
        };
        if let Err(cause) = self.preflight_charge(accounting) {
            return Err(RetirementRejection::new(cause, fence, retained));
        }
        self.charge(accounting);
        self.retired.borrow_mut().push(RetiredCpuFence {
            fence,
            _retained: retained,
            accounting: Some(accounting),
        });
        Ok(())
    }

    fn quarantine_rejected(&self, rejected: RetirementRejection<CpuCompletion<()>, CpuBuffer>) {
        let (_, fence, retained) = rejected.into_parts();
        self.retired.borrow_mut().push(RetiredCpuFence {
            fence,
            _retained: retained,
            accounting: None,
        });
    }

    fn poll_retired(&self) {
        let mut retired = core::mem::take(&mut *self.retired.borrow_mut());
        let mut pending = Vec::new();
        for mut entry in retired.drain(..) {
            match entry.fence.poll_step() {
                CompletionPoll::Pending => pending.push(entry),
                CompletionPoll::Ready(_) => {
                    if let Some(accounting) = entry.accounting {
                        self.release_accounting(accounting);
                    }
                    drop(entry);
                }
            }
        }
        self.retired.borrow_mut().extend(pending);
    }
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
        let retirement = Rc::new(CpuFenceRetirement {
            tracker: Rc::clone(&tracker),
            retired: RefCell::new(Vec::new()),
        });
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

    fn check_submit(&self) -> Result<()> {
        if self.tracker.borrow().cancellation_requested {
            return Err(ExecutorError::Cancelled);
        }
        Ok(())
    }

    fn check_operation(&self, operation: OperationKind) -> Result<()> {
        self.check_submit()?;
        let capabilities = self.capabilities;
        if !capabilities.operations.contains(operation) {
            return Err(ExecutorError::Unsupported(
                "operation is unsupported by CPU backend",
            ));
        }
        Ok(())
    }

    fn check_buffer(&self, buffer: &CpuBuffer) -> Result<()> {
        if !Rc::ptr_eq(&buffer.tracker, &self.tracker) {
            return Err(ExecutorError::WrongBackend);
        }
        buffer.descriptor.validate_for(self.identity())?;
        if buffer.descriptor.layout.dtype() != DType::F32 {
            return Err(ExecutorError::InvalidDType(
                "CPU foundation supports only f32",
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
        Ok(())
    }

    fn check_output_shape(&self, output: &CpuBuffer, shape: Shape) -> Result<()> {
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
            tracker: Rc::clone(&self.tracker),
        })
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

    pub fn copy(&self, output: &mut CpuBuffer, input: &CpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::Copy)?;
        self.check_buffer(input)?;
        self.check_output_shape(output, input.descriptor.layout.shape())?;
        output.values.copy_from_slice(&input.values);
        Ok(())
    }

    /// Gather selected rows from a contiguous [rows, columns] f32 table.
    pub fn gather_rows(
        &self,
        output: &mut CpuBuffer,
        table: &CpuBuffer,
        ids: &[u32],
    ) -> Result<()> {
        self.check_operation(OperationKind::GatherRows)?;
        self.check_buffer(table)?;
        let table_shape = table.descriptor.layout.shape();
        if table_shape.rank() != 2 {
            return Err(ExecutorError::InvalidShape("gather table must be rank two"));
        }
        let rows = usize::try_from(table_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("row count exceeds usize"))?;
        let columns = usize::try_from(table_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("column count exceeds usize"))?;
        let output_shape = Shape::new(&[
            u64::try_from(ids.len())
                .map_err(|_| ExecutorError::Overflow("id count overflows u64"))?,
            u64::try_from(columns)
                .map_err(|_| ExecutorError::Overflow("column count overflows u64"))?,
        ])?;
        self.check_output_shape(output, output_shape)?;
        for (destination_row, id) in ids.iter().copied().enumerate() {
            let source_row = usize::try_from(id)
                .map_err(|_| ExecutorError::OutOfBounds("gather identifier exceeds usize"))?;
            if source_row >= rows {
                return Err(ExecutorError::OutOfBounds(
                    "gather identifier exceeds row count",
                ));
            }
            let source_start = source_row
                .checked_mul(columns)
                .ok_or(ExecutorError::Overflow(
                    "gather source offset overflows usize",
                ))?;
            let destination_start =
                destination_row
                    .checked_mul(columns)
                    .ok_or(ExecutorError::Overflow(
                        "gather destination offset overflows usize",
                    ))?;
            output.values[destination_start..destination_start + columns]
                .copy_from_slice(&table.values[source_start..source_start + columns]);
        }
        Ok(())
    }

    pub fn add(&self, output: &mut CpuBuffer, left: &CpuBuffer, right: &CpuBuffer) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Add, |a, b| a + b)
    }

    pub fn multiply(
        &self,
        output: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
    ) -> Result<()> {
        self.elementwise(output, left, right, OperationKind::Multiply, |a, b| a * b)
    }

    fn elementwise(
        &self,
        output: &mut CpuBuffer,
        left: &CpuBuffer,
        right: &CpuBuffer,
        operation: OperationKind,
        function: impl Fn(f32, f32) -> f32,
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
        for ((destination, lhs), rhs) in output
            .values
            .iter_mut()
            .zip(left.values.iter())
            .zip(right.values.iter())
        {
            let value = function(*lhs, *rhs);
            if !value.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "elementwise operation produced a non-finite value",
                ));
            }
            *destination = value;
        }
        Ok(())
    }

    /// Row-major linear projection: input [m, k] times weight [n, k] yields output [m, n].
    /// Accumulation is sequential f32 in increasing k order.
    pub fn linear(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
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
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("linear row count exceeds usize"))?;
        let inner = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("linear inner width exceeds usize"))?;
        let output_width = usize::try_from(weight_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("linear output width exceeds usize"))?;
        let weight_inner = usize::try_from(weight_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("linear weight width exceeds usize"))?;
        if inner != weight_inner {
            return Err(ExecutorError::InvalidShape(
                "linear input width differs from weight width",
            ));
        }
        let output_shape = Shape::new(&[
            u64::try_from(rows).map_err(|_| ExecutorError::Overflow("linear rows overflow u64"))?,
            u64::try_from(output_width)
                .map_err(|_| ExecutorError::Overflow("linear width overflows u64"))?,
        ])?;
        self.check_output_shape(output, output_shape)?;
        for row in 0..rows {
            for column in 0..output_width {
                let mut accumulator = 0.0_f32;
                for index in 0..inner {
                    let input_offset = row
                        .checked_mul(inner)
                        .and_then(|value| value.checked_add(index))
                        .ok_or(ExecutorError::Overflow(
                            "linear input offset overflows usize",
                        ))?;
                    let weight_offset = column
                        .checked_mul(inner)
                        .and_then(|value| value.checked_add(index))
                        .ok_or(ExecutorError::Overflow(
                            "linear weight offset overflows usize",
                        ))?;
                    accumulator += input.values[input_offset] * weight.values[weight_offset];
                }
                if !accumulator.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "linear projection produced a non-finite value",
                    ));
                }
                let output_offset = row
                    .checked_mul(output_width)
                    .and_then(|value| value.checked_add(column))
                    .ok_or(ExecutorError::Overflow(
                        "linear output offset overflows usize",
                    ))?;
                output.values[output_offset] = accumulator;
            }
        }
        Ok(())
    }

    /// Row RMS norm over a [rows, width] input and a [width] weight.
    /// The sum and square root use f32 to define the scalar reference rounding path.
    pub fn row_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
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
        let rows = usize::try_from(input_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("RMS row count exceeds usize"))?;
        let width = usize::try_from(input_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("RMS width exceeds usize"))?;
        let weight_width = usize::try_from(weight_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("RMS weight width exceeds usize"))?;
        if width != weight_width {
            return Err(ExecutorError::InvalidShape(
                "RMS input width differs from weight width",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        if width == 0 {
            return Ok(());
        }
        if width > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "RMS width exceeds exact f32 divisor range",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let width_as_f32 = width as f32;
        for row in 0..rows {
            let start = row
                .checked_mul(width)
                .ok_or(ExecutorError::Overflow("RMS row offset overflows usize"))?;
            let mut squared_sum = 0.0_f32;
            for value in &input.values[start..start + width] {
                squared_sum += value * value;
            }
            let reciprocal = (squared_sum / width_as_f32 + epsilon).sqrt().recip();
            if !reciprocal.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "RMS normalization reciprocal is non-finite",
                ));
            }
            for column in 0..width {
                let normalized = input.values[start + column] * reciprocal;
                let value = normalized * weight.values[column];
                if !value.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "RMS normalization produced a non-finite value",
                    ));
                }
                output.values[start + column] = value;
            }
        }
        Ok(())
    }

    /// Copy a checked row-major rectangle between distinct rank-two CPU buffers.
    pub fn copy_rect_2d(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
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
        let source_width = usize::try_from(source_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("rectangular source width exceeds usize"))?;
        let destination_width = usize::try_from(destination_shape.dim(1)?)
            .map_err(|_| ExecutorError::Overflow("rectangular destination width exceeds usize"))?;
        let source_row = usize::try_from(rectangle.source_row())
            .map_err(|_| ExecutorError::Overflow("rectangular source row exceeds usize"))?;
        let source_column = usize::try_from(rectangle.source_column())
            .map_err(|_| ExecutorError::Overflow("rectangular source column exceeds usize"))?;
        let destination_row = usize::try_from(rectangle.destination_row())
            .map_err(|_| ExecutorError::Overflow("rectangular destination row exceeds usize"))?;
        let destination_column = usize::try_from(rectangle.destination_column())
            .map_err(|_| ExecutorError::Overflow("rectangular destination column exceeds usize"))?;
        let rows = usize::try_from(rectangle.rows())
            .map_err(|_| ExecutorError::Overflow("rectangular row count exceeds usize"))?;
        let columns = usize::try_from(rectangle.columns())
            .map_err(|_| ExecutorError::Overflow("rectangular column count exceeds usize"))?;
        for row in 0..rows {
            let source_start = source_row
                .checked_add(row)
                .and_then(|value| value.checked_mul(source_width))
                .and_then(|value| value.checked_add(source_column))
                .ok_or(ExecutorError::Overflow(
                    "rectangular source offset overflows usize",
                ))?;
            let destination_start = destination_row
                .checked_add(row)
                .and_then(|value| value.checked_mul(destination_width))
                .and_then(|value| value.checked_add(destination_column))
                .ok_or(ExecutorError::Overflow(
                    "rectangular destination offset overflows usize",
                ))?;
            let source_end = source_start
                .checked_add(columns)
                .ok_or(ExecutorError::Overflow(
                    "rectangular source end overflows usize",
                ))?;
            let destination_end =
                destination_start
                    .checked_add(columns)
                    .ok_or(ExecutorError::Overflow(
                        "rectangular destination end overflows usize",
                    ))?;
            output.values[destination_start..destination_end]
                .copy_from_slice(&input.values[source_start..source_end]);
        }
        Ok(())
    }

    /// Apply head-local RMS normalization to packed `[tokens, heads * head_dim]` rows.
    pub fn head_rms_norm(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
        weight: &CpuBuffer,
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
        let weight_shape = weight.descriptor.layout.shape();
        if weight_shape != Shape::new(&[u64::from(heads.head_dim())])? {
            return Err(ExecutorError::InvalidShape(
                "head RMS weight must have head_dim entries",
            ));
        }
        self.check_output_shape(output, input_shape)?;
        let tokens = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("head RMS token count exceeds usize"))?;
        let head_count = usize::try_from(heads.heads())
            .map_err(|_| ExecutorError::Overflow("head RMS head count exceeds usize"))?;
        let head_dim = usize::try_from(heads.head_dim())
            .map_err(|_| ExecutorError::Overflow("head RMS head dimension exceeds usize"))?;
        let packed_width = usize::try_from(heads.packed_width()?)
            .map_err(|_| ExecutorError::Overflow("head RMS packed width exceeds usize"))?;
        if head_dim > (1_usize << 24) {
            return Err(ExecutorError::Unsupported(
                "RMS width exceeds exact f32 divisor range",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let head_dim_f32 = head_dim as f32;
        for token in 0..tokens {
            for head in 0..head_count {
                let start = token
                    .checked_mul(packed_width)
                    .and_then(|value| value.checked_add(head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("head RMS offset overflows usize"))?;
                let mut squared_sum = 0.0_f32;
                for value in &input.values[start..start + head_dim] {
                    squared_sum += value * value;
                }
                let reciprocal = (squared_sum / head_dim_f32 + epsilon).sqrt().recip();
                if !reciprocal.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "head RMS reciprocal is non-finite",
                    ));
                }
                for column in 0..head_dim {
                    let normalized = input.values[start + column] * reciprocal;
                    let value = normalized * weight.values[column];
                    if !value.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "head RMS normalization produced a non-finite value",
                        ));
                    }
                    output.values[start + column] = value;
                }
            }
        }
        Ok(())
    }

    /// Apply split-half `RoPE` to packed `[tokens, heads * head_dim]` rows.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn split_half_rotary(
        &self,
        output: &mut CpuBuffer,
        input: &CpuBuffer,
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
        let head_count = usize::try_from(spec.heads().heads())
            .map_err(|_| ExecutorError::Overflow("RoPE head count exceeds usize"))?;
        let head_dim = usize::try_from(spec.heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("RoPE head dimension exceeds usize"))?;
        let half = head_dim / 2;
        let packed_width = usize::try_from(spec.heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("RoPE packed width exceeds usize"))?;
        for (token, position) in positions.iter().copied().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_f32 = position as f32;
            if !position_f32.is_finite() {
                return Err(ExecutorError::InvalidArgument(
                    "RoPE position is not representable as finite f32",
                ));
            }
            for head in 0..head_count {
                let head_start = token
                    .checked_mul(packed_width)
                    .and_then(|value| value.checked_add(head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("RoPE offset overflows usize"))?;
                for column in 0..half {
                    let exponent = -2.0_f64 * (column as f64) / (head_dim as f64);
                    let frequency = (f64::from(spec.theta()).powf(exponent)) as f32;
                    let angle = position_f32 * frequency;
                    let sine = f64::from(angle).sin() as f32;
                    let cosine = f64::from(angle).cos() as f32;
                    let first = input.values[head_start + column];
                    let second = input.values[head_start + half + column];
                    // Frequency and trig use the explicitly documented f64-to-f32 boundaries;
                    // vector products and combinations are sequential f32 model arithmetic.
                    let rotated_first = first * cosine - second * sine;
                    let rotated_second = second * cosine + first * sine;
                    if !rotated_first.is_finite() || !rotated_second.is_finite() {
                        return Err(ExecutorError::BackendFailure(
                            "RoPE produced a non-finite value",
                        ));
                    }
                    output.values[head_start + column] = rotated_first;
                    output.values[head_start + half + column] = rotated_second;
                }
            }
        }
        Ok(())
    }

    /// Causal grouped-query attention over packed rows and explicit non-repeated KV caches.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn causal_gqa(
        &self,
        output: &mut CpuBuffer,
        query: &CpuBuffer,
        key: &CpuBuffer,
        value: &CpuBuffer,
        key_cache: &mut CpuBuffer,
        value_cache: &mut CpuBuffer,
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
        let query_shape = query.descriptor.layout.shape();
        let tokens = spec.query_heads().validate_packed(query_shape)?;
        let key_shape = key.descriptor.layout.shape();
        let value_shape = value.descriptor.layout.shape();
        if spec.key_value_heads().validate_packed(key_shape)? != tokens
            || spec.key_value_heads().validate_packed(value_shape)? != tokens
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
        let token_count = usize::try_from(tokens)
            .map_err(|_| ExecutorError::Overflow("GQA token count exceeds usize"))?;
        let query_heads = usize::try_from(spec.query_heads().heads())
            .map_err(|_| ExecutorError::Overflow("GQA query head count exceeds usize"))?;
        let head_dim = usize::try_from(spec.query_heads().head_dim())
            .map_err(|_| ExecutorError::Overflow("GQA head dimension exceeds usize"))?;
        let group_size = usize::try_from(spec.group_size())
            .map_err(|_| ExecutorError::Overflow("GQA group size exceeds usize"))?;
        let query_width = usize::try_from(spec.query_heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("GQA query width exceeds usize"))?;
        let kv_width = usize::try_from(spec.key_value_heads().packed_width()?)
            .map_err(|_| ExecutorError::Overflow("GQA key/value width exceeds usize"))?;
        let initial_len = usize::try_from(*cache_len)
            .map_err(|_| ExecutorError::Overflow("GQA cache length exceeds usize"))?;
        let staged_elements =
            token_count
                .checked_mul(query_width)
                .ok_or(ExecutorError::Overflow(
                    "GQA staging element count overflows usize",
                ))?;
        let mut staged = self.stage_f32(staged_elements)?;
        staged.resize(staged_elements, 0.0);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        if !scale.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA scale is non-finite"));
        }
        for token in 0..token_count {
            let visible = initial_len
                .checked_add(token)
                .and_then(|count| count.checked_add(1))
                .ok_or(ExecutorError::Overflow(
                    "GQA visible length overflows usize",
                ))?;
            for query_head in 0..query_heads {
                let kv_head = query_head / group_size;
                let query_start = token
                    .checked_mul(query_width)
                    .and_then(|offset| offset.checked_add(query_head.checked_mul(head_dim)?))
                    .ok_or(ExecutorError::Overflow("GQA query offset overflows usize"))?;
                let mut maximum = f32::NEG_INFINITY;
                for key_index in 0..visible {
                    maximum = maximum.max(Self::gqa_score(
                        query,
                        key,
                        key_cache,
                        initial_len,
                        key_index,
                        query_start,
                        kv_head,
                        kv_width,
                        head_dim,
                        scale,
                    )?);
                }
                let mut denominator = 0.0_f32;
                for key_index in 0..visible {
                    let score = Self::gqa_score(
                        query,
                        key,
                        key_cache,
                        initial_len,
                        key_index,
                        query_start,
                        kv_head,
                        kv_width,
                        head_dim,
                        scale,
                    )?;
                    denominator += (score - maximum).exp();
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(ExecutorError::BackendFailure(
                        "GQA softmax denominator is invalid",
                    ));
                }
                for dimension in 0..head_dim {
                    let mut weighted = 0.0_f32;
                    for key_index in 0..visible {
                        let score = Self::gqa_score(
                            query,
                            key,
                            key_cache,
                            initial_len,
                            key_index,
                            query_start,
                            kv_head,
                            kv_width,
                            head_dim,
                            scale,
                        )?;
                        let probability = (score - maximum).exp() / denominator;
                        let value_at_key = if key_index < initial_len {
                            value_cache.values
                                [key_index * kv_width + kv_head * head_dim + dimension]
                        } else {
                            value.values[(key_index - initial_len) * kv_width
                                + kv_head * head_dim
                                + dimension]
                        };
                        weighted += probability * value_at_key;
                    }
                    if !weighted.is_finite() {
                        return Err(ExecutorError::BackendFailure("GQA output is non-finite"));
                    }
                    staged[token * query_width + query_head * head_dim + dimension] = weighted;
                }
            }
        }
        let appended = token_count
            .checked_mul(kv_width)
            .ok_or(ExecutorError::Overflow(
                "GQA cache copy count overflows usize",
            ))?;
        let cache_offset = initial_len
            .checked_mul(kv_width)
            .ok_or(ExecutorError::Overflow("GQA cache offset overflows usize"))?;
        key_cache.values[cache_offset..cache_offset + appended]
            .copy_from_slice(&key.values[..appended]);
        value_cache.values[cache_offset..cache_offset + appended]
            .copy_from_slice(&value.values[..appended]);
        output.values.copy_from_slice(&staged);
        *cache_len = new_cache_len;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_score(
        query: &CpuBuffer,
        key: &CpuBuffer,
        key_cache: &CpuBuffer,
        initial_len: usize,
        key_index: usize,
        query_start: usize,
        kv_head: usize,
        kv_width: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<f32> {
        let mut dot = 0.0_f32;
        for dimension in 0..head_dim {
            let key_value = if key_index < initial_len {
                key_cache.values[key_index * kv_width + kv_head * head_dim + dimension]
            } else {
                key.values[(key_index - initial_len) * kv_width + kv_head * head_dim + dimension]
            };
            dot += query.values[query_start + dimension] * key_value;
        }
        let score = dot * scale;
        if !score.is_finite() {
            return Err(ExecutorError::BackendFailure("GQA score is non-finite"));
        }
        Ok(score)
    }

    fn stage_f32(&self, elements: usize) -> Result<Vec<f32>> {
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

    /// Apply B*V gated short convolution and update its old-to-new rolling U history.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn gated_short_convolution(
        &self,
        output: &mut CpuBuffer,
        b: &CpuBuffer,
        c: &CpuBuffer,
        v: &CpuBuffer,
        kernel: &CpuBuffer,
        history: &mut CpuBuffer,
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
        if history.descriptor.layout.shape()
            != Shape::new(&[spec.history_rows()?, u64::from(spec.hidden())])?
        {
            return Err(ExecutorError::InvalidShape(
                "short convolution history must be [width - 1, hidden]",
            ));
        }
        let tokens = usize::try_from(token_shape.dim(0)?)
            .map_err(|_| ExecutorError::Overflow("short convolution token count exceeds usize"))?;
        let hidden = usize::try_from(spec.hidden())
            .map_err(|_| ExecutorError::Overflow("short convolution hidden width exceeds usize"))?;
        let width = usize::try_from(spec.width())
            .map_err(|_| ExecutorError::Overflow("short convolution kernel width exceeds usize"))?;
        let history_rows = width.checked_sub(1).ok_or(ExecutorError::Overflow(
            "short convolution history underflows",
        ))?;
        let row_elements = tokens.checked_mul(hidden).ok_or(ExecutorError::Overflow(
            "short convolution element count overflows usize",
        ))?;
        let staged_elements = row_elements.checked_mul(2).ok_or(ExecutorError::Overflow(
            "short convolution staging count overflows usize",
        ))?;
        let mut staged = self.stage_f32(staged_elements)?;
        staged.resize(staged_elements, 0.0);
        let (u, produced) = staged.split_at_mut(row_elements);
        for (index, destination) in u.iter_mut().enumerate() {
            let gate = b.values[index] * v.values[index];
            if !gate.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "short convolution gate product is non-finite",
                ));
            }
            *destination = gate;
        }
        for token in 0..tokens {
            for channel in 0..hidden {
                let mut total = 0.0_f32;
                for tap in 0..width {
                    let relative = token as i128 + tap as i128 - history_rows as i128;
                    let u_value = if relative < 0 {
                        let history_row = usize::try_from(relative + history_rows as i128)
                            .map_err(|_| {
                                ExecutorError::Overflow(
                                    "short convolution history index exceeds usize",
                                )
                            })?;
                        history.values[history_row * hidden + channel]
                    } else {
                        let current = usize::try_from(relative).map_err(|_| {
                            ExecutorError::Overflow("short convolution token index exceeds usize")
                        })?;
                        u[current * hidden + channel]
                    };
                    total += kernel.values[channel * width + tap] * u_value;
                }
                let result = total * c.values[token * hidden + channel];
                if !result.is_finite() {
                    return Err(ExecutorError::BackendFailure(
                        "short convolution output is non-finite",
                    ));
                }
                produced[token * hidden + channel] = result;
            }
        }
        if history_rows > 0 {
            if tokens >= history_rows {
                let source_start = (tokens - history_rows) * hidden;
                history
                    .values
                    .copy_from_slice(&u[source_start..source_start + history_rows * hidden]);
            } else {
                let keep_rows = history_rows - tokens;
                history.values.copy_within(
                    (history_rows - keep_rows) * hidden..history_rows * hidden,
                    0,
                );
                history.values[keep_rows * hidden..].copy_from_slice(u);
            }
        }
        output.values.copy_from_slice(produced);
        Ok(())
    }

    /// Apply SiLU(gate) * up over matching contiguous f32 layouts.
    pub fn swiglu(&self, output: &mut CpuBuffer, gate: &CpuBuffer, up: &CpuBuffer) -> Result<()> {
        self.check_operation(OperationKind::SwiGlu)?;
        self.check_buffer(gate)?;
        self.check_buffer(up)?;
        if gate.descriptor.layout != up.descriptor.layout {
            return Err(ExecutorError::InvalidShape(
                "SwiGLU gate and up layouts differ",
            ));
        }
        self.check_output_shape(output, gate.descriptor.layout.shape())?;
        for ((destination, gate_value), up_value) in
            output.values.iter_mut().zip(&gate.values).zip(&up.values)
        {
            let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate_value).exp());
            let result = (*gate_value * sigmoid) * *up_value;
            if !result.is_finite() {
                return Err(ExecutorError::BackendFailure(
                    "SwiGLU produced a non-finite value",
                ));
            }
            *destination = result;
        }
        Ok(())
    }

    /// Submit a CPU completion fence. This baseline is ready on its first poll.
    pub fn fence(&self) -> Result<CpuCompletion<()>> {
        self.submit_ready(Ok(()), Vec::new())
    }

    /// Submit f32 readback without exposing a blocking requirement through the shared trait.
    /// CPU preflights and accounts the completion-owned host result before it fallibly copies.
    /// A successful poll transfers that Vec to the caller and removes it from backend accounting.
    pub fn read_f32_async(&self, buffer: &CpuBuffer) -> Result<CpuCompletion<Vec<f32>>> {
        self.check_submit()?;
        self.check_buffer(buffer)?;
        let result_bytes = buffer.byte_len();
        preflight_completion_bytes(&self.tracker, result_bytes)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(buffer.values.len())
            .map_err(|_| ExecutorError::ResourceLimit("CPU readback allocation failed"))?;
        values.extend_from_slice(&buffer.values);
        CpuCompletion::new(
            Rc::clone(&self.tracker),
            Ok(values),
            Vec::new(),
            0,
            result_bytes,
        )
    }

    /// Submit an already-computed CPU result. The first poll is ready, and retained buffers
    /// stay owned until that poll transfers success or failure to the caller.
    pub fn submit_ready<T>(
        &self,
        result: Result<T>,
        retained: Vec<CpuBuffer>,
    ) -> Result<CpuCompletion<T>> {
        let result = if self.tracker.borrow().cancellation_requested {
            Err(ExecutorError::Cancelled)
        } else {
            result
        };
        CpuCompletion::new(Rc::clone(&self.tracker), result, retained, 0, 0)
    }
}

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
        ids: &[u32],
    ) -> Result<()> {
        CpuBackend::gather_rows(self, output, table, ids)
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
        b: &Self::Buffer,
        c: &Self::Buffer,
        v: &Self::Buffer,
        kernel: &Self::Buffer,
        history: &mut Self::Buffer,
        spec: GatedShortConvSpec,
    ) -> Result<()> {
        CpuBackend::gated_short_convolution(self, output, b, c, v, kernel, history, spec)
    }

    fn swiglu(
        &self,
        output: &mut Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
    ) -> Result<()> {
        CpuBackend::swiglu(self, output, gate, up)
    }
}

fn preflight_completion_bytes(tracker: &Rc<RefCell<Tracker>>, result_bytes: u64) -> Result<()> {
    let state = tracker.borrow();
    if state.pending_operations >= state.limits.max_pending_operations {
        return Err(ExecutorError::ResourceLimit(
            "pending operation count exceeds configured limit",
        ));
    }
    let current_owned = state
        .live_bytes
        .checked_add(state.pending_result_bytes)
        .ok_or(ExecutorError::Overflow(
            "CPU completion resource total overflows u64",
        ))?;
    state
        .limits
        .validate_allocation(result_bytes, current_owned)
}

/// Completion owned by the CPU backend. CPU submissions become ready immediately; the
/// private deferred constructor exists only to exercise portable pending retention semantics.
#[derive(Debug)]
pub struct CpuCompletion<T> {
    tracker: Option<Rc<RefCell<Tracker>>>,
    result: Option<Result<T>>,
    retained: Vec<CpuBuffer>,
    retained_by_class: ClassBytes,
    result_bytes: u64,
    remaining_pending_steps: u8,
}

impl<T> CpuCompletion<T> {
    fn new(
        tracker: Rc<RefCell<Tracker>>,
        result: Result<T>,
        retained: Vec<CpuBuffer>,
        remaining_pending_steps: u8,
        result_bytes: u64,
    ) -> Result<Self> {
        let identity = tracker.borrow().identity;
        for buffer in &retained {
            if !Rc::ptr_eq(&buffer.tracker, &tracker) {
                return Err(ExecutorError::WrongBackend);
            }
            buffer.descriptor.validate_for(identity)?;
        }
        let mut retained_by_class = ClassBytes::default();
        for buffer in &retained {
            retained_by_class.checked_add(buffer.class, buffer.byte_len())?;
        }
        let retained_bytes = retained_by_class.total()?;
        preflight_completion_bytes(&tracker, result_bytes)?;
        {
            let mut state = tracker.borrow_mut();
            state.pending_operations =
                state
                    .pending_operations
                    .checked_add(1)
                    .ok_or(ExecutorError::Overflow(
                        "pending operation count overflows u32",
                    ))?;
            state.pending_retained_bytes = state
                .pending_retained_bytes
                .checked_add(retained_bytes)
                .ok_or(ExecutorError::Overflow(
                    "completion retained bytes overflow u64",
                ))?;
            for class in [
                AllocationClass::Weight,
                AllocationClass::Cache,
                AllocationClass::Scratch,
                AllocationClass::Branch,
            ] {
                let bytes = match class {
                    AllocationClass::Weight => retained_by_class.weight,
                    AllocationClass::Cache => retained_by_class.cache,
                    AllocationClass::Scratch => retained_by_class.scratch,
                    AllocationClass::Branch => retained_by_class.branch,
                };
                state.pending_retained_by_class.checked_add(class, bytes)?;
            }
            state.pending_result_bytes =
                state.pending_result_bytes.checked_add(result_bytes).ok_or(
                    ExecutorError::Overflow("completion result bytes overflow u64"),
                )?;
        }
        Ok(Self {
            tracker: Some(tracker),
            result: Some(result),
            retained,
            retained_by_class,
            result_bytes,
            remaining_pending_steps,
        })
    }

    #[cfg(test)]
    fn deferred_for_test(
        tracker: Rc<RefCell<Tracker>>,
        result: Result<T>,
        retained: Vec<CpuBuffer>,
        result_bytes: u64,
    ) -> Result<Self> {
        Self::new(tracker, result, retained, 1, result_bytes)
    }

    /// Release completion-owned backend accounting. A ready successful result transfers its
    /// result Vec to the caller after this release; a dropped or cancelled completion drops it.
    fn release_retained(&mut self) {
        let retained_bytes = self.retained_by_class.total().unwrap_or(u64::MAX);
        let retained_by_class = self.retained_by_class;
        self.retained.clear();
        self.retained_by_class = ClassBytes::default();
        let result_bytes = self.result_bytes;
        self.result_bytes = 0;
        if let Some(tracker) = self.tracker.take() {
            let mut state = tracker.borrow_mut();
            debug_assert!(state.pending_operations > 0);
            debug_assert!(state.pending_retained_bytes >= retained_bytes);
            debug_assert!(state.pending_result_bytes >= result_bytes);
            state.pending_operations -= 1;
            state.pending_retained_bytes -= retained_bytes;
            state
                .pending_retained_by_class
                .saturating_sub(AllocationClass::Weight, retained_by_class.weight);
            state
                .pending_retained_by_class
                .saturating_sub(AllocationClass::Cache, retained_by_class.cache);
            state
                .pending_retained_by_class
                .saturating_sub(AllocationClass::Scratch, retained_by_class.scratch);
            state
                .pending_retained_by_class
                .saturating_sub(AllocationClass::Branch, retained_by_class.branch);
            state.pending_result_bytes -= result_bytes;
        }
    }
}

impl<T> Drop for CpuCompletion<T> {
    fn drop(&mut self) {
        self.release_retained();
    }
}

impl<T> InferenceCompletion for CpuCompletion<T> {
    type Output = T;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.result.is_none() {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        if self.remaining_pending_steps > 0 {
            self.remaining_pending_steps -= 1;
            return CompletionPoll::Pending;
        }
        self.release_retained();
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.result.is_none() {
            return Err(ExecutorError::CompletionConsumed);
        }
        self.result = Some(Err(ExecutorError::Cancelled));
        self.remaining_pending_steps = 0;
        self.release_retained();
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn limits() -> ResourceLimits {
        ResourceLimits {
            max_allocation_bytes: 4096,
            max_total_bytes: 4096,
            max_pending_operations: 2,
        }
    }

    #[test]
    fn completion_retains_buffers_for_pending_success_and_error() {
        let mut backend = CpuBackend::new(7, limits());
        let buffer = backend
            .upload_f32(Shape::new(&[2]).expect("shape"), &[1.0, 2.0])
            .expect("buffer");
        let before = backend.resource_report();
        assert_eq!(before.total_owned_bytes().expect("total"), 8);

        let mut pending = CpuCompletion::deferred_for_test(
            Rc::clone(&backend.tracker),
            Ok(5_u32),
            vec![buffer],
            0,
        )
        .expect("completion");
        assert_eq!(backend.resource_report().pending_operation_bytes, 8);
        assert_eq!(pending.poll_step(), CompletionPoll::Pending);
        assert_eq!(backend.resource_report().pending_operation_bytes, 8);
        assert_eq!(pending.poll_step(), CompletionPoll::Ready(Ok(5)));
        assert_eq!(
            backend
                .resource_report()
                .total_owned_bytes()
                .expect("total"),
            0
        );

        let buffer = backend
            .upload_f32(Shape::new(&[1]).expect("shape"), &[3.0])
            .expect("buffer");
        let mut failed = CpuCompletion::deferred_for_test(
            Rc::clone(&backend.tracker),
            Err::<u32, _>(ExecutorError::BackendFailure("synthetic failure")),
            vec![buffer],
            0,
        )
        .expect("completion");
        assert_eq!(failed.poll_step(), CompletionPoll::Pending);
        assert_eq!(
            failed.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::BackendFailure("synthetic failure")))
        );
        assert_eq!(
            backend
                .resource_report()
                .total_owned_bytes()
                .expect("total"),
            0
        );
    }

    #[test]
    fn abandoned_fence_retirement_keeps_buffers_accounted_until_terminal_poll() {
        let mut backend = CpuBackend::new(7, limits());
        let buffer = backend
            .upload_f32_classified(
                Shape::new(&[2]).expect("shape"),
                &[1.0, 2.0],
                AllocationClass::Weight,
            )
            .expect("buffer");
        let fence =
            CpuCompletion::deferred_for_test(Rc::clone(&backend.tracker), Ok(()), Vec::new(), 0)
                .expect("deferred fence");
        backend
            .fence_retirement()
            .retire(fence, vec![buffer])
            .expect("same-instance retirement");
        assert_eq!(backend.resource_report().resident_weight_bytes, 0);
        assert_eq!(backend.resource_report().pending_operation_bytes, 8);
        assert_eq!(backend.resource_report().pending_operations, 1);

        backend
            .poll_retired_fences()
            .expect("pending retirement poll");
        assert_eq!(backend.resource_report().pending_operation_bytes, 8);
        backend
            .poll_retired_fences()
            .expect("terminal retirement poll");
        assert_eq!(backend.resource_report().total_owned_bytes(), Ok(0));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn retirement_admission_rejects_foreign_or_mixed_payloads_without_losing_inputs() {
        let mut owner = CpuBackend::new(
            0xCAFE,
            ResourceLimits {
                max_allocation_bytes: 8,
                max_total_bytes: 8,
                max_pending_operations: 4,
            },
        );
        let mut foreign = CpuBackend::new(
            0xCAFE,
            ResourceLimits {
                max_allocation_bytes: 8,
                max_total_bytes: 8,
                max_pending_operations: 4,
            },
        );
        assert_eq!(owner.identity(), foreign.identity());

        let foreign_buffer = foreign
            .upload_f32_classified(
                Shape::new(&[2]).expect("shape"),
                &[1.0, 2.0],
                AllocationClass::Weight,
            )
            .expect("foreign buffer");
        let owner_fence = owner.fence().expect("owner fence");
        let rejected = owner
            .fence_retirement()
            .retire(owner_fence, vec![foreign_buffer])
            .expect_err("foreign buffer must not enter owner accounting");
        assert_eq!(rejected.cause(), &ExecutorError::WrongBackend);
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));
        assert_eq!(foreign.resource_report().total_owned_bytes(), Ok(8));
        let (_, owner_fence, foreign_buffers) = rejected.into_parts();
        assert_eq!(foreign_buffers.len(), 1);
        drop(owner_fence);
        drop(foreign_buffers);
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));
        assert_eq!(foreign.resource_report().total_owned_bytes(), Ok(0));

        let owner_buffer = owner
            .upload_f32_classified(
                Shape::new(&[2]).expect("shape"),
                &[3.0, 4.0],
                AllocationClass::Weight,
            )
            .expect("owner buffer");
        let foreign_fence = foreign.fence().expect("foreign fence");
        let rejected = owner
            .fence_retirement()
            .retire(foreign_fence, vec![owner_buffer])
            .expect_err("foreign fence must not enter owner queue");
        assert_eq!(rejected.cause(), &ExecutorError::WrongBackend);
        let (_, foreign_fence, owner_buffers) = rejected.into_parts();
        assert_eq!(owner_buffers.len(), 1);
        drop(foreign_fence);
        drop(owner_buffers);
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));
        assert_eq!(foreign.resource_report().total_owned_bytes(), Ok(0));

        let owner_buffer = owner
            .upload_f32_classified(
                Shape::new(&[1]).expect("shape"),
                &[5.0],
                AllocationClass::Weight,
            )
            .expect("owner mixed buffer");
        let foreign_buffer = foreign
            .upload_f32_classified(
                Shape::new(&[1]).expect("shape"),
                &[6.0],
                AllocationClass::Weight,
            )
            .expect("foreign mixed buffer");
        let owner_fence = owner.fence().expect("mixed fence");
        let rejected = owner
            .fence_retirement()
            .retire(owner_fence, vec![owner_buffer, foreign_buffer])
            .expect_err("mixed retained set must not partially enter owner queue");
        assert_eq!(rejected.cause(), &ExecutorError::WrongBackend);
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(4));
        assert_eq!(foreign.resource_report().total_owned_bytes(), Ok(4));
        let (_, owner_fence, mut mixed) = rejected.into_parts();
        let owner_buffer = mixed.remove(0);
        let foreign_buffer = mixed.remove(0);
        owner
            .fence_retirement()
            .retire(owner_fence, vec![owner_buffer])
            .expect("returned owner payload admits without a foreign element");
        drop(foreign_buffer);
        owner.poll_retired_fences().expect("owner terminal poll");
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));
        assert_eq!(foreign.resource_report().total_owned_bytes(), Ok(0));

        let stale_buffer = owner
            .upload_f32_classified(
                Shape::new(&[2]).expect("shape"),
                &[7.0, 8.0],
                AllocationClass::Weight,
            )
            .expect("stale generation buffer");
        let stale_fence = owner.fence().expect("stale generation fence");
        owner.advance_generation().expect("generation advance");
        owner
            .fence_retirement()
            .retire(stale_fence, vec![stale_buffer])
            .expect("same actual instance accepts an older generation");
        assert_eq!(owner.resource_report().pending_operation_bytes, 8);
        owner.poll_retired_fences().expect("stale terminal poll");
        assert_eq!(owner.resource_report().total_owned_bytes(), Ok(0));
    }

    #[test]
    fn cancellation_before_submit_produces_a_ready_cancelled_completion() {
        let mut backend = CpuBackend::new(9, limits());
        backend.request_cancel();
        let mut completion = backend
            .submit_ready::<()>(Ok(()), Vec::new())
            .expect("submit");
        assert_eq!(
            completion.poll_step(),
            CompletionPoll::Ready(Err(ExecutorError::Cancelled))
        );
    }

    #[test]
    fn deferred_readback_withholds_host_output_until_completion_is_ready() {
        let mut backend = CpuBackend::new(10, limits());
        let buffer = backend
            .upload_f32(Shape::new(&[2]).expect("shape"), &[8.0, 13.0])
            .expect("buffer");
        let expected = buffer.as_slice().to_vec();
        let mut readback = CpuCompletion::deferred_for_test(
            Rc::clone(&backend.tracker),
            Ok(expected),
            vec![buffer],
            8,
        )
        .expect("deferred readback");

        assert_eq!(backend.resource_report().pending_operation_bytes, 16);
        assert_eq!(readback.poll_step(), CompletionPoll::Pending);
        assert_eq!(backend.resource_report().pending_operation_bytes, 16);
        assert_eq!(
            readback.poll_step(),
            CompletionPoll::Ready(Ok(vec![8.0, 13.0]))
        );
        assert_eq!(backend.resource_report().pending_operation_bytes, 0);
    }
}
