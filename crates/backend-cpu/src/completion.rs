// Completion ownership, readback, and fence retirement.

use super::storage::{ClassBytes, Tracker};
use super::{CpuBackend, CpuBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, ExecutorError, FenceRetirement, InferenceCompletion, Result,
    RetirementRejection,
};
use std::{cell::RefCell, rc::Rc};

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
    pub(super) fn new(tracker: Rc<RefCell<Tracker>>) -> Self {
        Self {
            tracker,
            retired: RefCell::new(Vec::new()),
        }
    }

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

    fn has_unresolved(&self) -> bool {
        !self.retired.borrow().is_empty()
    }
}

impl CpuBackend {
    /// Submit a CPU completion fence. This baseline is ready on its first poll.
    pub fn fence(&self) -> Result<CpuCompletion<()>> {
        self.submit_ready(Ok(()), Vec::new())
    }

    /// Submit f32 readback without exposing a blocking requirement through the shared trait.
    /// CPU preflights and accounts the completion-owned host result before it fallibly copies.
    /// A successful poll transfers that Vec to the caller and removes it from backend accounting.
    pub fn read_f32_async(&self, buffer: &CpuBuffer) -> Result<CpuCompletion<Vec<f32>>> {
        self.check_submit()?;
        self.check_f32_buffer(buffer)?;
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
    pub(super) fn deferred_for_test(
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
