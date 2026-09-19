//! Completion types for the wgpu backend.
//!
//! `fence()` snapshots the pending batch, submits it, and returns a serial.
//! Completion is confirmed by `on_submitted_work_done` callbacks that bump a
//! shared counter; `poll_step` pumps device maintenance and compares serials,
//! so it never blocks.
//!
//! `read_f32_async` records a staging copy and a deferred `map_buffer_on_submit`
//! into the same batch, then submits. The completion owns the staging buffer
//! until a ready `poll_step` copies the mapped range out and returns the
//! buffer to its pool. A readback dropped or cancelled before its map resolves
//! parks the staging buffer until the callback lands, so it can never be
//! recycled under an in-flight map.
//!
//! The retirement queue mirrors the CPU backend: `retire` admits an unresolved
//! fence plus the buffers its submission still references after ownership
//! checks, `poll_retired` drops each entry once its fence reports a terminal
//! result, and accounting moves between live and pending buckets on admission
//! and release.

use std::{cell::RefCell, rc::Rc, sync::Arc};

use minifield_engine_api::{
    AllocationClass, CompletionPoll, ExecutorError, FenceRetirement, InferenceCompletion, Result,
    RetirementRejection,
};
use portable_atomic::{AtomicBool, Ordering};

use crate::{
    WgpuBuffer,
    device::{ClassBytes, DeviceInner, PooledBuf},
};

/// Fence returned by `fence()`. Becomes ready when its submission serial is
/// confirmed complete by the `on_submitted_work_done` callback.
pub struct WgpuFence {
    pub(crate) device: Rc<DeviceInner>,
    pub(crate) serial: u64,
    /// Submit-time stamp for `MINIFIELD_WGPU_STATS` wait-latency accounting.
    created: Option<std::time::Instant>,
    terminal: Option<Result<()>>,
    consumed: bool,
    charged: bool,
}

impl WgpuFence {
    pub(crate) fn new(device: Rc<DeviceInner>, serial: u64) -> Self {
        let created = device.stats_timing().then(std::time::Instant::now);
        Self {
            device,
            serial,
            created,
            terminal: None,
            consumed: false,
            charged: true,
        }
    }

    /// Drop the pending-operation charge once. Called on Ready, cancel, and drop.
    fn release_charge(&mut self) {
        if self.charged {
            self.charged = false;
            let mut tracker = self.device.tracker.borrow_mut();
            debug_assert!(tracker.pending_operations > 0);
            tracker.pending_operations = tracker.pending_operations.saturating_sub(1);
        }
    }
}

impl Drop for WgpuFence {
    fn drop(&mut self) {
        self.release_charge();
    }
}

impl InferenceCompletion for WgpuFence {
    type Output = ();

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.consumed {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        if self.terminal.is_none() {
            self.device.reap();
            if self.device.confirmed_serial() >= self.serial {
                self.terminal = Some(Ok(()));
            }
        }
        match self.terminal.take() {
            Some(result) => {
                self.consumed = true;
                self.release_charge();
                let mut stats = self.device.stats.borrow_mut();
                stats.fences += 1;
                if let Some(created) = self.created {
                    stats.fence_wait_ns +=
                        u64::try_from(created.elapsed().as_nanos()).unwrap_or(u64::MAX);
                }
                CompletionPoll::Ready(result)
            }
            None => CompletionPoll::Pending,
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.consumed {
            return Err(ExecutorError::CompletionConsumed);
        }
        if self.terminal.is_none() {
            // The batch is already submitted; the fence resolves cancelled on
            // the next poll rather than blocking on GPU work.
            self.terminal = Some(Err(ExecutorError::Cancelled));
        }
        Ok(())
    }
}

/// Readback returned by `read_f32_async`. Owns the staging buffer until a
/// ready poll copies the mapped range out or the object is dropped.
pub struct WgpuReadback {
    device: Rc<DeviceInner>,
    staging: Option<PooledBuf>,
    flag: Arc<AtomicBool>,
    outcome: Arc<std::sync::Mutex<Option<bool>>>,
    bytes: u64,
    /// Submit-time stamp for `MINIFIELD_WGPU_STATS` wait-latency accounting.
    created: Option<std::time::Instant>,
    terminal: Option<Result<Vec<f32>>>,
    consumed: bool,
    charged: bool,
}

impl WgpuReadback {
    pub(crate) fn new(
        device: Rc<DeviceInner>,
        staging: PooledBuf,
        flag: Arc<AtomicBool>,
        outcome: Arc<std::sync::Mutex<Option<bool>>>,
        bytes: u64,
    ) -> Self {
        let created = device.stats_timing().then(std::time::Instant::now);
        Self {
            device,
            staging: Some(staging),
            flag,
            outcome,
            bytes,
            created,
            terminal: None,
            consumed: false,
            charged: true,
        }
    }

    fn release_charge(&mut self) {
        if self.charged {
            self.charged = false;
            let mut tracker = self.device.tracker.borrow_mut();
            debug_assert!(tracker.pending_operations > 0);
            debug_assert!(tracker.pending_result_bytes >= self.bytes);
            tracker.pending_operations = tracker.pending_operations.saturating_sub(1);
            tracker.pending_result_bytes = tracker.pending_result_bytes.saturating_sub(self.bytes);
        }
    }

    /// Resolve the staged map into `terminal`, releasing the staging buffer.
    /// Does nothing while the map callback has not fired.
    fn settle(&mut self) {
        if self.terminal.is_some() || !self.flag.load(Ordering::Acquire) {
            return;
        }
        let ok = *self
            .outcome
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(staging) = self.staging.take() else {
            self.terminal = Some(Err(ExecutorError::BackendFailure(
                "wgpu readback staging buffer missing",
            )));
            return;
        };
        if ok != Some(true) {
            self.device.release_staging(staging, false);
            self.terminal = Some(Err(ExecutorError::BackendFailure(
                "wgpu readback map failed",
            )));
            return;
        }
        if self.bytes == 0 {
            self.device.release_staging(staging, true);
            self.terminal = Some(Ok(Vec::new()));
            return;
        }
        let range = staging.buffer.slice(0..self.bytes);
        if let Ok(view) = range.get_mapped_range() {
            let values = view
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<f32>>();
            drop(view);
            self.device.release_staging(staging, true);
            self.terminal = Some(Ok(values));
        } else {
            self.device.release_staging(staging, true);
            self.terminal = Some(Err(ExecutorError::BackendFailure(
                "wgpu mapped readback range unavailable",
            )));
        }
    }
}

impl Drop for WgpuReadback {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            if self.flag.load(Ordering::Acquire) {
                let ok = *self
                    .outcome
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                self.device.release_staging(staging, ok == Some(true));
            } else {
                self.device.zombie_staging(
                    staging,
                    Arc::clone(&self.flag),
                    Arc::clone(&self.outcome),
                );
            }
        }
        self.release_charge();
    }
}

impl InferenceCompletion for WgpuReadback {
    type Output = Vec<f32>;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        if self.consumed {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        self.device.reap();
        self.settle();
        match self.terminal.take() {
            Some(result) => {
                self.consumed = true;
                self.release_charge();
                let mut stats = self.device.stats.borrow_mut();
                stats.readbacks += 1;
                if let Some(created) = self.created {
                    stats.readback_wait_ns +=
                        u64::try_from(created.elapsed().as_nanos()).unwrap_or(u64::MAX);
                }
                CompletionPoll::Ready(result)
            }
            None => CompletionPoll::Pending,
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.consumed {
            return Err(ExecutorError::CompletionConsumed);
        }
        if self.terminal.is_none() {
            if let Some(staging) = self.staging.take() {
                if self.flag.load(Ordering::Acquire) {
                    let ok = *self
                        .outcome
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    self.device.release_staging(staging, ok == Some(true));
                } else {
                    self.device.zombie_staging(
                        staging,
                        Arc::clone(&self.flag),
                        Arc::clone(&self.outcome),
                    );
                }
            }
            self.terminal = Some(Err(ExecutorError::Cancelled));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct RetirementAccounting {
    bytes: u64,
    by_class: ClassBytes,
}

struct RetiredWgpuFence {
    fence: WgpuFence,
    _retained: Vec<WgpuBuffer>,
    accounting: Option<RetirementAccounting>,
}

/// Backend-owned quarantine for fences whose original task was abandoned.
///
/// Retained buffers stay alive (and accounted) until their fence reports a
/// terminal result; `poll_retired` advances every admitted fence once.
pub struct WgpuFenceRetirement {
    device: Rc<DeviceInner>,
    retired: RefCell<Vec<RetiredWgpuFence>>,
}

impl WgpuFenceRetirement {
    pub(crate) fn new(device: Rc<DeviceInner>) -> Self {
        Self {
            device,
            retired: RefCell::new(Vec::new()),
        }
    }

    fn retained_accounting(&self, retained: &[WgpuBuffer]) -> Result<RetirementAccounting> {
        let mut by_class = ClassBytes::default();
        for buffer in retained {
            if !Rc::ptr_eq(&buffer.device, &self.device) {
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
        let tracker = self.device.tracker.borrow();
        let next_total = tracker
            .pending_retained_bytes
            .checked_add(accounting.bytes)
            .ok_or(ExecutorError::Overflow(
                "retired wgpu buffer bytes overflow pending accounting",
            ))?;
        let next_weight = tracker
            .pending_retained_by_class
            .weight
            .checked_add(accounting.by_class.weight)
            .ok_or(ExecutorError::Overflow(
                "retired wgpu weight bytes overflow pending accounting",
            ))?;
        let next_cache = tracker
            .pending_retained_by_class
            .cache
            .checked_add(accounting.by_class.cache)
            .ok_or(ExecutorError::Overflow(
                "retired wgpu cache bytes overflow pending accounting",
            ))?;
        let next_scratch = tracker
            .pending_retained_by_class
            .scratch
            .checked_add(accounting.by_class.scratch)
            .ok_or(ExecutorError::Overflow(
                "retired wgpu scratch bytes overflow pending accounting",
            ))?;
        let next_branch = tracker
            .pending_retained_by_class
            .branch
            .checked_add(accounting.by_class.branch)
            .ok_or(ExecutorError::Overflow(
                "retired wgpu branch bytes overflow pending accounting",
            ))?;
        if next_total > tracker.live_bytes
            || next_weight > tracker.live_by_class.weight
            || next_cache > tracker.live_by_class.cache
            || next_scratch > tracker.live_by_class.scratch
            || next_branch > tracker.live_by_class.branch
        {
            return Err(ExecutorError::BackendFailure(
                "retired wgpu buffers exceed actual backend live ownership",
            ));
        }
        Ok(())
    }

    fn charge(&self, accounting: RetirementAccounting) {
        let mut tracker = self.device.tracker.borrow_mut();
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
        let mut tracker = self.device.tracker.borrow_mut();
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

impl FenceRetirement<WgpuFence, WgpuBuffer> for WgpuFenceRetirement {
    fn retire(
        &self,
        fence: WgpuFence,
        retained: Vec<WgpuBuffer>,
    ) -> core::result::Result<(), RetirementRejection<WgpuFence, WgpuBuffer>> {
        if !Rc::ptr_eq(&fence.device, &self.device) {
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
        self.retired.borrow_mut().push(RetiredWgpuFence {
            fence,
            _retained: retained,
            accounting: Some(accounting),
        });
        Ok(())
    }

    fn quarantine_rejected(&self, rejected: RetirementRejection<WgpuFence, WgpuBuffer>) {
        let (_, fence, retained) = rejected.into_parts();
        self.retired.borrow_mut().push(RetiredWgpuFence {
            fence,
            _retained: retained,
            accounting: None,
        });
    }

    fn poll_retired(&self) {
        self.device.reap();
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
