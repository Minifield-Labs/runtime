//! Nonblocking submission observation and abandoned-task ownership.
use crate::{Allocation, Bucket, Device, MetalBuffer, Submission};
use minifield_engine_api::{
    CompletionPoll, ExecutorError, FenceRetirement, InferenceCompletion, Result,
    RetirementRejection,
};
use std::{cell::RefCell, rc::Rc};

/// One submitted native Metal completion boundary.
pub struct MetalFence {
    device: Rc<Device>,
    submission: Rc<Submission>,
    consumed: bool,
}
impl MetalFence {
    pub(crate) fn new(device: Rc<Device>, submission: Rc<Submission>) -> Self {
        Self {
            device,
            submission,
            consumed: false,
        }
    }
}
impl InferenceCompletion for MetalFence {
    type Output = ();
    fn poll_step(&mut self) -> CompletionPoll<()> {
        if self.consumed {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        self.device.reap();
        match self.submission.poll() {
            Some(result) => {
                self.consumed = true;
                CompletionPoll::Ready(result)
            }
            None => CompletionPoll::Pending,
        }
    }
    fn cancel(&mut self) -> Result<()> {
        if self.consumed {
            return Err(ExecutorError::CompletionConsumed);
        }
        // Submission is irrevocable. Keeping a real fence unresolved lets the
        // model executor transfer its retained state into retirement safely.
        Err(ExecutorError::Unsupported(
            "submitted Metal work cannot be cancelled",
        ))
    }
}

/// Snapshot readback. Its private staging allocation stays alive until completion.
pub struct MetalReadback {
    device: Rc<Device>,
    submission: Rc<Submission>,
    staging: Option<Rc<Allocation>>,
    bytes: u64,
    reserved: u64,
    consumed: bool,
}
impl MetalReadback {
    pub(crate) fn new(
        device: Rc<Device>,
        submission: Rc<Submission>,
        staging: Rc<Allocation>,
        bytes: u64,
        reserved: u64,
    ) -> Self {
        Self {
            device,
            submission,
            staging: Some(staging),
            bytes,
            reserved,
            consumed: false,
        }
    }
    fn release(&mut self) {
        if self.reserved > 0 {
            self.device
                .accounting
                .borrow_mut()
                .subtract(Bucket::Pending, self.reserved);
            self.reserved = 0;
        }
        self.staging = None;
    }
}
impl Drop for MetalReadback {
    fn drop(&mut self) {
        self.release();
    }
}
impl InferenceCompletion for MetalReadback {
    type Output = Vec<f32>;
    fn poll_step(&mut self) -> CompletionPoll<Vec<f32>> {
        if self.consumed {
            return CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed));
        }
        self.device.reap();
        let Some(result) = self.submission.poll() else {
            return CompletionPoll::Pending;
        };
        self.consumed = true;
        let output = result.and_then(|()| {
            let staging = self.staging.as_ref().ok_or(ExecutorError::BackendFailure(
                "Metal readback staging missing",
            ))?;
            let length = usize::try_from(self.bytes).map_err(|_| {
                ExecutorError::ResourceLimit("Metal readback exceeds address space")
            })?;
            let bytes = staging
                .raw
                .read_completed(length, &self.submission.command)?;
            let mut values = Vec::new();
            values.try_reserve_exact(length / 4).map_err(|_| {
                ExecutorError::ResourceLimit("Metal readback result allocation failed")
            })?;
            for word in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([word[0], word[1], word[2], word[3]]));
            }
            Ok(values)
        });
        self.release();
        CompletionPoll::Ready(output)
    }
    fn cancel(&mut self) -> Result<()> {
        if self.consumed {
            return Err(ExecutorError::CompletionConsumed);
        }
        Err(ExecutorError::Unsupported(
            "submitted Metal readback cannot be cancelled",
        ))
    }
}

struct Retired {
    fence: MetalFence,
    _retained: Vec<MetalBuffer>,
}
/// Backend-owned unresolved fences, including rejected ownership payloads.
pub struct MetalFenceRetirement {
    device: Rc<Device>,
    retired: RefCell<Vec<Retired>>,
    rejected: RefCell<Vec<Retired>>,
}
impl MetalFenceRetirement {
    pub(crate) fn new(device: Rc<Device>) -> Self {
        Self {
            device,
            retired: RefCell::new(Vec::new()),
            rejected: RefCell::new(Vec::new()),
        }
    }
    fn poll(entries: &RefCell<Vec<Retired>>) {
        entries
            .borrow_mut()
            .retain_mut(|entry| matches!(entry.fence.poll_step(), CompletionPoll::Pending));
    }
}
impl FenceRetirement<MetalFence, MetalBuffer> for MetalFenceRetirement {
    fn retire(
        &self,
        fence: MetalFence,
        retained: Vec<MetalBuffer>,
    ) -> core::result::Result<(), RetirementRejection<MetalFence, MetalBuffer>> {
        let foreign = !fence.device.lease.same_actual_instance(&self.device.lease)
            || retained
                .iter()
                .any(|b| !b.lease.same_actual_instance(&self.device.lease));
        if foreign {
            return Err(RetirementRejection::new(
                ExecutorError::WrongBackend,
                fence,
                retained,
            ));
        }
        self.retired.borrow_mut().push(Retired {
            fence,
            _retained: retained,
        });
        Ok(())
    }
    fn quarantine_rejected(&self, rejected: RetirementRejection<MetalFence, MetalBuffer>) {
        let (_, fence, retained) = rejected.into_parts();
        self.rejected.borrow_mut().push(Retired {
            fence,
            _retained: retained,
        });
    }
    fn poll_retired(&self) {
        Self::poll(&self.retired);
        Self::poll(&self.rejected);
    }
    fn has_unresolved(&self) -> bool {
        !self.retired.borrow().is_empty() || !self.rejected.borrow().is_empty()
    }
}
