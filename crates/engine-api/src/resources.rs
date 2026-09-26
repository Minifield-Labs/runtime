//! Resource accounting and nonblocking completion ownership.

use crate::{ExecutorError, Result};

/// Limits counted by a backend-owned resource arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceLimits {
    pub max_allocation_bytes: u64,
    pub max_total_bytes: u64,
    pub max_pending_operations: u32,
}

/// Lifetime/accounting class for backend-owned buffers.
///
/// The class describes the caller's logical ownership purpose; it does not loosen a backend's
/// aggregate allocation limit. Existing unclassified allocation helpers deliberately default to
/// `Scratch` so older callers cannot accidentally publish mutable model state as weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationClass {
    Weight,
    Cache,
    Scratch,
    Branch,
}

impl ResourceLimits {
    pub fn validate_allocation(self, allocation_bytes: u64, current_bytes: u64) -> Result<()> {
        if allocation_bytes > self.max_allocation_bytes {
            return Err(ExecutorError::ResourceLimit(
                "allocation exceeds configured per-buffer limit",
            ));
        }
        let total = current_bytes
            .checked_add(allocation_bytes)
            .ok_or(ExecutorError::Overflow("resource total overflows u64"))?;
        if total > self.max_total_bytes {
            return Err(ExecutorError::ResourceLimit(
                "allocation exceeds configured total resource limit",
            ));
        }
        Ok(())
    }
}

/// Resource accounting includes owned resident, cache, scratch, branch, and pending bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceReport {
    pub resident_weight_bytes: u64,
    pub cache_bytes: u64,
    pub scratch_bytes: u64,
    pub staged_branch_bytes: u64,
    pub pending_operation_bytes: u64,
    pub pending_operations: u32,
}

impl ResourceReport {
    pub fn total_owned_bytes(self) -> Result<u64> {
        self.resident_weight_bytes
            .checked_add(self.cache_bytes)
            .and_then(|value| value.checked_add(self.scratch_bytes))
            .and_then(|value| value.checked_add(self.staged_branch_bytes))
            .and_then(|value| value.checked_add(self.pending_operation_bytes))
            .ok_or(ExecutorError::Overflow(
                "resource report total overflows u64",
            ))
    }

    pub fn validate(self, limits: ResourceLimits) -> Result<()> {
        if self.pending_operations > limits.max_pending_operations {
            return Err(ExecutorError::ResourceLimit(
                "pending operation count exceeds configured limit",
            ));
        }
        if self.total_owned_bytes()? > limits.max_total_bytes {
            return Err(ExecutorError::ResourceLimit(
                "owned resource bytes exceed configured limit",
            ));
        }
        Ok(())
    }
}

/// One nonblocking observation of an inference completion.
#[derive(Debug, PartialEq)]
pub enum CompletionPoll<T> {
    Pending,
    Ready(Result<T>),
}

/// A portable poll/step boundary. Implementations need not be Send, Sync, threaded, or blocking.
pub trait InferenceCompletion {
    type Output;

    /// Advance or observe one bounded completion step.
    fn poll_step(&mut self) -> CompletionPoll<Self::Output>;

    /// Request cancellation before an operation is irrevocably submitted.
    fn cancel(&mut self) -> Result<()>;
}

/// A retirement admission rejection that preserves the complete unresolved payload.
///
/// A queue must return this value before changing any ownership or accounting when the fence or
/// any retained buffer belongs to another actual backend instance. The caller can inspect the
/// cause and route or retain the original fence and buffers without dropping them.
#[derive(Debug)]
pub struct RetirementRejection<Fence, Buffer> {
    cause: ExecutorError,
    fence: Fence,
    retained: Vec<Buffer>,
}

impl<Fence, Buffer> RetirementRejection<Fence, Buffer> {
    /// Build an ownership-preserving rejection before retirement admission mutates state.
    #[must_use]
    pub const fn new(cause: ExecutorError, fence: Fence, retained: Vec<Buffer>) -> Self {
        Self {
            cause,
            fence,
            retained,
        }
    }

    #[must_use]
    pub const fn cause(&self) -> &ExecutorError {
        &self.cause
    }

    /// Recover the cause and every input exactly as supplied to retirement admission.
    #[must_use]
    pub fn into_parts(self) -> (ExecutorError, Fence, Vec<Buffer>) {
        (self.cause, self.fence, self.retained)
    }
}

/// Backend-owned quarantine for an unresolved submission fence and the buffers the submission
/// still references.
///
/// A caller may abandon a higher-level task after cancellation cannot be confirmed. In that
/// case, the task transfers the fence and buffers here instead of dropping either one. The
/// backend keeps them alive until `poll_retired` observes a terminal completion result. This
/// trait deliberately does not require threads, Send, or Sync.
pub trait FenceRetirement<Fence, Buffer> {
    /// Admit one unresolved fence and every buffer retained by that submission.
    ///
    /// Admission checks actual backend ownership before changing accounting. A rejection returns
    /// every consumed input so a normal caller can route it safely.
    fn retire(
        &self,
        fence: Fence,
        retained: Vec<Buffer>,
    ) -> core::result::Result<(), RetirementRejection<Fence, Buffer>>;

    /// Conservatively retain a payload that a task cannot return because it is being dropped.
    ///
    /// The payload must have come from this queue's `retire` rejection. Implementations retain it
    /// without cross-instance accounting and release it only after the carried fence is terminal.
    fn quarantine_rejected(&self, rejected: RetirementRejection<Fence, Buffer>);

    /// Advance all abandoned fences once. A ready success or error releases its retained buffers.
    fn poll_retired(&self);

    /// Return whether any fence may still reference an abandoned task's source snapshots.
    ///
    /// Backends that cannot expose this information must keep the conservative default. Shared
    /// callers then retain source snapshots until the executor itself is dropped or reloaded.
    fn has_unresolved(&self) -> bool {
        true
    }
}

/// Common explicit completion state used by immediate portable adapters and tests.
#[derive(Debug)]
pub struct ReadyCompletion<T> {
    result: Option<Result<T>>,
}

impl<T> ReadyCompletion<T> {
    #[must_use]
    pub fn new(result: Result<T>) -> Self {
        Self {
            result: Some(result),
        }
    }
}

impl<T> InferenceCompletion for ReadyCompletion<T> {
    type Output = T;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.result.is_some() {
            self.result = Some(Err(ExecutorError::Cancelled));
            Ok(())
        } else {
            Err(ExecutorError::CompletionConsumed)
        }
    }
}
