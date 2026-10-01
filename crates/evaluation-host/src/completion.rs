//! Frozen completion policy: checked deadline and a 1 ms sleep per Pending.
use crate::HostResult;
use minifield_engine_api::{CompletionPoll, InferenceCompletion};
use minifield_executor_core::LoaderPoll;
use std::time::{Duration, Instant};

pub(crate) fn deadline(seconds: f64) -> HostResult<Instant> {
    Instant::now()
        .checked_add(Duration::try_from_secs_f64(seconds)?)
        .ok_or_else(|| "completion deadline overflows clock".into())
}

pub(crate) fn check(limit: Instant) -> HostResult<()> {
    if Instant::now() >= limit {
        Err("operation exceeded its completion deadline".into())
    } else {
        Ok(())
    }
}

pub(crate) fn wait<C: InferenceCompletion>(task: &mut C, limit: Instant) -> HostResult<C::Output> {
    loop {
        if let Err(error) = check(limit) {
            let _ = task.cancel();
            return Err(error);
        }
        let poll = task.poll_step();
        // A CPU backend may perform the full pass inside poll_step.
        if let Err(error) = check(limit) {
            let _ = task.cancel();
            return Err(error);
        }
        match poll {
            CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            CompletionPoll::Ready(result) => return Ok(result?),
        }
    }
}

pub(crate) fn wait_loader<T>(
    limit: Instant,
    mut poll: impl FnMut() -> LoaderPoll<T>,
) -> HostResult<T> {
    loop {
        check(limit)?;
        let observed = poll();
        check(limit)?;
        match observed {
            LoaderPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            LoaderPoll::Ready(result) => return Ok(result?),
        }
    }
}
