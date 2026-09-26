// CPU ownership and operation contract regressions.

use super::{CpuBackend, CpuCompletion};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, ExecutorError, FenceRetirement, InferenceCompletion,
    InferenceOps, ResourceLimits, Shape, TokenIds,
};
use std::rc::Rc;

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

    let mut pending =
        CpuCompletion::deferred_for_test(Rc::clone(&backend.tracker), Ok(5_u32), vec![buffer], 0)
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

#[test]
fn argmax_picks_first_strict_max_and_feeds_device_gather() {
    let mut backend = CpuBackend::new(11, limits());
    let logits = backend
        .upload_f32(
            Shape::new(&[3, 4]).expect("shape"),
            &[
                0.0, 9.0, -1.0, 9.0, // tie at 1 and 3: first wins
                5.0, 1.0, 2.0, 3.0, // clean max at 0
                1.0, 2.0, 3.0, 4.0, // max at 3
            ],
        )
        .expect("logits");
    let mut out = backend
        .allocate_f32(Shape::new(&[3]).expect("out"))
        .expect("out");
    backend.argmax(&mut out, &logits).expect("argmax");
    assert_eq!(out.as_slice(), &[1.0, 0.0, 3.0]);

    // The device-id path resolves the argmax buffer into gather selectors.
    let table = backend
        .upload_f32(
            Shape::new(&[4, 2]).expect("table"),
            &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        )
        .expect("table");
    let mut gathered = backend
        .allocate_f32(Shape::new(&[3, 2]).expect("gathered"))
        .expect("gathered");
    backend
        .gather_rows(&mut gathered, &table, TokenIds::Device(&out))
        .expect("device gather");
    assert_eq!(gathered.as_slice(), &[2.0, 2.5, 1.0, 1.5, 4.0, 4.5]);

    // Invalid operands: rank-1 input, mismatched output. Non-finite input
    // is rejected at operand validation on this eager backend rather than
    // propagating NaN like the deferred wgpu path.
    let flat = backend
        .upload_f32(Shape::new(&[4]).expect("flat"), &[0.0; 4])
        .expect("flat");
    assert!(backend.argmax(&mut out, &flat).is_err());
    let mut wrong = backend
        .allocate_f32(Shape::new(&[2]).expect("wrong"))
        .expect("wrong");
    assert!(backend.argmax(&mut wrong, &logits).is_err());

    let mut poisoned = backend
        .upload_f32(Shape::new(&[3, 4]).expect("shape"), &[1.0; 12])
        .expect("poisoned");
    poisoned.values[5] = f32::NAN;
    assert!(backend.argmax(&mut out, &poisoned).is_err());
    let mut bad_ids = backend
        .upload_f32(Shape::new(&[1]).expect("ids"), &[1.0])
        .expect("ids");
    bad_ids.values[0] = f32::NAN;
    assert!(
        backend
            .gather_rows(&mut gathered, &table, TokenIds::Device(&bad_ids))
            .is_err()
    );
}

#[test]
fn gather_columns_preserves_caller_order_duplicates_and_rows() {
    let mut backend = CpuBackend::new(12, limits());
    let input = backend
        .upload_f32(
            Shape::new(&[3, 4]).expect("input"),
            &[
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
            ],
        )
        .expect("input");
    let mut out = backend
        .allocate_f32(Shape::new(&[3, 3]).expect("out"))
        .expect("out");
    backend
        .gather_columns(&mut out, &input, &[2, 0, 2])
        .expect("gather");
    assert_eq!(
        out.as_slice(),
        &[2.0, 0.0, 2.0, 12.0, 10.0, 12.0, 22.0, 20.0, 22.0]
    );

    let mut empty = backend
        .allocate_f32(Shape::new(&[3, 0]).expect("empty"))
        .expect("empty");
    backend
        .gather_columns(&mut empty, &input, &[])
        .expect("empty gather");

    assert!(backend.gather_columns(&mut out, &input, &[1, 4]).is_err());
    assert!(
        backend
            .gather_columns(&mut out, &input, &[u32::MAX])
            .is_err()
    );
    let flat = backend
        .upload_f32(Shape::new(&[4]).expect("flat"), &[0.0; 4])
        .expect("flat");
    assert!(backend.gather_columns(&mut out, &flat, &[0]).is_err());
    let mut wrong = backend
        .allocate_f32(Shape::new(&[3, 3]).expect("wrong"))
        .expect("wrong");
    assert!(backend.gather_columns(&mut wrong, &input, &[0, 1]).is_err());
    let mut wrong_rows = backend
        .allocate_f32(Shape::new(&[2, 2]).expect("wrong rows"))
        .expect("wrong rows");
    assert!(
        backend
            .gather_columns(&mut wrong_rows, &input, &[0, 1])
            .is_err()
    );
}
