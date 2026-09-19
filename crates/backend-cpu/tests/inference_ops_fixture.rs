#![allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::match_wild_err_arm
)]

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{
    CompletionPoll, ExecutorError, InferenceCompletion, InferenceOps, OperationKind,
    ResourceLimits, Shape, TokenIds,
};
use serde_json::{Map, Value};

const CASES: &str = include_str!("fixtures/inference-ops-001/cases.json");

fn must<T, E: core::fmt::Display>(result: core::result::Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("unexpected error: {error}"),
    }
}

fn object(value: &Value) -> &Map<String, Value> {
    match value.as_object() {
        Some(object) => object,
        None => panic!("fixture value is not an object"),
    }
}

fn field<'a>(object: &'a Map<String, Value>, name: &str) -> &'a Value {
    match object.get(name) {
        Some(value) => value,
        None => panic!("fixture field is missing: {name}"),
    }
}

fn f32_values(value: &Value) -> Vec<f32> {
    match value.as_array() {
        Some(values) => values
            .iter()
            .map(|item| match item.as_f64() {
                Some(number) => number as f32,
                None => panic!("fixture f32 value is not numeric"),
            })
            .collect(),
        None => panic!("fixture f32 values are not an array"),
    }
}

fn u32_values(value: &Value) -> Vec<u32> {
    match value.as_array() {
        Some(values) => values
            .iter()
            .map(|item| match item.as_u64() {
                Some(number) => match u32::try_from(number) {
                    Ok(value) => value,
                    Err(_) => panic!("fixture token ID does not fit u32"),
                },
                None => panic!("fixture token ID is not unsigned"),
            })
            .collect(),
        None => panic!("fixture token IDs are not an array"),
    }
}

fn shape(tensor: &Value) -> Shape {
    let object = object(tensor);
    let dimensions = match field(object, "shape").as_array() {
        Some(values) => values
            .iter()
            .map(|item| match item.as_u64() {
                Some(dimension) => dimension,
                None => panic!("fixture dimension is not unsigned"),
            })
            .collect::<Vec<_>>(),
        None => panic!("fixture shape is not an array"),
    };
    must(Shape::new(&dimensions))
}

fn values(tensor: &Value) -> Vec<f32> {
    f32_values(field(object(tensor), "f32_values"))
}

fn backend() -> CpuBackend {
    CpuBackend::new(
        0xC0DE,
        ResourceLimits {
            max_allocation_bytes: 8 * 1024 * 1024,
            max_total_bytes: 16 * 1024 * 1024,
            max_pending_operations: 4,
        },
    )
}

fn assert_within(actual: &[f32], expected: &[f32], absolute: f32, relative: f32, case_id: &str) {
    assert_eq!(actual.len(), expected.len(), "{case_id}: result length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{case_id}[{index}] is non-finite");
        let permitted = absolute + relative * expected.abs();
        assert!(
            (*actual - *expected).abs() <= permitted,
            "{case_id}[{index}] actual={actual:?} expected={expected:?} permitted={permitted:?}"
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn independent_operator_fixture_covers_every_cr01_cpu_kernel() {
    let document: Value = must(serde_json::from_str(CASES));
    let root = object(&document);
    let tolerance = object(field(root, "tolerance"));
    let absolute = match field(tolerance, "absolute").as_f64() {
        Some(value) => value as f32,
        None => panic!("fixture absolute tolerance is missing"),
    };
    let relative = match field(tolerance, "relative").as_f64() {
        Some(value) => value as f32,
        None => panic!("fixture relative tolerance is missing"),
    };
    let cases = match field(root, "cases").as_array() {
        Some(cases) => cases,
        None => panic!("fixture cases are not an array"),
    };

    let mut executed = 0_usize;
    let unsupported = 0_usize;
    for case in cases {
        let case = object(case);
        let case_id = match field(case, "id").as_str() {
            Some(id) => id,
            None => panic!("fixture case ID is not text"),
        };
        let operation = match field(case, "operation").as_str() {
            Some(operation) => operation,
            None => panic!("fixture operation is not text"),
        };
        let inputs = object(field(case, "inputs"));
        let expected = field(case, "expected");
        let mut cpu = backend();

        match operation {
            "linear" => {
                let x = field(inputs, "x");
                let weight = field(inputs, "weight");
                let input = must(cpu.upload_f32(shape(x), &values(x)));
                let weights = must(cpu.upload_f32(shape(weight), &values(weight)));
                let mut output = must(cpu.allocate_f32(shape(expected)));
                must(cpu.linear(&mut output, &input, &weights));
                assert_within(
                    &must(cpu.read_f32(&output)),
                    &values(expected),
                    absolute,
                    relative,
                    case_id,
                );
                executed += 1;
            }
            "rms_norm" => {
                let x = field(inputs, "x");
                let weight = field(inputs, "weight");
                let epsilon = match field(inputs, "epsilon_f32").as_f64() {
                    Some(value) => value as f32,
                    None => panic!("fixture RMS epsilon is missing"),
                };
                let input = must(cpu.upload_f32(shape(x), &values(x)));
                let weights = must(cpu.upload_f32(shape(weight), &values(weight)));
                let mut output = must(cpu.allocate_f32(shape(expected)));
                must(cpu.row_rms_norm(&mut output, &input, &weights, epsilon));
                assert_within(
                    &must(cpu.read_f32(&output)),
                    &values(expected),
                    absolute,
                    relative,
                    case_id,
                );
                executed += 1;
            }
            "embedding" => {
                let weight = field(inputs, "weight");
                let ids = u32_values(field(inputs, "token_ids"));
                let table = must(cpu.upload_f32(shape(weight), &values(weight)));
                let mut output = must(cpu.allocate_f32(shape(expected)));
                must(cpu.gather_rows(&mut output, &table, TokenIds::Host(&ids)));
                assert_within(
                    &must(cpu.read_f32(&output)),
                    &values(expected),
                    absolute,
                    relative,
                    case_id,
                );
                executed += 1;
            }
            "add" | "multiply" => {
                let left = field(inputs, "left");
                let right = field(inputs, "right");
                let left_buffer = must(cpu.upload_f32(shape(left), &values(left)));
                let right_buffer = must(cpu.upload_f32(shape(right), &values(right)));
                let mut output = must(cpu.allocate_f32(shape(expected)));
                if operation == "add" {
                    must(cpu.add(&mut output, &left_buffer, &right_buffer));
                } else {
                    must(cpu.multiply(&mut output, &left_buffer, &right_buffer));
                }
                assert_within(
                    &must(cpu.read_f32(&output)),
                    &values(expected),
                    absolute,
                    relative,
                    case_id,
                );
                executed += 1;
            }
            "swiglu" => {
                let gate = field(inputs, "gate");
                let up = field(inputs, "up");
                let gate_buffer = must(cpu.upload_f32(shape(gate), &values(gate)));
                let up_buffer = must(cpu.upload_f32(shape(up), &values(up)));
                let mut output = must(cpu.allocate_f32(shape(expected)));
                must(cpu.swiglu(&mut output, &gate_buffer, &up_buffer));
                assert_within(
                    &must(cpu.read_f32(&output)),
                    &values(expected),
                    absolute,
                    relative,
                    case_id,
                );
                executed += 1;
            }
            unexpected => panic!("unexpected fixture operation: {unexpected}"),
        }
    }

    assert_eq!(
        executed, 12,
        "every implemented CPU operation must use this oracle"
    );
    assert_eq!(unsupported, 0, "no fixture operation remains declared-only");
}

#[test]
fn cpu_storage_rejects_colliding_owners_stale_buffers_limits_and_bad_gathers() {
    let narrow_limits = ResourceLimits {
        max_allocation_bytes: 8,
        max_total_bytes: 12,
        max_pending_operations: 1,
    };
    let mut first = CpuBackend::new(77, narrow_limits);
    let mut second = CpuBackend::new(77, narrow_limits);
    let pair = must(first.upload_f32(must(Shape::new(&[2])), &[1.0, 2.0]));

    assert_eq!(second.read_f32(&pair), Err(ExecutorError::WrongBackend));
    match second.submit_ready::<()>(Ok(()), vec![pair]) {
        Err(ExecutorError::WrongBackend) => {}
        Err(other) => panic!("foreign retained buffer returned {other:?}"),
        Ok(_) => panic!("foreign retained buffer was accepted"),
    }
    assert_eq!(second.resource_report().pending_operations, 0);
    assert_eq!(InferenceOps::identity(&first), first.identity());

    match first.allocate_f32(must(Shape::new(&[3]))) {
        Err(ExecutorError::ResourceLimit("allocation exceeds backend capability")) => {}
        Err(other) => panic!("over-limit allocation returned {other:?}"),
        Ok(_) => panic!("over-limit allocation succeeded"),
    }
    assert_eq!(first.resource_report().total_owned_bytes(), Ok(0));

    let stale_buffer = must(first.upload_f32(must(Shape::new(&[2])), &[4.0, 5.0]));
    match first.allocate_f32(must(Shape::new(&[2]))) {
        Err(ExecutorError::ResourceLimit("allocation exceeds configured total resource limit")) => {
        }
        Err(other) => panic!("aggregate over-limit allocation returned {other:?}"),
        Ok(_) => panic!("aggregate over-limit allocation succeeded"),
    }
    must(first.advance_generation());
    assert_eq!(
        first.read_f32(&stale_buffer),
        Err(ExecutorError::StaleBuffer)
    );

    let table = must(second.upload_f32(must(Shape::new(&[2, 1])), &[0.0, 1.0]));
    let mut output = must(second.allocate_f32(must(Shape::new(&[1, 1]))));
    assert_eq!(
        second.gather_rows(&mut output, &table, TokenIds::Host(&[2])),
        Err(ExecutorError::OutOfBounds(
            "gather identifier exceeds row count"
        ))
    );
    assert!(
        second
            .capabilities()
            .operations
            .contains(OperationKind::SwiGlu)
    );
}

#[test]
fn cancellation_preserves_retention_accounting_on_success_and_error_paths() {
    let mut cpu = backend();
    let retained = must(cpu.upload_f32(must(Shape::new(&[1])), &[1.0]));
    cpu.request_cancel();
    let mut cancelled = must(cpu.submit_ready::<u32>(Ok(9), vec![retained]));
    assert_eq!(cpu.resource_report().pending_operation_bytes, 4);
    assert_eq!(
        cancelled.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::Cancelled))
    );
    assert_eq!(cpu.resource_report().pending_operation_bytes, 0);

    cpu.clear_cancel();
    let retained = must(cpu.upload_f32(must(Shape::new(&[1])), &[2.0]));
    let mut failed = must(cpu.submit_ready::<u32>(
        Err(ExecutorError::BackendFailure("synthetic operation error")),
        vec![retained],
    ));
    assert_eq!(
        failed.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::BackendFailure(
            "synthetic operation error"
        )))
    );
    assert_eq!(cpu.resource_report().pending_operations, 0);
}

#[test]
fn portable_async_fence_and_readback_complete_without_sync_api() {
    let mut cpu = backend();
    let buffer = must(cpu.upload_f32(must(Shape::new(&[2])), &[3.0, 5.0]));

    let mut fence = must(InferenceOps::fence(&cpu));
    assert_eq!(fence.poll_step(), CompletionPoll::Ready(Ok(())));

    let mut readback = must(InferenceOps::read_f32_async(&cpu, &buffer));
    assert_eq!(
        readback.poll_step(),
        CompletionPoll::Ready(Ok(vec![3.0, 5.0]))
    );
    assert_eq!(
        readback.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed))
    );
}

#[test]
fn readback_preflights_bytes_and_pending_count_before_host_copy() {
    let full_limits = ResourceLimits {
        max_allocation_bytes: 8,
        max_total_bytes: 8,
        max_pending_operations: 1,
    };
    let mut full = CpuBackend::new(900, full_limits);
    let buffer = must(full.upload_f32(must(Shape::new(&[2])), &[1.0, 2.0]));
    match full.read_f32_async(&buffer) {
        Err(ExecutorError::ResourceLimit("allocation exceeds configured total resource limit")) => {
        }
        Err(other) => panic!("full readback returned {other:?}"),
        Ok(_) => panic!("full readback unexpectedly allocated host output"),
    }
    assert_eq!(full.resource_report().total_owned_bytes(), Ok(8));
    assert_eq!(full.resource_report().pending_operations, 0);
    assert_eq!(
        full.read_f32(&buffer),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    );

    let no_pending_limits = ResourceLimits {
        max_allocation_bytes: 16,
        max_total_bytes: 16,
        max_pending_operations: 0,
    };
    let mut no_pending = CpuBackend::new(901, no_pending_limits);
    let buffer = must(no_pending.upload_f32(must(Shape::new(&[2])), &[3.0, 4.0]));
    match no_pending.read_f32_async(&buffer) {
        Err(ExecutorError::ResourceLimit("pending operation count exceeds configured limit")) => {}
        Err(other) => panic!("zero-pending readback returned {other:?}"),
        Ok(_) => panic!("zero-pending readback unexpectedly allocated host output"),
    }
    assert_eq!(no_pending.resource_report().total_owned_bytes(), Ok(8));
    assert_eq!(no_pending.resource_report().pending_operations, 0);
}

#[test]
fn readback_result_bytes_release_on_ready_drop_and_cancel() {
    let mut cpu = CpuBackend::new(
        902,
        ResourceLimits {
            max_allocation_bytes: 32,
            max_total_bytes: 64,
            max_pending_operations: 2,
        },
    );
    let buffer = must(cpu.upload_f32(must(Shape::new(&[2])), &[8.0, 13.0]));

    let mut ready = must(cpu.read_f32_async(&buffer));
    let report = cpu.resource_report();
    assert_eq!(report.scratch_bytes, 8);
    assert_eq!(report.pending_operation_bytes, 8);
    assert_eq!(report.total_owned_bytes(), Ok(16));
    assert_eq!(
        ready.poll_step(),
        CompletionPoll::Ready(Ok(vec![8.0, 13.0]))
    );
    assert_eq!(cpu.resource_report().total_owned_bytes(), Ok(8));
    assert_eq!(cpu.resource_report().pending_operations, 0);

    let dropped = must(cpu.read_f32_async(&buffer));
    assert_eq!(cpu.resource_report().total_owned_bytes(), Ok(16));
    drop(dropped);
    assert_eq!(cpu.resource_report().total_owned_bytes(), Ok(8));
    assert_eq!(cpu.resource_report().pending_operations, 0);

    let mut cancelled = must(cpu.read_f32_async(&buffer));
    assert_eq!(cpu.resource_report().total_owned_bytes(), Ok(16));
    must(cancelled.cancel());
    assert_eq!(cpu.resource_report().total_owned_bytes(), Ok(8));
    assert_eq!(cpu.resource_report().pending_operations, 0);
    assert_eq!(
        cancelled.poll_step(),
        CompletionPoll::Ready(Err(ExecutorError::Cancelled))
    );
}

#[test]
fn held_readback_result_bytes_block_allocations_until_ready_drop_or_cancel() {
    let limits = ResourceLimits {
        max_allocation_bytes: 16,
        max_total_bytes: 16,
        max_pending_operations: 2,
    };

    let mut ready_cpu = CpuBackend::new(0xA11C, limits);
    let ready_source = must(ready_cpu.upload_f32(must(Shape::new(&[2])), &[1.0, 2.0]));
    let mut ready = must(ready_cpu.read_f32_async(&ready_source));
    assert_eq!(ready_cpu.resource_report().total_owned_bytes(), Ok(16));
    assert!(matches!(
        ready_cpu.allocate_f32(must(Shape::new(&[1]))),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    ));
    assert_eq!(ready.poll_step(), CompletionPoll::Ready(Ok(vec![1.0, 2.0])));
    let ready_recovered = must(ready_cpu.allocate_f32(must(Shape::new(&[1]))));
    drop(ready_recovered);

    let mut dropped_cpu = CpuBackend::new(0xA11D, limits);
    let dropped_source = must(dropped_cpu.upload_f32(must(Shape::new(&[2])), &[3.0, 4.0]));
    let dropped = must(dropped_cpu.read_f32_async(&dropped_source));
    assert!(matches!(
        dropped_cpu.allocate_f32(must(Shape::new(&[1]))),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    ));
    drop(dropped);
    let dropped_recovered = must(dropped_cpu.allocate_f32(must(Shape::new(&[1]))));
    drop(dropped_recovered);

    let mut cancelled_cpu = CpuBackend::new(0xA11E, limits);
    let cancelled_source = must(cancelled_cpu.upload_f32(must(Shape::new(&[2])), &[5.0, 6.0]));
    let mut cancelled = must(cancelled_cpu.read_f32_async(&cancelled_source));
    assert!(matches!(
        cancelled_cpu.allocate_f32(must(Shape::new(&[1]))),
        Err(ExecutorError::ResourceLimit(
            "allocation exceeds configured total resource limit"
        ))
    ));
    must(cancelled.cancel());
    let cancelled_recovered = must(cancelled_cpu.allocate_f32(must(Shape::new(&[1]))));
    drop(cancelled_recovered);
}
