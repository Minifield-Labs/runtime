//! Ordered phases: load and prepare, warm, snapshot, measure, snapshot.
use crate::{
    HostResult,
    backend::HostBackend,
    completion::deadline,
    loading, measurement,
    preparation::{Classifier, Pointer, PreparedModel},
    request::{Inputs, Request},
};
use serde_json::{Value, json};
use std::time::Instant;

pub(crate) fn evaluate<B: HostBackend>(
    backend: B,
    request: &Request,
    started: Instant,
) -> HostResult<Value> {
    request.validate()?;
    let limit = deadline(request.deadline_seconds)?;
    let assets = loading::assets(request, limit)?;
    let identity = assets.identity;
    if request.task == "classifier" {
        let model = Classifier::prepare(
            backend,
            assets.model,
            &assets.inputs.cases,
            &assets.tokenizer,
            request,
            limit,
        )?;
        phases(model, &assets.inputs, request, started, &identity)
    } else {
        let model = Pointer::prepare(backend, assets.model, &assets.inputs.cases, request, limit)?;
        phases(model, &assets.inputs, request, started, &identity)
    }
}

fn phases(
    mut model: impl PreparedModel,
    inputs: &Inputs,
    request: &Request,
    started: Instant,
    artifacts: &Value,
) -> HostResult<Value> {
    let initialization = started.elapsed().as_secs_f64();
    let warmup_start = Instant::now();
    for _ in 0..request.warmups {
        for index in 0..inputs.cases.len() {
            model.predict(index, request.deadline_seconds)?;
        }
    }
    let warmup = warmup_start.elapsed().as_secs_f64();
    let before = model.snapshot()?;
    let cases = inputs
        .cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            measurement::case(case, request.measured_cycles, || {
                model.predict(index, request.deadline_seconds)
            })
        })
        .collect::<HostResult<Vec<_>>>()?;
    let after = model.snapshot()?;
    let counts = measurement::dispatch_delta(&before, &after)?;
    Ok(
        json!({"schema_version":1,"backend":after.evidence,"artifacts":artifacts,"initialization_seconds":initialization,
        "warmup_seconds":warmup,"cases":cases,"dispatch_counts":counts,"resources":{"accounted_bytes":after.accounted,"peak_accounted_bytes":after.peak}}),
    )
}
