//! Fixed-work measurement. Warmups and controller spacing stay outside samples.
use crate::{
    HostResult,
    backend::HostBackend,
    prediction::{Prediction, finite},
    request::Case,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Instant};

pub(crate) struct Snapshot {
    pub evidence: Value,
    pub counts: BTreeMap<String, u64>,
    pub accounted: u64,
    pub peak: u64,
}

pub(crate) fn snapshot<B: HostBackend>(backend: &B) -> HostResult<Snapshot> {
    Ok(Snapshot {
        evidence: backend.backend_evidence(),
        counts: serde_json::from_value(backend.dispatch_counts())?,
        accounted: backend.resource_report().total_owned_bytes()?,
        peak: backend.peak_accounted_bytes()?,
    })
}

pub(crate) fn dispatch_delta(
    before: &Snapshot,
    after: &Snapshot,
) -> HostResult<BTreeMap<String, u64>> {
    if before.evidence != after.evidence {
        return Err("backend identity changed during measurement".into());
    }
    let mut delta = BTreeMap::new();
    for (name, &count) in &after.counts {
        let previous = *before.counts.get(name).unwrap_or(&0);
        delta.insert(
            name.clone(),
            count
                .checked_sub(previous)
                .ok_or("dispatch counter decreased during measurement")?,
        );
    }
    if before
        .counts
        .iter()
        .any(|(name, &count)| count > 0 && !after.counts.contains_key(name))
    {
        return Err("dispatch counter disappeared during measurement".into());
    }
    if matches!(
        after.evidence["implementation"].as_str(),
        Some("wgpu_metal" | "native_metal")
    ) {
        if delta.contains_key("gpu_dispatches") {
            return Err("raw kernel name collides with the GPU aggregate counter".into());
        }
        let total = delta.values().try_fold(0_u64, |total, &count| {
            total
                .checked_add(count)
                .ok_or("GPU dispatch total overflow")
        })?;
        delta.insert("gpu_dispatches".into(), total);
    }
    Ok(delta)
}

pub(crate) fn case(
    case: &Case,
    cycles: usize,
    mut predict: impl FnMut() -> HostResult<Prediction>,
) -> HostResult<Value> {
    let mut outputs = Vec::new();
    let mut predictions = Vec::new();
    let mut latencies = Vec::new();
    outputs
        .try_reserve_exact(cycles)
        .map_err(|_| "output records allocation failed")?;
    predictions
        .try_reserve_exact(cycles)
        .map_err(|_| "decision records allocation failed")?;
    latencies
        .try_reserve_exact(cycles)
        .map_err(|_| "latency records allocation failed")?;
    let started = Instant::now();
    for _ in 0..cycles {
        let single = Instant::now();
        let (output, prediction) = predict()?;
        finite(&output)?;
        latencies.push(single.elapsed().as_secs_f64());
        outputs.push(output);
        predictions.push(prediction);
    }
    let elapsed = started.elapsed().as_secs_f64();
    Ok(
        json!({"id":case.id,"completed_predictions":cycles,"elapsed_seconds":elapsed,"latencies_seconds":latencies,"outputs":outputs,"predictions":predictions}),
    )
}
