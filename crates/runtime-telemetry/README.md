# Runtime telemetry

Content-free records for one high-level inference. This crate builds model identity, TypeID
inference IDs, elapsed timings, and work-counter deltas. It has no HTTP client, environment
configuration, or inference content API. Hosts own delivery and deployment settings.

`Measurement` wraps a serialized inference on one executor. Snapshot `inference_work()` at
the start, at the prefill/decode boundary, and on completion. Use `tokenized` and `emitted`
for logical input/output counts. `finish` consumes the measurement and creates one terminal
record. A failed call uses partial estimate coverage because recorded device work may not
have completed. Core counters never contain token IDs, logits, prompts, or output text.

The browser and native adapters populate origin, hardware, and platform. The browser host
reports automatically after its public inference methods. The CLI binary reports after
stdout has been flushed. Native library embedders use `run_with_reporter` or build records
around their own executor calls; library functions don't initiate network requests.

See the [full record and deployment configuration](../../docs/runtime-telemetry.md).
