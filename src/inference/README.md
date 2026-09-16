# Inference

The audited foundation contains:

- resident-model.mjs: bounded FIFO model loading, inference, replacement, unload, and disposal.
- wasm-memory.mjs: fresh linear-memory views, copied FP32 outputs, bounds and finite-value checks.
- sequence-buckets.mjs: explicit compiled-shape selection, with overflow rejected.

See [the audit](../../docs/wires-audit.md) and tests/ for the concurrency and memory cases. Model loading, tokenization, decoder execution, and actual GPU resource management remain engine-adapter work.

## Adapter requirements

A resident-model factory returns a fresh resource with an asynchronous or synchronous dispose() method. Include the artifact hash, engine version, and numerical configuration in its key. Repeated loads of the current key reuse that resource.

The manager serializes operations. Each operation must settle only after its final GPU consumer/readback completes, including failure and cancellation. Abort notification alone doesn't prove that buffers can be reused.

Replacement builds a candidate before disposing the old resource. Budget for that temporary overlap, or explicitly unload before loading when memory is tight. A failed factory retains the old model; a failed old-model disposal clears the active slot and attempts candidate cleanup.

`training-model.mjs` verifies complete local LFM2 training-model exports before
an engine sees their paths. The first native backend is the Rust mistral.rs
process adapter under `engines/mistralrs/`. It keeps checkpoint merging outside
runtime and passes the trained serializer's raw prompt to the engine.

The lower-level lifecycle helpers remain Worker-compatible. They don't start a
browser Worker, implement device-loss fallback, or expose product tools. Those
steps belong to the runtime procedure.
