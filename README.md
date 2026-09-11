# Minifield Runtime

Run a delivered product specialist and connect its supported actions to the host application.

This is an independent repository. It currently contains execution boundaries, the pinned model-bundle contract, synthetic fixtures, and validation checks. Model loading and product tool integration remain to be implemented.

Read the [detailed procedure and success criteria](docs/procedure.md) for implementation order, required artifacts, validation, failure handling, and the first milestone.

## Ownership

- src/inference/: model loading, tokenizer/template application, and decoding interface.
- src/context/: authorized observations, product policy, history, and context limits.
- src/tools/: schema validation, tool dispatch, results, and error handling.
- src/sessions/: the model/action loop, cancellation, budgets, and undo boundaries.
- engines/: backend-specific implementations and compatibility tests.
- products/: host integrations and supported-action mappings.
- configs/: reviewed configuration templates.

The target browser/native runtime and implementation language will be selected during the deployment proof. The only current Python dependency supports development-time contract checks; it doesn't select the inference stack.

## Input contract

Load self-contained model bundles matching the pinned schema in contracts/model-bundle/. Check checksums, supported engine/format, product version, and memory/context constraints before starting a session. Refuse unknown versions and fixtures in a real loader.

The host app enforces permissions and confirmation rules. Keep private training metadata and judgments outside model-visible context. Logging and any cloud fallback need an explicit data-flow decision consistent with local AI processing.

See [bundle rules](contracts/model-bundle/v0.1.0/README.md). Model packing belongs to the training repository; this repository owns evidence that the delivered artifact runs correctly on the target device.

## Local checks

Use Python 3.11 or newer for the contract checks:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/check_contracts.py
```

The check validates schemas, pinned snapshots, example hashes, and handoff consistency. It also checks that malformed records are rejected. It doesn't run a teacher, a trainer, or model inference.

## First implementation

Choose one reference device and engine. Load an intact candidate, round-trip a tool call, and measure downloaded bytes, cold load, peak memory, and complete-task latency. Integrate a trained product bundle after this path works.
