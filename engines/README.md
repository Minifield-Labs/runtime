# Engine adapters

Keep each inference backend in its own directory with compatibility notes and smoke tests for the exact packed formats it supports.

Choose the first backend against the reference device and delivery budget. Avoid adding a frontend framework or multiple engines before that experiment. Runtime source code can share one interface while backend dependencies remain local to each adapter.

`mistralrs/` is the first complete-model adapter. It starts a private local
mistral.rs server for a verified `minifield.training-model/1` LFM2 export,
sends raw prompts serialized exactly like training, and shuts the process down
through the runtime's resident-model owner. The adapter never merges training
checkpoints. Export remains a training or experiment operation.
