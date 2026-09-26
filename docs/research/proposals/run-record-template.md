# Experiment run record

Copy this template into the named development experiment or the platform's persistent run record. Fill the setup before execution and the evidence/decision afterward. This file is a template; it contains no results.

## Identity and setup

| Field | Value to record |
| --- | --- |
| Experiment | Number and protocol revision |
| Run | ID, date, owner, status, artifact location |
| Question | Hypothesis and primary metric |
| Bundle | Registry ID, digest, model/export revision, tensor formats |
| Runtime | Revision, build flags, engine/dependencies, dirty patch digest if applicable |
| Host | Device, OS, browser/native version, CPU/GPU, driver, memory, power mode |
| Product | Contract/adapter version, permissions, context and serializer identity |
| Inputs | Immutable dataset IDs/digests, workload families, development/acceptance split |
| Control and treatment | Exact settings and the single intended difference |
| Execution | Reproducible invocation or product job configuration and worker image |
| Decoding | Tokenizer/template, policy, seed, stop rules, context/output limits |
| Cache | Download, pipeline, model, prefix state, capacity, reset procedure |
| Sampling | Warmup, repetitions, order, cold states, sustained duration |
| Numerical gate | Operations, reference, error tolerances, token/action parity requirements |
| Product gate | Task/rejection thresholds, allowed regression margins, forbidden-effect checks |
| Resource gate | Download, RAM/accelerator memory, latency, responsiveness, cancellation limits |
| Improvement gate | Minimum useful change, uncertainty method, stop conditions |

## Measurements

Record control and treatment values, sample counts, absolute/relative differences, and uncertainty for the primary metric and relevant diagnostics. Include failures, timeouts, invalid outputs, missing measurements, and any deviation from the planned run.

Attach stage timings, memory accounting, cold/warm results, sustained traces, numerical comparisons, and product results by family. Include full-request latency, false rejection, and forbidden effects even if the experiment targets one kernel.

## Decision

- Outcome: accepted, rejected, deferred, or inconclusive, with the applicable gates listed.
- Explanation: measured benefit/cost, remaining uncertainty, and the supported device/workload scope.
- Evidence: immutable report/artifact IDs and checksums; keep customer content out of committed summaries.
- Next action: chosen configuration, follow-up comparison or defer trigger, and rollback/control configuration.
- Product qualification: current state and link to the separate platform validation/release record.
