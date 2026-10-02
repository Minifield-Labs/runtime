// Actual WASM entry points and HTTP delivery, against the runner's loopback sink.

function check(condition, message) { if (!condition) throw new Error(`Telemetry: ${message}`); }
async function records() { return (await fetch("/telemetry-test")).json(); }

export async function qualifyTelemetry(classifier, profile, runtimeModule) {
  const { load, configure_telemetry, flush_telemetry } = runtimeModule;
  const initial = await records();
  check(initial.length === 2 * profile.prompts.length + 2, "one terminal record per classifier call");
  check(initial.filter(r => r.status === "failed").length === 1, "one rejected-input record");
  check(initial.every(r => r.mode === "single_step" && r.autoregressive === null), "single-step shape");
  check(initial.every(r => r.execution.tokens.output === 0), "classifier output is not generated tokens");
  check(initial.filter(r => r.status === "completed").every(r => r.single_step.predictions_produced === 1 && r.single_step.alternatives_evaluated === profile.classes), "classification counts");

  const firstCached = initial[profile.prompts.length].execution;
  if (firstCached.tokens.input > 16) {
    check(firstCached.cache.rebuilds === 1 && firstCached.cache.token_positions_reused === firstCached.tokens.input - 1, "cache reserves one tail token");
    check(firstCached.prefill.token_positions_processed === firstCached.tokens.input, "cache creation does not repeat the full prompt");
    if (profile.prompts[0] === profile.prompts[1]) {
      const repeated = initial[profile.prompts.length + 1].execution;
      check(repeated.cache.rebuilds === 0 && repeated.prefill.forward_passes === 1 && repeated.prefill.token_positions_processed === 1, "repeated cached prompt executes one tail token");
    }
  }

  configure_telemetry({ enabled: false });
  await classifier.classify(profile.prompts[0]);
  await flush_telemetry();
  check((await records()).length === initial.length, "deployment opt-out");
  configure_telemetry({ enabled: true, endpoint: `${location.origin}/telemetry-unavailable` });
  await classifier.classify(profile.prompts[0]);
  await flush_telemetry();
  configure_telemetry({ endpoint: `http://localhost:${location.port}/telemetry-test` });

  if (profile.generation) {
    const bytes = await Promise.all(["config.json", "model.safetensors", "tokenizer.json"].map(async name =>
      new Uint8Array(await (await fetch(`/generation-assets/${name}`)).arrayBuffer())));
    const runtime = await load(...bytes);
    try {
      const generated = await runtime.generate("a", 3, () => {});
      check(generated.tokens === 3, "synthetic autoregressive fixture must emit 3 tokens");
      await runtime.choose("a", ["first", "second"], ["x", "b"]);
      await runtime.warm_tools("a");
      await runtime.generate_json("a", "x", "example", 6, () => {});
      let rejected = false;
      try { await runtime.generate("a", 0, () => {}); } catch { rejected = true; }
      check(rejected, "zero-budget generation rejected");
      await flush_telemetry();
    } finally { runtime.free(); }
  }
  const all = await records();
  check(all.length === initial.length + (profile.generation ? 4 : 0), "one record per inference; warm_tools excluded");
  check(new Set(all.map(r => r.inference_id)).size === all.length, "unique inference IDs");
  for (const record of all) {
    check(/^inf_[0-7][0-9a-hjkmnp-tv-z]{25}$/.test(record.inference_id), "TypeID encoding");
    check(record.schema_version === "minifield.runtime-inference/1", "schema version");
    check(record.origin.environment === "test", "test traffic isolation");
    check(record.runtime.version && record.runtime.build_id.startsWith("sha256:"), "runtime build identity");
    check(record.model.bundle_sha256.length === 64 && record.model.parameter_count > 0, "model identity and parameters");
    check(record.execution.elapsed_ms >= 0, "elapsed time");
    check(!("prompt" in record) && !("text" in record) && !("choice" in record), "content-free envelope");
    if (record.status === "completed") {
      const execution = record.execution;
      check(execution.flops_estimate_coverage === "complete", "complete work coverage");
      check(BigInt(execution.estimated_flops) === BigInt(execution.prefill.estimated_flops) + BigInt(execution.decode?.estimated_flops ?? "0"), "phase FLOPs sum");
    }
  }
  if (profile.generation) {
    const [generated, choice, constrained, failed] = all.slice(initial.length);
    check(generated.execution.tokens.output === 3 && generated.execution.decode.forward_passes === 2, "autoregressive pass count");
    check(choice.mode === "single_step" && choice.single_step.alternatives_evaluated === 2, "choice counts");
    check(constrained.autoregressive.constraint === "tool_call" && constrained.execution.fallback_used, "constrained fallback counted");
    check(constrained.execution.cache.token_positions_reused > 0, "warm prefix reused");
    check(failed.status === "failed" && failed.execution.tokens.output === 0, "failed generation");
  }
  return { records: all.length, classifier: true, generation: Boolean(profile.generation), optOut: true, deliveryFailure: true };
}
