import { qualifyTelemetry } from "./telemetry-qualification.mjs";
import { createYieldScheduler } from "./yield.mjs";

const scheduler = createYieldScheduler();
globalThis.__minifieldYield = scheduler.yield;

async function bytes(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Asset fetch failed: ${path} (${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}

async function qualify() {
  if (!navigator.gpu) throw new Error("WebGPU is required for browser qualification");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter is required for browser qualification");
  const info = adapter.info;
  const adapterInfo = Object.fromEntries(
    ["vendor", "architecture", "device", "description"].map(key => [key, info?.[key] ?? "unknown"]),
  );
  const profile = await (await fetch("/qualification-profile.json")).json();
  let runtime;
  if (profile.module === "blob") {
    const [script, wasm] = await Promise.all([bytes("/web/pkg/minifield_web_demo.js"), bytes("/web/pkg/minifield_web_demo_bg.wasm")]);
    const url = URL.createObjectURL(new Blob([script], { type: "text/javascript" }));
    try {
      runtime = await import(url);
      await runtime.default({ module_or_path: wasm });
    } finally { URL.revokeObjectURL(url); }
  } else {
    runtime = await import("./pkg/minifield_web_demo.js");
    await runtime.default();
  }
  const { load_classifier, configure_telemetry, flush_telemetry } = runtime;
  configure_telemetry({ endpoint: `http://localhost:${location.port}/telemetry-test`, environment: "test", integrationId: "browser-qualification" });
  const assets = await Promise.all(["config.json", "model.safetensors", "tokenizer.json"].map(name => bytes(`/assets/${name}`)));
  const started = performance.now();
  const classifier = await load_classifier(...assets, profile.classes);
  const initializationSeconds = (performance.now() - started) / 1000;
  try {
    // The same scheduler is used by Rust polling. Concurrent requests must
    // all return before model execution starts.
    await Promise.all(Array.from({ length: 64 }, () => scheduler.yield()));
    const full = [];
    const cached = [];
    for (const [method, output] of [["classify", full], ["classify_cached", cached]]) {
      for (const prompt of profile.prompts) {
        const start = performance.now();
        const logits = Array.from(await classifier[method](prompt));
        if (logits.length !== profile.classes || logits.some(x => !Number.isFinite(x))) {
          throw new Error(`${method} returned invalid logits`);
        }
        output.push({ logits, seconds: (performance.now() - start) / 1000 });
      }
    }
    let emptyRejected = false;
    try { await classifier.classify(""); } catch { emptyRejected = true; }
    if (!emptyRejected) throw new Error("Empty classifier input was accepted");
    const recovery = Array.from(await classifier.classify(profile.prompts[0]));
    await flush_telemetry();
    const telemetry = await qualifyTelemetry(classifier, profile, runtime);
    return { browser: navigator.userAgent, adapter: adapterInfo, moduleSource: profile.module, initializationSeconds, full, cached, recovery, emptyRejected, telemetry };
  } finally {
    classifier.free();
    scheduler.close();
  }
}

globalThis.__minifieldQualification = qualify().then(
  result => {
    document.querySelector("#status").textContent = "Browser execution complete.";
    return result;
  },
  error => {
    document.querySelector("#status").textContent = String(error);
    return { error: String(error), stack: error?.stack };
  },
);
