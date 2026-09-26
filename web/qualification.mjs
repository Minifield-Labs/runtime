import init, { load_classifier } from "./pkg/minifield_web_demo.js";
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
  await init();
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
    return { browser: navigator.userAgent, adapter: adapterInfo, initializationSeconds, full, cached, recovery, emptyRejected };
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
