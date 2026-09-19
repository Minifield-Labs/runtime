import init, { load } from "./pkg/minifield_web_demo.js";

// Rust polls GPU completions cooperatively; this hands the event loop back one
// macrotask so WebGPU device-timeline promises can resolve. MessageChannel is
// unclamped, unlike nested setTimeout(0).
{
  const { port1, port2 } = new MessageChannel();
  globalThis.__minifieldYield = () =>
    new Promise((resolve) => {
      port1.onmessage = resolve;
      port2.postMessage(0);
    });
}

const MODEL_DIR = "../tmp/models/lfm2.5-230m";
const FILES = {
  config: `${MODEL_DIR}/config.json`,
  weights: `${MODEL_DIR}/lfm2.5-230m-ternary-v1.safetensors`,
  tokenizer: `${MODEL_DIR}/tokenizer.json`,
};

const status = document.getElementById("status");
const out = document.getElementById("out");
const go = document.getElementById("go");
const prompt = document.getElementById("prompt");
const max = document.getElementById("max");

function fail(what, error) {
  status.innerHTML = `<span class="err">${what}: ${error?.message ?? error}</span>`;
  console.error(what, error);
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length") ?? 0);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status.textContent = total
      ? `${label}: ${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
      : `${label}: ${(received / 1e6).toFixed(1)} MB`;
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

let demo = null;

async function boot() {
  if (!navigator.gpu) {
    fail("WebGPU unavailable", "this browser has no navigator.gpu (Chrome/Edge 113+ required)");
    return;
  }
  try {
    await init();
    const [config, weights, tokenizer] = [
      await fetchBytes(FILES.config, "config"),
      await fetchBytes(FILES.weights, "weights"),
      await fetchBytes(FILES.tokenizer, "tokenizer"),
    ];
    status.textContent = "uploading weights to GPU…";
    const t0 = performance.now();
    demo = await load(config, weights, tokenizer);
    status.innerHTML = `<span class="ok">model loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s</span>`;
    go.disabled = false;
  } catch (error) {
    fail("load failed", error);
  }
}

go.addEventListener("click", async () => {
  go.disabled = true;
  out.textContent = "";
  const t0 = performance.now();
  let tokens = 0;
  try {
    const stats = await demo.generate(
      prompt.value,
      Number(max.value),
      (fragment, id) => {
        const chip = document.createElement("span");
        chip.className = "tok";
        const idEl = document.createElement("span");
        idEl.className = "id";
        idEl.textContent = id;
        chip.append(idEl, document.createTextNode(fragment || "∅"));
        out.appendChild(chip);
        tokens += 1;
      },
    );
    const seconds = (performance.now() - t0) / 1000;
    status.textContent =
      `${stats.tokens} tokens in ${seconds.toFixed(1)}s ` +
      `(${(stats.tokens / seconds).toFixed(1)} tok/s` +
      `${stats.stopped ? ", EOS" : ""})`;
  } catch (error) {
    fail("generate failed", error);
  } finally {
    go.disabled = false;
  }
});

boot();
