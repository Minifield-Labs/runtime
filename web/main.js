import init, { load } from "./pkg/minifield_web_demo.js";
import { canonicalModelJson } from "../src/context/lfm2-chatml.mjs";

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
  weights: `${MODEL_DIR}/qat.safetensors`,
  tokenizer: `${MODEL_DIR}/tokenizer.json`,
};

// Task-manager schemas for the demo's default tool names, in the same
// {type, function:{name, description, parameters}} wire shape the generator
// writes into training records. Names outside this catalog get a bare def.
const TOOL_DEFS = {
  create_task: {
    type: "function",
    function: {
      name: "create_task",
      description: "Create a new task in the list.",
      parameters: {
        type: "object",
        properties: {
          title: { type: "string", description: "Short task title." },
          due_date: {
            type: "string",
            description: "Optional due date, ISO 8601 (YYYY-MM-DD).",
          },
          priority: {
            type: "string",
            enum: ["low", "medium", "high"],
            description: "Task priority.",
          },
        },
        required: ["title"],
      },
    },
  },
  archive_task: {
    type: "function",
    function: {
      name: "archive_task",
      description: "Archive a task so it leaves the active list.",
      parameters: {
        type: "object",
        properties: {
          task_id: { type: "string", description: "ID of the task to archive." },
        },
        required: ["task_id"],
      },
    },
  },
  reorder_list: {
    type: "function",
    function: {
      name: "reorder_list",
      description: "Reorder the list's items into the given order.",
      parameters: {
        type: "object",
        properties: {
          item_ids: {
            type: "array",
            items: { type: "string" },
            description: "Every item ID in its new position order.",
          },
        },
        required: ["item_ids"],
      },
    },
  },
  assign_owner: {
    type: "function",
    function: {
      name: "assign_owner",
      description: "Assign a task to a team member.",
      parameters: {
        type: "object",
        properties: {
          task_id: { type: "string", description: "ID of the task." },
          owner: { type: "string", description: "Name of the assignee." },
        },
        required: ["task_id", "owner"],
      },
    },
  },
};

// Match the lfm2-chatml-tool-json-v1 serializer used by the training worker:
// the tokenizer prepends <|startoftext|>, each turn is
// <|im_start|>role\nbody<|im_end|>\n, and generation follows
// <|im_start|>assistant\n. The system block is kept separate from the
// user/assistant tail so generate_json can cache its prefilled KV state.
function chatmlSys(toolNames) {
  if (!toolNames.length) return "";
  const tools = toolNames.map(
    (name) =>
      TOOL_DEFS[name] ?? {
        type: "function",
        function: { name, parameters: { type: "object" } },
      },
  );
  return `<|im_start|>system\nAvailable tools:\n${canonicalModelJson(tools)}<|im_end|>\n`;
}

function chatmlTail(userText) {
  return `<|im_start|>user\n${userText}<|im_end|>\n<|im_start|>assistant\n`;
}

function chatml(userText, toolNames) {
  return chatmlSys(toolNames) + chatmlTail(userText);
}

function toolNames() {
  return tools.value
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
}

const status = document.getElementById("status");
const out = document.getElementById("out");
const go = document.getElementById("go");
const goJson = document.getElementById("goJson");
const prompt = document.getElementById("prompt");
const tools = document.getElementById("tools");
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
    status.textContent = "prefilling tool list…";
    await demo.warm_tools(chatmlSys(toolNames()));
    status.innerHTML = `<span class="ok">model loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s</span>`;
    go.disabled = false;
    goJson.disabled = false;
  } catch (error) {
    fail("load failed", error);
  }
}

function run(generate, label) {
  return async () => {
    go.disabled = true;
    goJson.disabled = true;
    out.textContent = "";
    const t0 = performance.now();
    try {
      const stats = await generate(
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
        },
      );
      const seconds = (performance.now() - t0) / 1000;
      status.textContent =
        `${label} · ${stats.tokens} tokens in ${seconds.toFixed(1)}s ` +
        `(${(stats.tokens / seconds).toFixed(1)} tok/s` +
        `${stats.stopped ? ", stopped" : ""})`;
    } catch (error) {
      fail("generate failed", error);
    } finally {
      go.disabled = false;
      goJson.disabled = false;
    }
  };
}

go.addEventListener(
  "click",
  run((p, m, cb) => demo.generate(chatml(p, []), m, cb), "free"),
);
goJson.addEventListener(
  "click",
  run(
    (p, m, cb) =>
      demo.generate_json(
        chatmlSys(toolNames()),
        chatmlTail(p),
        tools.value,
        m,
        cb,
      ),
    "tool",
  ),
);

boot();
