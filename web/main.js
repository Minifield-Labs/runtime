import init, { load } from "./pkg/minifield_web_demo.js";
import { canonicalModelJson } from "./canonical-json.mjs";
import { createYieldScheduler } from "./yield.mjs";

const scheduler = createYieldScheduler();
globalThis.__minifieldYield = scheduler.yield;
addEventListener("pagehide", () => scheduler.close(), { once: true });

const modelDir = new URLSearchParams(location.search).get("bundle") ?? "../models/demo";
const base = new URL(`${modelDir.replace(/\/$/, "")}/`, location.href);
const FILES = {
  config: new URL("config.json", base),
  weights: new URL("model.safetensors", base),
  tokenizer: new URL("tokenizer/tokenizer.json", base),
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
const choiceState = document.getElementById("choiceState");
const choiceInstructions = document.getElementById("choiceInstructions");
const choiceOptions = document.getElementById("choiceOptions");
const goChoice = document.getElementById("goChoice");
const choiceStatus = document.getElementById("choiceStatus");
const choiceOut = document.getElementById("choiceOut");

function setButtons(disabled) {
  go.disabled = disabled;
  goJson.disabled = disabled;
  goChoice.disabled = disabled;
}

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
    setButtons(false);
    choiceStatus.textContent = "ready";
  } catch (error) {
    fail("load failed", error);
  }
}

function run(generate, label) {
  return async () => {
    setButtons(true);
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
      setButtons(false);
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

function parseCriteria(text) {
  const criteria = [];
  const seen = new Set();
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const sep = line.indexOf(":");
    const name = sep < 0 ? "" : line.slice(0, sep).trim();
    const description = sep < 0 ? "" : line.slice(sep + 1).trim();
    if (!name || !description) {
      return { error: `each line needs a nonempty name: description` };
    }
    if (seen.has(name)) {
      return { error: `duplicate criterion name: ${name}` };
    }
    seen.add(name);
    criteria.push({ name, description });
  }
  if (criteria.length < 2 || criteria.length > 8) {
    return { error: `enter 2 to 8 criteria (got ${criteria.length})` };
  }
  return { criteria };
}

function choiceBase(state, instructions, criteria) {
  const map = Object.create(null);
  for (const criterion of criteria) {
    map[criterion.name] = criterion.description;
  }
  return (
    `<|im_start|>user\n` +
    `Evaluate which choice best matches the state. ` +
    `Each listed choice will be checked independently as true or false.\n` +
    canonicalModelJson({ state, instructions, criteria: map }) +
    `<|im_end|>\n` +
    `<|im_start|>assistant\n` +
    `{"choice":"`
  );
}

// The base ends on the name's opening quote so the `:"` merge stays inside
// the base encoding and every tail remains a compositional continuation.
function choiceTail(criterion) {
  return `${JSON.stringify(criterion.name).slice(1)},"selected":`;
}

function renderChoice(result, criteria, ms) {
  choiceOut.textContent = "";
  const pre = document.createElement("pre");
  pre.className = "cjson";
  pre.textContent = JSON.stringify(result, null, 2);
  choiceOut.appendChild(pre);
  const winner = document.createElement("p");
  winner.className = "cwinner";
  winner.textContent = `${result.choice} · ${ms.toFixed(0)} ms`;
  choiceOut.appendChild(winner);
  for (const criterion of criteria) {
    const probability = result.probabilities[criterion.name] ?? 0;
    const row = document.createElement("div");
    row.className =
      criterion.name === result.choice ? "crow selected" : "crow";
    const label = document.createElement("span");
    label.className = "clabel";
    label.textContent = criterion.name;
    const desc = document.createElement("span");
    desc.className = "cdesc";
    desc.textContent = criterion.description;
    const track = document.createElement("span");
    track.className = "ctrack";
    const bar = document.createElement("span");
    bar.className = "cbar";
    const pct = Math.min(100, Math.max(0, probability * 100));
    bar.style.width = `${pct}%`;
    track.appendChild(bar);
    const pctEl = document.createElement("span");
    pctEl.className = "cpct";
    pctEl.textContent = `${pct.toFixed(1)}%`;
    row.append(label, desc, track, pctEl);
    choiceOut.appendChild(row);
  }
}

goChoice.addEventListener("click", async () => {
  const parsed = parseCriteria(choiceOptions.value);
  if (parsed.error) {
    choiceOut.textContent = "";
    choiceStatus.textContent = parsed.error;
    return;
  }
  const criteria = parsed.criteria;
  setButtons(true);
  const t0 = performance.now();
  try {
    const state = choiceState.value.trim();
    const instructions = choiceInstructions.value.trim();
    const result = await demo.choose(
      choiceBase(state, instructions, criteria),
      criteria.map((criterion) => criterion.name),
      criteria.map(choiceTail),
    );
    const ms = performance.now() - t0;
    renderChoice(result, criteria, ms);
    choiceStatus.textContent =
      `${criteria.length} binary predictions · ${ms.toFixed(0)} ms`;
  } catch (error) {
    choiceOut.textContent = "";
    choiceStatus.textContent = `choice failed: ${error?.message ?? error}`;
    console.error("choice failed", error);
  } finally {
    setButtons(false);
  }
});

boot();
