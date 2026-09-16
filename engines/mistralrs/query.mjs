#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { renderLfm2Prompt } from "../../src/context/lfm2-chatml.mjs";
import { loadTrainingModel } from "../../src/inference/training-model.mjs";
import { ResidentModel } from "../../src/inference/resident-model.mjs";
import { startMistralRs } from "./process.mjs";

const { values } = parseArgs({
  options: {
    binary: { type: "string" },
    model: { type: "string" },
    prompt: { type: "string", multiple: true },
    system: { type: "string" },
    tool: { type: "string", multiple: true },
    tools: { type: "string" },
    output: { type: "string" },
    "max-tokens": { type: "string", default: "64" },
  },
  strict: true,
});
if (!values.binary || !values.model || !values.prompt?.length) {
  throw new Error("usage: query.mjs --binary PATH --model DIRECTORY --prompt TEXT [--prompt TEXT] [--system TEXT] [--tools FILE] [--tool NAME] [--output FILE] [--max-tokens N]");
}
const maxTokens = Number(values["max-tokens"]);
if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) throw new Error("--max-tokens must be a positive integer");
const rawTools = values.tools ? JSON.parse(await readFile(resolve(values.tools), "utf8")) : [];
let tools;
if (Array.isArray(rawTools)) {
  tools = rawTools;
} else if (Array.isArray(rawTools?.tools)) {
  tools = rawTools.tools.map((tool) => ({
    type: "function",
    function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
  }));
} else {
  throw new Error("--tools must contain an array or a catalog with a tools array");
}
if (values.tool?.length) {
  const selected = new Set(values.tool);
  tools = tools.filter((tool) => selected.has(tool?.function?.name));
  const found = new Set(tools.map((tool) => tool.function.name));
  const missing = [...selected].filter((name) => !found.has(name));
  if (missing.length) throw new Error(`Requested tools are absent from the catalog: ${missing.join(", ")}`);
}
const model = await loadTrainingModel(resolve(values.model));
const owner = new ResidentModel({ maxPending: 2 });
try {
  await owner.load(model.key, () => startMistralRs({ binary: resolve(values.binary), model }));
  const results = [];
  for (const prompt of values.prompt) {
    const messages = [];
    if (values.system) messages.push({ role: "system", content: values.system });
    messages.push({ role: "user", content: prompt });
    const rendered = renderLfm2Prompt({ serializer: model.serializer, messages, tools });
    const result = await owner.run((engine) => engine.generate(rendered, { maxTokens }));
    results.push({ prompt, ...result });
  }
  const report = {
    checkpoint_step: model.manifest.checkpoint.step,
    model_sha256: model.assets.get("model.safetensors").sha256,
    engine: "mistralrs",
    results,
  };
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (values.output) await writeFile(resolve(values.output), serialized, { flag: "wx" });
  process.stdout.write(serialized);
} finally {
  await owner.dispose();
}
