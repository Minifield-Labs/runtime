import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import assert from "node:assert/strict";

import { renderLfm2Prompt } from "../src/context/lfm2-chatml.mjs";
import { loadTrainingModel } from "../src/inference/training-model.mjs";

const sha = (value) => createHash("sha256").update(value).digest("hex");

async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "minifield-model-"));
  const values = {
    "chat_template.jinja": "template",
    "config.json": JSON.stringify({ model_type: "lfm2" }),
    "model.safetensors": "weights",
    "serializer.json": JSON.stringify({ implementation: "lfm2-chatml-tool-json-v1", bos_token: "<s>", assistant_prefix: "<|im_start|>assistant\n", turn_suffix: "<|im_end|>\n" }),
    "tokenizer.json": "tokenizer",
  };
  for (const [name, value] of Object.entries(values)) await writeFile(path.join(root, name), value);
  const files = Object.entries(values).map(([name, value]) => ({ path: name, sha256: sha(value), bytes: Buffer.byteLength(value) }));
  await writeFile(path.join(root, "manifest.json"), JSON.stringify({
    schema_version: "minifield.training-model/1",
    checkpoint: { step: 1700 },
    engine: { family: "mistralrs", architecture: "lfm2" },
    limits: { max_context_tokens: 512 },
    files,
  }));
  return root;
}

test("verified LFM2 exports load with an immutable engine key", async () => {
  const model = await loadTrainingModel(await fixture());
  assert.equal(model.manifest.checkpoint.step, 1700);
  assert.match(model.key, /^[0-9a-f]{64}:mistralrs:lfm2:1700$/);
  assert.equal(model.maxContextTokens, 512);
});

test("export checksum changes are rejected before engine startup", async () => {
  const root = await fixture();
  await writeFile(path.join(root, "model.safetensors"), "tampered");
  await assert.rejects(loadTrainingModel(root), /path or size|checksum/);
});

test("training prompt serialization preserves canonical tool JSON", () => {
  const prompt = renderLfm2Prompt({
    serializer: { bos_token: "<s>", assistant_prefix: "<|im_start|>assistant\n" },
    tools: [{ name: "move", inputSchema: { z: 2, a: 1 } }],
    messages: [{ role: "user", content: "Move it" }],
  });
  assert.equal(prompt, '<s><|im_start|>system\nAvailable tools:\n[{"inputSchema":{"a":1,"z":2},"name":"move"}]<|im_end|>\n<|im_start|>user\nMove it<|im_end|>\n<|im_start|>assistant\n');
});
