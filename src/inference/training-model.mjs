import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile, realpath } from "node:fs/promises";
import path from "node:path";

const REQUIRED_FILES = new Set([
  "chat_template.jinja",
  "config.json",
  "model.safetensors",
  "serializer.json",
  "tokenizer.json",
]);
const SHA256 = /^[0-9a-f]{64}$/;

async function digestFile(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

function parseObject(bytes, name) {
  let value;
  try { value = JSON.parse(bytes); }
  catch { throw new Error(`${name} isn't valid JSON`); }
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error(`${name} must contain a JSON object`);
  }
  return value;
}

function safeAssetName(value) {
  return typeof value === "string" && value.length > 0 &&
    value === path.basename(value) && value !== "." && value !== "..";
}

/** Verify one complete, manually exported training model before inference. */
export async function loadTrainingModel(directory) {
  if (typeof directory !== "string" || directory.length === 0) {
    throw new TypeError("A model directory is required");
  }
  const root = await realpath(directory);
  if (!(await lstat(root)).isDirectory()) throw new Error("Model path isn't a directory");
  const manifest = parseObject(await readFile(path.join(root, "manifest.json"), "utf8"), "manifest.json");
  if (manifest.schema_version !== "minifield.training-model/1") {
    throw new Error("Unsupported training-model manifest version");
  }
  if (manifest.engine?.family !== "mistralrs" || manifest.engine?.architecture !== "lfm2") {
    throw new Error("Export requires an unsupported inference engine");
  }
  if (!Number.isSafeInteger(manifest.limits?.max_context_tokens) || manifest.limits.max_context_tokens < 1) {
    throw new Error("Export has an invalid context limit");
  }
  if (!Array.isArray(manifest.files) || manifest.files.length < REQUIRED_FILES.size) {
    throw new Error("Export file inventory is incomplete");
  }
  const assets = new Map();
  for (const record of manifest.files) {
    if (!record || !safeAssetName(record.path) || !SHA256.test(record.sha256) ||
        !Number.isSafeInteger(record.bytes) || record.bytes < 1 || assets.has(record.path)) {
      throw new Error("Export contains an invalid file record");
    }
    const file = path.join(root, record.path);
    const stat = await lstat(file);
    const resolved = await realpath(file);
    if (!stat.isFile() || resolved !== file || stat.size !== record.bytes) {
      throw new Error(`Export asset failed path or size validation: ${record.path}`);
    }
    if (await digestFile(file) !== record.sha256) {
      throw new Error(`Export asset failed checksum validation: ${record.path}`);
    }
    assets.set(record.path, { path: file, sha256: record.sha256, bytes: record.bytes });
  }
  for (const name of REQUIRED_FILES) {
    if (!assets.has(name)) throw new Error(`Export is missing required asset: ${name}`);
  }
  const config = parseObject(await readFile(assets.get("config.json").path, "utf8"), "config.json");
  const serializer = parseObject(await readFile(assets.get("serializer.json").path, "utf8"), "serializer.json");
  if (config.model_type !== "lfm2" || serializer.implementation !== "lfm2-chatml-tool-json-v1") {
    throw new Error("Export architecture and serializer aren't compatible");
  }
  for (const name of ["bos_token", "assistant_prefix", "turn_suffix"]) {
    if (typeof serializer[name] !== "string" || serializer[name].length === 0) {
      throw new Error(`Serializer is missing ${name}`);
    }
  }
  const checkpointStep = manifest.checkpoint?.step;
  if (!Number.isSafeInteger(checkpointStep) || checkpointStep < 1) {
    throw new Error("Export has an invalid checkpoint step");
  }
  return Object.freeze({
    directory: root,
    manifest,
    config,
    serializer,
    assets,
    maxContextTokens: manifest.limits.max_context_tokens,
    key: `${assets.get("model.safetensors").sha256}:mistralrs:lfm2:${checkpointStep}`,
  });
}
