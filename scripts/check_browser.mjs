#!/usr/bin/env node
// Runs the actual wasm-bindgen/WebGPU path in an isolated Chrome profile.
// Uses Chrome's DevTools Protocol and Node built-ins; no downloaded browser.
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { access, mkdtemp, readFile, rm, stat, writeFile, mkdir } from "node:fs/promises";
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, resolve, sep, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const options = { classes: 8, timeout: 180, atol: 0.000025, rtol: 0.00001 };
for (let i = 0; i < argv.length; i += 2) {
  const key = argv[i].replace(/^--/, "");
  if (!["bundle", "prompts", "classes", "expected", "out", "chrome", "timeout", "atol", "rtol"].includes(key) || argv[i + 1] === undefined) {
    throw new Error("Usage: node scripts/check_browser.mjs --bundle DIR --prompts FILE [--classes N] [--expected NATIVE_JSON] [--out JSON] [--chrome EXECUTABLE]");
  }
  options[key] = ["classes", "timeout", "atol", "rtol"].includes(key) ? Number(argv[i + 1]) : argv[i + 1];
}
if (!options.bundle || !options.prompts || !Number.isInteger(options.classes) || options.classes < 1 || !Number.isFinite(options.timeout) || options.timeout <= 0) {
  throw new Error("Bundle, nonempty prompts, positive classes, and a finite positive timeout are required");
}
for (const key of ["atol", "rtol"]) {
  if (!Number.isFinite(options[key]) || options[key] < 0) throw new Error(`Invalid ${key}`);
}
const prompts = JSON.parse(await readFile(resolve(options.prompts), "utf8"));
if (!Array.isArray(prompts) || prompts.length === 0 || prompts.some(p => typeof p !== "string" || p.length === 0)) {
  throw new Error("Prompts must be a nonempty array of nonempty strings");
}
const bundle = resolve(options.bundle);
let tokenizer = resolve(bundle, "tokenizer/tokenizer.json");
try { await access(tokenizer); } catch { tokenizer = resolve(bundle, "tokenizer.json"); }
const assets = new Map([
  ["/assets/config.json", resolve(bundle, "config.json")],
  ["/assets/model.safetensors", resolve(bundle, "model.safetensors")],
  ["/assets/tokenizer.json", tokenizer],
]);
for (const path of [...assets.values(), resolve(root, "web/pkg/minifield_web_demo_bg.wasm")]) await access(path);
const hashes = {};
for (const [url, path] of assets) hashes[url.split("/").at(-1)] = createHash("sha256").update(await readFile(path)).digest("hex");
hashes.prompts = createHash("sha256").update(await readFile(resolve(options.prompts))).digest("hex");
hashes.wasm = createHash("sha256").update(await readFile(resolve(root, "web/pkg/minifield_web_demo_bg.wasm"))).digest("hex");
const server = createServer(async (request, response) => {
  try {
    const path = new URL(request.url, "http://localhost").pathname;
    if (path === "/qualification-profile.json") {
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ prompts, classes: options.classes }));
      return;
    }
    let file = assets.get(path);
    if (!file && path.startsWith("/web/")) {
      file = resolve(root, `.${decodeURIComponent(path)}`);
      if (!file.startsWith(resolve(root, "web") + sep)) file = null;
    }
    if (!file) { response.writeHead(404); response.end(); return; }
    const metadata = await stat(file);
    response.setHeader("Content-Length", metadata.size);
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Content-Type", ({ ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".json": "application/json" })[extname(file)] ?? "application/octet-stream");
    createReadStream(file).pipe(response);
  } catch (error) { response.writeHead(500); response.end(String(error)); }
});
await new Promise(done => server.listen(0, "127.0.0.1", done));
const profile = await mkdtemp(resolve(tmpdir(), "minifield-browser-"));
const chrome = options.chrome ?? process.env.CHROME_BIN ?? (process.platform === "darwin" ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "google-chrome");
const child = spawn(chrome, ["--headless", "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
let stderr = "";
let spawnError;
child.on("error", error => { spawnError = error; });
child.stderr.on("data", data => { stderr = (stderr + data).slice(-8000); });
let socket;
const deadline = Date.now() + options.timeout * 1000;
try {
  let port;
  while (!port) {
    if (spawnError) throw spawnError;
    if (Date.now() > deadline || child.exitCode !== null) throw new Error(`Chrome failed to start: ${stderr}`);
    try { port = Number((await readFile(resolve(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]); } catch { await delay(50); }
  }
  const url = `http://127.0.0.1:${server.address().port}/web/qualification.html`;
  const page = await (await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, { method: "PUT", signal: AbortSignal.timeout(Math.max(1, deadline - Date.now())) })).json();
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((done, fail) => {
    const timer = setTimeout(() => fail(new Error("Chrome debugger connection timed out")), Math.max(1, deadline - Date.now()));
    socket.addEventListener("open", () => { clearTimeout(timer); done(); }, { once: true });
    socket.addEventListener("error", error => { clearTimeout(timer); fail(error); }, { once: true });
  });
  const call = (method, params) => new Promise((done, fail) => {
    const timeout = setTimeout(() => fail(new Error(`Browser qualification exceeded ${options.timeout}s`)), Math.max(1, deadline - Date.now()));
    const message = event => {
      const result = JSON.parse(event.data);
      if (result.id !== 1) return;
      clearTimeout(timeout);
      socket.removeEventListener("message", message);
      result.error ? fail(new Error(JSON.stringify(result.error))) : done(result.result);
    };
    socket.addEventListener("message", message);
    socket.send(JSON.stringify({ id: 1, method, params }));
  });
  const evaluated = await call("Runtime.evaluate", {
    expression: "(async () => { for (let i=0; i<2000 && !globalThis.__minifieldQualification; i++) await new Promise(r=>setTimeout(r,50)); if (!globalThis.__minifieldQualification) throw new Error('Qualification module did not load'); return await globalThis.__minifieldQualification; })()",
    awaitPromise: true, returnByValue: true,
  });
  if (evaluated.exceptionDetails) throw new Error(JSON.stringify(evaluated.exceptionDetails));
  const result = evaluated.result.value;
  if (!result || result.error) throw new Error(JSON.stringify(result));
  const argmax = values => values.reduce((best, value, i) => value > values[best] ? i : best, 0);
  let maxAbs = 0;
  function compare(got, expected, label) {
    if (got.length !== options.classes || expected.length !== got.length || [...got, ...expected].some(x => !Number.isFinite(x))) throw new Error(`${label}: invalid logits`);
    if (argmax(got) !== argmax(expected)) throw new Error(`${label}: argmax mismatch`);
    got.forEach((value, i) => {
      const delta = Math.abs(value - expected[i]);
      maxAbs = Math.max(maxAbs, delta);
      if (delta > options.atol + options.rtol * Math.abs(expected[i])) throw new Error(`${label}: logit ${i} delta ${delta} exceeds tolerance`);
    });
  }
  result.full.forEach((row, i) => compare(result.cached[i].logits, row.logits, `cached prompt ${i}`));
  compare(result.recovery, result.full[0].logits, "recovery after rejected input");
  if (options.expected) {
    const reference = JSON.parse(await readFile(resolve(options.expected), "utf8"));
    if (reference.schema_version !== 1 || reference.classes !== options.classes || (reference.status && reference.status !== "passed")) throw new Error("Expected evidence has an incompatible schema, class count, or failed status");
    for (const [field, key] of [["weights_sha256", "model.safetensors"], ["config_sha256", "config.json"], ["tokenizer_sha256", "tokenizer.json"], ["prompts_sha256", "prompts"]]) {
      if (reference.artifacts?.[field] !== hashes[key]) throw new Error(`Expected ${field} differs from browser inputs`);
    }
    const expected = reference.samples ?? reference.runs?.[0]?.result?.samples ?? reference.logits?.map(logits => ({ logits }));
    if (!Array.isArray(expected)) throw new Error("Expected evidence has no logit samples");
    if (expected.length !== result.full.length) throw new Error("Native expected result count differs");
    expected.forEach((row, i) => compare(result.full[i].logits, row.logits, `native/browser prompt ${i}`));
  }
  const report = { schema: "minifield.browser-qualification/1", passed: true, revision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), dirty: Boolean(execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim()), hashes, classes: options.classes, prompts: prompts.length, atol: options.atol, rtol: options.rtol, maxAbs, nativeComparison: Boolean(options.expected), ...result };
  if (options.out) {
    await mkdir(dirname(resolve(options.out)), { recursive: true });
    await writeFile(resolve(options.out), JSON.stringify(report, null, 2) + "\n");
  }
  console.log(JSON.stringify({ passed: true, prompts: prompts.length, maxAbs, nativeComparison: report.nativeComparison, browser: report.browser, adapter: report.adapter }));
} finally {
  socket?.close();
  child.kill("SIGTERM");
  await Promise.race([new Promise(done => child.once("exit", done)), delay(2000)]);
  if (child.exitCode === null) child.kill("SIGKILL");
  server.closeAllConnections();
  await new Promise(done => server.close(done));
  await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
}
