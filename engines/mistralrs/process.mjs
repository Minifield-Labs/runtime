import { spawn } from "node:child_process";
import { createServer } from "node:net";

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : null;
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  if (!port) throw new Error("Couldn't allocate a local inference port");
  return port;
}

/** Start a private mistral.rs process for one verified local model. */
export async function startMistralRs({ binary, model, startupTimeoutMs = 120_000 }) {
  if (typeof binary !== "string" || !binary.length || !model?.directory) {
    throw new TypeError("A mistral.rs binary and verified model are required");
  }
  const port = await freePort();
  const args = [
    "serve", "--no-ui", "-p", String(port), "-m", model.directory,
    "-c", model.assets.get("chat_template.jinja").path,
  ];
  const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });
  let logs = "";
  const append = (chunk) => { logs = (logs + chunk.toString()).slice(-32_768); };
  child.stdout.on("data", append);
  child.stderr.on("data", append);
  let exited = false;
  let exitCode = null;
  child.once("exit", (code) => { exited = true; exitCode = code; });
  const baseUrl = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + startupTimeoutMs;
  while (Date.now() < deadline) {
    if (exited) throw new Error(`mistral.rs exited during startup (${exitCode})\n${logs}`);
    try {
      const response = await fetch(`${baseUrl}/v1/models`);
      if (response.ok) break;
    } catch {}
    await delay(250);
  }
  if (Date.now() >= deadline) {
    child.kill("SIGTERM");
    throw new Error(`mistral.rs didn't become ready\n${logs}`);
  }

  return {
    logs: () => logs,
    async generate(prompt, { maxTokens = 64, temperature = 0, seed = 42 } = {}) {
      if (typeof prompt !== "string" || !prompt.length || !Number.isSafeInteger(maxTokens) || maxTokens < 1) {
        throw new TypeError("A prompt and positive generation limit are required");
      }
      const response = await fetch(`${baseUrl}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "default", prompt, max_tokens: maxTokens, temperature, seed }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(`mistral.rs generation failed (${response.status}): ${JSON.stringify(body)}`);
      const text = body?.choices?.[0]?.text;
      if (typeof text !== "string") throw new Error("mistral.rs returned no completion text");
      return { text, usage: body.usage ?? null, response: body };
    },
    async dispose() {
      if (exited) return;
      child.kill("SIGTERM");
      const stopped = new Promise((resolve) => child.once("exit", resolve));
      await Promise.race([stopped, delay(5_000)]);
      if (!exited) {
        child.kill("SIGKILL");
        await stopped;
      }
    },
  };
}
