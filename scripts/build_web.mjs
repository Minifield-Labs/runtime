import { execFileSync } from "node:child_process";
import { mkdtemp, rm, mkdir, cp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundleWeb } from "./bundle_web.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scratch = await mkdtemp(resolve(tmpdir(), "minifield-web-build-"));
try {
  execFileSync("cargo", ["+1.89.0", "build", "--release", "--locked", "-p", "minifield-web-demo", "--target", "wasm32-unknown-unknown"], { cwd: root, stdio: "inherit" });
  const bindings = resolve(scratch, "bindings");
  execFileSync("wasm-bindgen", ["target/wasm32-unknown-unknown/release/minifield_web_demo.wasm", "--target", "web", "--out-dir", bindings], { cwd: root, stdio: "inherit" });
  const output = resolve(scratch, "pkg");
  const assets = await bundleWeb(bindings, output);
  const destination = resolve(root, "web/pkg");
  // Replace only generated output after compilation and bundling succeed.
  await rm(destination, { recursive: true, force: true });
  await mkdir(destination, { recursive: true });
  await cp(output, destination, { recursive: true });
  console.log(JSON.stringify({ directory: destination, assets }, null, 2));
} finally {
  await rm(scratch, { recursive: true, force: true });
}
