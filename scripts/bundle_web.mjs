import { build } from "esbuild";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve } from "node:path";

// Keep the deployable module self-contained so verified bytes can be imported
// from a blob URL in a Worker. WASM remains a separate, explicitly loaded file.
export async function bundleWeb(bindings, destination) {
  await mkdir(destination, { recursive: true });
  const javascript = "minifield_web_demo.js";
  const wasm = "minifield_web_demo_bg.wasm";
  const result = await build({
    absWorkingDir: bindings,
    entryPoints: [javascript],
    outfile: resolve(destination, javascript),
    bundle: true,
    format: "esm",
    platform: "browser",
    target: "es2022",
    metafile: true,
    write: false,
  });
  if (Object.values(result.metafile.outputs).some(output => output.imports.length)) {
    throw new Error("Browser runtime must not contain external module imports");
  }
  for (const output of result.outputFiles) await writeFile(output.path, output.contents);
  for (const file of [wasm, "minifield_web_demo.d.ts", "minifield_web_demo_bg.wasm.d.ts"]) {
    await copyFile(resolve(bindings, file), resolve(destination, file));
  }
  const assets = {};
  for (const [name, contentType] of [[javascript, "text/javascript"], [wasm, "application/wasm"]]) {
    const bytes = await readFile(resolve(destination, name));
    assets[name] = { bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"), contentType };
  }
  await writeFile(resolve(destination, "assets.json"), JSON.stringify(assets, null, 2) + "\n");
  return assets;
}
