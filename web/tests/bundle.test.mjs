import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { bundleWeb } from "../../scripts/bundle_web.mjs";

test("browser package imports without a filesystem base and pins the delivered bytes", async () => {
  const root = await mkdtemp(resolve(tmpdir(), "minifield-bundle-test-"));
  try {
    const bindings = resolve(root, "bindings");
    const output = resolve(root, "pkg");
    await mkdir(resolve(bindings, "snippets"), { recursive: true });
    await writeFile(resolve(bindings, "snippets/helper.mjs"), "export const value = 42;");
    await writeFile(resolve(bindings, "minifield_web_demo.js"), 'import { value } from "./snippets/helper.mjs"; export { value }; export default () => new URL("minifield_web_demo_bg.wasm", import.meta.url);');
    await writeFile(resolve(bindings, "minifield_web_demo_bg.wasm"), new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
    for (const file of ["minifield_web_demo.d.ts", "minifield_web_demo_bg.wasm.d.ts"]) await writeFile(resolve(bindings, file), "export {};\n");
    const assets = await bundleWeb(bindings, output);
    const source = await readFile(resolve(output, "minifield_web_demo.js"));
    // A data URL, like a blob URL, cannot resolve a relative module import.
    const runtime = await import(`data:text/javascript;base64,${source.toString("base64")}`);
    assert.equal(runtime.value, 42);
    assert.ok(!source.toString().includes(root));
    assert.ok(!source.toString().includes("import {"));
    assert.ok(!(await readdir(output)).includes("snippets"));
    assert.deepEqual(JSON.parse(await readFile(resolve(output, "assets.json"), "utf8")), assets);
    for (const [name, asset] of Object.entries(assets)) {
      const bytes = await readFile(resolve(output, name));
      assert.equal(asset.bytes, bytes.length);
      assert.equal(asset.sha256, createHash("sha256").update(bytes).digest("hex"));
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});
