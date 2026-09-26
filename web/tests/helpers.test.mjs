import { test } from "node:test";
import assert from "node:assert/strict";
import { createYieldScheduler } from "../yield.mjs";
import { canonicalModelJson } from "../canonical-json.mjs";

test("concurrent yields all resolve in order and the queue can be reused", { timeout: 1000 }, async () => {
  const scheduler = createYieldScheduler();
  try {
    const resolved = [];
    await Promise.all(Array.from({ length: 64 }, (_, i) => scheduler.yield().then(() => resolved.push(i))));
    assert.deepEqual(resolved, Array.from({ length: 64 }, (_, i) => i));
    await scheduler.yield();
  } finally {
    scheduler.close();
  }
});

test("closing a scheduler rejects pending and future work", async () => {
  const scheduler = createYieldScheduler();
  const pending = scheduler.yield();
  scheduler.close();
  await assert.rejects(pending, /closed/);
  await assert.rejects(scheduler.yield(), /closed/);
});

test("canonical model JSON sorts nested keys and rejects non-finite values", () => {
  assert.equal(canonicalModelJson({ z: [1, { b: true, a: null }], a: "x" }), '{"a":"x","z":[1,{"a":null,"b":true}]}');
  for (const value of [NaN, Infinity, undefined, 1n]) {
    assert.throws(() => canonicalModelJson(value), /finite JSON/);
  }
});
