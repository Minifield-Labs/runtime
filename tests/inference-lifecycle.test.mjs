import { test } from "node:test";
import assert from "node:assert/strict";
import { ResidentModel } from "../src/inference/resident-model.mjs";

const deferred = () => {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
};

test("concurrent same-key loads initialize once and disposal is idempotent", async () => {
  const owner = new ResidentModel();
  let loads = 0, frees = 0;
  const factory = async () => { loads++; return { dispose() { frees++; } }; };
  await Promise.all([owner.load("hash-a:engine-v1", factory), owner.load("hash-a:engine-v1", factory)]);
  assert.equal(loads, 1);
  await Promise.all([owner.dispose(), owner.dispose()]);
  assert.equal(frees, 1);
  await assert.rejects(owner.load("b", factory), /disposed/);
});

test("replacement and clear wait for inference and cannot restore stale weights", async () => {
  const owner = new ResidentModel();
  const started = deferred(), finish = deferred(), events = [];
  await owner.load("a", async () => ({ id: "a", dispose() { events.push("free-a"); } }));
  const inference = owner.run(async (model) => { started.resolve(); await finish.promise; events.push(`run-${model.id}`); return model.id; });
  await started.promise;
  const replacement = owner.load("b", async () => ({ id: "b", dispose() { events.push("free-b"); } }));
  const clear = owner.unload();
  assert.deepEqual(events, []);
  finish.resolve();
  assert.equal(await inference, "a");
  await Promise.all([replacement, clear]);
  assert.deepEqual(events, ["run-a", "free-a", "free-b"]);
  assert.equal(owner.key, null);
  await assert.rejects(owner.run(() => {}), /Load a model/);
});

test("request and factory failures leave the queue usable", async () => {
  const owner = new ResidentModel();
  await owner.load("a", async () => ({ id: "a", dispose() {} }));
  await assert.rejects(owner.load("b", async () => { throw Error("bad download"); }), /bad download/);
  assert.equal(owner.key, "a");
  await assert.rejects(owner.run(() => { throw Error("bad inference"); }), /bad inference/);
  assert.equal(await owner.run((model) => model.id), "a");
  await owner.dispose();
});

test("cancellation waits for in-flight consumers before freeing resources", async () => {
  const owner = new ResidentModel();
  const controller = new AbortController(), started = deferred(), finish = deferred();
  let freed = false;
  await owner.load("a", async () => ({ dispose() { freed = true; } }));
  const cancelled = assert.rejects(owner.run(async () => { started.resolve(); await finish.promise; }, { signal: controller.signal }), { name: "AbortError" });
  await started.promise;
  controller.abort();
  const disposed = owner.dispose();
  assert.equal(freed, false);
  finish.resolve();
  await cancelled;
  await disposed;
  assert.equal(freed, true);
});

test("aborted queued work never executes and backpressure is bounded", async () => {
  const owner = new ResidentModel({ maxPending: 1 });
  const finish = deferred();
  const loaded = owner.load("a", async () => { await finish.promise; return { dispose() {} }; });
  await assert.rejects(owner.run(() => {}), /queue is full/);
  finish.resolve(); await loaded;
  const controller = new AbortController(); controller.abort();
  let called = false;
  await assert.rejects(owner.run(() => { called = true; }, { signal: controller.signal }), { name: "AbortError" });
  assert.equal(called, false);
  await owner.dispose();
});

test("failed old-model disposal also cleans the replacement", async () => {
  const owner = new ResidentModel();
  let replacementFreed = false;
  await owner.load("a", async () => ({ dispose() { throw Error("free failed"); } }));
  await assert.rejects(owner.load("b", async () => ({ dispose() { replacementFreed = true; } })), /free failed/);
  assert.equal(owner.key, null);
  assert.equal(replacementFreed, true);
  await owner.dispose();
});

test("a new key cannot relabel and dispose the current resource", async () => {
  const owner = new ResidentModel();
  let freed = false;
  const model = { dispose() { freed = true; } };
  await owner.load("a", async () => model);
  await assert.rejects(owner.load("b", async () => model), /distinct model/);
  assert.equal(owner.key, "a");
  assert.equal(freed, false);
  await owner.dispose();
});
