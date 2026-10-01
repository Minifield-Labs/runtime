import test from "node:test";
import assert from "node:assert/strict";
import { configureTelemetry, reportTelemetry, flushTelemetry } from "../telemetry.mjs";

test("telemetry batches records, omits credentials/referrer, bounds input, and tolerates delivery failures", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, request) => { requests.push({ url, ...request }); return new Response(); };
  try {
    configureTelemetry({ enabled: true, endpoint: "http://127.0.0.1:8000/insert/jsonline", environment: "test",
      websiteOrigin: "https://example.com/private/account?secret=value#fragment", integrationId: "test-sdk" });
    reportTelemetry('{"inference_id":"one"}');
    reportTelemetry('{"inference_id":"two"}');
    await flushTelemetry();
    assert.equal(requests.length, 1);
    const request = requests[0];
    const records = request.body.trim().split("\n").map(line => JSON.parse(line));
    assert.deepEqual(records.map(record => record.inference_id), ["one", "two"]);
    assert.equal(records[0].origin.website_origin, "https://example.com");
    assert.equal(records[0].origin.environment, "test");
    assert.equal(request.credentials, "omit");
    assert.equal(request.mode, "no-cors");
    assert.equal(request.referrerPolicy, "no-referrer");
    assert.equal(request.redirect, "follow");
    assert.ok(!request.body.includes("secret"));
    reportTelemetry("invalid JSON");
    reportTelemetry(JSON.stringify({ oversized: "x".repeat(50 * 1024) }));
    await flushTelemetry();
    assert.equal(requests.length, 1);
    for (let i = 0; i < 200; i++) reportTelemetry(JSON.stringify({ inference_id: i }));
    await flushTelemetry();
    assert.equal(requests[1].body.trim().split("\n").length, 64);
    assert.ok(Buffer.byteLength(requests[1].body) <= 48 * 1024);
    configureTelemetry({ enabled: false });
    reportTelemetry('{"inference_id":"disabled"}');
    await flushTelemetry();
    assert.equal(requests.length, 2);
    configureTelemetry({ enabled: true });
    globalThis.fetch = async () => { throw new Error("offline"); };
    reportTelemetry('{"inference_id":"offline"}');
    await assert.doesNotReject(flushTelemetry());
    // Disabling clears pending data, including a queued batch from before the switch.
    globalThis.fetch = originalFetch;
    reportTelemetry('{"inference_id":"queued"}');
    configureTelemetry({ enabled: false });
    await flushTelemetry();
    assert.throws(() => configureTelemetry({ endpoint: "http://example.com/" }));
    assert.throws(() => configureTelemetry({ endpoint: "https://user:password@example.com/" }));
    assert.throws(() => configureTelemetry({ environment: "unknown" }));
    assert.throws(() => configureTelemetry({ integrationId: "private request content" }));
  } finally { configureTelemetry({ enabled: false }); globalThis.fetch = originalFetch; }
});

test("telemetry serializes large keepalive batches", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const releases = [];
  let first;
  let second;
  globalThis.fetch = async (_url, request) => {
    requests.push(request);
    await new Promise(resolve => releases.push(resolve));
    return new Response();
  };
  try {
    configureTelemetry({ enabled: true, endpoint: "http://127.0.0.1:8000/insert/jsonline", environment: "test" });
    const record = JSON.stringify({ inference_id: "large", padding: "x".repeat(40 * 1024) });
    reportTelemetry(record);
    first = flushTelemetry();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, 1);
    reportTelemetry(record);
    second = flushTelemetry();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, 1, "second batch waits for the first request");
    releases[0]();
    await first;
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, 2);
    assert.ok(requests.every(request => request.keepalive && Buffer.byteLength(request.body) <= 48 * 1024));
    releases[1]();
    await second;
  } finally {
    configureTelemetry({ enabled: false });
    for (const release of releases) release();
    await Promise.allSettled([first, second].filter(Boolean));
    globalThis.fetch = originalFetch;
  }
});
