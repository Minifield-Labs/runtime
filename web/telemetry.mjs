// Delivery stays in the browser host. Only explicitly assembled runtime records enter here.
const DEFAULT_ENDPOINT = "https://telemetry.minifieldlabs.com/insert/jsonline";
const MAX_QUEUED = 64;
const MAX_BATCH_BYTES = 48 * 1024;
let options = { enabled: true, endpoint: DEFAULT_ENDPOINT };
let pending = [];
let pendingBytes = 0;
let timer;
const active = new Map();

function label(value) {
  return typeof value === "string" && /^[a-zA-Z0-9._:/@+-]{1,128}$/.test(value) ? value : null;
}
function origin(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.origin : null;
  } catch { return null; }
}

export function configureTelemetry(next = {}) {
  if (typeof next !== "object" || next === null) throw new TypeError("Telemetry options must be an object");
  const updated = { ...options };
  if ("enabled" in next) {
    if (typeof next.enabled !== "boolean") throw new TypeError("enabled must be boolean");
    updated.enabled = next.enabled;
  }
  if ("endpoint" in next) {
    const url = new URL(next.endpoint);
    if (url.username || url.password || url.search || url.hash ||
        (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)))) {
      throw new TypeError("Telemetry endpoint requires HTTPS (HTTP is allowed on loopback)");
    }
    updated.endpoint = url.href;
  }
  if ("environment" in next) {
    if (!["production", "preview", "development", "test"].includes(next.environment)) throw new TypeError("Invalid telemetry environment");
    updated.environment = next.environment;
  }
  for (const key of ["integrationId", "productId", "applicationId", "applicationVersion"]) {
    if (key in next) {
      if (next[key] !== null && !label(next[key])) throw new TypeError(`Invalid ${key}`);
      updated[key] = next[key];
    }
  }
  if ("websiteOrigin" in next) updated.websiteOrigin = origin(next.websiteOrigin);
  // Configuration changes never send old queued data to a new destination.
  clearTimeout(timer);
  timer = undefined;
  pending = [];
  pendingBytes = 0;
  for (const controller of active.keys()) controller.abort();
  options = updated;
}

export function telemetryEnabled() { return options.enabled; }

function platform() {
  const ua = globalThis.navigator?.userAgent ?? "";
  const browser = [["edge", /Edg\/(\d+)/], ["chrome", /(?:Chrome|CriOS)\/(\d+)/],
    ["firefox", /(?:Firefox|FxiOS)\/(\d+)/], ["safari", /Version\/(\d+).*Safari/]]
    .map(([family, pattern]) => ({ family, match: ua.match(pattern) })).find(item => item.match);
  const os = /Android/.test(ua) ? "android" : /iPhone|iPad/.test(ua) ? "ios" :
    /Macintosh|Mac OS X/.test(ua) ? "macos" : /Windows/.test(ua) ? "windows" : /Linux/.test(ua) ? "linux" : null;
  return { host: "browser", os: os ? { family: os, major_version: null } : null,
    browser: browser ? { family: browser.family, major_version: browser.match[1] } : null };
}

export function reportTelemetry(encoded) {
  if (!options.enabled || typeof globalThis.fetch !== "function") return;
  try {
    const record = JSON.parse(encoded);
    const website = options.websiteOrigin ?? origin(globalThis.location?.origin);
    const local = website && ["localhost", "127.0.0.1", "[::1]"].includes(new URL(website).hostname);
    record.origin = {
      integration_id: options.integrationId ?? "minifield-web",
      product_id: options.productId ?? null, website_origin: website,
      application_id: options.applicationId ?? null, application_version: options.applicationVersion ?? null,
      environment: options.environment ?? (local ? "development" : "production"),
    };
    record.platform = platform();
    const line = JSON.stringify(record) + "\n";
    const size = new TextEncoder().encode(line).byteLength;
    if (size > MAX_BATCH_BYTES || pending.length >= MAX_QUEUED || pendingBytes + size > MAX_BATCH_BYTES) return;
    pending.push(line);
    pendingBytes += size;
    if (!timer) timer = setTimeout(() => { void flushTelemetry(); }, 1000);
  } catch { /* Reporting must never affect inference. */ }
}

export async function flushTelemetry() {
  clearTimeout(timer);
  timer = undefined;
  if (!options.enabled) return;
  // Keepalive bodies share a 64 KiB quota, so 48 KiB batches must run serially.
  if (active.size >= 1 && pending.length) {
    await Promise.race(active.values());
    return flushTelemetry();
  }
  if (pending.length) {
    const body = pending.join("");
    pending = [];
    pendingBytes = 0;
    const endpoint = options.endpoint;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const request = Promise.resolve().then(() => fetch(endpoint, {
      method: "POST", body, headers: { "Content-Type": "text/plain;charset=UTF-8" },
      // Cross-origin no-cors requests require follow redirect mode in the Fetch standard.
      mode: "no-cors", credentials: "omit", referrerPolicy: "no-referrer", redirect: "follow",
      keepalive: true, signal: controller.signal,
    })).catch(() => {}).finally(() => { clearTimeout(timeout); active.delete(controller); });
    active.set(controller, request);
  }
  // Also await a batch that the timer already dispatched before the host called flush.
  await Promise.all(active.values());
}

globalThis.addEventListener?.("pagehide", () => { void flushTelemetry(); });
