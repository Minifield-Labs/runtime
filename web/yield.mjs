/** Give WebGPU callbacks a macrotask without losing concurrent callers. */
export function createYieldScheduler() {
  const { port1, port2 } = new MessageChannel();
  const waiting = [];
  let cursor = 0;
  let closed = false;
  port1.onmessage = () => {
    const next = waiting[cursor++];
    if (cursor === waiting.length) {
      waiting.length = 0;
      cursor = 0;
    }
    next?.resolve();
  };
  return {
    yield() {
      if (closed) return Promise.reject(new Error("Yield scheduler is closed"));
      return new Promise((resolve, reject) => {
        waiting.push({ resolve, reject });
        port2.postMessage(0);
      });
    },
    close() {
      closed = true;
      port1.close();
      port2.close();
      for (const { reject } of waiting.slice(cursor)) {
        reject(new Error("Yield scheduler is closed"));
      }
      waiting.length = 0;
      cursor = 0;
    },
  };
}
