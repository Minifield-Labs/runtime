/**
 * Serialize model lifetime changes and inference. Adapted from the ownership
 * pattern in Wires; replaces its take/await/restore race. No GPU API is assumed.
 */
export class ResidentModel {
  #model = null;
  #key = null;
  #tail = Promise.resolve();
  #pending = 0;
  #limit;
  #closed = false;
  #disposal = null;

  constructor({ maxPending = 8 } = {}) {
    if (!Number.isSafeInteger(maxPending) || maxPending < 1) {
      throw new RangeError("maxPending must be a positive integer");
    }
    this.#limit = maxPending;
  }

  get key() { return this.#key; }
  get pending() { return this.#pending; }

  #enqueue(operation, closing = false) {
    if (!closing && this.#closed) return Promise.reject(new Error("Model owner is disposed"));
    if (!closing && this.#pending >= this.#limit) return Promise.reject(new Error("Model queue is full"));
    this.#pending += 1;
    const result = this.#tail.then(operation).finally(() => { this.#pending -= 1; });
    // A rejected request mustn't poison later work or become an unhandled tail.
    this.#tail = result.catch(() => {});
    return result;
  }

  load(key, factory) {
    if (typeof key !== "string" || key.length === 0 || typeof factory !== "function") {
      return Promise.reject(new TypeError("Load requires a versioned key and model factory"));
    }
    return this.#enqueue(async () => {
      if (this.#model && this.#key === key) return;
      const next = await factory();
      if (!next || typeof next.dispose !== "function") throw new TypeError("Model factory must return a disposable resource");
      if (next === this.#model) throw new TypeError("A replacement key requires a distinct model resource");
      const previous = this.#model;
      this.#model = null;
      this.#key = null;
      try {
        if (previous) await previous.dispose();
      } catch (error) {
        try { await next.dispose(); }
        catch (cleanup) { throw new AggregateError([error, cleanup], "Model replacement cleanup failed"); }
        throw error;
      }
      this.#model = next;
      this.#key = key;
    });
  }

  run(operation, { signal } = {}) {
    if (typeof operation !== "function") return Promise.reject(new TypeError("Inference operation must be callable"));
    return this.#enqueue(async () => {
      if (signal?.aborted) throw new DOMException("Inference cancelled", "AbortError");
      if (!this.#model) throw new Error("Load a model before inference");
      // Completion must mean the backend's final GPU consumer/readback settled.
      // An abort notification alone mustn't release resources still in use.
      const output = await operation(this.#model, signal);
      if (signal?.aborted) throw new DOMException("Inference cancelled", "AbortError");
      return output;
    });
  }

  async #release() {
    const previous = this.#model;
    this.#model = null;
    this.#key = null;
    if (previous) await previous.dispose();
  }

  unload() { return this.#enqueue(() => this.#release()); }

  dispose() {
    if (!this.#disposal) {
      this.#closed = true;
      this.#disposal = this.#enqueue(() => this.#release(), true);
    }
    return this.#disposal;
  }
}
