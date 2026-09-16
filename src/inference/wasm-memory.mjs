/** Reacquire linear-memory views after inference and detach returned results. */
export function copyFiniteF32(memory, pointer, length) {
  if (!(memory instanceof WebAssembly.Memory)) throw new TypeError("Expected WebAssembly.Memory");
  if (!Number.isSafeInteger(pointer) || pointer < 0 || pointer % 4 !== 0 ||
      !Number.isSafeInteger(length) || length < 1) {
    throw new RangeError("Invalid FP32 output range");
  }
  const buffer = memory.buffer;
  if (pointer > buffer.byteLength || length > (buffer.byteLength - pointer) / 4) {
    throw new RangeError("FP32 output exceeds linear memory");
  }
  const output = new Float32Array(buffer, pointer, length).slice();
  if (!output.every(Number.isFinite)) throw new Error("Inference returned non-finite values");
  return output;
}
