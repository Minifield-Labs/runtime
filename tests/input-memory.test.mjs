import { test } from "node:test";
import assert from "node:assert/strict";
import { copyFiniteF32 } from "../src/inference/wasm-memory.mjs";
import { prepareTokenInput } from "../src/context/token-input.mjs";
import { sequenceBucket } from "../src/inference/sequence-buckets.mjs";

test("output views are reacquired after WASM growth and detached from storage", () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
  const old = new Float32Array(memory.buffer, 4, 2);
  memory.grow(1);
  assert.equal(old.byteLength, 0);
  const current = new Float32Array(memory.buffer, 4, 2); current.set([1, 2]);
  const result = copyFiniteF32(memory, 4, 2);
  current[0] = 8;
  assert.deepEqual([...result], [1, 2]);
});

test("output reads reject misalignment, overflow and non-finite results", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  for (const [pointer, length] of [[1, 1], [-4, 1], [0, 0], [65536, 1], [0, Number.MAX_SAFE_INTEGER]]) {
    assert.throws(() => copyFiniteF32(memory, pointer, length), RangeError);
  }
  new Float32Array(memory.buffer, 0, 1)[0] = NaN;
  assert.throws(() => copyFiniteF32(memory, 0, 1), /non-finite/);
});

test("raw token IDs and masks are checked before typed conversion", () => {
  for (const value of [-1, .5, NaN, Infinity, 2 ** 32, "1", 1n]) {
    assert.throws(() => prepareTokenInput([value], { maxTokens: 4 }), RangeError);
  }
  for (const mask of [[2], [256], [0], [1, 1]]) {
    assert.throws(() => prepareTokenInput([1], { mask, maxTokens: 4 }), RangeError);
  }
  const tokens = [1, 2], mask = [1, 0];
  const result = prepareTokenInput(tokens, { mask, maxTokens: 2 });
  tokens[0] = 99; mask[0] = 0;
  assert.deepEqual([...result.tokenIds], [1, 2]);
  assert.deepEqual([...result.mask], [1, 0]);
});

test("sequence buckets have explicit limits and don't change input semantics", () => {
  assert.equal(sequenceBucket(63), 64);
  assert.equal(sequenceBucket(64), 64);
  assert.equal(sequenceBucket(65), 128);
  for (const size of [0, -1, 1.5, 513]) assert.throws(() => sequenceBucket(size), RangeError);
  assert.throws(() => sequenceBucket(4, [8, 4]), RangeError);
  assert.throws(() => sequenceBucket(4, [8, 8]), RangeError);
  assert.equal(sequenceBucket(513, [512, 1024]), 1024);
});
