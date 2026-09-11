/** Select a bounded compiled shape. The caller owns masks and position IDs. */
export function sequenceBucket(length, buckets = [8, 16, 32, 64, 128, 256, 512]) {
  if (!Number.isSafeInteger(length) || length < 1 || !Array.isArray(buckets) || !buckets.length) {
    throw new RangeError("Expected a positive length and non-empty bucket list");
  }
  let previous = 0;
  for (const size of buckets) {
    if (!Number.isSafeInteger(size) || size <= previous) throw new RangeError("Buckets must be strictly increasing positive integers");
    previous = size;
  }
  const selected = buckets.find((size) => size >= length);
  if (selected === undefined) throw new RangeError("Sequence exceeds compiled shapes");
  return selected;
}
