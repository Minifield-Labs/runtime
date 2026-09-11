/** Validate before typed-array coercion so negative/fractional IDs can't wrap. */
export function prepareTokenInput(tokenIds, { maxTokens, mask = null } = {}) {
  const arrayLike = (x) => Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView));
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) throw new RangeError("A positive token budget is required");
  if (!arrayLike(tokenIds) || tokenIds.length < 1 || tokenIds.length > maxTokens) {
    throw new RangeError("Token input exceeds its bounds");
  }
  for (const value of tokenIds) {
    if (!Number.isInteger(value) || value < 0 || value > 0xFFFFFFFF) throw new RangeError("Token IDs must be unsigned 32-bit integers");
  }
  if (mask !== null) {
    if (!arrayLike(mask) || mask.length !== tokenIds.length) throw new RangeError("Mask length differs from token count");
    let active = false;
    for (const value of mask) {
      if (value !== 0 && value !== 1) throw new RangeError("Mask values must be 0 or 1");
      active ||= value === 1;
    }
    if (!active) throw new RangeError("At least one token must be active");
  }
  return { tokenIds: Uint32Array.from(tokenIds), mask: mask === null ? null : Uint8Array.from(mask) };
}
