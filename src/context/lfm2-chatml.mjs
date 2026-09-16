function canonical(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  throw new TypeError("Model-visible JSON must contain finite JSON values");
}

function renderBody(message) {
  if (!message || typeof message !== "object" || !["system", "user", "assistant", "tool"].includes(message.role)) {
    throw new TypeError("Conversation contains an invalid message");
  }
  if (message.role === "assistant") {
    return canonical({ content: message.content ?? "", tool_calls: message.tool_calls ?? [] });
  }
  if (message.role === "tool") {
    if (typeof message.tool_call_id !== "string" || typeof message.is_error !== "boolean") {
      throw new TypeError("Tool result requires a call ID and error flag");
    }
    return canonical({ tool_call_id: message.tool_call_id, is_error: message.is_error, content: message.content });
  }
  return typeof message.content === "string" ? message.content : canonical(message.content);
}

/** Reproduce the serializer used by the LFM2 training worker. */
export function renderLfm2Prompt({ serializer, messages, tools = [], product = null, initialObservation = null }) {
  if (!serializer || typeof serializer.bos_token !== "string" || !Array.isArray(messages)) {
    throw new TypeError("Serializer and messages are required");
  }
  let text = serializer.bos_token;
  if (tools.length) text += `<|im_start|>system\nAvailable tools:\n${canonical(tools)}<|im_end|>\n`;
  if (product !== null) text += `<|im_start|>system\nPublic product contract:\n${canonical(product)}<|im_end|>\n`;
  if (initialObservation !== null) text += `<|im_start|>system\nInitial public observation:\n${canonical(initialObservation)}<|im_end|>\n`;
  for (const message of messages) {
    text += `<|im_start|>${message.role}\n${renderBody(message)}<|im_end|>\n`;
  }
  return text + serializer.assistant_prefix;
}

export { canonical as canonicalModelJson };
