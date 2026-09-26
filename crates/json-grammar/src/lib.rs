#![forbid(unsafe_code)]
//! Byte-level grammar enforcers for masked greedy decode.
//!
//! An enforcer simulates a grammar's acceptor over each vocab token's raw
//! bytes. At every decode step it yields a bitset of token ids whose bytes
//! keep the document acceptable, so the executor's masked argmax can only
//! emit tokens that continue a valid document. Masks for recent machine
//! states are cached; a cache miss scans the vocabulary, while
//! a hit costs one hashmap lookup plus the mask's upload to the backend.
//!
//! Three grammars ship here: [`JsonMachine`], a byte-level JSON parser
//! (objects, arrays, strings with standard escapes and `\uXXXX`, spec
//! numbers, `true`/`false`/`null`, JSON whitespace, unbounded nesting),
//! [`ToolMachine`], the fixed tool-call shape `{"<name>":true|false}` with
//! no whitespace, and [`AssistantCallMachine`], the serialized assistant
//! body `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
//! emitted by the lfm2-chatml-tool-json training serializer. UTF-8
//! continuation bytes are accepted inside strings byte-wise; the model's
//! tokenizer still owns real UTF-8 assembly.

mod assistant;
mod enforcer;
mod json;
mod parser;
mod tool;

pub use assistant::{AssistantCallEnforcer, AssistantCallMachine};
pub use enforcer::{Enforcer, Machine};
pub use json::{JsonEnforcer, JsonMachine};
pub use tool::{ToolCallEnforcer, ToolMachine};

#[cfg(test)]
mod tests;
