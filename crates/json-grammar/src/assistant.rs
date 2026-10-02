use std::rc::Rc;

use minifield_engine_api::TokenId;

use crate::enforcer::{Enforcer, Machine};
use crate::parser::{Frame, feed_on, frames_key};
use crate::tool::validate_tool_names;

/// Fixed literals of the serialized assistant body, matching the
/// lfm2-chatml-tool-json training serializer's compact canonical form.
const ASSIST_CONTENT_KEY: &[u8] = b"{\"content\":";
const ASSIST_CALLS_KEY: &[u8] = b",\"tool_calls\":[";
const ASSIST_ARGS_KEY: &[u8] = b"{\"arguments\":";
const ASSIST_ID_KEY: &[u8] = b",\"id\":\"";
const ASSIST_NAME_KEY: &[u8] = b",\"name\":\"";

/// Linear phases of the serialized assistant body
/// `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`.
/// `content` accepts any JSON value, `arguments` must be an object, and
/// `name` must close on a registered name. The fixed envelope has no
/// whitespace; JSON value slots retain ordinary JSON whitespace rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssistPhase {
    /// Inside `{"content":`.
    ContentKey(usize),
    /// Inside the `content` value; the JSON stack tracks it.
    ContentValue,
    /// Inside `,"tool_calls":[`.
    CallsKey(usize),
    /// Inside the calls array after `[`: `{` opens a call, `]` ends it.
    CallOrClose,
    /// Inside `{"arguments":`.
    ArgsKey(usize),
    /// Inside the `arguments` object; the JSON stack tracks it.
    ArgsValue,
    /// Inside `,"id":"`.
    IdKey(usize),
    /// Inside the id string.
    IdBody,
    /// Inside an id escape.
    IdEsc,
    /// `\u` escape with `left` hex digits remaining.
    IdHex(u8),
    /// Inside `,"name":"`.
    NameKey(usize),
    /// Inside the name string; `key` holds the bytes emitted so far.
    NameBody,
    /// Name closed; `}` ends the call object.
    CallEnd,
    /// After a call object: `,` starts another call, `]` ends the array.
    AfterCall,
    /// Array closed; `}` ends the document.
    CloseBrace,
    /// Document complete.
    Done,
}

/// One byte through the assistant-body machine. `key` is the emitted name
/// bytes; `stack` is the JSON parser stack for the active value slot;
/// `names` is the registered set `name` must match exactly.
fn step_assist(
    state: &mut AssistPhase,
    key: &mut Vec<u8>,
    stack: &mut Vec<Frame>,
    names: &[Vec<u8>],
    byte: u8,
) -> bool {
    match *state {
        AssistPhase::ContentKey(_)
        | AssistPhase::CallsKey(_)
        | AssistPhase::ArgsKey(_)
        | AssistPhase::IdKey(_)
        | AssistPhase::NameKey(_) => {
            return step_literal(state, stack, byte);
        }
        AssistPhase::ContentValue => {
            // A bare number remains active until its delimiter arrives.
            // Its comma belongs to the assistant envelope, so consume it
            // here rather than re-feeding it into a completed JSON root.
            if byte == b',' && matches!(stack.as_slice(), [Frame::Num(phase)] if phase.terminable())
            {
                stack.clear();
                *state = AssistPhase::CallsKey(1);
                return true;
            }
            if !feed_on(stack, byte) {
                return false;
            }
            if stack.is_empty() {
                *state = AssistPhase::CallsKey(0);
            }
        }
        AssistPhase::CallOrClose => match byte {
            b'{' => *state = AssistPhase::ArgsKey(1),
            b']' => *state = AssistPhase::CloseBrace,
            _ => return false,
        },
        AssistPhase::ArgsValue => {
            if !feed_on(stack, byte) {
                return false;
            }
            if stack.is_empty() {
                *state = AssistPhase::IdKey(0);
            }
        }
        AssistPhase::IdBody => match byte {
            b'"' => *state = AssistPhase::NameKey(0),
            b'\\' => *state = AssistPhase::IdEsc,
            0x00..=0x1f => return false,
            _ => {}
        },
        AssistPhase::IdEsc => match byte {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                *state = AssistPhase::IdBody;
            }
            b'u' => *state = AssistPhase::IdHex(4),
            _ => return false,
        },
        AssistPhase::IdHex(left) => match byte {
            b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => {
                *state = if left == 1 {
                    AssistPhase::IdBody
                } else {
                    AssistPhase::IdHex(left - 1)
                };
            }
            _ => return false,
        },
        AssistPhase::NameBody => {
            if byte == b'"' {
                if !names.iter().any(|name| name.as_slice() == key.as_slice()) {
                    return false;
                }
                key.clear();
                *state = AssistPhase::CallEnd;
            } else {
                let pos = key.len();
                if !names.iter().any(|name| {
                    name.len() > pos && name.starts_with(key.as_slice()) && name[pos] == byte
                }) {
                    return false;
                }
                key.push(byte);
            }
        }
        AssistPhase::CallEnd => {
            if byte != b'}' {
                return false;
            }
            *state = AssistPhase::AfterCall;
        }
        AssistPhase::AfterCall => match byte {
            b',' => *state = AssistPhase::ArgsKey(0),
            b']' => *state = AssistPhase::CloseBrace,
            _ => return false,
        },
        AssistPhase::CloseBrace => {
            if byte != b'}' {
                return false;
            }
            *state = AssistPhase::Done;
        }
        AssistPhase::Done => return false,
    }
    true
}

/// Advance a fixed envelope literal and initialize its JSON slot, if any.
fn step_literal(state: &mut AssistPhase, stack: &mut Vec<Frame>, byte: u8) -> bool {
    match *state {
        AssistPhase::ContentKey(pos) => {
            if byte != ASSIST_CONTENT_KEY[pos] {
                return false;
            }
            *state = if pos + 1 == ASSIST_CONTENT_KEY.len() {
                stack.clear();
                stack.push(Frame::Value);
                AssistPhase::ContentValue
            } else {
                AssistPhase::ContentKey(pos + 1)
            };
        }
        AssistPhase::CallsKey(pos) => {
            if byte != ASSIST_CALLS_KEY[pos] {
                return false;
            }
            *state = if pos + 1 == ASSIST_CALLS_KEY.len() {
                AssistPhase::CallOrClose
            } else {
                AssistPhase::CallsKey(pos + 1)
            };
        }
        AssistPhase::ArgsKey(pos) => {
            if byte != ASSIST_ARGS_KEY[pos] {
                return false;
            }
            *state = if pos + 1 == ASSIST_ARGS_KEY.len() {
                stack.clear();
                stack.push(Frame::RootObject);
                AssistPhase::ArgsValue
            } else {
                AssistPhase::ArgsKey(pos + 1)
            };
        }
        AssistPhase::IdKey(pos) => {
            if byte != ASSIST_ID_KEY[pos] {
                return false;
            }
            *state = if pos + 1 == ASSIST_ID_KEY.len() {
                AssistPhase::IdBody
            } else {
                AssistPhase::IdKey(pos + 1)
            };
        }
        AssistPhase::NameKey(pos) => {
            if byte != ASSIST_NAME_KEY[pos] {
                return false;
            }
            *state = if pos + 1 == ASSIST_NAME_KEY.len() {
                AssistPhase::NameBody
            } else {
                AssistPhase::NameKey(pos + 1)
            };
        }
        _ => unreachable!("step_literal requires an envelope literal phase"),
    }
    true
}

/// Serialized-assistant-body acceptor used by [`AssistantCallEnforcer`]:
/// `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
/// exactly, with a compact fixed envelope, where `<name>` is a registered
/// tool name. `content` and `arguments` reuse the JSON parser stack; `id` is a
/// free string; multi-call arrays are accepted.
pub struct AssistantCallMachine {
    names: Rc<[Vec<u8>]>,
    state: AssistPhase,
    /// Bytes emitted inside the current `name` string.
    key: Vec<u8>,
    /// JSON parser stack for the active `content`/`arguments` slot.
    stack: Vec<Frame>,
    /// Reusable fork buffers for `accepts`.
    scratch_stack: Vec<Frame>,
    scratch_key: Vec<u8>,
}

impl AssistantCallMachine {
    /// `names` is the registered tool-name set the `name` member must match.
    #[must_use]
    pub fn new(names: Vec<Vec<u8>>) -> Self {
        validate_tool_names(&names);
        Self {
            names: names.into(),
            state: AssistPhase::ContentKey(0),
            key: Vec::new(),
            stack: Vec::new(),
            scratch_stack: Vec::new(),
            scratch_key: Vec::new(),
        }
    }
}

impl Machine for AssistantCallMachine {
    fn feed(&mut self, byte: u8) -> bool {
        step_assist(
            &mut self.state,
            &mut self.key,
            &mut self.stack,
            &self.names,
            byte,
        )
    }

    /// Simulate on scratch copies; `AssistPhase` itself is `Copy`.
    fn accepts(&mut self, bytes: &[u8]) -> bool {
        self.scratch_stack.clear();
        self.scratch_stack.extend_from_slice(&self.stack);
        self.scratch_key.clear();
        self.scratch_key.extend_from_slice(&self.key);
        let mut state = self.state;
        let names = &self.names;
        let stack = &mut self.scratch_stack;
        let key = &mut self.scratch_key;
        bytes
            .iter()
            .all(|&byte| step_assist(&mut state, key, stack, names, byte))
    }

    fn complete(&self) -> bool {
        self.state == AssistPhase::Done
    }

    fn finished(&self) -> bool {
        self.complete()
    }

    /// The mask at `NameBody` depends on which names the emitted bytes can
    /// still reach, and slot masks depend on the JSON stack; both are part
    /// of the cache key.
    fn key(&self, out: &mut Vec<u8>) {
        match self.state {
            AssistPhase::ContentKey(pos) => out.extend([0, u8::try_from(pos).unwrap_or(u8::MAX)]),
            AssistPhase::ContentValue => {
                out.push(1);
                frames_key(&self.stack, out);
            }
            AssistPhase::CallsKey(pos) => out.extend([2, u8::try_from(pos).unwrap_or(u8::MAX)]),
            AssistPhase::CallOrClose => out.push(3),
            AssistPhase::ArgsKey(pos) => out.extend([4, u8::try_from(pos).unwrap_or(u8::MAX)]),
            AssistPhase::ArgsValue => {
                out.push(5);
                frames_key(&self.stack, out);
            }
            AssistPhase::IdKey(pos) => out.extend([6, u8::try_from(pos).unwrap_or(u8::MAX)]),
            AssistPhase::IdBody => out.push(7),
            AssistPhase::IdEsc => out.push(8),
            AssistPhase::IdHex(left) => out.extend([9, left]),
            AssistPhase::NameKey(pos) => out.extend([10, u8::try_from(pos).unwrap_or(u8::MAX)]),
            AssistPhase::NameBody => {
                out.push(11);
                out.push(u8::try_from(self.key.len()).unwrap_or(u8::MAX));
                out.extend_from_slice(&self.key);
            }
            AssistPhase::CallEnd => out.push(12),
            AssistPhase::AfterCall => out.push(13),
            AssistPhase::CloseBrace => out.push(14),
            AssistPhase::Done => out.push(15),
        }
    }
}

/// Assistant-body grammar enforcer: the lfm2-chatml-tool-json serialized
/// shape `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
/// where `<name>` is one of the registered tool names.
pub type AssistantCallEnforcer = Enforcer<AssistantCallMachine>;

impl Enforcer<AssistantCallMachine> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `names` is the registered tool-name set.
    /// `eos` becomes allowed once the document completes.
    #[must_use]
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId, names: Vec<Vec<u8>>) -> Self {
        Self::with_machine(vocab, eos, AssistantCallMachine::new(names))
    }
}
