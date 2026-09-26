use std::rc::Rc;

use minifield_engine_api::TokenId;

use crate::enforcer::{Enforcer, Machine};

/// Linear states for the tool-call shape `{"<name>":true|false}`.
/// The emitted key bytes live on the machine (`key`), not the state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolState {
    /// Need `{`.
    OpenBrace,
    /// Need `"` to open the key.
    KeyQuote,
    /// In the key string; `key` holds the bytes emitted so far.
    Key,
    /// Need `:`.
    Colon,
    /// Need `t` or `f`.
    Bool,
    /// Remainder of `true`/`false`: expected word plus consumed count.
    InLit(&'static [u8], usize),
    /// Need `}`.
    CloseBrace,
    /// Document complete.
    Done,
}

/// One byte through the tool-call machine. `key` is the emitted key bytes;
/// `names` is the registered tool-name set the key must match exactly.
/// Observable state is only mutated on a successful transition.
fn step_tool(state: &mut ToolState, key: &mut Vec<u8>, names: &[Vec<u8>], byte: u8) -> bool {
    match *state {
        ToolState::OpenBrace => {
            if byte != b'{' {
                return false;
            }
            *state = ToolState::KeyQuote;
        }
        ToolState::KeyQuote => {
            if byte != b'"' {
                return false;
            }
            *state = ToolState::Key;
        }
        ToolState::Key => {
            if byte == b'"' {
                // The key closes only on an exact registered name.
                if !names.iter().any(|name| name.as_slice() == key.as_slice()) {
                    return false;
                }
                *state = ToolState::Colon;
            } else {
                // The byte must keep the emitted key a prefix of a single
                // registered name; checking `name[pos]` alone would let the
                // key drift across different names position by position.
                let pos = key.len();
                if !names.iter().any(|name| {
                    name.len() > pos && name.starts_with(key.as_slice()) && name[pos] == byte
                }) {
                    return false;
                }
                key.push(byte);
            }
        }
        ToolState::Colon => {
            if byte != b':' {
                return false;
            }
            *state = ToolState::Bool;
        }
        ToolState::Bool => match byte {
            b't' => *state = ToolState::InLit(b"true", 1),
            b'f' => *state = ToolState::InLit(b"false", 1),
            _ => return false,
        },
        ToolState::InLit(word, pos) => {
            if byte != word[pos] {
                return false;
            }
            *state = if pos + 1 == word.len() {
                ToolState::CloseBrace
            } else {
                ToolState::InLit(word, pos + 1)
            };
        }
        ToolState::CloseBrace => {
            if byte != b'}' {
                return false;
            }
            *state = ToolState::Done;
        }
        ToolState::Done => return false,
    }
    true
}

/// Fixed-shape tool-call acceptor used by [`ToolCallEnforcer`]:
/// `{"<name>":true}` or `{"<name>":false}` exactly, where `<name>` is one of
/// the registered tool names: compact JSON, no whitespace, one boolean
/// member. The key can never exceed the longest registered name, so every
/// accepted prefix reaches a complete document. Non-ASCII bytes and escapes
/// are impossible in a name, so emitted text is always valid UTF-8.
pub struct ToolMachine {
    names: Rc<[Vec<u8>]>,
    state: ToolState,
    /// Bytes emitted inside the key so far.
    key: Vec<u8>,
    /// Reusable fork buffer for `accepts`.
    scratch: Vec<u8>,
}

/// Every registered name must be nonempty printable ASCII without `"` or
/// `\` (a name that could never close or continue would be dead grammar).
pub(crate) fn validate_tool_names(names: &[Vec<u8>]) {
    assert!(
        !names.is_empty(),
        "tool grammar requires at least one registered name"
    );
    for name in names {
        assert!(
            !name.is_empty()
                && name
                    .iter()
                    .all(|b| (0x20..=0x7e).contains(b) && *b != b'"' && *b != b'\\'),
            "tool names must be nonempty printable ASCII without '\"' or '\\'"
        );
    }
}

impl ToolMachine {
    /// `names` is the registered tool-name set; every name must be nonempty
    /// printable ASCII without `"` or `\` (a name that could never close or
    /// continue would be dead grammar).
    #[must_use]
    pub fn new(names: Vec<Vec<u8>>) -> Self {
        validate_tool_names(&names);
        Self {
            names: names.into(),
            state: ToolState::OpenBrace,
            key: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl Machine for ToolMachine {
    fn feed(&mut self, byte: u8) -> bool {
        step_tool(&mut self.state, &mut self.key, &self.names, byte)
    }

    /// Simulate on a scratch copy of the key; `ToolState` itself is `Copy`.
    fn accepts(&mut self, bytes: &[u8]) -> bool {
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.key);
        let mut state = self.state;
        let names = &self.names;
        let key = &mut self.scratch;
        bytes
            .iter()
            .all(|&byte| step_tool(&mut state, key, names, byte))
    }

    fn complete(&self) -> bool {
        self.state == ToolState::Done
    }

    /// The mask at `Key` depends on which names the emitted bytes can still
    /// reach, so the emitted key is part of the cache key.
    fn key(&self, out: &mut Vec<u8>) {
        match self.state {
            ToolState::OpenBrace => out.push(0),
            ToolState::KeyQuote => out.push(1),
            ToolState::Key => {
                out.push(2);
                out.push(u8::try_from(self.key.len()).unwrap_or(u8::MAX));
                out.extend_from_slice(&self.key);
            }
            ToolState::Colon => out.push(3),
            ToolState::Bool => out.push(4),
            ToolState::InLit(word, pos) => {
                out.push(5);
                out.push(u8::from(word != b"true"));
                out.push(u8::try_from(pos).unwrap_or(u8::MAX));
            }
            ToolState::CloseBrace => out.push(6),
            ToolState::Done => out.push(7),
        }
    }
}

/// Tool-call grammar enforcer: `{"<name>":true|false}` where `<name>` is
/// one of the registered names.
pub type ToolCallEnforcer = Enforcer<ToolMachine>;

impl Enforcer<ToolMachine> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `names` is the registered tool-name set.
    /// `eos` becomes allowed once the closing `}` is emitted.
    #[must_use]
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId, names: Vec<Vec<u8>>) -> Self {
        Self::with_machine(vocab, eos, ToolMachine::new(names))
    }
}
