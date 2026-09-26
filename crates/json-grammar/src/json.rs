use minifield_engine_api::TokenId;

use crate::enforcer::{Enforcer, Machine};
use crate::parser::{Frame, feed_on, frames_key};

/// Byte-level JSON parser used by [`JsonEnforcer`].
pub struct JsonMachine {
    stack: Vec<Frame>,
    /// Reusable fork buffer for `accepts`; never observable state.
    scratch: Vec<Frame>,
}

impl JsonMachine {
    /// Any JSON value as the root document.
    #[must_use]
    pub fn any_value() -> Self {
        Self {
            stack: vec![Frame::Value],
            scratch: Vec::new(),
        }
    }

    /// The root document must be an object whose first byte is `{`; a
    /// response document has no leading whitespace.
    #[must_use]
    pub fn object_root() -> Self {
        Self {
            stack: vec![Frame::RootObject],
            scratch: Vec::new(),
        }
    }
}

impl Machine for JsonMachine {
    fn feed(&mut self, byte: u8) -> bool {
        feed_on(&mut self.stack, byte)
    }

    fn accepts(&mut self, bytes: &[u8]) -> bool {
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.stack);
        bytes.iter().all(|&byte| feed_on(&mut self.scratch, byte))
    }

    /// A complete JSON value has been emitted (or is completable, for a
    /// terminable root-level number).
    fn complete(&self) -> bool {
        self.stack.is_empty()
            || (self.stack.len() == 1
                && matches!(self.stack[0], Frame::Num(phase) if phase.terminable()))
    }

    fn key(&self, out: &mut Vec<u8>) {
        frames_key(&self.stack, out);
    }
}

/// JSON grammar enforcer: any-value root via [`JsonEnforcer::new`], or an
/// object root via [`JsonEnforcer::object`].
pub type JsonEnforcer = Enforcer<JsonMachine>;

impl Enforcer<JsonMachine> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `eos` becomes allowed once a complete JSON
    /// value has been emitted.
    #[must_use]
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId) -> Self {
        Self::with_machine(vocab, eos, JsonMachine::any_value())
    }

    /// Same grammar but the root document must be an object whose first
    /// byte is `{`.
    #[must_use]
    pub fn object(vocab: Vec<Vec<u8>>, eos: TokenId) -> Self {
        Self::with_machine(vocab, eos, JsonMachine::object_root())
    }
}
