#![forbid(unsafe_code)]
//! Byte-level grammar enforcers for masked greedy decode.
//!
//! An enforcer simulates a grammar's acceptor over each vocab token's raw
//! bytes. At every decode step it yields a bitset of token ids whose bytes
//! keep the document acceptable, so the executor's masked argmax can only
//! emit tokens that continue a valid document. Masks are computed once per
//! distinct machine state and cached; per-step cost is one hashmap lookup
//! plus the mask's upload to the backend.
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

use std::collections::HashMap;
use std::rc::Rc;

use minifield_engine_api::{DecodeConstraint, TokenId};

/// Parser frame on the value stack. `Str`/`Esc`/`Hex` carry `key`: a key
/// string closes into `ObjColon`, a value string completes a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Frame {
    /// Expecting a value start.
    Value,
    /// Document root that must be an object: `{` opens it. Leading
    /// whitespace is not allowed: a response document starts immediately.
    RootObject,
    /// In an object after `{`: a key string or `}`.
    ObjKeyOrEnd,
    /// In an object after `,`: a key string only (no trailing comma).
    ObjKey,
    /// In an object after a key string: `:`.
    ObjColon,
    /// In an object after a member value: `,` or `}`.
    ObjCommaOrEnd,
    /// In an array after `[`: first element or `]`.
    ArrElemOrEnd,
    /// In an array after `,`: an element value only.
    ArrElem,
    /// In an array after an element: `,` or `]`.
    ArrCommaOrEnd,
    Str {
        key: bool,
    },
    Esc {
        key: bool,
    },
    /// `\u` escape with `left` hex digits remaining.
    Hex {
        key: bool,
        left: u8,
    },
    Num(NumPhase),
    /// `true`/`false`/`null`: expected word plus consumed count.
    Lit(&'static [u8], usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumPhase {
    /// Need the first integer digit.
    IntFirst,
    /// Integer digits seen; `.`/`e` or a delimiter may follow.
    Int,
    /// Leading `0`: no more integer digits allowed.
    IntZero,
    /// Need the first fraction digit.
    FracFirst,
    /// Fraction digits seen.
    Frac,
    /// `e`/`E` seen: optional sign or first exponent digit.
    ExpFirst,
    /// Exponent sign seen: digit required.
    ExpSign,
    /// Exponent digits seen.
    Exp,
}

impl NumPhase {
    /// The number may legally end here (a delimiter completes it).
    fn terminable(self) -> bool {
        matches!(self, Self::Int | Self::IntZero | Self::Frac | Self::Exp)
    }
}

fn is_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_delimiter(byte: u8) -> bool {
    is_ws(byte) || matches!(byte, b',' | b'}' | b']')
}

/// A value just completed: resume the enclosing frame.
fn value_end(stack: &mut Vec<Frame>) {
    match stack.last_mut() {
        Some(Frame::ObjColon) => {
            // The object frame below the member value resumes at comma-or-end.
            *stack.last_mut().expect("checked") = Frame::ObjCommaOrEnd;
        }
        Some(Frame::ArrCommaOrEnd) | None => {}
        Some(other) => {
            debug_assert!(false, "value_end above unexpected frame {other:?}");
        }
    }
}

/// Feed one byte through the parser stack; returns false when the byte is
/// invalid here.
fn feed_on(stack: &mut Vec<Frame>, byte: u8) -> bool {
    loop {
        let Some(top) = stack.last_mut() else {
            // Document complete: only trailing whitespace is legal.
            return is_ws(byte);
        };
        match *top {
            Frame::RootObject => match byte {
                b'{' => *top = Frame::ObjKeyOrEnd,
                _ => return false,
            },
            Frame::Value => match byte {
                b if is_ws(b) => return true,
                b'{' => *top = Frame::ObjKeyOrEnd,
                b'[' => *top = Frame::ArrElemOrEnd,
                b'"' => *top = Frame::Str { key: false },
                b'-' => *top = Frame::Num(NumPhase::IntFirst),
                b'0' => *top = Frame::Num(NumPhase::IntZero),
                b'1'..=b'9' => *top = Frame::Num(NumPhase::Int),
                b't' => *top = Frame::Lit(b"true", 1),
                b'f' => *top = Frame::Lit(b"false", 1),
                b'n' => *top = Frame::Lit(b"null", 1),
                _ => return false,
            },
            Frame::ObjKeyOrEnd => match byte {
                b if is_ws(b) => return true,
                b'"' => *top = Frame::Str { key: true },
                b'}' => {
                    stack.pop();
                    value_end(stack);
                }
                _ => return false,
            },
            Frame::ObjKey => match byte {
                b if is_ws(b) => return true,
                b'"' => *top = Frame::Str { key: true },
                _ => return false,
            },
            Frame::ObjColon => match byte {
                b if is_ws(b) => return true,
                b':' => stack.push(Frame::Value),
                _ => return false,
            },
            Frame::ObjCommaOrEnd => match byte {
                b if is_ws(b) => return true,
                b',' => *top = Frame::ObjKey,
                b'}' => {
                    stack.pop();
                    value_end(stack);
                }
                _ => return false,
            },
            Frame::ArrElemOrEnd => match byte {
                b if is_ws(b) => return true,
                b']' => {
                    stack.pop();
                    value_end(stack);
                }
                _ => {
                    // Element starts on this byte: resume here on comma or
                    // ']', and re-feed the byte to the value machine.
                    *top = Frame::ArrCommaOrEnd;
                    stack.push(Frame::Value);
                    continue;
                }
            },
            Frame::ArrElem => {
                if is_ws(byte) {
                    return true;
                }
                *top = Frame::ArrCommaOrEnd;
                stack.push(Frame::Value);
                continue;
            }
            Frame::ArrCommaOrEnd => match byte {
                b if is_ws(b) => return true,
                b',' => *top = Frame::ArrElem,
                b']' => {
                    stack.pop();
                    value_end(stack);
                }
                _ => return false,
            },
            Frame::Str { key } => match byte {
                b'"' => {
                    if key {
                        *top = Frame::ObjColon;
                    } else {
                        stack.pop();
                        value_end(stack);
                    }
                }
                b'\\' => *top = Frame::Esc { key },
                0x00..=0x1f => return false,
                _ => return true,
            },
            Frame::Esc { key } => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    *top = Frame::Str { key };
                }
                b'u' => *top = Frame::Hex { key, left: 4 },
                _ => return false,
            },
            Frame::Hex { key, left } => match byte {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => {
                    if left == 1 {
                        *top = Frame::Str { key };
                    } else {
                        *top = Frame::Hex {
                            key,
                            left: left - 1,
                        };
                    }
                }
                _ => return false,
            },
            Frame::Num(phase) => {
                if phase.terminable() && is_delimiter(byte) {
                    stack.pop();
                    value_end(stack);
                    continue;
                }
                match (phase, byte) {
                    (NumPhase::IntFirst | NumPhase::Int, b'0'..=b'9') => {
                        *top = Frame::Num(if byte == b'0' && phase == NumPhase::IntFirst {
                            NumPhase::IntZero
                        } else {
                            NumPhase::Int
                        });
                    }
                    (NumPhase::Int | NumPhase::IntZero, b'.') => {
                        *top = Frame::Num(NumPhase::FracFirst);
                    }
                    (NumPhase::Int | NumPhase::IntZero | NumPhase::Frac, b'e' | b'E') => {
                        *top = Frame::Num(NumPhase::ExpFirst);
                    }
                    (NumPhase::FracFirst | NumPhase::Frac, b'0'..=b'9') => {
                        *top = Frame::Num(NumPhase::Frac);
                    }
                    (NumPhase::ExpFirst, b'+' | b'-') => {
                        *top = Frame::Num(NumPhase::ExpSign);
                    }
                    (NumPhase::ExpFirst | NumPhase::ExpSign | NumPhase::Exp, b'0'..=b'9') => {
                        *top = Frame::Num(NumPhase::Exp);
                    }
                    _ => return false,
                }
                return true;
            }
            Frame::Lit(word, pos) => {
                if byte != word[pos] {
                    return false;
                }
                if pos + 1 == word.len() {
                    stack.pop();
                    value_end(stack);
                } else {
                    *top = Frame::Lit(word, pos + 1);
                }
            }
        }
        return true;
    }
}

/// A byte-level document acceptor driven by [`Enforcer`].
///
/// `feed` mutates observable state; `accepts` must leave it unchanged
/// (implementations may use internal scratch space).
pub trait Machine {
    /// Feed one byte; returns false when the byte is invalid here.
    fn feed(&mut self, byte: u8) -> bool;
    /// Would feeding `bytes` succeed from the current state?
    fn accepts(&mut self, bytes: &[u8]) -> bool;
    /// The document is complete (or completable) in this state.
    fn complete(&self) -> bool;
    /// Append a byte-identity of the state for mask caching.
    fn key(&self, out: &mut Vec<u8>);
}

/// Grammar-constrained decode: maintains a byte-level acceptor over
/// emitted tokens and yields allowed-id bitsets.
pub struct Enforcer<M> {
    /// Raw byte string per token id; entries may be empty for special or
    /// unmapped ids, which are never allowed.
    vocab: Vec<Vec<u8>>,
    machine: M,
    cache: HashMap<Vec<u8>, Rc<[u64]>>,
    /// Stop token allowed once the document is complete.
    eos: TokenId,
}

impl<M: Machine> Enforcer<M> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `eos` becomes allowed only once the machine
    /// reports a complete document.
    pub fn with_machine(vocab: Vec<Vec<u8>>, eos: TokenId, machine: M) -> Self {
        Self {
            vocab,
            machine,
            cache: HashMap::new(),
            eos,
        }
    }

    /// The machine's document is complete or completable.
    pub fn complete(&self) -> bool {
        self.machine.complete()
    }
}

impl<M: Machine> DecodeConstraint for Enforcer<M> {
    fn allowed(&mut self) -> Rc<[u64]> {
        let mut key = Vec::new();
        self.machine.key(&mut key);
        if let Some(mask) = self.cache.get(&key) {
            return Rc::clone(mask);
        }
        // Gate the vocab scan on each token's first byte: only a handful of
        // bytes can open a valid continuation from this state, so the full
        // simulation runs just for those tokens.
        let mut first = [false; 256];
        for byte in 0..=255_u8 {
            if self.machine.accepts(&[byte]) {
                first[usize::from(byte)] = true;
            }
        }
        let mut mask = vec![0_u64; self.vocab.len().div_ceil(64)];
        for (id, word) in mask.iter_mut().enumerate() {
            for bit in 0..64 {
                let token = id * 64 + bit;
                if token < self.vocab.len()
                    && !self.vocab[token].is_empty()
                    && first[usize::from(self.vocab[token][0])]
                    && self.machine.accepts(&self.vocab[token])
                {
                    *word |= 1_u64 << bit;
                }
            }
        }
        if self.machine.complete() {
            let eos = usize::try_from(self.eos).unwrap_or(usize::MAX);
            if let Some(word) = mask.get_mut(eos / 64) {
                *word |= 1_u64 << (eos % 64);
            }
        }
        let mask: Rc<[u64]> = mask.into();
        self.cache.insert(key, Rc::clone(&mask));
        mask
    }

    fn advance(&mut self, token: TokenId) {
        let Some(bytes) = usize::try_from(token)
            .ok()
            .and_then(|index| self.vocab.get(index))
        else {
            return;
        };
        for index in 0..bytes.len() {
            let byte = bytes[index];
            // The executor's masked argmax only emits allowed ids; a feed
            // failure here would mean the mask and machine disagree.
            let _ = self.machine.feed(byte);
        }
    }
}

/// Byte-level JSON parser used by [`JsonEnforcer`].
pub struct JsonMachine {
    stack: Vec<Frame>,
    /// Reusable fork buffer for `accepts`; never observable state.
    scratch: Vec<Frame>,
}

impl JsonMachine {
    /// Any JSON value as the root document.
    pub fn any_value() -> Self {
        Self {
            stack: vec![Frame::Value],
            scratch: Vec::new(),
        }
    }

    /// The root document must be an object whose first byte is `{`; a
    /// response document has no leading whitespace.
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

/// Append a byte-identity of a JSON parser stack for mask caching.
fn frames_key(stack: &[Frame], out: &mut Vec<u8>) {
    for frame in stack {
        match *frame {
            Frame::Value => out.push(0),
            Frame::RootObject => out.push(16),
            Frame::ObjKeyOrEnd => out.push(1),
            Frame::ObjKey => out.push(2),
            Frame::ObjColon => out.push(3),
            Frame::ObjCommaOrEnd => out.push(4),
            Frame::ArrElemOrEnd => out.push(5),
            Frame::ArrElem => out.push(6),
            Frame::ArrCommaOrEnd => out.push(7),
            Frame::Str { key: is_key } => out.push(if is_key { 8 } else { 9 }),
            Frame::Esc { key: is_key } => out.push(if is_key { 10 } else { 11 }),
            Frame::Hex { key: is_key, left } => {
                out.push(if is_key { 12 } else { 13 });
                out.push(left);
            }
            Frame::Num(phase) => {
                out.push(14);
                out.push(phase as u8);
            }
            Frame::Lit(word, pos) => {
                out.push(15);
                out.push(match word {
                    b"true" => 0,
                    b"false" => 1,
                    _ => 2,
                });
                out.push(u8::try_from(pos).unwrap_or(u8::MAX));
            }
        }
    }
}

/// JSON grammar enforcer: any-value root via [`JsonEnforcer::new`], or an
/// object root via [`JsonEnforcer::object`].
pub type JsonEnforcer = Enforcer<JsonMachine>;

impl Enforcer<JsonMachine> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `eos` becomes allowed once a complete JSON
    /// value has been emitted.
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId) -> Self {
        Self::with_machine(vocab, eos, JsonMachine::any_value())
    }

    /// Same grammar but the root document must be an object whose first
    /// byte is `{`.
    pub fn object(vocab: Vec<Vec<u8>>, eos: TokenId) -> Self {
        Self::with_machine(vocab, eos, JsonMachine::object_root())
    }
}

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
fn validate_tool_names(names: &[Vec<u8>]) {
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
                out.push(if word == b"true" { 0 } else { 1 });
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
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId, names: Vec<Vec<u8>>) -> Self {
        Self::with_machine(vocab, eos, ToolMachine::new(names))
    }
}

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
/// `name` must close on a registered name. No whitespace anywhere.
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
        AssistPhase::ContentValue => {
            if !feed_on(stack, byte) {
                return false;
            }
            // A root-level bare number only terminates on whitespace: the
            // comma after `5` feeds into the enclosing object instead, so
            // `"content":5 ,` is reachable but `"content":5,` is not.
            if stack.is_empty() {
                *state = AssistPhase::CallsKey(0);
            }
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
        AssistPhase::CallOrClose => match byte {
            b'{' => *state = AssistPhase::ArgsKey(1),
            b']' => *state = AssistPhase::CloseBrace,
            _ => return false,
        },
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
        AssistPhase::ArgsValue => {
            if !feed_on(stack, byte) {
                return false;
            }
            if stack.is_empty() {
                *state = AssistPhase::IdKey(0);
            }
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

/// Serialized-assistant-body acceptor used by [`AssistantCallEnforcer`]:
/// `{"content":<value>,"tool_calls":[{"arguments":<object>,"id":"<string>","name":"<name>"}]}`
/// exactly, compact with no whitespace, where `<name>` is a registered tool
/// name. `content` and `arguments` reuse the JSON parser stack; `id` is a
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
    pub fn new(vocab: Vec<Vec<u8>>, eos: TokenId, names: Vec<Vec<u8>>) -> Self {
        Self::with_machine(vocab, eos, AssistantCallMachine::new(names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab_from(strings: &[&str]) -> Vec<Vec<u8>> {
        strings.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    /// Toy vocab: 0..4 specials (empty), then byte strings.
    fn toy_vocab() -> Vec<Vec<u8>> {
        let mut vocab = vec![Vec::new(); 4];
        vocab.extend(vocab_from(&[
            "{",
            "}",
            "[",
            "]",
            "\"",
            ":",
            ",",
            " ",
            "a",
            "b",
            "1",
            "2",
            "-",
            ".",
            "e",
            "t",
            "true",
            "false",
            "null",
            "nul",
            "\\",
            "\\n",
            "\\u0041",
            "\n",
            "x",
            "ab",
            "\":\"",
            "{\"a\":1}",
            "e5",
            ",\"",
            "zzz",
            "f",
            "rue",
            "alse",
            "{\"a\":true}",
            "a\":true}",
        ]));
        vocab
    }

    fn allows<M: Machine>(enforcer: &mut Enforcer<M>, id: usize) -> bool {
        let mask = enforcer.allowed();
        mask[id / 64] & (1_u64 << (id % 64)) != 0
    }

    fn id_of(vocab: &[Vec<u8>], s: &str) -> usize {
        vocab
            .iter()
            .position(|entry| entry == s.as_bytes())
            .unwrap_or_else(|| panic!("{s} not in toy vocab"))
    }

    #[test]
    fn start_allows_value_starts_only() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(allows(&mut enforcer, id_of(&vocab, "[")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(allows(&mut enforcer, id_of(&vocab, "t")));
        // Whole-document tokens are fine too.
        assert!(allows(&mut enforcer, id_of(&vocab, "true")));
        assert!(allows(&mut enforcer, id_of(&vocab, "{\"a\":1}")));
        // Structural mid-document tokens are not value starts.
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, ":")));
        assert!(!allows(&mut enforcer, id_of(&vocab, ",")));
        // Special (byte-less) ids are never allowed.
        assert!(!allows(&mut enforcer, 0));
        // Not complete yet: EOS excluded.
        assert!(!allows(&mut enforcer, 7));
    }

    #[test]
    fn object_flow_enforces_key_colon_value_comma() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        for id in [id_of(&vocab, "{")] {
            enforcer.advance(u32::try_from(id).unwrap());
        }
        // After '{': only a key string, '}', or whitespace.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "}")));
        assert!(allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));

        enforcer.advance(id_of(&vocab, "\"") as u32);
        // Inside a string: content tokens allowed, raw '"' closes it.
        assert!(allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(allows(&mut enforcer, id_of(&vocab, "ab")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\\")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\\n")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\\u0041")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\n")));

        enforcer.advance(id_of(&vocab, "a") as u32);
        enforcer.advance(id_of(&vocab, "\"") as u32);
        // Key closed: colon only.
        assert!(allows(&mut enforcer, id_of(&vocab, ":")));
        assert!(allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

        enforcer.advance(id_of(&vocab, ":") as u32);
        // Value position.
        assert!(allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(!allows(&mut enforcer, id_of(&vocab, ":")));

        enforcer.advance(id_of(&vocab, "1") as u32);
        // Number is terminable: ',' or '}' resume the object; digits continue it.
        assert!(allows(&mut enforcer, id_of(&vocab, ",")));
        assert!(allows(&mut enforcer, id_of(&vocab, "}")));
        assert!(allows(&mut enforcer, id_of(&vocab, "2")));
        assert!(allows(&mut enforcer, id_of(&vocab, ".")));
        // Multi-byte tokens spanning the boundary work: ',"' and '\":"'.
        assert!(allows(&mut enforcer, id_of(&vocab, ",\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));

        enforcer.advance(id_of(&vocab, "}") as u32);
        // Document complete: whitespace and EOS only.
        assert!(allows(&mut enforcer, 7));
        assert!(allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(allows(&mut enforcer, id_of(&vocab, "\n")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    }

    #[test]
    fn array_and_nesting_resume_correctly() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        enforcer.advance(id_of(&vocab, "[") as u32);
        // First element or ']'.
        assert!(allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(allows(&mut enforcer, id_of(&vocab, "]")));
        enforcer.advance(id_of(&vocab, "[") as u32);
        enforcer.advance(id_of(&vocab, "1") as u32);
        enforcer.advance(id_of(&vocab, "]") as u32);
        // Inner array closed inside outer element: comma-or-end resumes.
        assert!(allows(&mut enforcer, id_of(&vocab, ",")));
        assert!(allows(&mut enforcer, id_of(&vocab, "]")));
        enforcer.advance(id_of(&vocab, "]") as u32);
        assert!(allows(&mut enforcer, 7));
    }

    #[test]
    fn literals_split_across_tokens() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        enforcer.advance(id_of(&vocab, "nul") as u32);
        // 'nul' consumed; only 'l' (or a token starting with 'l') continues.
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
        let mut vocab2 = vocab.clone();
        vocab2.push(b"l".to_vec());
        let last = vocab2.len() - 1;
        let mut enforcer = JsonEnforcer::new(vocab2.clone(), 7);
        enforcer.advance(id_of(&vocab2, "nul") as u32);
        assert!(allows(&mut enforcer, last));
        enforcer.advance(last as u32);
        assert!(allows(&mut enforcer, 7));
    }

    #[test]
    fn numbers_reject_leading_zero_and_require_digits() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        enforcer.advance(id_of(&vocab, "-") as u32);
        // After '-': a digit is required.
        assert!(allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(!allows(&mut enforcer, id_of(&vocab, ".")));
        enforcer.advance(id_of(&vocab, "2") as u32);
        // 'e' continues toward exponent; digits continue.
        assert!(allows(&mut enforcer, id_of(&vocab, "e")));
        assert!(allows(&mut enforcer, id_of(&vocab, "e5")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "-")));
    }

    #[test]
    fn object_root_rejects_scalars_and_arrays() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::object(vocab.clone(), 7);
        // Tool-call shape: the first token must open the object.
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\n")));
        assert!(allows(&mut enforcer, id_of(&vocab, "{\"a\":1}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "[")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "true")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "e5")));
        assert!(!allows(&mut enforcer, 7));
        assert!(!enforcer.complete());
    }

    /// Registered names for the tool tests: `a` is both a full name and a
    /// prefix of `ab`; `x` is independent.
    fn tool_enforcer(vocab: &[Vec<u8>]) -> ToolCallEnforcer {
        let names = [b"a".to_vec(), b"ab".to_vec(), b"x".to_vec()];
        ToolCallEnforcer::new(vocab.to_vec(), 7, names.to_vec())
    }

    #[test]
    fn tool_call_shape_enforces_the_fixed_sequence() {
        let vocab = toy_vocab();
        let mut enforcer = tool_enforcer(&vocab);
        // Only '{'-starting tokens may open the document.
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(allows(&mut enforcer, id_of(&vocab, "{\"a\":true}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "[")));
        assert!(!allows(&mut enforcer, 7));
        assert!(!enforcer.complete());

        enforcer.advance(id_of(&vocab, "{") as u32);
        // Need the key's opening quote.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a\":true}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));

        enforcer.advance(id_of(&vocab, "\"") as u32);
        // In the key: only bytes extending toward a registered name, and
        // '"' only when the emitted bytes are exactly a name.
        assert!(allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(allows(&mut enforcer, id_of(&vocab, "ab")));
        assert!(allows(&mut enforcer, id_of(&vocab, "x")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "b")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\\u0041")));

        enforcer.advance(id_of(&vocab, "a") as u32);
        // Key 'a' is a complete name AND a prefix of 'ab'.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "b")));
        // 'a":true}' is rejected: the 'a' byte would make the key 'aa',
        // which no registered name prefixes.
        assert!(!allows(&mut enforcer, id_of(&vocab, "a\":true}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "x")));

        enforcer.advance(id_of(&vocab, "b") as u32);
        // Key 'ab' is complete and extends nothing: '"' is forced.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "x")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));

        enforcer.advance(id_of(&vocab, "\"") as u32);
        // Need ':' exactly; '\":"' dies because '"' cannot start a bool.
        assert!(allows(&mut enforcer, id_of(&vocab, ":")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\":\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));

        enforcer.advance(id_of(&vocab, ":") as u32);
        // Need a boolean literal.
        assert!(allows(&mut enforcer, id_of(&vocab, "t")));
        assert!(allows(&mut enforcer, id_of(&vocab, "true")));
        assert!(allows(&mut enforcer, id_of(&vocab, "f")));
        assert!(allows(&mut enforcer, id_of(&vocab, "false")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "null")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

        enforcer.advance(id_of(&vocab, "f") as u32);
        // Mid 'false': 'alse' completes it; 'a' alone also continues it.
        assert!(allows(&mut enforcer, id_of(&vocab, "alse")));
        assert!(allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "rue")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

        enforcer.advance(id_of(&vocab, "alse") as u32);
        // Need '}' exactly.
        assert!(allows(&mut enforcer, id_of(&vocab, "}")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        assert!(!allows(&mut enforcer, 7));

        enforcer.advance(id_of(&vocab, "}") as u32);
        // Document complete: only EOS remains.
        assert!(enforcer.complete());
        assert!(allows(&mut enforcer, 7));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
    }

    #[test]
    fn tool_call_key_cannot_drift_across_names() {
        let vocab = toy_vocab();
        // 'x' then 'b' prefixes only 'xbe'; a 't' byte matches 'abt' at
        // position 2 but would leave the key a prefix of nothing.
        let names = [b"abt".to_vec(), b"xbe".to_vec()];
        let mut enforcer = ToolCallEnforcer::new(vocab.clone(), 7, names.to_vec());
        enforcer.advance(id_of(&vocab, "{") as u32);
        enforcer.advance(id_of(&vocab, "\"") as u32);
        enforcer.advance(id_of(&vocab, "x") as u32);
        enforcer.advance(id_of(&vocab, "b") as u32);
        assert!(!allows(&mut enforcer, id_of(&vocab, "t")));
        assert!(allows(&mut enforcer, id_of(&vocab, "e")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
    }

    #[test]
    fn tool_call_literal_split_across_tokens() {
        let vocab = toy_vocab();
        let mut enforcer = tool_enforcer(&vocab);
        for id in ["{", "\"", "a", "\"", ":"] {
            enforcer.advance(id_of(&vocab, id) as u32);
        }
        enforcer.advance(id_of(&vocab, "t") as u32);
        // Mid 'true': only tokens continuing the literal pass.
        assert!(allows(&mut enforcer, id_of(&vocab, "rue")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "alse")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "e5")));
        enforcer.advance(id_of(&vocab, "rue") as u32);
        assert!(allows(&mut enforcer, id_of(&vocab, "}")));
        enforcer.advance(id_of(&vocab, "}") as u32);
        assert!(enforcer.complete());
        assert!(allows(&mut enforcer, 7));
    }

    #[test]
    fn mask_is_cached_per_state() {
        let vocab = toy_vocab();
        let mut enforcer = JsonEnforcer::new(vocab.clone(), 7);
        let first = enforcer.allowed();
        let second = enforcer.allowed();
        assert!(Rc::ptr_eq(&first, &second));
    }

    // Temporary perf probe: synthesize a 65k-ish vocab of printable tokens
    // and time allowed() per visited state while feeding a full document.
    #[test]
    #[ignore]
    fn bench_assistant_mask() {
        use std::time::Instant;
        let mut vocab = vec![Vec::new(); 4];
        for b in 0x20u8..=0x7e {
            vocab.push(vec![b]);
        }
        let mut x: u64 = 0x9e3779b97f4a7c15;
        let mut rng = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..65_000 {
            let len = 2 + (rng() % 6) as usize;
            let tok: Vec<u8> = (0..len).map(|_| 0x20 + (rng() % 0x5f) as u8).collect();
            vocab.push(tok);
        }
        let names = [
            b"create_task".to_vec(),
            b"archive_task".to_vec(),
            b"reorder_list".to_vec(),
            b"assign_owner".to_vec(),
        ];
        let doc = concat!(
            "{\"content\":\"Reorder List\",\"tool_calls\":[{\"arguments\":{\"l\":1},",
            "\"id\":\"call_0\",\"name\":\"reorder_list\"}]}"
        );
        let mut enforcer = AssistantCallEnforcer::new(vocab.clone(), 7, names.to_vec());
        // Single-char id map for feeding the document.
        let mut singles = std::collections::HashMap::new();
        for (i, t) in vocab.iter().enumerate() {
            if t.len() == 1 {
                singles.insert(t[0], i as u32);
            }
        }
        let start = Instant::now();
        let mut states = 0;
        let mut worst = (0.0, Vec::new());
        let mut report = |e: &mut AssistantCallEnforcer| {
            let t = Instant::now();
            let _ = e.allowed();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            states += 1;
            let mut k = Vec::new();
            e.machine.key(&mut k);
            if ms > worst.0 {
                worst = (ms, k);
            }
        };
        report(&mut enforcer);
        for b in doc.bytes() {
            let id = singles[&b];
            enforcer.advance(id);
            report(&mut enforcer);
        }
        println!(
            "states={} total={:.1}ms worst={:.2}ms worst-key={:?}",
            states,
            start.elapsed().as_secs_f64() * 1000.0,
            worst.0,
            worst.1
        );
    }

    /// Toy vocab plus the letters the assistant-body literals need and a
    /// couple of multi-byte tokens that span literal boundaries.
    fn assist_vocab() -> Vec<Vec<u8>> {
        let mut vocab = toy_vocab();
        vocab.extend(vocab_from(&[
            "c",
            "o",
            "n",
            "l",
            "_",
            "s",
            "i",
            "d",
            "r",
            "g",
            "u",
            "m",
            "j",
            "k",
            "0",
            "{\"content\":",
            ",\"tool_calls\":[",
        ]));
        vocab
    }

    /// Registered names for the assistant tests: `a` is both a full name
    /// and a prefix of `ab`; `x` is independent.
    fn assist_enforcer(vocab: &[Vec<u8>]) -> AssistantCallEnforcer {
        let names = [b"a".to_vec(), b"ab".to_vec(), b"x".to_vec()];
        AssistantCallEnforcer::new(vocab.to_vec(), 7, names.to_vec())
    }

    /// Advance one single-character token per byte of `text`.
    fn feed_str<M: Machine>(enforcer: &mut Enforcer<M>, vocab: &[Vec<u8>], text: &str) {
        for ch in text.chars() {
            enforcer.advance(id_of(vocab, &ch.to_string()) as u32);
        }
    }

    #[test]
    fn assistant_body_enforces_the_serialized_shape() {
        let vocab = assist_vocab();
        let mut enforcer = assist_enforcer(&vocab);
        // Only `{`-starting tokens may open the document; a token carrying
        // the whole first literal is fine too.
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(allows(&mut enforcer, id_of(&vocab, "{\"content\":")));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "[")));
        assert!(!allows(&mut enforcer, 7));
        assert!(!enforcer.complete());
        enforcer.advance(id_of(&vocab, "{") as u32);
        // The `{"content":` literal is strict: only `"` continues it, so
        // `"tool_calls"` can never open first.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "t")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
        feed_str(&mut enforcer, &vocab, "\"content\":\"sure\",\"tool_calls\":[");
        // Inside the calls array: `{` opens a call, `]` ends it.
        assert!(allows(&mut enforcer, id_of(&vocab, "{")));
        assert!(allows(&mut enforcer, id_of(&vocab, "]")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!enforcer.complete());
        feed_str(
            &mut enforcer,
            &vocab,
            "{\"arguments\":{\"t\":1},\"id\":\"call_0_0\",\"name\":\"ab\"}]}",
        );
        assert!(enforcer.complete());
        assert!(allows(&mut enforcer, 7));
        assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
    }

    #[test]
    fn assistant_body_accepts_empty_and_multi_call_arrays() {
        let vocab = assist_vocab();
        let mut enforcer = assist_enforcer(&vocab);
        feed_str(&mut enforcer, &vocab, "{\"content\":\"ok\",\"tool_calls\":[]}");
        assert!(enforcer.complete());
        assert!(allows(&mut enforcer, 7));

        let mut enforcer = assist_enforcer(&vocab);
        feed_str(
            &mut enforcer,
            &vocab,
            "{\"content\":\"\",\"tool_calls\":[{\"arguments\":{},\"id\":\"i\",\"name\":\"x\"},{\"arguments\":{\"b\":false},\"id\":\"j\",\"name\":\"a\"}]}",
        );
        assert!(enforcer.complete());
    }

    #[test]
    fn assistant_body_name_must_stay_a_registered_prefix() {
        let vocab = assist_vocab();
        let mut enforcer = assist_enforcer(&vocab);
        feed_str(
            &mut enforcer,
            &vocab,
            "{\"content\":\"\",\"tool_calls\":[{\"arguments\":{},\"id\":\"i\",\"name\":\"a",
        );
        // `a` is a complete name and a prefix of `ab`.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(allows(&mut enforcer, id_of(&vocab, "b")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "x")));
        enforcer.advance(id_of(&vocab, "b") as u32);
        // `ab` extends nothing: `"` is forced.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    }

    #[test]
    fn assistant_body_rejects_wrong_key_order() {
        let vocab = assist_vocab();
        let mut enforcer = assist_enforcer(&vocab);
        feed_str(&mut enforcer, &vocab, "{\"content\":\"x\",");
        // Canonical order requires "tool_calls" next; "id" can't appear here.
        assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
        assert!(!allows(&mut enforcer, id_of(&vocab, "i")));
    }
}
