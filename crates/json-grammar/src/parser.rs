/// Parser frame on the value stack. `Str`/`Esc`/`Hex` carry `key`: a key
/// string closes into `ObjColon`, a value string completes a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Frame {
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
pub(crate) enum NumPhase {
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
    pub(crate) fn terminable(self) -> bool {
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
fn value_end(stack: &mut [Frame]) {
    match stack.last_mut() {
        Some(frame @ Frame::ObjColon) => {
            // The object frame below the member value resumes at comma-or-end.
            *frame = Frame::ObjCommaOrEnd;
        }
        Some(Frame::ArrCommaOrEnd) | None => {}
        Some(other) => {
            debug_assert!(false, "value_end above unexpected frame {other:?}");
        }
    }
}

/// Feed one byte through the parser stack; returns false when the byte is
/// invalid here.
pub(crate) fn feed_on(stack: &mut Vec<Frame>, byte: u8) -> bool {
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
            Frame::Num(phase) if phase.terminable() && is_delimiter(byte) => {
                stack.pop();
                value_end(stack);
                continue;
            }
            Frame::Str { .. }
            | Frame::Esc { .. }
            | Frame::Hex { .. }
            | Frame::Num(_)
            | Frame::Lit(_, _) => return feed_lexeme(stack, byte),
        }
        return true;
    }
}

/// Advance the active string, escape, number, or keyword. Number delimiters
/// are handled by `feed_on` so the enclosing frame can consume the same byte.
fn feed_lexeme(stack: &mut Vec<Frame>, byte: u8) -> bool {
    let Some(top) = stack.last_mut() else {
        return false;
    };
    match *top {
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
        _ => unreachable!("feed_lexeme requires a lexical frame"),
    }
    true
}

/// Append a byte-identity of a JSON parser stack for mask caching.
pub(crate) fn frames_key(stack: &[Frame], out: &mut Vec<u8>) {
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
