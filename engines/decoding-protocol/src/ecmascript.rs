//! Small bounded ECMAScript-pattern subset for the declared corpus forms.
//!
//! JSON Schema patterns use ECMAScript search semantics. The draft only admits
//! the ASCII forms exercised by the pinned V8 oracle; unsupported grammar is an
//! installation failure rather than an accidental Rust-regex approximation.

use crate::{ProtocolError, Result};
use std::collections::BTreeSet;

const MAX_PATTERN_CHARS: usize = 4_096;
const MAX_AST_NODES: usize = 1_024;
const MAX_MATCH_WORK: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct EcmaPattern {
    root: Expr,
}

#[derive(Clone, Debug)]
enum Expr {
    Sequence(Vec<Expr>),
    Alternation(Vec<Expr>),
    Repeat {
        expression: Box<Expr>,
        minimum: usize,
        maximum: Option<usize>,
    },
    Character(char),
    Class(Vec<(char, char)>),
    Dot,
    Start,
    End,
}

impl EcmaPattern {
    /// Compile the finite, supported ECMAScript subset.
    pub fn compile(pattern: &str) -> Result<Self> {
        if !pattern.is_ascii() {
            return Err(ProtocolError::Schema(
                "non-ASCII ECMAScript pattern syntax is outside the admitted subset".to_owned(),
            ));
        }
        if pattern.len() > MAX_PATTERN_CHARS {
            return Err(ProtocolError::Schema(format!(
                "pattern exceeds {MAX_PATTERN_CHARS} ASCII bytes"
            )));
        }
        let mut parser = Parser {
            chars: pattern.chars().collect(),
            index: 0,
            nodes: 0,
            group_depth: 0,
        };
        let root = parser.expression()?;
        if parser.peek().is_some() {
            return Err(parser.error("unexpected trailing pattern syntax"));
        }
        Ok(Self { root })
    }

    /// Test a decoded string using ECMAScript search behavior with no flags.
    pub fn is_match(&self, value: &str) -> Result<bool> {
        let input: Vec<u16> = value.encode_utf16().collect();
        let mut work = 0usize;
        for start in 0..=input.len() {
            if !self
                .matches_expression(&self.root, &input, start, &mut work)?
                .is_empty()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn matches_expression(
        &self,
        expression: &Expr,
        input: &[u16],
        start: usize,
        work: &mut usize,
    ) -> Result<BTreeSet<usize>> {
        self.matches_node(expression, input, start, work)
    }

    fn matches_node(
        &self,
        expression: &Expr,
        input: &[u16],
        start: usize,
        work: &mut usize,
    ) -> Result<BTreeSet<usize>> {
        *work = work.checked_add(1).ok_or_else(|| {
            ProtocolError::InputLimit("ECMAScript match work counter overflow".to_owned())
        })?;
        if *work > MAX_MATCH_WORK {
            return Err(ProtocolError::InputLimit(format!(
                "ECMAScript pattern match exceeds {MAX_MATCH_WORK} work steps"
            )));
        }
        match expression {
            Expr::Sequence(expressions) => {
                let mut positions = BTreeSet::from([start]);
                for expression in expressions {
                    let mut next = BTreeSet::new();
                    for position in positions {
                        next.extend(self.matches_node(expression, input, position, work)?);
                    }
                    if next.is_empty() {
                        return Ok(next);
                    }
                    positions = next;
                }
                Ok(positions)
            }
            Expr::Alternation(expressions) => {
                let mut output = BTreeSet::new();
                for expression in expressions {
                    output.extend(self.matches_node(expression, input, start, work)?);
                }
                Ok(output)
            }
            Expr::Repeat {
                expression,
                minimum,
                maximum,
            } => {
                let maximum = maximum.unwrap_or(input.len().saturating_add(1));
                let mut positions = BTreeSet::from([start]);
                let mut output = BTreeSet::new();
                for count in 0..=maximum {
                    if count >= *minimum {
                        output.extend(&positions);
                    }
                    if count == maximum || positions.is_empty() {
                        break;
                    }
                    let mut next = BTreeSet::new();
                    for position in positions {
                        next.extend(self.matches_node(expression, input, position, work)?);
                    }
                    if !next.is_empty() && next.iter().all(|position| *position == start) {
                        return Err(ProtocolError::Schema(
                            "quantified zero-width pattern is unsupported".to_owned(),
                        ));
                    }
                    positions = next;
                }
                Ok(output)
            }
            Expr::Character(character) => Ok(match input.get(start) {
                Some(value)
                    if *value == u16::try_from(u32::from(*character)).expect("ASCII pattern") =>
                {
                    BTreeSet::from([start + 1])
                }
                _ => BTreeSet::new(),
            }),
            Expr::Class(ranges) => Ok(match input.get(start) {
                Some(value)
                    if ranges.iter().any(|(lower, upper)| {
                        u16::try_from(u32::from(*lower)).expect("ASCII pattern") <= *value
                            && *value <= u16::try_from(u32::from(*upper)).expect("ASCII pattern")
                    }) =>
                {
                    BTreeSet::from([start + 1])
                }
                _ => BTreeSet::new(),
            }),
            Expr::Dot => Ok(match input.get(start) {
                Some(value) if !is_ecmascript_line_terminator(*value) => {
                    BTreeSet::from([start + 1])
                }
                _ => BTreeSet::new(),
            }),
            Expr::Start => Ok(if start == 0 {
                BTreeSet::from([start])
            } else {
                BTreeSet::new()
            }),
            Expr::End => Ok(if start == input.len() {
                BTreeSet::from([start])
            } else {
                BTreeSet::new()
            }),
        }
    }
}

fn is_ecmascript_line_terminator(character: u16) -> bool {
    matches!(character, 0x000a | 0x000d | 0x2028 | 0x2029)
}

fn is_nullable(expression: &Expr) -> bool {
    match expression {
        Expr::Sequence(expressions) => expressions.iter().all(is_nullable),
        Expr::Alternation(expressions) => expressions.iter().any(is_nullable),
        Expr::Repeat {
            expression,
            minimum,
            ..
        } => *minimum == 0 || is_nullable(expression),
        Expr::Start | Expr::End => true,
        Expr::Character(_) | Expr::Class(_) | Expr::Dot => false,
    }
}

struct Parser {
    chars: Vec<char>,
    index: usize,
    nodes: usize,
    group_depth: usize,
}

impl Parser {
    fn expression(&mut self) -> Result<Expr> {
        let mut alternatives = vec![self.sequence()?];
        while self.consume('|') {
            alternatives.push(self.sequence()?);
        }
        if alternatives.len() == 1 {
            Ok(alternatives.pop().expect("one sequence"))
        } else {
            self.node(Expr::Alternation(alternatives))
        }
    }

    fn sequence(&mut self) -> Result<Expr> {
        let mut expressions = Vec::new();
        while !matches!(self.peek(), None | Some(')' | '|')) {
            expressions.push(self.atom_with_quantifier()?);
        }
        self.node(Expr::Sequence(expressions))
    }

    fn atom_with_quantifier(&mut self) -> Result<Expr> {
        let atom = self.atom()?;
        let quantifier = match self.peek() {
            Some('+') => {
                self.index += 1;
                Some((1, None))
            }
            Some('*') => {
                self.index += 1;
                Some((0, None))
            }
            Some('{') => Some(self.bounded_quantifier()?),
            _ => None,
        };
        match quantifier {
            Some((_minimum, _maximum))
                if matches!(atom, Expr::Start | Expr::End) || is_nullable(&atom) =>
            {
                Err(self.error("quantified nullable or zero-width patterns are unsupported"))
            }
            Some((minimum, maximum)) => self.node(Expr::Repeat {
                expression: Box::new(atom),
                minimum,
                maximum,
            }),
            None => Ok(atom),
        }
    }

    fn atom(&mut self) -> Result<Expr> {
        let Some(character) = self.peek() else {
            return Err(self.error("expected pattern atom"));
        };
        self.index += 1;
        match character {
            '^' => self.node(Expr::Start),
            '$' => self.node(Expr::End),
            '.' => self.node(Expr::Dot),
            '[' => self.character_class(),
            '(' => {
                self.group_depth = self
                    .group_depth
                    .checked_add(1)
                    .ok_or_else(|| self.error("pattern group-depth overflow"))?;
                if self.group_depth > 64 {
                    return Err(self.error("pattern group-depth limit exceeded"));
                }
                if self.consume('?') && !self.consume(':') {
                    return Err(self.error("only noncapturing groups are supported"));
                }
                let expression = self.expression()?;
                if !self.consume(')') {
                    return Err(self.error("unclosed group"));
                }
                self.group_depth -= 1;
                Ok(expression)
            }
            '\\' => Err(self.error("escape sequences are outside the admitted pattern subset")),
            '*' | '?' | '{' | '}' | ']' | ')' | '|' | '+' => {
                Err(self.error("unsupported ECMAScript metacharacter"))
            }
            literal => self.node(Expr::Character(literal)),
        }
    }

    fn character_class(&mut self) -> Result<Expr> {
        if self.consume('^') {
            return Err(self.error("negated character classes are unsupported"));
        }
        let mut ranges = Vec::new();
        let mut closed = false;
        while let Some(first) = self.peek() {
            self.index += 1;
            if first == ']' {
                closed = true;
                break;
            }
            if matches!(first, '\\' | '[') {
                return Err(self.error("escaped or nested character class syntax is unsupported"));
            }
            if self.peek() == Some('-') && self.chars.get(self.index + 1).copied() != Some(']') {
                self.index += 1;
                let Some(last) = self.peek() else {
                    return Err(self.error("unterminated character-class range"));
                };
                self.index += 1;
                if matches!(last, ']' | '\\') || first > last {
                    return Err(self.error("invalid character-class range"));
                }
                ranges.push((first, last));
            } else {
                ranges.push((first, first));
            }
        }
        if !closed || ranges.is_empty() {
            return Err(self.error("empty or unclosed character class"));
        }
        self.node(Expr::Class(ranges))
    }

    fn bounded_quantifier(&mut self) -> Result<(usize, Option<usize>)> {
        self.expect('{')?;
        let minimum = self.decimal()?;
        let maximum = if self.consume(',') {
            if self.consume('}') {
                return Err(self.error("unbounded brace quantifiers are unsupported"));
            }
            Some(self.decimal()?)
        } else {
            Some(minimum)
        };
        self.expect('}')?;
        if maximum.is_some_and(|value| value < minimum) {
            return Err(self.error("quantifier maximum is below minimum"));
        }
        Ok((minimum, maximum))
    }

    fn decimal(&mut self) -> Result<usize> {
        let start = self.index;
        while matches!(self.peek(), Some('0'..='9')) {
            self.index += 1;
        }
        if start == self.index {
            return Err(self.error("quantifier requires an ASCII decimal"));
        }
        let text: String = self.chars[start..self.index].iter().collect();
        text.parse::<usize>()
            .map_err(|_| self.error("quantifier is outside host limits"))
    }

    fn node(&mut self, expression: Expr) -> Result<Expr> {
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| self.error("pattern node counter overflow"))?;
        if self.nodes > MAX_AST_NODES {
            return Err(self.error("pattern exceeds supported AST node limit"));
        }
        Ok(expression)
    }

    fn expect(&mut self, character: char) -> Result<()> {
        if self.consume(character) {
            Ok(())
        } else {
            Err(self.error("unexpected pattern syntax"))
        }
    }

    fn consume(&mut self, character: char) -> bool {
        if self.peek() == Some(character) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }

    fn error(&self, message: &str) -> ProtocolError {
        ProtocolError::Schema(format!(
            "unsupported ECMAScript pattern at scalar {}: {message}",
            self.index
        ))
    }
}
