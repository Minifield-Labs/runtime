use std::rc::Rc;

use minifield_engine_api::{DecodeConstraint, TokenId};

use crate::{AssistantCallEnforcer, Enforcer, JsonEnforcer, Machine, ToolCallEnforcer};

// EOS is a byte-less special, separate from the ordinary closing bracket.
const EOS: usize = 0;

fn token_id(id: usize) -> TokenId {
    TokenId::try_from(id).unwrap_or_else(|_| panic!("test token id {id} is out of range"))
}

#[test]
fn printable_eos_is_reserved_until_document_completion() {
    let mut vocab = toy_vocab();
    vocab[EOS] = b"x".to_vec();
    let ordinary_x = id_of(&vocab[1..], "x") + 1;
    let close_quote = id_of(&vocab, "\"");
    assert_ne!(EOS, id_of(&vocab, "]"));
    let mut enforcer = JsonEnforcer::new(vocab, token_id(EOS));
    assert!(!allows(&mut enforcer, EOS));
    enforcer.advance(token_id(close_quote));
    assert!(allows(&mut enforcer, ordinary_x));
    assert!(!allows(&mut enforcer, EOS));
    assert!(!enforcer.complete());
    enforcer.advance(token_id(ordinary_x));
    enforcer.advance(token_id(close_quote));
    assert!(enforcer.complete());
    assert!(allows(&mut enforcer, EOS));
    assert!(!allows(&mut enforcer, ordinary_x));
    let completed = enforcer.allowed();
    enforcer.advance(token_id(EOS));
    assert!(enforcer.complete());
    assert!(Rc::ptr_eq(&completed, &enforcer.allowed()));
}

#[test]
fn eos_outside_vocabulary_never_sets_padding_bits() {
    let vocab = vec![Vec::new(), b"true".to_vec(), b"x".to_vec()];
    for eos in [token_id(vocab.len()), TokenId::MAX] {
        let mut enforcer = JsonEnforcer::new(vocab.clone(), eos);
        assert!(allows(&mut enforcer, 1));
        enforcer.advance(1);
        assert!(enforcer.complete());
        assert_eq!(enforcer.allowed().as_ref(), &[0]);
    }
}

#[test]
fn assistant_numeric_content_accepts_every_token_boundary() {
    for number in [
        "5", "-5", "0", "-0", "5.25", "-0.5", "5e2", "-5E+2", "5e-2", "-0.5e-2",
    ] {
        let document = format!("{{\"content\":{number},\"tool_calls\":[]}}");
        // Splits include the number's leading sign, fraction, exponent,
        // and its terminating comma, as well as the fixed envelope.
        for split in 1..document.len() {
            let vocab = vec![
                Vec::new(),
                document.as_bytes()[..split].to_vec(),
                document.as_bytes()[split..].to_vec(),
            ];
            let mut enforcer = assist_enforcer(&vocab);
            assert!(allows(&mut enforcer, 1), "{number}, split {split}: prefix");
            enforcer.advance(1);
            assert!(allows(&mut enforcer, 2), "{number}, split {split}: suffix");
            enforcer.advance(2);
            assert!(enforcer.complete(), "{number}, split {split}: completion");
            assert!(allows(&mut enforcer, EOS));
        }
    }
}

#[test]
fn assistant_numeric_content_rejects_invalid_numbers_at_token_boundaries() {
    for number in ["-", "01", "-01", "5.", "5e", "5e+", "-.5", "+5", "5e--2"] {
        let document = format!("{{\"content\":{number},\"tool_calls\":[]}}");
        for split in 1..document.len() {
            let vocab = vec![
                Vec::new(),
                document.as_bytes()[..split].to_vec(),
                document.as_bytes()[split..].to_vec(),
            ];
            let mut enforcer = assist_enforcer(&vocab);
            if allows(&mut enforcer, 1) {
                enforcer.advance(1);
                assert!(!allows(&mut enforcer, 2), "{number}, split {split}");
                assert!(!enforcer.complete());
                assert!(!allows(&mut enforcer, EOS));
            }
        }
    }
}

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
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
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
    assert!(!allows(&mut enforcer, EOS));
}

#[test]
fn object_flow_enforces_key_colon_value_comma() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
    enforcer.advance(token_id(id_of(&vocab, "{")));
    // After '{': only a key string, '}', or whitespace.
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(allows(&mut enforcer, id_of(&vocab, "}")));
    assert!(allows(&mut enforcer, id_of(&vocab, " ")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "1")));

    enforcer.advance(token_id(id_of(&vocab, "\"")));
    // Inside a string: content tokens allowed, raw '"' closes it.
    assert!(allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(allows(&mut enforcer, id_of(&vocab, "ab")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\\")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\\n")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\\u0041")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "\n")));

    enforcer.advance(token_id(id_of(&vocab, "a")));
    enforcer.advance(token_id(id_of(&vocab, "\"")));
    // Key closed: colon only.
    assert!(allows(&mut enforcer, id_of(&vocab, ":")));
    assert!(allows(&mut enforcer, id_of(&vocab, " ")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

    enforcer.advance(token_id(id_of(&vocab, ":")));
    // Value position.
    assert!(allows(&mut enforcer, id_of(&vocab, "1")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(allows(&mut enforcer, id_of(&vocab, "{")));
    assert!(!allows(&mut enforcer, id_of(&vocab, ":")));

    enforcer.advance(token_id(id_of(&vocab, "1")));
    // Number is terminable: ',' or '}' resume the object; digits continue it.
    assert!(allows(&mut enforcer, id_of(&vocab, ",")));
    assert!(allows(&mut enforcer, id_of(&vocab, "}")));
    assert!(allows(&mut enforcer, id_of(&vocab, "2")));
    assert!(allows(&mut enforcer, id_of(&vocab, ".")));
    // Multi-byte tokens spanning the boundary work: ',"' and '\":"'.
    assert!(allows(&mut enforcer, id_of(&vocab, ",\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));

    enforcer.advance(token_id(id_of(&vocab, "}")));
    // Document complete: whitespace and EOS only.
    assert!(allows(&mut enforcer, EOS));
    assert!(allows(&mut enforcer, id_of(&vocab, " ")));
    assert!(allows(&mut enforcer, id_of(&vocab, "\n")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "{")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
}

#[test]
fn array_and_nesting_resume_correctly() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
    enforcer.advance(token_id(id_of(&vocab, "[")));
    // First element or ']'.
    assert!(allows(&mut enforcer, id_of(&vocab, "1")));
    assert!(allows(&mut enforcer, id_of(&vocab, "]")));
    enforcer.advance(token_id(id_of(&vocab, "[")));
    enforcer.advance(token_id(id_of(&vocab, "1")));
    enforcer.advance(token_id(id_of(&vocab, "]")));
    // Inner array closed inside outer element: comma-or-end resumes.
    assert!(allows(&mut enforcer, id_of(&vocab, ",")));
    assert!(allows(&mut enforcer, id_of(&vocab, "]")));
    enforcer.advance(token_id(id_of(&vocab, "]")));
    assert!(allows(&mut enforcer, EOS));
}

#[test]
fn literals_split_across_tokens() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
    enforcer.advance(token_id(id_of(&vocab, "nul")));
    // 'nul' consumed; only 'l' (or a token starting with 'l') continues.
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
    let mut vocab2 = vocab.clone();
    vocab2.push(b"l".to_vec());
    let last = vocab2.len() - 1;
    let mut enforcer = JsonEnforcer::new(vocab2.clone(), token_id(EOS));
    enforcer.advance(token_id(id_of(&vocab2, "nul")));
    assert!(allows(&mut enforcer, last));
    enforcer.advance(token_id(last));
    assert!(allows(&mut enforcer, EOS));
}

#[test]
fn numbers_reject_leading_zero_and_require_digits() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
    enforcer.advance(token_id(id_of(&vocab, "-")));
    // After '-': a digit is required.
    assert!(allows(&mut enforcer, id_of(&vocab, "1")));
    assert!(!allows(&mut enforcer, id_of(&vocab, ".")));
    enforcer.advance(token_id(id_of(&vocab, "2")));
    // 'e' continues toward exponent; digits continue.
    assert!(allows(&mut enforcer, id_of(&vocab, "e")));
    assert!(allows(&mut enforcer, id_of(&vocab, "e5")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "-")));
}

#[test]
fn object_root_rejects_scalars_and_arrays() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::object(vocab.clone(), token_id(EOS));
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
    assert!(!allows(&mut enforcer, EOS));
    assert!(!enforcer.complete());
}

/// Registered names for the tool tests: `a` is both a full name and a
/// prefix of `ab`; `x` is independent.
fn tool_enforcer(vocab: &[Vec<u8>]) -> ToolCallEnforcer {
    let names = [b"a".to_vec(), b"ab".to_vec(), b"x".to_vec()];
    ToolCallEnforcer::new(vocab.to_vec(), token_id(EOS), names.to_vec())
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
    assert!(!allows(&mut enforcer, EOS));
    assert!(!enforcer.complete());

    enforcer.advance(token_id(id_of(&vocab, "{")));
    // Need the key's opening quote.
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a\":true}")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
    assert!(!allows(&mut enforcer, id_of(&vocab, " ")));

    enforcer.advance(token_id(id_of(&vocab, "\"")));
    // In the key: only bytes extending toward a registered name, and
    // '"' only when the emitted bytes are exactly a name.
    assert!(allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(allows(&mut enforcer, id_of(&vocab, "ab")));
    assert!(allows(&mut enforcer, id_of(&vocab, "x")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "b")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "\\u0041")));

    enforcer.advance(token_id(id_of(&vocab, "a")));
    // Key 'a' is a complete name AND a prefix of 'ab'.
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(allows(&mut enforcer, id_of(&vocab, "b")));
    // 'a":true}' is rejected: the 'a' byte would make the key 'aa',
    // which no registered name prefixes.
    assert!(!allows(&mut enforcer, id_of(&vocab, "a\":true}")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "x")));

    enforcer.advance(token_id(id_of(&vocab, "b")));
    // Key 'ab' is complete and extends nothing: '"' is forced.
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "x")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "1")));

    enforcer.advance(token_id(id_of(&vocab, "\"")));
    // Need ':' exactly; '\":"' dies because '"' cannot start a bool.
    assert!(allows(&mut enforcer, id_of(&vocab, ":")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "\":\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));

    enforcer.advance(token_id(id_of(&vocab, ":")));
    // Need a boolean literal.
    assert!(allows(&mut enforcer, id_of(&vocab, "t")));
    assert!(allows(&mut enforcer, id_of(&vocab, "true")));
    assert!(allows(&mut enforcer, id_of(&vocab, "f")));
    assert!(allows(&mut enforcer, id_of(&vocab, "false")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "null")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "1")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

    enforcer.advance(token_id(id_of(&vocab, "f")));
    // Mid 'false': 'alse' completes it; 'a' alone also continues it.
    assert!(allows(&mut enforcer, id_of(&vocab, "alse")));
    assert!(allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "rue")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));

    enforcer.advance(token_id(id_of(&vocab, "alse")));
    // Need '}' exactly.
    assert!(allows(&mut enforcer, id_of(&vocab, "}")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    assert!(!allows(&mut enforcer, EOS));

    enforcer.advance(token_id(id_of(&vocab, "}")));
    // Document complete: only EOS remains.
    assert!(enforcer.complete());
    assert!(allows(&mut enforcer, EOS));
    assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
}

#[test]
fn tool_call_key_cannot_drift_across_names() {
    let vocab = toy_vocab();
    // 'x' then 'b' prefixes only 'xbe'; a 't' byte matches 'abt' at
    // position 2 but would leave the key a prefix of nothing.
    let names = [b"abt".to_vec(), b"xbe".to_vec()];
    let mut enforcer = ToolCallEnforcer::new(vocab.clone(), token_id(EOS), names.to_vec());
    enforcer.advance(token_id(id_of(&vocab, "{")));
    enforcer.advance(token_id(id_of(&vocab, "\"")));
    enforcer.advance(token_id(id_of(&vocab, "x")));
    enforcer.advance(token_id(id_of(&vocab, "b")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "t")));
    assert!(allows(&mut enforcer, id_of(&vocab, "e")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "\"")));
}

#[test]
fn tool_call_literal_split_across_tokens() {
    let vocab = toy_vocab();
    let mut enforcer = tool_enforcer(&vocab);
    for id in ["{", "\"", "a", "\"", ":"] {
        enforcer.advance(token_id(id_of(&vocab, id)));
    }
    enforcer.advance(token_id(id_of(&vocab, "t")));
    // Mid 'true': only tokens continuing the literal pass.
    assert!(allows(&mut enforcer, id_of(&vocab, "rue")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "alse")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "e5")));
    enforcer.advance(token_id(id_of(&vocab, "rue")));
    assert!(allows(&mut enforcer, id_of(&vocab, "}")));
    enforcer.advance(token_id(id_of(&vocab, "}")));
    assert!(enforcer.complete());
    assert!(allows(&mut enforcer, EOS));
}

#[test]
fn mask_is_cached_per_state() {
    let vocab = toy_vocab();
    let mut enforcer = JsonEnforcer::new(vocab.clone(), token_id(EOS));
    let first = enforcer.allowed();
    let second = enforcer.allowed();
    assert!(Rc::ptr_eq(&first, &second));
}

// Temporary perf probe: synthesize a 65k-ish vocab of printable tokens
// and time allowed() per visited state while feeding a full document.
#[test]
#[ignore = "manual mask-scan timing probe"]
fn bench_assistant_mask() {
    use std::time::Instant;
    let mut vocab = vec![Vec::new(); 4];
    for b in 0x20u8..=0x7e {
        vocab.push(vec![b]);
    }
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
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
    let mut enforcer = AssistantCallEnforcer::new(vocab.clone(), token_id(EOS), names.to_vec());
    // Single-char id map for feeding the document.
    let mut singles = std::collections::HashMap::new();
    for (i, t) in vocab.iter().enumerate() {
        if t.len() == 1 {
            singles.insert(t[0], token_id(i));
        }
    }
    let start = Instant::now();
    let mut states = 0;
    let mut worst = 0.0_f64;
    let mut report = |e: &mut AssistantCallEnforcer| {
        let t = Instant::now();
        let _ = e.allowed();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        states += 1;
        worst = worst.max(ms);
    };
    report(&mut enforcer);
    for b in doc.bytes() {
        let id = singles[&b];
        enforcer.advance(id);
        report(&mut enforcer);
    }
    println!(
        "states={} total={:.1}ms worst={:.2}ms",
        states,
        start.elapsed().as_secs_f64() * 1000.0,
        worst
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
    AssistantCallEnforcer::new(vocab.to_vec(), token_id(EOS), names.to_vec())
}

/// Advance one single-character token per byte of `text`.
fn feed_str<M: Machine>(enforcer: &mut Enforcer<M>, vocab: &[Vec<u8>], text: &str) {
    for ch in text.chars() {
        enforcer.advance(token_id(id_of(vocab, &ch.to_string())));
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
    assert!(!allows(&mut enforcer, EOS));
    assert!(!enforcer.complete());
    enforcer.advance(token_id(id_of(&vocab, "{")));
    // The `{"content":` literal is strict: only `"` continues it, so
    // `"tool_calls"` can never open first.
    assert!(allows(&mut enforcer, id_of(&vocab, "\"")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "t")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "a")));
    feed_str(
        &mut enforcer,
        &vocab,
        "\"content\":\"sure\",\"tool_calls\":[",
    );
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
    assert!(allows(&mut enforcer, EOS));
    assert!(!allows(&mut enforcer, id_of(&vocab, " ")));
    assert!(!allows(&mut enforcer, id_of(&vocab, "}")));
}

#[test]
fn assistant_body_accepts_empty_and_multi_call_arrays() {
    let vocab = assist_vocab();
    let mut enforcer = assist_enforcer(&vocab);
    feed_str(
        &mut enforcer,
        &vocab,
        "{\"content\":\"ok\",\"tool_calls\":[]}",
    );
    assert!(enforcer.complete());
    assert!(allows(&mut enforcer, EOS));

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
    enforcer.advance(token_id(id_of(&vocab, "b")));
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
