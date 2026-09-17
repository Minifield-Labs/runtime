use minifield_decoding_protocol::EcmaPattern;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Oracle {
    node: String,
    v8: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    pattern: String,
    flags: String,
    value: String,
    valid: bool,
}

#[test]
fn generic_matcher_agrees_with_every_v8_oracle_case() -> Result<(), Box<dyn Error>> {
    let oracle: Oracle = serde_json::from_str(include_str!(
        "../fixtures/ecmascript-patterns-001/expected.json"
    ))?;
    assert_eq!(oracle.node, "v21.6.1");
    assert_eq!(oracle.v8, "11.8.172.17-node.19");
    assert_eq!(oracle.cases.len(), 694);
    for case in oracle.cases {
        assert!(case.flags.is_empty(), "unexpected flags for {}", case.id);
        let pattern = EcmaPattern::compile(&case.pattern)?;
        assert_eq!(
            pattern.is_match(&case.value)?,
            case.valid,
            "V8 disagreement for {} with pattern {:?} and value {:?}",
            case.id,
            case.pattern,
            case.value
        );
    }
    Ok(())
}

#[test]
fn unsupported_ecmascript_extensions_fail_at_admission() {
    for pattern in [r"\d+", "(?=x)x", "[^x]", "a{1,}", r"\u0061"] {
        assert!(
            EcmaPattern::compile(pattern).is_err(),
            "pattern {pattern:?} must not be silently approximated"
        );
    }
}

#[test]
fn matcher_uses_ecmascript_utf16_units_for_astral_values() -> Result<(), Box<dyn Error>> {
    assert!(!EcmaPattern::compile("^.$")?.is_match("💻")?);
    assert!(EcmaPattern::compile("^..$")?.is_match("💻")?);
    assert!(EcmaPattern::compile("^.{2}$")?.is_match("💻")?);
    assert!(!EcmaPattern::compile("^.$")?.is_match("\n")?);
    Ok(())
}

#[test]
fn generic_matcher_agrees_with_supplemental_node_semantics_oracle() -> Result<(), Box<dyn Error>> {
    let oracle: Oracle = serde_json::from_str(include_str!(
        "../fixtures/ecmascript-semantics-002/expected.json"
    ))?;
    assert_eq!(oracle.cases.len(), 1_107);
    for case in oracle.cases {
        assert!(case.flags.is_empty(), "unexpected flags for {}", case.id);
        let pattern = EcmaPattern::compile(&case.pattern)?;
        assert_eq!(
            pattern.is_match(&case.value)?,
            case.valid,
            "supplemental V8 disagreement for {} with {:?} / {:?}",
            case.id,
            case.pattern,
            case.value
        );
    }
    Ok(())
}
