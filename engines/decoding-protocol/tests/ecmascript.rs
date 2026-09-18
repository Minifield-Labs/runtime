mod support;

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
fn generic_matcher_agrees_with_compact_v8_cases() -> Result<(), Box<dyn Error>> {
    let oracle: Oracle =
        serde_json::from_str(include_str!("../fixtures/ecmascript-patterns-compact.json"))?;
    assert_eq!(oracle.node, "v21.6.1");
    assert_eq!(oracle.v8, "11.8.172.17-node.19");
    assert_eq!(oracle.cases.len(), 97);
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
#[ignore = "requires MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT"]
fn generic_matcher_agrees_with_supplemental_node_semantics_oracle() -> Result<(), Box<dyn Error>> {
    let bundle = support::required_bundle(
        "ecmascript-semantics-002",
        "2eed75659b7277f2f83a819e74665dc428afb2a29cdc5a2ba52b1cde940feb40",
    )?;
    let oracle: Oracle = serde_json::from_slice(&bundle.read("expected.json")?)?;
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

#[test]
#[ignore = "requires MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT"]
fn generic_matcher_agrees_with_complete_v8_oracle() -> Result<(), Box<dyn Error>> {
    let bundle = support::required_bundle(
        "ecmascript-patterns-001",
        "b71b077228f832a656516855b8a7abd206ba8c611a5262fb0387b08c981fa55d",
    )?;
    let oracle: Oracle = serde_json::from_slice(&bundle.read("expected.json")?)?;
    assert_eq!(oracle.node, "v21.6.1");
    assert_eq!(oracle.v8, "11.8.172.17-node.19");
    assert_eq!(oracle.cases.len(), 694);
    for case in oracle.cases {
        assert!(case.flags.is_empty(), "unexpected flags for {}", case.id);
        assert_eq!(
            EcmaPattern::compile(&case.pattern)?.is_match(&case.value)?,
            case.valid,
            "full V8 oracle disagreement for {}",
            case.id
        );
    }
    Ok(())
}
