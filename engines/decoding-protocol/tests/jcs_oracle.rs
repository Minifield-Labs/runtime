use minifield_decoding_protocol::{
    NumericKind, RawJson, RawJsonLimits, admit_number, parse_json_document, safe_json,
};
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
    source_json: String,
    #[allow(dead_code)]
    category: String,
    binary64_hex: Option<String>,
    jcs: String,
    safe_json: String,
}

#[test]
fn strict_raw_admission_and_safe_json_match_independent_jcs_oracle() -> Result<(), Box<dyn Error>> {
    let oracle: Oracle =
        serde_json::from_str(include_str!("../fixtures/jcs-binary64-001/expected.json"))?;
    assert_eq!(oracle.node, "v21.6.1");
    assert_eq!(oracle.v8, "11.8.172.17-node.19");
    assert_eq!(oracle.cases.len(), 4_202);
    for case in oracle.cases {
        let raw = parse_json_document(case.source_json.as_bytes(), RawJsonLimits::default())?;
        if let (RawJson::Number(number), Some(hex)) = (&raw, &case.binary64_hex) {
            let bits = admit_number(number, NumericKind::Number)?.value.to_bits();
            let expected = u64::from_str_radix(hex, 16)?;
            assert_eq!(bits, expected, "binary64 mismatch for {}", case.id);
        }
        let value = raw.into_value()?;
        assert_eq!(
            serde_jcs::to_string(&value)?,
            case.jcs,
            "JCS mismatch for {}",
            case.id
        );
        assert_eq!(
            String::from_utf8(safe_json(&value)?)?,
            case.safe_json,
            "SafeJSON mismatch for {}",
            case.id
        );
    }
    Ok(())
}
