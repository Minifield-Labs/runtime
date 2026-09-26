mod support;

use minifield_decoding_protocol::{
    NumericKind, RawJson, RawJsonLimits, admit_number, parse_json_document, safe_json,
};
use serde::Deserialize;
use std::error::Error;

const MANIFEST_SHA256: &str = "7f01c49073b6687d857c6ba95eefc804461e9f8e4a80d878abf01d43606f8d54";

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
#[ignore = "requires MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT"]
fn strict_raw_admission_and_safe_json_match_independent_jcs_oracle() -> Result<(), Box<dyn Error>> {
    let bundle = support::required_bundle("jcs-binary64-001", MANIFEST_SHA256)?;
    let oracle: Oracle = serde_json::from_slice(&bundle.read("expected.json")?)?;
    assert_eq!(oracle.node, "v21.6.1");
    assert_eq!(oracle.v8, "11.8.172.17-node.19");
    assert_eq!(oracle.cases.len(), 4_202);
    for case in oracle.cases {
        let raw = parse_json_document(case.source_json.as_bytes(), RawJsonLimits::default())?;
        if let (RawJson::Number(number), Some(hex)) = (&raw, &case.binary64_hex) {
            let bits = admit_number(number, NumericKind::Number)?.value.to_bits();
            assert_eq!(bits, u64::from_str_radix(hex, 16)?, "{}", case.id);
        }
        let value = raw.into_value()?;
        assert_eq!(serde_jcs::to_string(&value)?, case.jcs, "{}", case.id);
        assert_eq!(
            String::from_utf8(safe_json(&value)?)?,
            case.safe_json,
            "{}",
            case.id
        );
    }
    Ok(())
}
