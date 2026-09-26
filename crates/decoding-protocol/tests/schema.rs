#![allow(clippy::expect_used)]
// This test asserts parser-admitted fixture syntax before testing schema behavior.
use minifield_decoding_protocol::{
    RawJson, RawJsonLimits, SchemaLimits, normalize_schema_document, parse_json_document,
    sha256_hex,
};
use std::error::Error;

#[test]
fn strict_document_normalization_preserves_source_hash_refs_and_property_order()
-> Result<(), Box<dyn Error>> {
    let source = br##"{
      "$defs":{"base":{"type":"object","properties":{"beta":{"type":"string"}},"required":["beta"]}},
      "properties":{"z":{"$ref":"#/$defs/base","additionalProperties":false},"emoji":{"type":"string"}},
      "additionalProperties":false
    }"##;
    let plan = normalize_schema_document("fixture_tool", source, SchemaLimits::default())?;
    let source_hash = sha256_hex(source);
    assert_eq!(plan.original_document.as_deref(), Some(source.as_slice()));
    assert_eq!(
        plan.original_document_sha256.as_deref(),
        Some(source_hash.as_str())
    );
    assert_eq!(
        plan.property_order
            .iter()
            .map(|entry| (entry.schema_path.as_str(), entry.properties.as_slice()))
            .collect::<Vec<_>>(),
        vec![
            ("", &["z".to_owned(), "emoji".to_owned()][..]),
            ("/properties/z/allOf/0", &["beta".to_owned()][..]),
        ]
    );
    let RawJson::Object(root) = &plan.resolved else {
        return Err("resolved root is not an object".into());
    };
    assert!(root.iter().all(|(key, _)| key != "$defs"));
    let properties = field(&plan.resolved, "properties").ok_or("properties missing")?;
    let z = field(properties, "z").ok_or("z missing")?;
    let RawJson::Object(z) = z else {
        return Err("z is not an object".into());
    };
    assert_eq!(z.len(), 1);
    assert_eq!(z[0].0, "allOf");
    assert_eq!(plan.interpreted_schema_sha256.len(), 64);
    assert_eq!(
        plan.interpreted_canonical_json()?,
        plan.original_safe_json()?
    );
    Ok(())
}

#[test]
fn normalizer_rejects_bad_keyword_shapes_patterns_and_local_references()
-> Result<(), Box<dyn Error>> {
    for source in [
        br#"{"type":"nonsense"}"#.as_slice(),
        br#"{"minLength":1.0000000000000001}"#,
        br#"{"multipleOf":0}"#,
        br#"{"pattern":"\\d+"}"#,
        br#"{"required":["a","a"]}"#,
        br##"{"$ref":"#/$defs/missing"}"##,
        br##"{"oneOf":[true],"$ref":"#/oneOf/+1"}"##,
        br##"{"$defs":{"self":{"$ref":"#/$defs/self"}},"$ref":"#/$defs/self"}"##,
    ] {
        let raw = parse_json_document(source, RawJsonLimits::default())?;
        assert!(
            minifield_decoding_protocol::normalize_schema(
                "fixture_tool",
                raw,
                SchemaLimits::default()
            )
            .is_err(),
            "must reject {source:?}"
        );
    }
    Ok(())
}

#[test]
fn source_document_accepts_rfc8259_whitespace_and_rejects_duplicate_keys() {
    assert!(
        normalize_schema_document(
            "fixture_tool",
            b" \n { \t \"type\" : \"string\" \r } ",
            SchemaLimits::default()
        )
        .is_ok()
    );
    assert!(
        normalize_schema_document(
            "fixture_tool",
            br#"{"type":"string","\u0074ype":"number"}"#,
            SchemaLimits::default()
        )
        .is_err()
    );
}

#[test]
fn pointer_paths_preserve_empty_tokens_and_budget_rejects_before_expansion()
-> Result<(), Box<dyn Error>> {
    let empty_name = br##"{"$defs":{"":{"type":"string"}},"properties":{"":{"$ref":"#/$defs/"}}}"##;
    let plan = normalize_schema_document("fixture_tool", empty_name, SchemaLimits::default())?;
    assert_eq!(plan.property_order[0].schema_path, "");
    assert_eq!(plan.property_order[0].properties, vec![String::new()]);
    let property = field(
        field(&plan.resolved, "properties").ok_or("properties missing")?,
        "",
    )
    .ok_or("empty property missing")?;
    let RawJson::Object(property) = property else {
        return Err("empty property was not resolved".into());
    };
    assert_eq!(property[0].0, "allOf");

    let repeated = br##"{
      "$defs":{"payload":{"default":"0123456789012345678901234567890123456789"}},
      "properties":{"a":{"$ref":"#/$defs/payload"},"b":{"$ref":"#/$defs/payload"}}
    }"##;
    let raw = parse_json_document(repeated, RawJsonLimits::default())?;
    let limits = SchemaLimits {
        max_resolved_bytes: 64,
        ..SchemaLimits::default()
    };
    assert!(minifield_decoding_protocol::normalize_schema("fixture_tool", raw, limits).is_err());
    Ok(())
}

fn field<'a>(value: &'a RawJson, wanted: &str) -> Option<&'a RawJson> {
    let RawJson::Object(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find(|(key, _)| key == wanted)
        .map(|(_, value)| value)
}

#[test]
fn nested_inert_schema_data_still_uses_strict_number_admission() {
    for source in [
        br#"{"const":{"inner":[1e309]}}"#.as_slice(),
        br#"{"enum":[{"inner":1e-999999999999999999999999999}]}"#,
        br#"{"default":{"inner":{"n":1e309}}}"#,
        br#"{"examples":[{"n":1e-999999999999999999999999999}]}"#,
    ] {
        let raw = parse_json_document(source, RawJsonLimits::default()).expect("syntax");
        assert!(
            minifield_decoding_protocol::normalize_schema(
                "strict_data",
                raw,
                SchemaLimits::default()
            )
            .is_err(),
            "{source:?}"
        );
    }
}

#[test]
fn unused_definitions_are_still_pattern_admitted() {
    for source in [
        br#"{"$defs":{"unused":{"pattern":"(?=a)a"}}}"#.as_slice(),
        br#"{"$defs":{"unused":{"pattern":"(a)\\1"}}}"#,
        br#"{"$defs":{"unused":{"pattern":"(a?)*"}}}"#,
    ] {
        assert!(
            normalize_schema_document("patterns", source, SchemaLimits::default()).is_err(),
            "{source:?}"
        );
    }
}
