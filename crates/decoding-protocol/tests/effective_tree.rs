use minifield_decoding_protocol::{
    EffectiveTree, RawJson, RawJsonLimits, parse_json_document, semantic_equal,
};
use serde_json::Value;
use std::error::Error;

fn raw(value: &Value) -> Result<RawJson, Box<dyn Error>> {
    Ok(parse_json_document(
        &serde_json::to_vec(value)?,
        RawJsonLimits::default(),
    )?)
}

#[test]
fn successive_choices_preserve_literal_all_of_wrappers_and_origins() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/schema-choice-wrapping-001.json"
    ))?;
    let resolved = raw(&fixture["resolved_schema"])?;
    let expected = fixture["choices"].as_array().ok_or("choices missing")?;
    let initial = EffectiveTree::from_resolved(&resolved)?;
    assert_eq!(
        initial.source_path_for("/properties/x"),
        Some("/properties/x")
    );

    let after_first = initial.select_choice("/allOf/0/anyOf", 0)?;
    let first = &after_first.provenance.selections[0];
    assert_eq!(first.source_keyword_path, "/allOf/0/anyOf");
    assert_eq!(first.current_keyword_path, "/allOf/0/anyOf");
    assert!(semantic_equal(
        &first.alternative.schema,
        &raw(&expected[0]["selected_alternative"]["schema"])?
    )?);
    assert_eq!(
        after_first.source_path_for("/allOf/0/allOf/1"),
        Some("/allOf/1")
    );
    assert_eq!(
        after_first.source_path_for("/allOf/0/properties/x"),
        Some("/properties/x")
    );

    let after_second = after_first.select_choice("/allOf/0/allOf/1/oneOf", 0)?;
    let second = &after_second.provenance.selections[1];
    assert_eq!(second.source_keyword_path, "/allOf/1/oneOf");
    assert_eq!(second.current_keyword_path, "/allOf/0/allOf/1/oneOf");
    assert_eq!(second.alternative.schema_path, "/allOf/1/oneOf/0");
    assert!(semantic_equal(
        &second.alternative.schema,
        &raw(&expected[1]["selected_alternative"]["schema"])?
    )?);
    assert!(
        after_second
            .source_path_for("/allOf/0/allOf/0/properties/x")
            .is_some()
    );
    Ok(())
}

#[test]
fn effective_tree_limits_reject_before_retained_clones() -> Result<(), Box<dyn Error>> {
    use minifield_decoding_protocol::EffectiveLimits;
    let schema = parse_json_document(
        br#"{"anyOf":[{"type":"string"},{"type":"number"}]}"#,
        RawJsonLimits::default(),
    )?;
    assert!(
        EffectiveTree::from_resolved_with_limits(&schema, EffectiveLimits { max_bytes: 1 })
            .is_err()
    );
    let tree = EffectiveTree::from_resolved(&schema)?;
    let limits = EffectiveLimits {
        max_bytes: tree.charged_bytes,
    };
    let tree = EffectiveTree::from_resolved_with_limits(&schema, limits)?;
    assert!(tree.select_choice("/anyOf", 0).is_err());
    Ok(())
}
