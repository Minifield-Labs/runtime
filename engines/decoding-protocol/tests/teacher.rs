use minifield_decoding_protocol::{
    RawJson, RawJsonLimits, SchemaLimits, discover_teacher_unions, normalize_schema,
    parse_json_document,
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
fn nested_union_discovery_validates_full_effective_root_and_preserves_source_paths()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-traces-003/expected.json"
    ))?;
    let case = &fixture["cases"][0];
    let plan = normalize_schema(
        "synthetic.nested_choice_source_path",
        raw(&case["original_schema"])?,
        SchemaLimits::default(),
    )?;
    let arguments = raw(&case["teacher_arguments"])?;
    let choices = discover_teacher_unions(&plan, &arguments)?;

    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0].argument_path, "");
    assert_eq!(choices[0].source_keyword_path, "/anyOf");
    assert_eq!(choices[0].current_keyword_path, "/anyOf");
    assert_eq!(choices[0].selected_index, 0);
    assert_eq!(choices[0].matching_indices, vec![0]);
    assert_eq!(choices[0].alternatives[0].schema_path, "/anyOf/0");

    assert_eq!(choices[1].argument_path, "/x");
    assert_eq!(choices[1].source_keyword_path, "/properties/x/anyOf");
    assert_eq!(
        choices[1].current_keyword_path,
        "/allOf/0/properties/x/anyOf"
    );
    assert_eq!(choices[1].selected_index, 0);
    assert_eq!(choices[1].matching_indices, vec![0]);
    assert_eq!(
        choices[1].alternatives[0].schema_path,
        "/properties/x/anyOf/0"
    );
    assert_eq!(
        choices[1].alternatives[1].schema_path,
        "/properties/x/anyOf/1"
    );
    Ok(())
}
