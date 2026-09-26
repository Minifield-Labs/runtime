use minifield_decoding_protocol::{
    ProbeOperation, RawJson, RawJsonLimits, SchemaLimits, SegmentTokenizer, TeacherOperation,
    TeacherTraceBuilder, TeacherTraceInput, TokenId, TokenPolicy, discover_teacher_unions,
    normalize_schema, parse_json_document,
};
use serde_json::Value;
use std::error::Error;

fn raw(value: &Value) -> Result<RawJson, Box<dyn Error>> {
    Ok(parse_json_document(
        &serde_json::to_vec(value)?,
        RawJsonLimits::default(),
    )?)
}

struct ByteTokenizer;
impl SegmentTokenizer for ByteTokenizer {
    type Error = std::convert::Infallible;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(segment.bytes().map(|byte| u32::from(byte) + 1000).collect())
    }
}

#[test]
fn repeated_property_conjunctions_retain_both_union_operations() -> Result<(), Box<dyn Error>> {
    let mut source = serde_json::json!({
        "allOf": [
            {"properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}},
            {"properties": {"x": {"anyOf": [{"minLength": 1}, {"type": "number"}]}}}
        ]
    });
    let schema = raw(&source)?;
    let arguments = raw(&serde_json::json!({"x": "yes"}))?;
    let plan = normalize_schema("conjoined_property", schema, SchemaLimits::default())?;
    let choices = discover_teacher_unions(&plan, &arguments)?;
    assert_eq!(choices.len(), 2);
    for (choice, branch) in choices.iter().zip(0..2) {
        assert_eq!(choice.argument_path, "/x");
        let keyword_path = format!("/allOf/{branch}/properties/x/anyOf");
        assert_eq!(choice.source_keyword_path, keyword_path);
        assert_eq!(choice.current_keyword_path, keyword_path);
        assert_eq!(choice.selected_index, 0);
        assert_eq!(choice.matching_indices, vec![0]);
    }

    // Structural tracing requires the declared object to be closed. Discovery
    // above uses the exact open-object reproduction.
    source["allOf"][0]["additionalProperties"] = serde_json::json!(false);
    let trace_plan =
        normalize_schema("conjoined_property", raw(&source)?, SchemaLimits::default())?;
    let policy = TokenPolicy::draft5()?;
    let input = TeacherTraceInput {
        public_events: &[],
        selected_route: "conjoined_property",
        schema_plan: &trace_plan,
    };
    let builder = TeacherTraceBuilder::new(&ByteTokenizer, &policy);
    let trace = builder.build(&input, &arguments)?;
    assert_eq!(trace, builder.build(&input, &arguments)?);
    let union_probes = trace
        .probes
        .iter()
        .enumerate()
        .filter(|(_, probe)| probe.operation == ProbeOperation::Union)
        .collect::<Vec<_>>();
    assert_eq!(union_probes.len(), 2);
    assert!(union_probes[0].1.global_operation_index < union_probes[1].1.global_operation_index);
    for (probe_index, probe) in union_probes {
        assert_eq!(probe.path, "/x");
        assert_eq!(probe.selected_index, 0);
        assert_eq!(
            trace.operation_log[probe.global_operation_index],
            TeacherOperation::Probe { probe_index }
        );
    }
    assert_eq!(
        trace
            .main
            .appends
            .iter()
            .filter(|append| append.source.as_deref() == Some(b"\"yes\"".as_slice()))
            .count(),
        1
    );
    Ok(())
}

#[test]
fn nested_union_discovery_validates_full_effective_root_and_preserves_source_paths()
-> Result<(), Box<dyn Error>> {
    let schema = raw(&serde_json::json!({
        "type": "object",
        "properties": {
            "x": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            }
        },
        "required": ["x"],
        "additionalProperties": false,
        "anyOf": [
            {"required": ["x"]},
            {"properties": {"x": {"type": "null"}}}
        ]
    }))?;
    let arguments = raw(&serde_json::json!({"x": "v"}))?;
    let plan = normalize_schema(
        "synthetic.nested_choice_source_path",
        schema,
        SchemaLimits::default(),
    )?;
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
