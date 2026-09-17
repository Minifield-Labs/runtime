#![allow(clippy::too_many_lines)]
// External fixture tests compare every pinned oracle field in one causal traversal.
mod support;
use minifield_decoding_protocol::{
    PublicEvent, RawJson, RawJsonLimits, SchemaLimits, SegmentTokenizer, TeacherTraceBuilder,
    TeacherTraceInput, TokenId, TokenPolicy, TraceOwnership, normalize_schema, parse_json_document,
    plan_teacher,
};
use serde_json::Value;
use std::{collections::HashMap, error::Error};

fn raw(value: &Value) -> Result<RawJson, Box<dyn Error>> {
    Ok(parse_json_document(
        &serde_json::to_vec(value)?,
        RawJsonLimits::default(),
    )?)
}

#[test]
#[ignore = "requires MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT"]
fn array_items_keep_independent_union_selections() -> Result<(), Box<dyn Error>> {
    let bundle = support::required_bundle(
        "argument-array-unions-004",
        "63e5a8aa683543d721cf7e800a58871811a2b5b075590cb51e2e093d45429cf8",
    )?;
    let cases: Value = serde_json::from_slice(&bundle.read("cases.json")?)?;
    let assertions: Value = serde_json::from_slice(&bundle.read("manual-assertions.json")?)?;
    let expected: Value = serde_json::from_slice(&bundle.read("expected.json")?)?;
    let tokenizer = array_union_tokenizer(&bundle)?;
    let policy = TokenPolicy::draft5()?;
    for (case, assertion) in cases
        .as_array()
        .ok_or("cases array")?
        .iter()
        .zip(assertions.as_array().ok_or("assertions array")?)
    {
        assert_eq!(case["name"], assertion["name"]);
        let plan = normalize_schema(
            case["selected_route"].as_str().ok_or("route")?,
            raw(&case["schema"])?,
            SchemaLimits::default(),
        )?;
        let target = raw(&case["arguments"])?;
        let teacher = plan_teacher(&plan, &target)?;
        let actual = teacher
            .unions
            .iter()
            .map(|choice| (choice.argument_path.as_str(), choice.selected_index))
            .collect::<Vec<_>>();
        let expected_selection = assertion["expected_union_path_selection"]
            .as_array()
            .ok_or("selections")?
            .iter()
            .map(|row| {
                Ok((
                    row[0].as_str().ok_or("path")?,
                    usize::try_from(row[1].as_u64().ok_or("index")?)?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        assert_eq!(actual, expected_selection, "{}", case["name"]);
        let events = case["public_events"]
            .as_array()
            .ok_or("array fixture events")?
            .iter()
            .map(fixture_event)
            .collect::<Result<Vec<_>, _>>()?;
        let trace = TeacherTraceBuilder::new(&tokenizer, &policy).build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: case["selected_route"]
                    .as_str()
                    .ok_or("array fixture route")?,
                schema_plan: &plan,
            },
            &target,
        )?;
        let oracle = expected
            .as_array()
            .ok_or("array expected")?
            .iter()
            .find(|row| row["name"] == case["name"])
            .and_then(|row| row.get("oracle"))
            .ok_or("array oracle")?;
        let mut main = Vec::new();
        for operation in oracle["expected_operations"]
            .as_array()
            .ok_or("array operations")?
        {
            if operation["kind"] == "main_append" {
                for id in operation["token_ids"].as_array().ok_or("array main IDs")? {
                    main.push(u32::try_from(id.as_u64().ok_or("array main ID")?)?);
                }
            }
        }
        assert_eq!(trace.main.token_ids, main, "{} exact main", case["name"]);
    }
    Ok(())
}

struct ByteTokenizer;
impl SegmentTokenizer for ByteTokenizer {
    type Error = std::convert::Infallible;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(segment.bytes().map(|byte| u32::from(byte) + 1000).collect())
    }
}

#[test]
fn wholly_dynamic_array_is_one_learned_value_not_structural_array() -> Result<(), Box<dyn Error>> {
    let schema = raw(&serde_json::json!({
        "type": "object",
        "properties": {"value": {"allOf": [{}, {"title": "dynamic"}]}},
        "required": ["value"],
        "additionalProperties": false,
    }))?;
    let target = raw(&serde_json::json!({"value": [1, 2]}))?;
    let plan = normalize_schema("dynamic_array", schema, SchemaLimits::default())?;
    let policy = TokenPolicy::draft5()?;
    let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
        &TeacherTraceInput {
            public_events: &[PublicEvent::User {
                content: "x".to_owned(),
            }],
            selected_route: "dynamic_array",
            schema_plan: &plan,
        },
        &target,
    )?;
    assert!(trace.probes.is_empty());
    let append = trace
        .main
        .appends
        .iter()
        .find(|append| append.source.as_deref() == Some(b"[1,2]".as_slice()))
        .ok_or("dynamic array append missing")?;
    assert_eq!(append.ownership, TraceOwnership::Learned);
    Ok(())
}

struct FixtureTokenizer(HashMap<String, Vec<TokenId>>);
impl SegmentTokenizer for FixtureTokenizer {
    type Error = String;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        self.0
            .get(segment)
            .cloned()
            .ok_or_else(|| format!("fixture has no segment {segment:?}"))
    }
}

fn tokenizer_from_segments(lines: &str) -> Result<FixtureTokenizer, Box<dyn Error>> {
    let mut segments = HashMap::new();
    for line in lines.lines() {
        let value: Value = serde_json::from_str(line)?;
        let ids = value["token_ids"]
            .as_array()
            .ok_or("fixture token IDs")?
            .iter()
            .map(|value| Ok(u32::try_from(value.as_u64().ok_or("fixture token")?)?))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        segments.insert(
            value["source"].as_str().ok_or("fixture source")?.to_owned(),
            ids,
        );
    }
    Ok(FixtureTokenizer(segments))
}

fn fixture_tokenizer(bundle: &support::Bundle) -> Result<FixtureTokenizer, Box<dyn Error>> {
    let lines = String::from_utf8(bundle.read("segments.jsonl")?)?;
    tokenizer_from_segments(&lines)
}

fn array_union_tokenizer(bundle: &support::Bundle) -> Result<FixtureTokenizer, Box<dyn Error>> {
    let lines = String::from_utf8(bundle.read("segments.jsonl")?)?;
    tokenizer_from_segments(&lines)
}

fn fixture_event(value: &Value) -> Result<PublicEvent, Box<dyn Error>> {
    match value["type"].as_str().ok_or("fixture event type")? {
        "system" => Ok(PublicEvent::System {
            policy: value["policy"].as_str().ok_or("fixture policy")?.to_owned(),
            observation: raw(&value["observation"])?,
        }),
        "user" => Ok(PublicEvent::User {
            content: value["content"]
                .as_str()
                .ok_or("fixture content")?
                .to_owned(),
        }),
        other => Err(format!("unsupported fixture event {other}").into()),
    }
}

#[test]
#[ignore = "requires MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT"]
fn dynamic_container_fixture_keeps_dynamic_arrays_atomic() -> Result<(), Box<dyn Error>> {
    let bundle = support::required_bundle(
        "argument-dynamic-containers-005",
        "9d9d2183526bab15906326ebbe36dd1ff11785acd49ddbda830479b1fba8bb14",
    )?;
    let cases: Value = serde_json::from_slice(&bundle.read("cases.json")?)?;
    let assertions: Value = serde_json::from_slice(&bundle.read("manual-assertions.json")?)?;
    let expected: Value = serde_json::from_slice(&bundle.read("expected.json")?)?;
    let tokenizer = fixture_tokenizer(&bundle)?;
    let policy = TokenPolicy::draft5()?;
    for (case, assertion) in cases
        .as_array()
        .ok_or("dynamic cases")?
        .iter()
        .zip(assertions.as_array().ok_or("dynamic assertions")?)
    {
        assert_eq!(case["name"], assertion["name"]);
        let events = case["public_events"]
            .as_array()
            .ok_or("fixture events")?
            .iter()
            .map(fixture_event)
            .collect::<Result<Vec<_>, _>>()?;
        let plan = normalize_schema(
            case["selected_route"].as_str().ok_or("fixture route")?,
            raw(&case["schema"])?,
            SchemaLimits::default(),
        )?;
        let arguments = raw(&case["arguments"])?;
        let trace = TeacherTraceBuilder::new(&tokenizer, &policy).build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: case["selected_route"].as_str().ok_or("fixture route")?,
                schema_plan: &plan,
            },
            &arguments,
        )?;
        assert_eq!(
            trace.learned_token_count,
            usize::try_from(
                assertion["learned_argument_tokens"]
                    .as_u64()
                    .ok_or("learned count")?
            )?,
            "{}",
            case["name"]
        );
        assert_eq!(
            trace.main.token_ids.len(),
            usize::try_from(assertion["main_tokens"].as_u64().ok_or("main count")?)?,
            "{}",
            case["name"]
        );
        let oracle = expected
            .as_array()
            .ok_or("dynamic expected")?
            .iter()
            .find(|row| row["name"] == case["name"])
            .and_then(|row| row.get("oracle"))
            .ok_or("dynamic oracle")?;
        let direct_ids = |value: &Value| -> Result<Vec<TokenId>, Box<dyn Error>> {
            value
                .as_array()
                .ok_or("direct token IDs")?
                .iter()
                .map(|id| Ok(u32::try_from(id.as_u64().ok_or("direct token")?)?))
                .collect()
        };
        let mut expected_main = Vec::new();
        let expected_probes = oracle["expected_operations"]
            .as_array()
            .ok_or("dynamic operations")?
            .iter()
            .filter(|operation| operation["kind"] == "probe")
            .collect::<Vec<_>>();
        for operation in oracle["expected_operations"]
            .as_array()
            .ok_or("dynamic operations")?
        {
            if operation["kind"] == "main_append" {
                expected_main.extend(direct_ids(&operation["token_ids"])?);
            }
        }
        assert_eq!(
            trace.main.token_ids, expected_main,
            "{} main IDs",
            case["name"]
        );
        assert_eq!(
            trace.probes.len(),
            expected_probes.len(),
            "{} probes",
            case["name"]
        );
        for (actual, expected) in trace.probes.iter().zip(expected_probes) {
            assert_eq!(
                actual.operation.wire(),
                expected["operation"].as_str().ok_or("probe op")?
            );
            assert_eq!(actual.path, expected["path"].as_str().ok_or("probe path")?);
            assert_eq!(
                actual.selected_index,
                usize::try_from(expected["selected_index"].as_u64().ok_or("probe choice")?)?
            );
            let mut prefix = actual.prefix_token_ids.clone();
            for suffix in &actual.suffix_segments {
                prefix.extend_from_slice(&suffix.token_ids);
            }
            assert_eq!(prefix, direct_ids(&expected["prefix_token_ids"])?);
            for (candidate, expected_candidate) in actual.candidates.iter().zip(
                expected["candidates"]
                    .as_array()
                    .ok_or("probe candidates")?,
            ) {
                assert_eq!(
                    candidate.token_ids,
                    direct_ids(&expected_candidate["token_ids"])?
                );
                let positions = expected_candidate["prediction_positions"]
                    .as_array()
                    .ok_or("probe positions")?
                    .iter()
                    .map(|position| {
                        Ok(usize::try_from(position.as_u64().ok_or("probe position")?)?)
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                assert_eq!(candidate.prediction_positions, positions);
            }
        }
        let paths = trace
            .probes
            .iter()
            .map(|probe| probe.path.as_str())
            .collect::<Vec<_>>();
        match case["name"].as_str().ok_or("fixture name")? {
            "typed_array_dynamic_items" => assert_eq!(paths, ["/value/1", "/value/2"]),
            _ => assert!(paths.is_empty(), "{}: {paths:?}", case["name"]),
        }
        if matches!(
            case["name"].as_str(),
            Some("dynamic_array" | "dynamic_wrapped_array" | "dynamic_empty_array")
        ) {
            let dynamic_source = serde_jcs::to_string(&case["arguments"]["value"])?;
            assert!(
                trace.main.appends.iter().any(|append| {
                    append.ownership == TraceOwnership::Learned
                        && append.source.as_deref() == Some(dynamic_source.as_bytes())
                }),
                "{}",
                case["name"]
            );
        }
    }
    Ok(())
}
