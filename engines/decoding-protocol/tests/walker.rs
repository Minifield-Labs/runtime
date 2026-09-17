#![allow(
    clippy::cast_possible_truncation,
    clippy::cloned_ref_to_slice_refs,
    clippy::expect_used,
    clippy::manual_let_else,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
// Fixture assertions use explicit failure messages and retain literal wire fragments.
use minifield_decoding_protocol::{
    ProtocolError, PublicEvent, RawJson, RawJsonLimits, SchemaLimits, SegmentTokenizer,
    TeacherLexicalValue, TeacherTraceBuilder, TeacherTraceInput, TokenByteMap, TokenId,
    TokenPolicy, TraceOwnership, normalize_schema, parse_json_document,
};
use serde_json::Value;
use std::{convert::Infallible, error::Error};

struct ByteTokenizer;
impl SegmentTokenizer for ByteTokenizer {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(segment
            .as_bytes()
            .iter()
            .map(|byte| u32::from(*byte) + 1000)
            .collect())
    }
}

fn raw(value: &Value) -> Result<RawJson, Box<dyn Error>> {
    Ok(parse_json_document(
        &serde_json::to_vec(value)?,
        RawJsonLimits::default(),
    )?)
}

fn event(value: &Value) -> Result<PublicEvent, Box<dyn Error>> {
    let object = value.as_object().ok_or("event must be an object")?;
    Ok(match object["type"].as_str().ok_or("event type missing")? {
        "system" => PublicEvent::System {
            policy: object["policy"]
                .as_str()
                .ok_or("policy missing")?
                .to_owned(),
            observation: raw(&object["observation"])?,
        },
        "user" => PublicEvent::User {
            content: object["content"]
                .as_str()
                .ok_or("content missing")?
                .to_owned(),
        },
        "tool_call" => PublicEvent::ToolCall {
            tool_call_id: object["tool_call_id"]
                .as_str()
                .ok_or("tool call id missing")?
                .to_owned(),
            tool: object["tool"].as_str().ok_or("tool missing")?.to_owned(),
            arguments: raw(&object["arguments"])?,
        },
        "tool_result" => PublicEvent::ToolResult {
            tool_call_id: object["tool_call_id"]
                .as_str()
                .ok_or("tool result id missing")?
                .to_owned(),
            result: raw(&object["result"])?,
        },
        "assistant_text" => PublicEvent::AssistantText {
            content: object["content"]
                .as_str()
                .ok_or("content missing")?
                .to_owned(),
        },
        other => return Err(format!("unknown event type {other}").into()),
    })
}

#[test]
fn nested_choice_trace_is_schema_derived_and_keeps_child_source_path() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-traces-003/expected.json"
    ))?;
    let case = &fixture["cases"][0];
    let events = case["public_events"]
        .as_array()
        .ok_or("public events missing")?
        .iter()
        .map(event)
        .collect::<Result<Vec<_>, _>>()?;
    let plan = normalize_schema(
        "synthetic.nested_choice_source_path",
        raw(&case["original_schema"])?,
        SchemaLimits::default(),
    )?;
    let arguments = raw(&case["teacher_arguments"])?;
    let policy = TokenPolicy::draft5()?;
    let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
        &TeacherTraceInput {
            public_events: &events,
            selected_route: "synthetic.nested_choice_source_path",
            schema_plan: &plan,
        },
        &arguments,
    )?;

    assert_eq!(trace.probes.len(), 2);
    assert_eq!(trace.probes[0].path, "");
    assert_eq!(trace.probes[1].path, "/x");
    let child_metadata =
        String::from_utf8(trace.probes[1].suffix_segments[1].source.clone().unwrap())?;
    assert!(child_metadata.contains(r#""schema_path":"/properties/x/anyOf/0""#));
    assert!(child_metadata.contains(r#""schema_path":"/properties/x/anyOf/1""#));
    assert_eq!(
        trace.probes[0].candidates[0].prediction_positions[0],
        trace.probes[0].suffix_segments.last().unwrap().branch_end - 1
    );
    assert_eq!(
        trace.probes[1].candidates[0].prediction_positions[0],
        trace.probes[1].suffix_segments.last().unwrap().branch_end - 1
    );

    let tail = trace
        .main
        .appends
        .iter()
        .skip_while(|append| append.label != "object-open")
        .collect::<Vec<_>>();
    assert_eq!(tail[0].source.as_deref(), Some(b"{".as_slice()));
    assert_eq!(tail[1].source.as_deref(), Some(br#""x""#.as_slice()));
    assert_eq!(tail[2].source.as_deref(), Some(b":".as_slice()));
    assert_eq!(tail[3].ownership, TraceOwnership::Learned);
    assert_eq!(tail[3].source.as_deref(), Some(br#""v""#.as_slice()));
    assert_eq!(tail[4].ownership, TraceOwnership::Learned);
    assert_eq!(tail[5].source.as_deref(), Some(b"}".as_slice()));
    assert_eq!(tail[6].source.as_deref(), Some(b"}".as_slice()));
    assert_eq!(trace.learned_token_count, 22);
    Ok(())
}

#[test]
fn wholly_open_objects_are_one_learned_span_even_when_explicitly_typed_or_empty()
-> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let events = [PublicEvent::User {
        content: "dynamic object".to_owned(),
    }];
    for (name, schema, argument) in [
        (
            "typed",
            br#"{"type":"object"}"#.as_slice(),
            br#"{"x":1}"#.as_slice(),
        ),
        (
            "explicit-empty-nonempty",
            br#"{"type":"object","properties":{},"additionalProperties":true}"#.as_slice(),
            br#"{"x":1}"#.as_slice(),
        ),
        (
            "explicit-empty-empty",
            br#"{"type":"object","properties":{},"additionalProperties":true}"#.as_slice(),
            br#"{}"#.as_slice(),
        ),
    ] {
        let plan = minifield_decoding_protocol::normalize_schema_document(
            name,
            schema,
            SchemaLimits::default(),
        )?;
        let arguments = parse_json_document(argument, RawJsonLimits::default())?;
        let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: "dynamic",
                schema_plan: &plan,
            },
            &arguments,
        )?;
        let envelope_close = trace.main.appends.last().ok_or("missing envelope close")?;
        assert_eq!(envelope_close.source.as_deref(), Some(b"}".as_slice()));
        let tail = &trace.main.appends[trace.main.appends.len() - 2];
        assert_eq!(tail.special_id, Some(policy.framing_ids().value_end));
        assert_eq!(tail.ownership, TraceOwnership::Learned);
        let dynamic = &trace.main.appends[trace.main.appends.len() - 3];
        assert_eq!(dynamic.ownership, TraceOwnership::Learned, "{name}");
        assert_eq!(dynamic.source.as_deref(), Some(argument), "{name}");
        assert!(trace.probes.is_empty(), "{name}");
        assert_eq!(trace.finite_choices.len(), 0, "{name}");
    }

    Ok(())
}

#[test]
fn resolved_unconstrained_additional_properties_reference_is_a_dynamic_object()
-> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let schema = br##"{
        "$defs":{
            "JsonValue":{},
            "JsonObject":{"type":"object","additionalProperties":{"$ref":"#/$defs/JsonValue"}}
        },
        "$ref":"#/$defs/JsonObject"
    }"##;
    let events = [PublicEvent::User {
        content: "dynamic ref".to_owned(),
    }];
    for argument in [br#"{}"#.as_slice(), br#"{"a":[1,null],"z":"v"}"#.as_slice()] {
        let plan = minifield_decoding_protocol::normalize_schema_document(
            "json-object-ref",
            schema,
            SchemaLimits::default(),
        )?;
        let arguments = parse_json_document(argument, RawJsonLimits::default())?;
        let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: "dynamic-ref",
                schema_plan: &plan,
            },
            &arguments,
        )?;
        let envelope_close = trace.main.appends.last().ok_or("missing envelope close")?;
        assert_eq!(envelope_close.source.as_deref(), Some(b"}".as_slice()));
        let dynamic = &trace.main.appends[trace.main.appends.len() - 3];
        assert_eq!(dynamic.ownership, TraceOwnership::Learned);
        assert_eq!(dynamic.source.as_deref(), Some(argument));
        assert_eq!(trace.probes.len(), 0);
    }
    Ok(())
}

fn raw_field<'a>(object: &'a RawJson, name: &str) -> Result<&'a RawJson, Box<dyn Error>> {
    object
        .object_entries()
        .and_then(|entries| {
            entries
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value)
        })
        .ok_or_else(|| format!("missing raw field {name}").into())
}
fn raw_string(object: &RawJson, name: &str) -> Result<String, Box<dyn Error>> {
    match raw_field(object, name)? {
        RawJson::String(value) => Ok(value.clone()),
        _ => Err(format!("raw field {name} is not a string").into()),
    }
}
fn raw_event(value: &RawJson) -> Result<PublicEvent, Box<dyn Error>> {
    let kind = raw_string(value, "type")?;
    Ok(match kind.as_str() {
        "system" => PublicEvent::System {
            policy: raw_string(value, "policy")?,
            observation: raw_field(value, "observation")?.clone(),
        },
        "user" => PublicEvent::User {
            content: raw_string(value, "content")?,
        },
        "tool_call" => PublicEvent::ToolCall {
            tool_call_id: raw_string(value, "tool_call_id")?,
            tool: raw_string(value, "tool")?,
            arguments: raw_field(value, "arguments")?.clone(),
        },
        "tool_result" => PublicEvent::ToolResult {
            tool_call_id: raw_string(value, "tool_call_id")?,
            result: raw_field(value, "result")?.clone(),
        },
        "assistant_text" => PublicEvent::AssistantText {
            content: raw_string(value, "content")?,
        },
        _ => return Err(format!("unknown raw event type {kind}").into()),
    })
}
fn expected_count(case: &Value, kind: &str) -> usize {
    case["expected_operations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|operation| operation["kind"].as_str() == Some(kind))
        .count()
}
fn run_fixture_smoke(bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_slice(bytes)?;
    let raw_fixture = parse_json_document(bytes, RawJsonLimits::default())?;
    let raw_cases = match raw_field(&raw_fixture, "cases")? {
        RawJson::Array(cases) => cases,
        _ => return Err("raw cases is not an array".into()),
    };
    for (index, expected_case) in expected["cases"].as_array().unwrap().iter().enumerate() {
        let raw_case = &raw_cases[index];
        let name = raw_string(raw_case, "name")?;
        if name == "synthetic.runtime_lexical" {
            continue;
        }
        let events = match raw_field(raw_case, "public_events")? {
            RawJson::Array(events) => events
                .iter()
                .map(raw_event)
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(format!("{name}: events are not an array").into()),
        };
        let plan = normalize_schema(
            &name,
            raw_field(raw_case, "original_schema")?.clone(),
            SchemaLimits::default(),
        )?;
        let arguments = raw_field(raw_case, "teacher_arguments")?.clone();
        let policy = TokenPolicy::draft5()?;
        let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: &raw_string(raw_case, "selected_route")?,
                schema_plan: &plan,
            },
            &arguments,
        )?;
        assert_eq!(
            trace.probes.len(),
            expected_count(expected_case, "probe"),
            "{name}: probe count"
        );
        assert_eq!(
            trace.finite_choices.len(),
            expected_count(expected_case, "finite_choice"),
            "{name}: finite count"
        );
        assert!(
            trace.main.token_ids.len() > trace.main.argument_start,
            "{name}"
        );
    }
    Ok(())
}

#[test]
fn all_nonlexical_teacher_synthetic_cases_reach_schema_driven_trace_planning()
-> Result<(), Box<dyn Error>> {
    run_fixture_smoke(include_bytes!(
        "../fixtures/argument-traces-001/expected.json"
    ))?;
    run_fixture_smoke(include_bytes!(
        "../fixtures/argument-traces-002/expected.json"
    ))?;
    run_fixture_smoke(include_bytes!(
        "../fixtures/argument-traces-003/expected.json"
    ))
}
use std::collections::BTreeMap;

fn runtime_lexical_fixture_bytes() -> Result<TokenByteMap, Box<dyn Error>> {
    Ok(TokenByteMap::new(
        vec![
            (511, "\"".to_owned()),
            (520, "+".to_owned()),
            (523, ".".to_owned()),
            (525, "0".to_owned()),
            (526, "1".to_owned()),
            (569, "\\".to_owned()),
            (578, "e".to_owned()),
            (594, "u".to_owned()),
            (1242, "xt".to_owned()),
            (4171, "ee".to_owned()),
            (19942, "006".to_owned()),
        ],
        Vec::<(TokenId, String)>::new(),
    )?)
}

struct OracleTokenizer {
    ids: BTreeMap<String, Vec<TokenId>>,
}
impl SegmentTokenizer for OracleTokenizer {
    type Error = String;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        self.ids
            .get(segment)
            .cloned()
            .ok_or_else(|| format!("fixture oracle has no independent segment for {segment:?}"))
    }
}
fn ids(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as TokenId)
        .collect()
}
fn record_oracle_segment(map: &mut BTreeMap<String, Vec<TokenId>>, segment: &Value) {
    let Some(text) = segment["text"].as_str() else {
        return;
    };
    let token_ids = ids(&segment["token_ids"]);
    if let Some(existing) = map.insert(text.to_owned(), token_ids.clone()) {
        assert_eq!(
            existing, token_ids,
            "fixture tokenization must be segment-local"
        );
    }
}
fn oracle_tokenizer(case: &Value) -> OracleTokenizer {
    let mut ids = BTreeMap::new();
    for operation in case["expected_operations"].as_array().unwrap() {
        match operation["kind"].as_str().unwrap() {
            "main_append" => record_oracle_segment(&mut ids, operation),
            "probe" => {
                for segment in operation["suffix_segments"].as_array().unwrap() {
                    record_oracle_segment(&mut ids, segment);
                }
                for candidate in operation["candidates"].as_array().unwrap() {
                    for segment in candidate["segments"].as_array().unwrap() {
                        record_oracle_segment(&mut ids, segment);
                    }
                }
            }
            "finite_choice" => {
                for candidate in operation["candidates"].as_array().unwrap() {
                    for segment in candidate["segments"].as_array().unwrap() {
                        record_oracle_segment(&mut ids, segment);
                    }
                }
            }
            "forced" => {}
            other => panic!("unknown expected operation {other}"),
        }
    }
    OracleTokenizer { ids }
}
fn flatten_segments(segments: &[minifield_decoding_protocol::BranchAppend]) -> Vec<TokenId> {
    segments
        .iter()
        .flat_map(|segment| segment.token_ids.iter().copied())
        .collect()
}
fn expected_flatten_segments(segments: &[Value]) -> Vec<TokenId> {
    segments
        .iter()
        .flat_map(|segment| ids(&segment["token_ids"]))
        .collect()
}
fn run_fixture_parity(bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_slice(bytes)?;
    let raw_fixture = parse_json_document(bytes, RawJsonLimits::default())?;
    let raw_cases = match raw_field(&raw_fixture, "cases")? {
        RawJson::Array(cases) => cases,
        _ => return Err("raw cases is not an array".into()),
    };
    for (index, expected_case) in expected["cases"].as_array().unwrap().iter().enumerate() {
        let raw_case = &raw_cases[index];
        let name = raw_string(raw_case, "name")?;
        let events = match raw_field(raw_case, "public_events")? {
            RawJson::Array(events) => events
                .iter()
                .map(raw_event)
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(format!("{name}: events are not an array").into()),
        };
        let plan = normalize_schema(
            &name,
            raw_field(raw_case, "original_schema")?.clone(),
            SchemaLimits::default(),
        )?;
        let is_runtime_lexical = name == "synthetic.runtime_lexical";
        let arguments = raw_field(
            raw_case,
            if is_runtime_lexical {
                "expected_arguments"
            } else {
                "teacher_arguments"
            },
        )?
        .clone();
        let route = raw_string(raw_case, "selected_route")?;
        let policy = TokenPolicy::draft5()?;
        let tokenizer = oracle_tokenizer(expected_case);
        let mut lexical = Vec::new();
        if is_runtime_lexical {
            for operation in expected_case["hand_authored_operations"]
                .as_array()
                .unwrap()
            {
                let Some(source) = operation["lexical"].as_str() else {
                    continue;
                };
                lexical.push(TeacherLexicalValue {
                    path: operation["path"].as_str().unwrap().to_owned(),
                    source: source.as_bytes().to_vec(),
                    token_ids: tokenizer.ids.get(source).cloned().ok_or_else(|| {
                        format!("{name}: fixture has no lexical token IDs for {source:?}")
                    })?,
                });
            }
        }
        let lexical_bytes = runtime_lexical_fixture_bytes()?;
        let builder = if lexical.is_empty() {
            TeacherTraceBuilder::new(&tokenizer, &policy)
        } else {
            TeacherTraceBuilder::with_token_bytes(&tokenizer, &policy, &lexical_bytes)
        };
        let trace = if lexical.is_empty() {
            builder.build(
                &TeacherTraceInput {
                    public_events: &events,
                    selected_route: &route,
                    schema_plan: &plan,
                },
                &arguments,
            )?
        } else {
            builder.build_with_lexical(
                &TeacherTraceInput {
                    public_events: &events,
                    selected_route: &route,
                    schema_plan: &plan,
                },
                &arguments,
                &lexical,
            )?
        };
        let expected_main_ids = ids(&expected_case["main_token_ids"]);
        if trace.main.token_ids != expected_main_ids {
            let first = trace
                .main
                .token_ids
                .iter()
                .zip(&expected_main_ids)
                .position(|(left, right)| left != right)
                .unwrap_or(trace.main.token_ids.len().min(expected_main_ids.len()));
            let append = trace
                .main
                .appends
                .iter()
                .find(|append| append.main_start <= first && first < append.main_end);
            panic!(
                "{name}: main ids differ at {first}; actual {} expected {}; actual_len {} expected_len {}; append {append:?}",
                trace.main.token_ids.get(first).copied().unwrap_or_default(),
                expected_main_ids.get(first).copied().unwrap_or_default(),
                trace.main.token_ids.len(),
                expected_main_ids.len(),
            );
        }
        assert_eq!(
            minifield_decoding_protocol::sha256_hex(
                &trace
                    .main
                    .token_ids
                    .iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect::<Vec<_>>()
            ),
            expected_case["main_sha256_u32le"].as_str().unwrap(),
            "{name}: main hash"
        );
        let expected_probes = expected_case["expected_operations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|operation| operation["kind"].as_str() == Some("probe"))
            .collect::<Vec<_>>();
        assert_eq!(trace.probes.len(), expected_probes.len(), "{name}: probes");
        for (actual, expected_probe) in trace.probes.iter().zip(expected_probes) {
            assert_eq!(
                actual.operation.wire(),
                expected_probe["operation"].as_str().unwrap(),
                "{name}: probe operation"
            );
            assert_eq!(
                actual.path,
                expected_probe["path"].as_str().unwrap(),
                "{name}: probe path"
            );
            assert_eq!(
                actual.fork_main_length,
                expected_probe["fork_main_length"].as_u64().unwrap() as usize,
                "{name}: probe fork"
            );
            assert_eq!(
                actual.fork_main_sha256_u32le,
                expected_probe["fork_main_sha256_u32le"].as_str().unwrap(),
                "{name}: probe hash"
            );
            assert_eq!(
                flatten_segments(&actual.suffix_segments),
                expected_flatten_segments(expected_probe["suffix_segments"].as_array().unwrap()),
                "{name}: probe suffix"
            );
            assert_eq!(
                actual.selected_index,
                expected_probe["selected_index"].as_u64().unwrap() as usize,
                "{name}: probe selected"
            );
            for (candidate, expected_candidate) in actual
                .candidates
                .iter()
                .zip(expected_probe["candidates"].as_array().unwrap())
            {
                assert_eq!(
                    candidate.label,
                    expected_candidate["label"].as_str().unwrap(),
                    "{name}: probe label"
                );
                assert_eq!(
                    candidate.token_ids,
                    ids(&expected_candidate["token_ids"]),
                    "{name}: probe candidate ids"
                );
                assert_eq!(
                    candidate.prediction_positions,
                    expected_candidate["prediction_positions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>(),
                    "{name}: probe positions"
                );
            }
        }
        let expected_finite = expected_case["expected_operations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|operation| operation["kind"].as_str() == Some("finite_choice"))
            .collect::<Vec<_>>();
        assert_eq!(
            trace.finite_choices.len(),
            expected_finite.len(),
            "{name}: finite choices"
        );
        for (actual, expected_finite) in trace.finite_choices.iter().zip(expected_finite) {
            assert_eq!(
                actual.path,
                expected_finite["path"].as_str().unwrap(),
                "{name}: finite path"
            );
            assert_eq!(
                actual.fork_main_length,
                expected_finite["fork_main_length"].as_u64().unwrap() as usize,
                "{name}: finite fork"
            );
            assert_eq!(
                actual.fork_main_sha256_u32le,
                expected_finite["fork_main_sha256_u32le"].as_str().unwrap(),
                "{name}: finite hash"
            );
            assert_eq!(
                actual.selected_index,
                expected_finite["selected_index"].as_u64().unwrap() as usize,
                "{name}: finite selected"
            );
            for (candidate, expected_candidate) in actual
                .candidates
                .iter()
                .zip(expected_finite["candidates"].as_array().unwrap())
            {
                assert_eq!(
                    candidate.token_ids,
                    ids(&expected_candidate["token_ids"]),
                    "{name}: finite candidate ids"
                );
                assert_eq!(
                    candidate.prediction_positions,
                    expected_candidate["prediction_positions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>(),
                    "{name}: finite positions"
                );
            }
        }
        let actual_main_targets = trace
            .main
            .appends
            .iter()
            .filter(|append| append.ownership == TraceOwnership::Learned)
            .flat_map(|append| {
                append
                    .prediction_positions
                    .iter()
                    .copied()
                    .zip(append.token_ids.iter().copied())
            })
            .collect::<Vec<_>>();
        let expected_main_targets = expected_case["learned_main_targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|target| {
                (
                    target["position"].as_u64().unwrap() as usize,
                    target["token_id"].as_u64().unwrap() as TokenId,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual_main_targets, expected_main_targets,
            "{name}: main causal targets"
        );
        assert_eq!(
            trace.learned_token_count,
            expected_case["learned_argument_token_count"]
                .as_u64()
                .unwrap() as usize,
            "{name}: learned denominator"
        );
    }
    Ok(())
}

#[test]
fn all_synthetic_teacher_goldens_replay_through_injected_segment_oracle()
-> Result<(), Box<dyn Error>> {
    run_fixture_parity(include_bytes!(
        "../fixtures/argument-traces-001/expected.json"
    ))?;
    run_fixture_parity(include_bytes!(
        "../fixtures/argument-traces-002/expected.json"
    ))?;
    run_fixture_parity(include_bytes!(
        "../fixtures/argument-traces-003/expected.json"
    ))
}

#[test]
fn committed_lexical_ids_must_decode_to_source_and_retain_alternate_valid_segmentation()
-> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let tokenizer = ByteTokenizer;
    let token_bytes = TokenByteMap::new(
        vec![
            (1001, "1.".to_owned()),
            (1002, "0e+0".to_owned()),
            (1101, "1.0e".to_owned()),
            (1102, "+0".to_owned()),
            (1300, "wrong".to_owned()),
        ],
        Vec::<(TokenId, String)>::new(),
    )?;
    let plan = minifield_decoding_protocol::normalize_schema_document(
        "lexical",
        br#"{"type":"object","properties":{"x":{"type":"number"}},"required":["x"],"additionalProperties":false}"#,
        SchemaLimits::default(),
    )?;
    let arguments = parse_json_document(br#"{"x":1}"#, RawJsonLimits::default())?;
    let events = [PublicEvent::User {
        content: "lexical".to_owned(),
    }];
    let input = TeacherTraceInput {
        public_events: &events,
        selected_route: "lexical",
        schema_plan: &plan,
    };
    let alternate = TeacherLexicalValue {
        path: "/x".to_owned(),
        source: b"1.0e+0".to_vec(),
        token_ids: vec![1101, 1102],
    };
    let missing_map = TeacherTraceBuilder::new(&tokenizer, &policy)
        .build_with_lexical(&input, &arguments, &[alternate.clone()])
        .expect_err("lexical IDs without an inverse byte map must reject");
    assert!(matches!(missing_map, ProtocolError::Schema(_)));
    let trace = TeacherTraceBuilder::with_token_bytes(&tokenizer, &policy, &token_bytes)
        .build_with_lexical(&input, &arguments, &[alternate])?;
    let append = trace
        .main
        .appends
        .iter()
        .find(|append| append.label == "value")
        .ok_or("missing committed lexical append")?;
    assert_eq!(append.source.as_deref(), Some(b"1.0e+0".as_slice()));
    assert_eq!(append.token_ids, vec![1101, 1102]);
    assert_eq!(token_bytes.token_stream_bytes(&[1001, 1002])?, b"1.0e+0");
    assert_eq!(token_bytes.token_stream_bytes(&[1101, 1102])?, b"1.0e+0");

    let bad = |token_ids| {
        TeacherTraceBuilder::with_token_bytes(&tokenizer, &policy, &token_bytes)
            .build_with_lexical(
                &input,
                &arguments,
                &[TeacherLexicalValue {
                    path: "/x".to_owned(),
                    source: b"1.0e+0".to_vec(),
                    token_ids,
                }],
            )
            .expect_err("invalid committed lexical history must reject")
    };
    assert!(matches!(bad(vec![1001, 1300]), ProtocolError::Schema(_)));
    assert!(matches!(bad(Vec::new()), ProtocolError::Schema(_)));
    assert!(matches!(bad(vec![1999]), ProtocolError::UnknownToken(1999)));
    Ok(())
}

#[test]
fn committed_lexical_byte_checks_reject_oversized_sources_and_pieces_before_parsing()
-> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let tokenizer = ByteTokenizer;
    let plan = minifield_decoding_protocol::normalize_schema_document(
        "lexical-bounds",
        br#"{"type":"object","properties":{"x":{"type":"number"}},"required":["x"],"additionalProperties":false}"#,
        SchemaLimits::default(),
    )?;
    let arguments = parse_json_document(br#"{"x":1}"#, RawJsonLimits::default())?;
    let events = [PublicEvent::User {
        content: "lexical bounds".to_owned(),
    }];
    let input = TeacherTraceInput {
        public_events: &events,
        selected_route: "lexical-bounds",
        schema_plan: &plan,
    };
    let runtime_limit = RawJsonLimits::draft5_value().max_bytes;
    let empty_map = TokenByteMap::default();
    let oversized_source = TeacherLexicalValue {
        path: "/x".to_owned(),
        source: vec![b'0'; runtime_limit + 1],
        token_ids: vec![2001],
    };
    let source_error = TeacherTraceBuilder::with_token_bytes(&tokenizer, &policy, &empty_map)
        .build_with_lexical(&input, &arguments, &[oversized_source])
        .expect_err("over-limit lexical source must reject before unknown token lookup");
    assert!(matches!(source_error, ProtocolError::InputLimit(_)));

    let oversized_piece_map = TokenByteMap::new(
        Vec::<(TokenId, String)>::new(),
        vec![(2001, "0".repeat(runtime_limit + 1))],
    )?;
    let oversized_piece = TeacherLexicalValue {
        path: "/x".to_owned(),
        source: vec![b'0'; runtime_limit],
        token_ids: vec![2001],
    };
    let piece_error =
        TeacherTraceBuilder::with_token_bytes(&tokenizer, &policy, &oversized_piece_map)
            .build_with_lexical(&input, &arguments, &[oversized_piece])
            .expect_err("oversized token piece must reject before source slicing or parsing");
    assert!(matches!(piece_error, ProtocolError::Schema(_)));
    Ok(())
}
