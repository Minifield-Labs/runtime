use minifield_decoding_protocol::{
    BranchAppend, MainAppend, PublicEvent, RawJson, RawJsonLimits, SchemaLimits, SegmentTokenizer,
    TeacherTrace, TeacherTraceBuilder, TeacherTraceInput, TokenId, TokenPolicy, TraceOwnership,
    normalize_schema, parse_json_document, plan_teacher, semantic_equal,
};
use serde_json::Value;
use std::{collections::HashMap, error::Error};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct FixtureTokenizer(HashMap<String, Vec<TokenId>>);

impl SegmentTokenizer for FixtureTokenizer {
    type Error = String;

    fn encode_without_special_tokens(&self, source: &str) -> Result<Vec<TokenId>, Self::Error> {
        self.0
            .get(source)
            .cloned()
            .ok_or_else(|| format!("fixture has no segment {source:?}"))
    }
}

struct ByteTokenizer;

impl SegmentTokenizer for ByteTokenizer {
    type Error = std::convert::Infallible;

    fn encode_without_special_tokens(&self, source: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(source.bytes().map(|byte| u32::from(byte) + 1000).collect())
    }
}

fn raw(value: &Value) -> TestResult<RawJson> {
    Ok(parse_json_document(
        &serde_json::to_vec(value)?,
        RawJsonLimits::default(),
    )?)
}

fn fixture_array<'a>(value: &'a Value, description: &str) -> TestResult<&'a Vec<Value>> {
    value.as_array().ok_or_else(|| description.into())
}

fn fixture_ids(value: &Value, description: &str) -> TestResult<Vec<TokenId>> {
    fixture_array(value, description)?
        .iter()
        .map(|token| {
            Ok(u32::try_from(
                token.as_u64().ok_or("fixture token must be u64")?,
            )?)
        })
        .collect()
}

fn fixture_positions(value: &Value, description: &str) -> TestResult<Vec<usize>> {
    fixture_array(value, description)?
        .iter()
        .map(|position| {
            Ok(usize::try_from(
                position.as_u64().ok_or("fixture position must be u64")?,
            )?)
        })
        .collect()
}

fn fixture_ownership(value: &Value) -> TestResult<TraceOwnership> {
    match value.as_str().ok_or("fixture ownership")? {
        "fixed" => Ok(TraceOwnership::Fixed),
        "learned" => Ok(TraceOwnership::Learned),
        other => Err(format!("unknown fixture ownership {other}").into()),
    }
}

fn fixture_source(value: &Value) -> TestResult<Option<Vec<u8>>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(
        value.as_str().ok_or("fixture text")?.as_bytes().to_vec(),
    ))
}

fn fixture_special(value: &Value) -> TestResult<Option<TokenId>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(u32::try_from(
        value.as_u64().ok_or("fixture special token")?,
    )?))
}

fn fixture_events(value: &Value) -> TestResult<Vec<PublicEvent>> {
    fixture_array(value, "fixture events")?
        .iter()
        .map(
            |event| match event["type"].as_str().ok_or("fixture event type")? {
                "system" => Ok(PublicEvent::System {
                    policy: event["policy"].as_str().ok_or("fixture policy")?.to_owned(),
                    observation: raw(&event["observation"])?,
                }),
                "user" => Ok(PublicEvent::User {
                    content: event["content"]
                        .as_str()
                        .ok_or("fixture content")?
                        .to_owned(),
                }),
                other => Err(format!("unsupported fixture event {other}").into()),
            },
        )
        .collect()
}

fn fixture_tokenizer() -> TestResult<FixtureTokenizer> {
    let mut segments = HashMap::new();
    for line in include_str!("../fixtures/argument-finite-containers-006/segments.jsonl").lines() {
        let entry: Value = serde_json::from_str(line)?;
        segments.insert(
            entry["source"].as_str().ok_or("fixture source")?.to_owned(),
            fixture_ids(&entry["token_ids"], "fixture token IDs")?,
        );
    }
    Ok(FixtureTokenizer(segments))
}

fn oracle_operations<'a>(oracle: &'a Value, kind: &str) -> TestResult<Vec<&'a Value>> {
    Ok(
        fixture_array(&oracle["expected_operations"], "fixture operations")?
            .iter()
            .filter(|operation| operation["kind"] == kind)
            .collect(),
    )
}

fn assert_main_append(actual: &MainAppend, expected: &Value) -> TestResult<()> {
    assert_eq!(actual.source, fixture_source(&expected["text"])?);
    assert_eq!(actual.special_id, fixture_special(&expected["special_id"])?);
    assert_eq!(
        actual.token_ids,
        fixture_ids(&expected["token_ids"], "main token IDs")?
    );
    assert_eq!(actual.ownership, fixture_ownership(&expected["ownership"])?);
    assert_eq!(
        actual.main_start,
        usize::try_from(expected["main_start"].as_u64().ok_or("main start")?)?
    );
    assert_eq!(
        actual.main_end,
        usize::try_from(expected["main_end"].as_u64().ok_or("main end")?)?
    );
    assert_eq!(
        actual.prediction_positions,
        fixture_positions(&expected["prediction_positions"], "main positions")?
    );
    // Labels are diagnostic-only and intentionally differ between the two implementations.
    Ok(())
}

fn assert_main(trace: &TeacherTrace, oracle: &Value) -> TestResult<()> {
    let expected = oracle_operations(oracle, "main_append")?;
    assert_eq!(trace.main.appends.len(), expected.len());
    for (actual, expected_append) in trace.main.appends.iter().zip(expected) {
        assert_main_append(actual, expected_append)?;
    }
    Ok(())
}

fn assert_branch(actual: &BranchAppend, expected: &Value) -> TestResult<()> {
    assert_eq!(actual.source, fixture_source(&expected["text"])?);
    assert_eq!(actual.special_id, fixture_special(&expected["special_id"])?);
    assert_eq!(
        actual.token_ids,
        fixture_ids(&expected["token_ids"], "branch token IDs")?
    );
    assert_eq!(actual.ownership, fixture_ownership(&expected["ownership"])?);
    Ok(())
}

fn assert_finite(trace: &TeacherTrace, oracle: &Value) -> TestResult<()> {
    let expected = oracle_operations(oracle, "finite_choice")?;
    assert_eq!(trace.finite_choices.len(), expected.len());
    for (actual, expected_choice) in trace.finite_choices.iter().zip(expected) {
        assert_eq!(
            actual.path,
            expected_choice["path"].as_str().ok_or("finite path")?
        );
        assert_eq!(
            actual.fork_main_length,
            usize::try_from(
                expected_choice["fork_main_length"]
                    .as_u64()
                    .ok_or("finite fork")?
            )?
        );
        assert_eq!(
            actual.fork_main_sha256_u32le,
            expected_choice["fork_main_sha256_u32le"]
                .as_str()
                .ok_or("finite hash")?
        );
        assert_eq!(
            actual.selected_index,
            usize::try_from(
                expected_choice["selected_index"]
                    .as_u64()
                    .ok_or("finite choice")?
            )?
        );
        let expected_candidates =
            fixture_array(&expected_choice["candidates"], "finite candidates")?;
        assert_eq!(actual.candidates.len(), expected_candidates.len());
        for (candidate, expected_candidate) in actual.candidates.iter().zip(expected_candidates) {
            assert!(semantic_equal(
                &candidate.value,
                &raw(&expected_candidate["value"])?
            )?);
            assert_eq!(
                candidate.token_ids,
                fixture_ids(
                    &expected_candidate["token_ids"],
                    "finite candidate token IDs"
                )?
            );
            assert_eq!(
                candidate.prediction_positions,
                fixture_positions(
                    &expected_candidate["prediction_positions"],
                    "finite positions"
                )?
            );
            let expected_segments =
                fixture_array(&expected_candidate["segments"], "finite segments")?;
            assert_eq!(candidate.segments.len(), expected_segments.len());
            for (actual_segment, expected_segment) in
                candidate.segments.iter().zip(expected_segments)
            {
                assert_branch(actual_segment, expected_segment)?;
            }
        }
    }
    Ok(())
}

fn assert_atomic_value(trace: &TeacherTrace, assertion: &Value) -> TestResult<()> {
    let source = assertion["atomic_value_text"]
        .as_str()
        .ok_or("atomic value text")?;
    let ownership = fixture_ownership(&assertion["atomic_value_ownership"])?;
    assert!(trace.main.appends.iter().any(|append| {
        append.source.as_deref() == Some(source.as_bytes()) && append.ownership == ownership
    }));
    Ok(())
}

fn assert_fixture_case(
    case: &Value,
    assertion: &Value,
    oracle: &Value,
    tokenizer: &FixtureTokenizer,
    policy: &TokenPolicy,
) -> TestResult<()> {
    assert_eq!(case["name"], assertion["name"]);
    let events = fixture_events(&case["public_events"])?;
    let plan = normalize_schema(
        case["selected_route"].as_str().ok_or("fixture route")?,
        raw(&case["schema"])?,
        SchemaLimits::default(),
    )?;
    let arguments = raw(&case["arguments"])?;
    let teacher = plan_teacher(&plan, &arguments)?;
    assert!(teacher.unions.is_empty(), "{}", case["name"]);
    let trace = TeacherTraceBuilder::new(tokenizer, policy).build(
        &TeacherTraceInput {
            public_events: &events,
            selected_route: case["selected_route"].as_str().ok_or("fixture route")?,
            schema_plan: &plan,
        },
        &arguments,
    )?;
    assert!(trace.probes.is_empty(), "{}", case["name"]);
    assert_eq!(
        trace.learned_token_count,
        usize::try_from(
            assertion["learned_argument_tokens"]
                .as_u64()
                .ok_or("learned count")?
        )?
    );
    assert_eq!(
        trace.main.token_ids.len(),
        usize::try_from(
            assertion["main_tokens"]
                .as_u64()
                .ok_or("main token count")?
        )?
    );
    assert_main(&trace, oracle)?;
    assert_finite(&trace, oracle)?;
    assert_atomic_value(&trace, assertion)
}

#[test]
fn finite_containers_consume_descendant_unions_against_oracle_006() -> TestResult {
    let cases: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-finite-containers-006/cases.json"
    ))?;
    let assertions: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-finite-containers-006/manual-assertions.json"
    ))?;
    let expected: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-finite-containers-006/expected.json"
    ))?;
    let tokenizer = fixture_tokenizer()?;
    let policy = TokenPolicy::draft5()?;
    for ((case, assertion), expected_row) in fixture_array(&cases, "finite cases")?
        .iter()
        .zip(fixture_array(&assertions, "finite assertions")?)
        .zip(fixture_array(&expected, "finite expected")?)
    {
        assert_eq!(case["name"], expected_row["name"]);
        assert_fixture_case(
            case,
            assertion,
            &expected_row["oracle"],
            &tokenizer,
            &policy,
        )?;
    }
    Ok(())
}

#[test]
fn finite_containers_do_not_bypass_original_instance_validation() -> TestResult {
    let cases: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/argument-finite-containers-006/negative-cases.json"
    ))?;
    for case in fixture_array(&cases, "finite negative cases")? {
        let plan = normalize_schema(
            case["name"].as_str().ok_or("negative case name")?,
            raw(&case["schema"])?,
            SchemaLimits::default(),
        )?;
        let arguments = raw(&case["arguments"])?;
        assert!(
            plan_teacher(&plan, &arguments).is_err(),
            "{} must reject its invalid complete original instance",
            case["name"]
        );
    }
    Ok(())
}

#[test]
fn selected_same_node_finite_branch_keeps_its_union_before_fixed_value() -> TestResult {
    let schema = raw(&serde_json::json!({
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "array", "const": ["x", 2], "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
                    {"type": "null"}
                ]
            }
        },
        "required": ["value"],
        "additionalProperties": false,
    }))?;
    let arguments = raw(&serde_json::json!({"value": ["x", 2]}))?;
    let plan = normalize_schema("selected_finite", schema, SchemaLimits::default())?;
    let teacher = plan_teacher(&plan, &arguments)?;
    assert_eq!(teacher.unions.len(), 1);
    assert_eq!(teacher.unions[0].argument_path, "/value");
    let policy = TokenPolicy::draft5()?;
    let trace = TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
        &TeacherTraceInput {
            public_events: &[],
            selected_route: "selected_finite",
            schema_plan: &plan,
        },
        &arguments,
    )?;
    assert_eq!(trace.probes.len(), 1);
    assert_eq!(trace.probes[0].operation.wire(), "union");
    assert!(trace.finite_choices.is_empty());
    let main_learned = trace
        .main
        .appends
        .iter()
        .map(|append| append.prediction_positions.len())
        .sum::<usize>();
    let selected_probe_learned = trace.probes[0].candidates[0].prediction_positions.len();
    assert_eq!(main_learned, 0);
    assert_eq!(trace.learned_token_count, selected_probe_learned);
    Ok(())
}
