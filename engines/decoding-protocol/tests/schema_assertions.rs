#![allow(clippy::expect_used)]
// Expected failures are the assertion under test.
use minifield_decoding_protocol::{
    RawJsonLimits, SchemaLimits, normalize_schema_document, parse_json_document,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    schema_source: String,
    instance_source: String,
    expected_protocol_valid: bool,
}

#[test]
fn original_schema_assertion_oracle_001() -> Result<(), Box<dyn Error>> {
    let fixture: Fixture = serde_json::from_slice(include_bytes!(
        "../fixtures/schema-assertions-001/cases.json"
    ))?;
    assert_eq!(fixture.cases.len(), 81);
    for case in fixture.cases {
        let plan = normalize_schema_document(
            &case.name,
            case.schema_source.as_bytes(),
            SchemaLimits::default(),
        )?;
        let instance =
            parse_json_document(case.instance_source.as_bytes(), RawJsonLimits::default())?;
        assert_eq!(
            plan.validate_original_instance(&instance).is_ok(),
            case.expected_protocol_valid,
            "{}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn one_of_propagates_operational_branch_failure() -> Result<(), Box<dyn Error>> {
    use minifield_decoding_protocol::{ValidationFailureKind, ValidationLimits};
    let schema = normalize_schema_document(
        "operational",
        br#"{"oneOf":[true,true]}"#,
        SchemaLimits::default(),
    )?;
    let instance = parse_json_document(b"true", RawJsonLimits::default())?;
    let failure = schema
        .validate_original_instance_with_limits(
            &instance,
            ValidationLimits {
                max_depth: 128,
                max_work: 1,
            },
        )
        .expect_err("branch work exhaustion must not become a nonmatching branch");
    assert_eq!(failure.class, ValidationFailureKind::Operational);
    Ok(())
}

#[test]
fn huge_zero_exponents_remain_admitted_integer_zero() -> Result<(), Box<dyn Error>> {
    let schema =
        normalize_schema_document("zero", br#"{"type":"integer"}"#, SchemaLimits::default())?;
    for source in [
        b"0e999999999999999999999999999".as_slice(),
        b"-0e-99999999999999999999999999999",
    ] {
        let instance = parse_json_document(source, RawJsonLimits::default())?;
        assert!(
            schema.validate_original_instance(&instance).is_ok(),
            "{source:?}"
        );
    }
    Ok(())
}

#[test]
fn admitted_plan_rejects_unsafe_numbers_even_under_true_schema() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document("true", b"true", SchemaLimits::default())?;
    let instance = parse_json_document(b"[1e400]", RawJsonLimits::default())?;
    let failure = plan
        .validate_original_instance(&instance)
        .expect_err("unsafe number");
    assert_eq!(
        failure.class,
        minifield_decoding_protocol::ValidationFailureKind::InstanceInvalid
    );
    Ok(())
}
