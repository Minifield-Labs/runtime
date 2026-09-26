#![allow(clippy::expect_used)]
// Expected failures are the assertions under test.
use minifield_decoding_protocol::{
    RawJson, RawJsonLimits, SchemaLimits, ValidationFailure, ValidationFailureKind,
    ValidationLimits, normalize_schema, normalize_schema_document, parse_json_document,
};
use std::error::Error;

fn limits(max_work: usize) -> ValidationLimits {
    ValidationLimits {
        max_work,
        ..ValidationLimits::default()
    }
}

fn assert_work_exhausted(failure: &ValidationFailure, keyword: Option<&str>) {
    assert_eq!(failure.class, ValidationFailureKind::Operational);
    assert_eq!(failure.keyword.as_deref(), keyword);
    assert_eq!(failure.message, "validation work limit exceeded");
}

#[test]
fn admission_and_schema_validation_share_one_work_budget() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document("shared_budget", b"true", SchemaLimits::default())?;
    let instance = RawJson::Array(vec![RawJson::Null]);
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(2))
            .expect_err("schema visit must spend the remaining admission budget"),
        None,
    );
    plan.validate_original_instance_with_limits(&instance, limits(3))?;
    Ok(())
}

#[test]
fn large_distinct_array_stops_at_admission_or_comparison_budget() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document(
        "distinct_array",
        br#"{"uniqueItems":true}"#,
        SchemaLimits::default(),
    )?;
    let instance = RawJson::Array(
        (0..5000)
            .map(|index| RawJson::String(format!("distinct-{index}")))
            .collect(),
    );
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(1))
            .expect_err("large input admission must honor a one-unit budget"),
        None,
    );
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(7000))
            .expect_err("pairwise comparisons must consume work after admission"),
        Some("uniqueItems"),
    );
    Ok(())
}

#[test]
fn enum_candidates_and_recursive_const_equality_consume_work() -> Result<(), Box<dyn Error>> {
    let enum_schema = RawJson::Object(vec![(
        "enum".to_owned(),
        RawJson::Array(
            (0..200)
                .map(|index| RawJson::String(format!("candidate-{index}")))
                .collect(),
        ),
    )]);
    let plan = normalize_schema("enum_budget", enum_schema, SchemaLimits::default())?;
    let instance = RawJson::String("absent".to_owned());
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(32))
            .expect_err("each unsuccessful enum comparison must consume work"),
        Some("enum"),
    );
    assert_eq!(
        plan.validate_original_instance(&instance)
            .expect_err("all candidates differ")
            .class,
        ValidationFailureKind::InstanceInvalid
    );

    let plan = normalize_schema_document(
        "recursive_equality",
        br#"{"const":[[null]]}"#,
        SchemaLimits::default(),
    )?;
    let instance = RawJson::Array(vec![RawJson::Array(vec![RawJson::Null])]);
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(6))
            .expect_err("equality must charge descendants as well as the root"),
        Some("const"),
    );
    plan.validate_original_instance_with_limits(&instance, limits(7))?;
    Ok(())
}

#[test]
fn object_equality_key_searches_consume_work_and_ignore_field_order() -> Result<(), Box<dyn Error>>
{
    let fields = (0..40)
        .map(|index| (format!("field-{index}"), RawJson::Null))
        .collect::<Vec<_>>();
    let plan = normalize_schema(
        "object_equality",
        RawJson::Object(vec![("const".to_owned(), RawJson::Object(fields.clone()))]),
        SchemaLimits::default(),
    )?;
    let instance = RawJson::Object(fields.into_iter().rev().collect());
    assert_work_exhausted(
        &plan
            .validate_original_instance_with_limits(&instance, limits(64))
            .expect_err("object key scans must also consume comparison work"),
        Some("const"),
    );
    plan.validate_original_instance(&instance)?;
    Ok(())
}

#[test]
fn programmatic_deep_values_are_bounded_before_schema_validation() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document("deep_value", b"true", SchemaLimits::default())?;
    for object in [false, true] {
        let mut instance = RawJson::Null;
        for _ in 0..512 {
            instance = if object {
                RawJson::Object(vec![("child".to_owned(), instance)])
            } else {
                RawJson::Array(vec![instance])
            };
        }
        let failure = plan
            .validate_original_instance(&instance)
            .expect_err("RawJson construction must not bypass validation depth bounds");
        assert_eq!(failure.class, ValidationFailureKind::Operational);
        assert_eq!(failure.message, "validation depth limit exceeded");
        assert_eq!(failure.schema_path, "");
        assert_eq!(failure.instance_path.matches('/').count(), 129);
    }
    Ok(())
}

#[test]
fn recursive_equality_honors_depth_after_schema_descent() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document(
        "equality_depth",
        br#"{"allOf":[{"const":[null]}]}"#,
        SchemaLimits::default(),
    )?;
    let instance = RawJson::Array(vec![RawJson::Null]);
    let failure = plan
        .validate_original_instance_with_limits(
            &instance,
            ValidationLimits {
                max_depth: 1,
                max_work: 100,
            },
        )
        .expect_err("comparison descendants must honor the remaining depth");
    assert_eq!(failure.class, ValidationFailureKind::Operational);
    assert_eq!(failure.keyword.as_deref(), Some("const"));
    assert_eq!(failure.instance_path, "/0");
    assert_eq!(failure.message, "validation depth limit exceeded");
    Ok(())
}

#[test]
fn budgeted_equality_preserves_binary64_rounding_and_signed_zero() -> Result<(), Box<dyn Error>> {
    for (schema, source, valid) in [
        (
            br#"{"const":1}"#.as_slice(),
            b"1.0000000000000001".as_slice(),
            true,
        ),
        (br#"{"enum":[0]}"#, b"-0", true),
        (br#"{"uniqueItems":true}"#, b"[1,1.0000000000000001]", false),
        (br#"{"uniqueItems":true}"#, b"[0,-0]", false),
        (
            br#"{"const":{"x":[1,0]}}"#,
            br#"{"x":[1.0000000000000001,-0]}"#,
            true,
        ),
    ] {
        let plan = normalize_schema_document("numeric_equality", schema, SchemaLimits::default())?;
        let instance = parse_json_document(source, RawJsonLimits::default())?;
        let result = plan.validate_original_instance(&instance);
        assert_eq!(result.is_ok(), valid, "{schema:?} {source:?}");
        if let Err(failure) = result {
            assert_eq!(failure.class, ValidationFailureKind::InstanceInvalid);
            assert_eq!(failure.keyword.as_deref(), Some("uniqueItems"));
        }
    }
    Ok(())
}
