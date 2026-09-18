use minifield_decoding_protocol::{
    CanonicalDecimal, NumericKind, RawJson, RawJsonLimits, admit_number, parse_runtime_value,
    semantic_equal,
};
use std::cmp::Ordering;
use std::error::Error;

fn admitted(source: &str) -> Result<minifield_decoding_protocol::AdmittedNumber, Box<dyn Error>> {
    let RawJson::Number(number) = parse_runtime_value(source.as_bytes())? else {
        return Err("expected numeric raw JSON".into());
    };
    Ok(admit_number(&number, NumericKind::Number)?)
}

fn decimal(source: &str) -> Result<CanonicalDecimal, Box<dyn Error>> {
    Ok(CanonicalDecimal::from_admitted(&admitted(source)?)?)
}

#[test]
fn exact_canonical_decimal_multiple_of_has_no_float_epsilon() -> Result<(), Box<dyn Error>> {
    assert!(decimal("0.3")?.is_multiple_of(&decimal("0.1")?)?);
    assert!(!decimal("0.30000000000000004")?.is_multiple_of(&decimal("0.1")?)?);
    assert!(decimal("0.07")?.is_multiple_of(&decimal("0.01")?)?);
    assert!(decimal("-0")?.is_multiple_of(&decimal("0.1")?)?);
    assert!(decimal("5e-324")?.compare(&decimal("0")?) == Ordering::Greater);
    Ok(())
}

#[test]
fn canonical_decimal_compares_wire_rationals_and_numeric_equality_is_binary64()
-> Result<(), Box<dyn Error>> {
    assert_eq!(
        decimal("1.0000000000000001")?.compare(&decimal("1")?),
        Ordering::Equal
    );
    assert_eq!(decimal("-0")?.compare(&decimal("0")?), Ordering::Equal);
    let RawJson::Number(one) = parse_runtime_value(b"1")? else {
        return Err("number missing".into());
    };
    let RawJson::Number(rounded) = parse_runtime_value(b"1.0000000000000001")? else {
        return Err("number missing".into());
    };
    assert!(semantic_equal(
        &RawJson::Number(one),
        &RawJson::Number(rounded)
    )?);
    assert!(!semantic_equal(
        &RawJson::Bool(true),
        &parse_runtime_value(b"1")?
    )?);
    assert_eq!(RawJsonLimits::default().max_bytes, 4 * 1024 * 1024);
    Ok(())
}
