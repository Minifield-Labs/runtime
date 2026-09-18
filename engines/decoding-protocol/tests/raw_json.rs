use minifield_decoding_protocol::{
    NumericKind, ProtocolError, RawJson, RawJsonLimits, admit_number, parse_json_document,
    parse_runtime_value,
};
use std::error::Error;

#[test]
fn parser_preserves_object_order_and_raw_number_lexemes() -> Result<(), Box<dyn Error>> {
    let parsed = parse_json_document(
        br#"{"z":5e-324,"a":1.0000000000000001,"nested":{"first":0,"second":-0}}"#,
        RawJsonLimits::default(),
    )?;
    let RawJson::Object(entries) = parsed else {
        return Err("root was not an object".into());
    };
    assert_eq!(
        entries
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        vec!["z", "a", "nested"]
    );
    assert!(matches!(
        &entries[0].1,
        RawJson::Number(number) if number.as_str() == "5e-324"
    ));
    assert!(matches!(
        &entries[1].1,
        RawJson::Number(number) if number.as_str() == "1.0000000000000001"
    ));
    let value = RawJson::Object(entries).into_value()?;
    assert_eq!(
        value
            .as_object()
            .ok_or("converted root was not an object")?
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["z", "a", "nested"]
    );
    Ok(())
}

#[test]
fn parser_rejects_duplicate_decoded_keys_invalid_utf8_and_unpaired_surrogates() {
    assert!(matches!(
        parse_json_document(br#"{"a":1,"\u0061":2}"#, RawJsonLimits::default()),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(matches!(
        parse_json_document(&[b'"', 0xff, b'"'], RawJsonLimits::default()),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(matches!(
        parse_json_document(br#""\ud800""#, RawJsonLimits::default()),
        Err(ProtocolError::InvalidJson(_))
    ));
}

#[test]
fn runtime_values_reject_outer_whitespace_trailing_values_and_raw_angles() {
    assert!(parse_runtime_value(br#"{"key":"value"}"#).is_ok());
    assert!(matches!(
        parse_runtime_value(br#" {"key":"value"}"#),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(matches!(
        parse_runtime_value(br#"{"key":"value"} null"#),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(matches!(
        parse_runtime_value(br#""<""#),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(matches!(
        parse_runtime_value(br#"{"<":"value"}"#),
        Err(ProtocolError::InvalidJson(_))
    ));
    assert!(parse_runtime_value(br#""\u003c""#).is_ok());
    assert!(parse_json_document(br#""<""#, RawJsonLimits::default()).is_ok());
}

#[test]
fn numeric_admission_keeps_lexemes_before_binary64_conversion() -> Result<(), Box<dyn Error>> {
    let rounded = number("1.0000000000000001")?;
    assert!(admit_number(&rounded, NumericKind::Number).is_ok());
    assert!(matches!(
        admit_number(&rounded, NumericKind::Integer),
        Err(ProtocolError::Numeric(_))
    ));

    let tiny = number("5e-324")?;
    assert_eq!(
        admit_number(&tiny, NumericKind::Number)?.value.to_bits(),
        5e-324_f64.to_bits()
    );
    assert!(matches!(
        admit_number(&number("1e-999")?, NumericKind::Number),
        Err(ProtocolError::Numeric(_))
    ));
    assert!(matches!(
        admit_number(&number("1e20")?, NumericKind::Number),
        Err(ProtocolError::Numeric(_))
    ));
    assert_eq!(
        admit_number(&number("-0")?, NumericKind::Integer)?
            .value
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(admit_number(&number("0e999999999999999999999")?, NumericKind::Integer).is_ok());
    assert!(matches!(
        admit_number(&number("1e999999999999999999999")?, NumericKind::Number),
        Err(ProtocolError::Numeric(_))
    ));
    assert_eq!(
        RawJson::Number(number("1.0000000000000001")?).into_value()?,
        serde_json::json!(1.0)
    );
    for lexical in [
        "51.248178375505404",
        "-93.31137037688033",
        "-36.573994842753436",
        "52.314008204106244",
        "97.45365320034685",
        "2.0030397744267762e-253",
    ] {
        let raw = number(lexical)?;
        let admitted = admit_number(&raw, NumericKind::Number)?;
        let value = RawJson::Number(raw).into_value()?;
        let safe = minifield_decoding_protocol::safe_json(&value)?;
        let round_trip = parse_json_document(&safe, RawJsonLimits::default())?;
        let RawJson::Number(round_trip) = round_trip else {
            return Err("safe JSON did not contain a number".into());
        };
        assert_eq!(
            admit_number(&round_trip, NumericKind::Number)?
                .value
                .to_bits(),
            admitted.value.to_bits(),
            "binary64 bits changed for {lexical}"
        );
    }
    Ok(())
}

fn number(source: &str) -> Result<minifield_decoding_protocol::RawNumber, Box<dyn Error>> {
    let parsed = parse_runtime_value(source.as_bytes())?;
    let RawJson::Number(number) = parsed else {
        return Err("numeric parser did not return a number".into());
    };
    Ok(number)
}
