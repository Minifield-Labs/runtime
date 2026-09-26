use minifield_decoding_protocol::{
    ProtocolError, Segment, SegmentTokenizer, TokenByteMap, TokenId, TokenPolicy, safe_json,
    tokenize_segments,
};
use serde::Deserialize;
use serde_json::Value;
use std::cell::RefCell;
use std::convert::Infallible;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    protocol_sha256: String,
    cases: Vec<FixtureCase>,
}

#[derive(Deserialize)]
struct FixtureCase {
    name: String,
    value: Value,
    safe_json_hex: String,
}

#[test]
fn safe_json_matches_draft5_primitive_fixture() -> Result<(), Box<dyn Error>> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/safe-json-primitives.draft-5.json"
    ))?;
    assert_eq!(
        fixture.protocol_sha256,
        minifield_decoding_protocol::DRAFT5_ARTIFACT_SHA256
    );
    for case in fixture.cases {
        let expected = decode_hex(&case.safe_json_hex)?;
        assert_eq!(safe_json(&case.value)?, expected, "case: {}", case.name);
    }
    Ok(())
}

#[derive(Default)]
struct ByteTokenizer {
    calls: RefCell<Vec<String>>,
}

impl SegmentTokenizer for ByteTokenizer {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        self.calls.borrow_mut().push(segment.to_owned());
        Ok(segment.bytes().map(|byte| 1000 + u32::from(byte)).collect())
    }
}

#[test]
fn segments_are_encoded_independently_and_only_explicit_framing_can_use_markers()
-> Result<(), Box<dyn Error>> {
    let tokenizer = ByteTokenizer::default();
    let policy = TokenPolicy::draft5()?;
    let dynamic = Value::String("<".to_owned());
    let stream = tokenize_segments(
        &tokenizer,
        &policy,
        [
            Segment::Framing(1),
            Segment::Text("A"),
            Segment::SafeJson(&dynamic),
        ],
    )?;

    assert_eq!(stream.segments.len(), 3);
    assert_eq!(stream.segments[0].start, 0);
    assert_eq!(stream.segments[0].end, 1);
    assert_eq!(stream.segments[1].start, 1);
    assert_eq!(stream.segments[1].end, 2);
    assert_eq!(stream.segments[2].start, 2);
    assert_eq!(stream.segments[2].end, stream.token_ids.len());
    assert_eq!(
        stream.segments[2].source.as_deref(),
        Some(br#""\u003c""#.as_slice())
    );
    assert_eq!(
        tokenizer.calls.into_inner(),
        vec!["A".to_owned(), r#""\u003c""#.to_owned()]
    );
    assert_eq!(stream.token_ids[0], 1);
    assert_eq!(stream.token_ids[1], 1065);
    Ok(())
}

struct MarkerTokenizer;

impl SegmentTokenizer for MarkerTokenizer {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, _segment: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(vec![6])
    }
}

#[test]
fn tokenizer_marker_in_a_payload_is_rejected() -> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let error = tokenize_segments(&MarkerTokenizer, &policy, [Segment::Text("plain")]);
    assert!(matches!(
        error,
        Err(ProtocolError::ForbiddenPayloadToken(6))
    ));
    Ok(())
}

#[test]
fn byte_map_uses_inverse_bytelevel_for_normal_and_utf8_for_added_tokens()
-> Result<(), Box<dyn Error>> {
    let map = TokenByteMap::new(
        vec![
            (101, "A".to_owned()),
            (102, "Ġ".to_owned()),
            (103, "Ã©".to_owned()),
        ],
        vec![(64_011, "Mathias".to_owned())],
    )?;
    assert_eq!(map.token_bytes(101)?, b"A");
    assert_eq!(map.token_bytes(102)?, b" ");
    assert_eq!(map.token_bytes(103)?, "é".as_bytes());
    assert_eq!(map.token_bytes(64_011)?, b"Mathias");
    assert_eq!(
        map.token_stream_bytes(&[101, 102, 103, 64_011])?,
        b"A \xc3\xa9Mathias"
    );
    assert!(matches!(
        TokenByteMap::new(vec![(1, "A".to_owned())], vec![(1, "added".to_owned())]),
        Err(ProtocolError::DuplicateTokenId(1))
    ));
    assert!(matches!(
        TokenByteMap::new(vec![(1, "\u{1f}".to_owned())], Vec::new()),
        Err(ProtocolError::InvalidByteLevel { token_id: 1, .. })
    ));
    Ok(())
}

fn decode_hex(encoded: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    if encoded.len() % 2 != 0 {
        return Err("hex fixture has an odd length".into());
    }
    let mut output = Vec::with_capacity(encoded.len() / 2);
    for index in (0..encoded.len()).step_by(2) {
        output.push(u8::from_str_radix(&encoded[index..index + 2], 16)?);
    }
    Ok(output)
}
