#![allow(clippy::unwrap_used)]
// The conversion is guarded by the test input's bounded UTF-8 length.
use minifield_decoding_protocol::{
    PublicEvent, RawJson, SchemaLimits, SegmentTokenizer, TeacherTraceInput, TokenId, TokenPolicy,
    TraceOwnership, TracePrefixBuilder, normalize_schema_document,
};
use std::convert::Infallible;
use std::error::Error;

struct Segments;
impl SegmentTokenizer for Segments {
    type Error = Infallible;
    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(vec![1000 + u32::try_from(text.len()).unwrap()])
    }
}

#[test]
fn typed_public_events_and_header_are_independently_segmented() -> Result<(), Box<dyn Error>> {
    let plan = normalize_schema_document(
        "trace",
        br#"{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}"#,
        SchemaLimits::default(),
    )?;
    let event = PublicEvent::System {
        policy: "Use <only> public data.".to_owned(),
        observation: RawJson::Object(vec![("state".to_owned(), RawJson::String("ok".to_owned()))]),
    };
    let policy = TokenPolicy::draft5()?;
    let builder = TracePrefixBuilder::new(&Segments, &policy);
    let mut trace = builder.build(&TeacherTraceInput {
        public_events: &[event],
        selected_route: "trace",
        schema_plan: &plan,
    })?;
    assert!(
        trace
            .appends
            .iter()
            .all(|append| append.main_start <= append.main_end)
    );
    assert!(
        trace
            .appends
            .windows(2)
            .all(|pair| pair[0].main_end == pair[1].main_start)
    );
    assert!(trace.appends.iter().any(|append| {
        append.source.as_deref()
            == Some(br#"{"observation":{"state":"ok"},"policy":"Use \u003conly\u003e public data.","type":"system"}"#)
    }));
    builder.append_value(
        &mut trace,
        &RawJson::String("v".to_owned()),
        TraceOwnership::Learned,
        "value:/x",
    )?;
    let learned = trace.appends.iter().rev().take(2).collect::<Vec<_>>();
    assert_eq!(learned[1].prediction_positions.len(), 1);
    assert_eq!(learned[0].prediction_positions.len(), 1);
    assert_eq!(
        learned[1].prediction_positions[0] + 1,
        learned[1].main_start
    );
    assert_eq!(
        learned[0].prediction_positions[0] + 1,
        learned[0].main_start
    );
    Ok(())
}
