use minifield_decoding_protocol::{
    PublicEvent, RawJson, RoutingTraceInput, SegmentTokenizer, TokenId, TokenPolicy,
    TracePrefixBuilder, control_description,
};
use std::convert::Infallible;
use std::error::Error;

struct Tokens;
impl SegmentTokenizer for Tokens {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<TokenId>, Self::Error> {
        if text.contains("marker") {
            return Ok(vec![2]);
        }
        Ok(match text {
            "true" => vec![10_567],
            "false" => vec![13_890],
            "marker" => vec![2],
            _ => vec![
                text.bytes()
                    .fold(1_000u32, |id, byte| id.wrapping_add(u32::from(byte))),
            ],
        })
    }
}

#[test]
fn route_only_framing_preserves_absent_and_null_system_observations() -> Result<(), Box<dyn Error>>
{
    let policy = TokenPolicy::draft5()?;
    let builder = TracePrefixBuilder::new(&Tokens, &policy);
    let absent = [PublicEvent::System {
        policy: "p".to_owned(),
        observation: None,
    }];
    let present_null = [PublicEvent::System {
        policy: "p".to_owned(),
        observation: Some(RawJson::Null),
    }];
    let absent_context = builder.build_public_context(&absent)?;
    let null_context = builder.build_public_context(&present_null)?;
    assert_ne!(absent_context.token_ids, null_context.token_ids);
    assert!(
        absent_context
            .appends
            .iter()
            .any(|append| append.source.as_deref() == Some(br#"{"policy":"p","type":"system"}"#))
    );
    assert!(
        null_context
            .appends
            .iter()
            .any(|append| append.source.as_deref()
                == Some(br#"{"observation":null,"policy":"p","type":"system"}"#))
    );

    let route = builder.build_routing(&RoutingTraceInput {
        public_events: &absent,
        candidate_name: "$finish",
        candidate_description: control_description("$finish").ok_or("control description")?,
    })?;
    assert_eq!(route.candidates.len(), 2);
    assert_eq!(route.candidates[0].label, "true");
    assert_eq!(route.candidates[0].token_ids, vec![10_567, 7]);
    assert_eq!(route.candidates[1].label, "false");
    assert_eq!(route.candidates[1].token_ids, vec![13_890, 7]);
    let last_prefix = route.prefix_token_ids.len() - 1;
    assert_eq!(
        route.candidates[0].prediction_positions,
        vec![last_prefix, last_prefix + 1]
    );
    assert!(route.suffix_segments.iter().any(|append| {
        append.source.as_deref()
            == Some(b"\nShould this be the next action? Answer true or false.\n")
    }));
    Ok(())
}

struct MultiTokenLabels;
impl SegmentTokenizer for MultiTokenLabels {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(match text {
            "true" => vec![41_001, 41_002],
            "false" => vec![42_001, 42_002, 42_003],
            _ => vec![50_000],
        })
    }
}

#[test]
fn routing_keeps_injected_multitoken_label_segmentations() -> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let trace =
        TracePrefixBuilder::new(&MultiTokenLabels, &policy).build_routing(&RoutingTraceInput {
            public_events: &[],
            candidate_name: "synthetic",
            candidate_description: "available route",
        })?;
    assert_eq!(trace.candidates[0].label, "true");
    assert_eq!(trace.candidates[0].token_ids, vec![41_001, 41_002, 7]);
    assert_eq!(trace.candidates[1].label, "false");
    assert_eq!(
        trace.candidates[1].token_ids,
        vec![42_001, 42_002, 42_003, 7]
    );
    let prefix_end = trace.prefix_token_ids.len() - 1;
    assert_eq!(
        trace.candidates[1].prediction_positions,
        vec![prefix_end, prefix_end + 1, prefix_end + 2, prefix_end + 3]
    );
    Ok(())
}

#[test]
fn empty_route_description_is_serialized_but_empty_route_name_is_rejected()
-> Result<(), Box<dyn Error>> {
    let policy = TokenPolicy::draft5()?;
    let builder = TracePrefixBuilder::new(&Tokens, &policy);
    let trace = builder.build_routing(&RoutingTraceInput {
        public_events: &[],
        candidate_name: "synthetic",
        candidate_description: "",
    })?;
    assert!(trace.suffix_segments.iter().any(|append| {
        append.source.as_deref()
            == Some(br#"{"available":true,"description":"","name":"synthetic"}"#)
    }));
    let Err(error) = builder.build_routing(&RoutingTraceInput {
        public_events: &[],
        candidate_name: "",
        candidate_description: "",
    }) else {
        return Err("empty route name unexpectedly accepted".into());
    };
    assert!(error.to_string().contains("name must be nonempty"));
    Ok(())
}

#[test]
fn frozen_control_descriptions_and_payload_marker_rejection_are_explicit()
-> Result<(), Box<dyn Error>> {
    assert_eq!(
        control_description("$clarify"),
        Some(
            "Ask a focused question because information needed for the next supported action is missing."
        )
    );
    assert_eq!(
        control_description("$explain_permission"),
        Some(
            "Explain that the requested action requires authority, a grant, or human approval that is not available."
        )
    );
    assert_eq!(control_description("$unknown"), None);

    let policy = TokenPolicy::draft5()?;
    let builder = TracePrefixBuilder::new(&Tokens, &policy);
    let Err(error) = builder.build_routing(&RoutingTraceInput {
        public_events: &[],
        candidate_name: "marker",
        candidate_description: "description",
    }) else {
        return Err("payload marker unexpectedly accepted route metadata".into());
    };
    assert!(error.to_string().contains("forbidden protocol token"));
    Ok(())
}
