use minifield_decoding_protocol::{
    ForcedArrayLabel, ForcedArrayReason, PublicEvent, RawJsonLimits, SchemaLimits,
    SegmentTokenizer, TeacherOperation, TeacherTraceBuilder, TeacherTraceInput, TokenId,
    TokenPolicy, normalize_schema_document, parse_json_document,
};
use std::convert::Infallible;
use std::error::Error;

struct ByteTokenizer;
impl SegmentTokenizer for ByteTokenizer {
    type Error = Infallible;

    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<TokenId>, Self::Error> {
        Ok(text.bytes().map(|byte| u32::from(byte) + 1000).collect())
    }
}

fn build(
    schema: &[u8],
    arguments: &[u8],
) -> Result<minifield_decoding_protocol::TeacherTrace, Box<dyn Error>> {
    let plan = normalize_schema_document("operation-log", schema, SchemaLimits::default())?;
    let arguments = parse_json_document(arguments, RawJsonLimits::default())?;
    let policy = TokenPolicy::draft5()?;
    Ok(TeacherTraceBuilder::new(&ByteTokenizer, &policy).build(
        &TeacherTraceInput {
            public_events: &[PublicEvent::User {
                content: "array".to_owned(),
            }],
            selected_route: "synthetic",
            schema_plan: &plan,
        },
        &arguments,
    )?)
}

#[test]
fn operation_log_is_interleaved_and_forced_array_transitions_are_zero_loss()
-> Result<(), Box<dyn Error>> {
    let trace = build(
        br#"{"type":"array","minItems":1,"maxItems":2,"items":{"oneOf":[{"type":"string"},{"type":"integer"}]}}"#,
        br#"["x",2]"#,
    )?;
    assert_eq!(trace.forced_arrays.len(), 2);
    assert_eq!(trace.forced_arrays[0].path, "/0");
    assert_eq!(trace.forced_arrays[0].label, ForcedArrayLabel::Continue);
    assert_eq!(trace.forced_arrays[0].reason, ForcedArrayReason::MinItems);
    assert_eq!(trace.forced_arrays[1].path, "/2");
    assert_eq!(trace.forced_arrays[1].label, ForcedArrayLabel::Stop);
    assert_eq!(trace.forced_arrays[1].reason, ForcedArrayReason::MaxItems);
    assert!(
        trace
            .forced_arrays
            .iter()
            .all(|forced| forced.direct_loss_tokens == 0)
    );

    let mut previous_main = None;
    let mut probe_globals = Vec::new();
    for (global, operation) in trace.operation_log.iter().enumerate() {
        match *operation {
            TeacherOperation::MainAppend { main_append_index } => {
                assert_eq!(
                    previous_main.map_or(0, |index| index + 1),
                    main_append_index
                );
                previous_main = Some(main_append_index);
            }
            TeacherOperation::Probe { probe_index } => {
                assert_eq!(trace.probes[probe_index].global_operation_index, global);
                probe_globals.push(global);
            }
            TeacherOperation::FiniteChoice {
                finite_choice_index,
            } => {
                assert!(finite_choice_index < trace.finite_choices.len());
            }
            TeacherOperation::ForcedArray { forced_array_index } => {
                assert!(forced_array_index < trace.forced_arrays.len());
            }
        }
    }
    assert_eq!(
        previous_main.map_or(0, |index| index + 1),
        trace.main.appends.len()
    );
    assert_eq!(probe_globals.len(), trace.probes.len());
    assert!(trace.operation_log.iter().any(|operation| matches!(
        operation,
        TeacherOperation::ForcedArray {
            forced_array_index: 0
        }
    )));
    assert!(
        trace
            .operation_log
            .iter()
            .any(|operation| matches!(operation, TeacherOperation::Probe { .. }))
    );
    Ok(())
}

#[test]
fn exact_zero_array_forces_stop_without_a_probe() -> Result<(), Box<dyn Error>> {
    let trace = build(br#"{"type":"array","minItems":0,"maxItems":0}"#, br"[]")?;
    assert!(trace.probes.is_empty());
    assert_eq!(trace.forced_arrays.len(), 1);
    let forced = &trace.forced_arrays[0];
    assert_eq!(forced.path, "/0");
    assert_eq!(forced.label, ForcedArrayLabel::Stop);
    assert_eq!(forced.reason, ForcedArrayReason::MaxItems);
    assert_eq!(forced.direct_loss_tokens, 0);
    assert!(trace.operation_log.iter().any(|operation| matches!(
        operation,
        TeacherOperation::ForcedArray {
            forced_array_index: 0
        }
    )));
    Ok(())
}
