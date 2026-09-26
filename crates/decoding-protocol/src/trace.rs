//! Generic independently segmented trace prefix construction.
use crate::{
    FramingIds, ProtocolError, PublicEvent, RawJson, SchemaPlan, SegmentTokenizer, TokenId,
    TokenPolicy, safe_json,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceOwnership {
    Fixed,
    Learned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MainAppend {
    pub source: Option<Vec<u8>>,
    pub special_id: Option<TokenId>,
    pub token_ids: Vec<TokenId>,
    pub ownership: TraceOwnership,
    pub label: String,
    pub main_start: usize,
    pub main_end: usize,
    pub prediction_positions: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct TracePrefix {
    pub token_ids: Vec<TokenId>,
    pub appends: Vec<MainAppend>,
    pub argument_start: usize,
}

pub struct TeacherTraceInput<'a> {
    pub public_events: &'a [PublicEvent],
    pub selected_route: &'a str,
    pub schema_plan: &'a SchemaPlan,
}

/// One caller-supplied available route. This layer exposes only public route
/// identity and description; it does not infer permission or route eligibility.
pub struct RoutingTraceInput<'a> {
    pub public_events: &'a [PublicEvent],
    pub candidate_name: &'a str,
    pub candidate_description: &'a str,
}

/// One complete true or false routing completion from a shared route prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingCandidate {
    pub label: String,
    pub appends: Vec<MainAppend>,
    pub token_ids: Vec<TokenId>,
    pub prediction_positions: Vec<usize>,
}

/// Independently segmented public route framing and both Boolean completions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingTrace {
    pub suffix_segments: Vec<MainAppend>,
    pub prefix_token_ids: Vec<TokenId>,
    pub candidates: Vec<RoutingCandidate>,
}

/// Return the exact frozen description for one control route.
#[must_use]
pub fn control_description(name: &str) -> Option<&'static str> {
    match name {
        "$clarify" => Some(
            "Ask a focused question because information needed for the next supported action is missing.",
        ),
        "$finish" => Some(
            "Conclude this turn and report the actual observed outcome, including failure, without another tool action.",
        ),
        "$reject_scope" => Some(
            "Explain that the requested task is outside this product interface supported scope.",
        ),
        "$explain_unavailable" => Some(
            "Explain that a supported action is unavailable because required current state or service is missing or invalid.",
        ),
        "$explain_permission" => Some(
            "Explain that the requested action requires authority, a grant, or human approval that is not available.",
        ),
        _ => None,
    }
}

pub struct TracePrefixBuilder<'a, T> {
    tokenizer: &'a T,
    policy: &'a TokenPolicy,
    framing: FramingIds,
}

impl<'a, T> TracePrefixBuilder<'a, T>
where
    T: SegmentTokenizer,
{
    pub fn new(tokenizer: &'a T, policy: &'a TokenPolicy) -> Self {
        Self {
            tokenizer,
            policy,
            framing: policy.framing_ids(),
        }
    }

    pub fn build(&self, input: &TeacherTraceInput<'_>) -> crate::Result<TracePrefix> {
        let mut trace = self.build_public_context(input.public_events)?;
        trace.argument_start = trace.token_ids.len();
        self.special(
            &mut trace,
            self.framing.message_start,
            TraceOwnership::Fixed,
            "argument-header",
        )?;
        self.text(
            &mut trace,
            "assistant\n",
            TraceOwnership::Fixed,
            "argument-header",
        )?;
        self.text(
            &mut trace,
            "minifield.arguments/1\n",
            TraceOwnership::Fixed,
            "argument-header",
        )?;
        let metadata = serde_json::json!({
            "name": input.selected_route,
            "parameters": input.schema_plan.original.clone().into_value()?,
            "property_order": serde_json::to_value(&input.schema_plan.property_order)
                .map_err(|error| ProtocolError::InvalidJson(error.to_string()))?,
        });
        self.value(
            &mut trace,
            &metadata,
            TraceOwnership::Fixed,
            "argument-header",
        )?;
        self.text(&mut trace, "\n", TraceOwnership::Fixed, "argument-header")?;
        self.text(&mut trace, "{\"name\":", TraceOwnership::Fixed, "envelope")?;
        self.value(
            &mut trace,
            &Value::String(input.selected_route.to_owned()),
            TraceOwnership::Fixed,
            "envelope",
        )?;
        self.text(
            &mut trace,
            ",\"arguments\":",
            TraceOwnership::Fixed,
            "envelope",
        )?;
        Ok(trace)
    }

    pub fn build_public_context(
        &self,
        public_events: &[PublicEvent],
    ) -> crate::Result<TracePrefix> {
        let mut trace = TracePrefix::default();
        self.special(&mut trace, self.framing.bos, TraceOwnership::Fixed, "bos")?;
        for (index, event) in public_events.iter().enumerate() {
            let label = format!("event:{index}");
            self.special(
                &mut trace,
                self.framing.message_start,
                TraceOwnership::Fixed,
                &label,
            )?;
            self.text(
                &mut trace,
                &format!("{}\n", event.role()),
                TraceOwnership::Fixed,
                &label,
            )?;
            self.raw(&mut trace, &event.as_raw(), TraceOwnership::Fixed, &label)?;
            self.special(
                &mut trace,
                self.framing.message_end,
                TraceOwnership::Fixed,
                &label,
            )?;
            self.text(&mut trace, "\n", TraceOwnership::Fixed, &label)?;
        }
        Ok(trace)
    }

    /// Build a route-only public prefix. It accepts no schema or teacher
    /// arguments and always represents a caller-supplied available candidate.
    pub fn build_routing(&self, input: &RoutingTraceInput<'_>) -> crate::Result<RoutingTrace> {
        if input.candidate_name.is_empty() {
            return Err(ProtocolError::Schema(
                "routing candidate name must be nonempty".to_owned(),
            ));
        }
        let mut prefix = self.build_public_context(input.public_events)?;
        let suffix_start = prefix.appends.len();
        self.special(
            &mut prefix,
            self.framing.message_start,
            TraceOwnership::Fixed,
            "route-header",
        )?;
        self.text(
            &mut prefix,
            "assistant\n",
            TraceOwnership::Fixed,
            "route-header",
        )?;
        self.text(
            &mut prefix,
            "minifield.route/1\n",
            TraceOwnership::Fixed,
            "route-header",
        )?;
        self.value(
            &mut prefix,
            &serde_json::json!({
                "name": input.candidate_name,
                "description": input.candidate_description,
                "available": true,
            }),
            TraceOwnership::Fixed,
            "route-header",
        )?;
        self.text(
            &mut prefix,
            "\nShould this be the next action? Answer true or false.\n",
            TraceOwnership::Fixed,
            "route-question",
        )?;
        let prefix_token_ids = prefix.token_ids.clone();
        let suffix_segments = prefix.appends[suffix_start..].to_vec();
        let candidates = ["true", "false"]
            .into_iter()
            .map(|label| self.routing_candidate(&prefix_token_ids, label))
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(RoutingTrace {
            suffix_segments,
            prefix_token_ids,
            candidates,
        })
    }

    fn routing_candidate(
        &self,
        prefix_token_ids: &[TokenId],
        label: &str,
    ) -> crate::Result<RoutingCandidate> {
        let mut completion = TracePrefix {
            token_ids: prefix_token_ids.to_vec(),
            appends: Vec::new(),
            argument_start: prefix_token_ids.len(),
        };
        self.text(
            &mut completion,
            label,
            TraceOwnership::Learned,
            "route-candidate",
        )?;
        self.special(
            &mut completion,
            self.framing.value_end,
            TraceOwnership::Learned,
            "route-candidate-end",
        )?;
        let prediction_positions = completion
            .appends
            .iter()
            .flat_map(|append| append.prediction_positions.iter().copied())
            .collect();
        Ok(RoutingCandidate {
            label: label.to_owned(),
            appends: completion.appends,
            token_ids: completion.token_ids[prefix_token_ids.len()..].to_vec(),
            prediction_positions,
        })
    }

    /// Append one independently tokenized SafeJSON value without a terminator.
    /// Structural object keys use this boundary; values use append_value.
    pub fn append_json(
        &self,
        trace: &mut TracePrefix,
        value: &RawJson,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.raw(trace, value, ownership, label)
    }

    pub fn append_value(
        &self,
        trace: &mut TracePrefix,
        value: &RawJson,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.raw(trace, value, ownership, label)?;
        self.special(
            trace,
            self.framing.value_end,
            ownership,
            &format!("{label}-end"),
        )
    }

    pub fn append_fixed(
        &self,
        trace: &mut TracePrefix,
        text: &str,
        label: &str,
    ) -> crate::Result<()> {
        self.text(trace, text, TraceOwnership::Fixed, label)
    }

    fn raw(
        &self,
        trace: &mut TracePrefix,
        value: &RawJson,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.value(trace, &value.clone().into_value()?, ownership, label)
    }

    fn value(
        &self,
        trace: &mut TracePrefix,
        value: &Value,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.append(trace, Some(safe_json(value)?), None, ownership, label)
    }

    fn text(
        &self,
        trace: &mut TracePrefix,
        text: &str,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.append(
            trace,
            Some(text.as_bytes().to_vec()),
            None,
            ownership,
            label,
        )
    }

    fn special(
        &self,
        trace: &mut TracePrefix,
        id: TokenId,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        self.policy.validate_framing(id)?;
        self.append(trace, None, Some(id), ownership, label)
    }

    fn append(
        &self,
        trace: &mut TracePrefix,
        source: Option<Vec<u8>>,
        special_id: Option<TokenId>,
        ownership: TraceOwnership,
        label: &str,
    ) -> crate::Result<()> {
        let start = trace.token_ids.len();
        let token_ids = match (&source, special_id) {
            (Some(source), None) => {
                let text =
                    std::str::from_utf8(source).map_err(|_| ProtocolError::InvalidCanonicalUtf8)?;
                let ids = self
                    .tokenizer
                    .encode_without_special_tokens(text)
                    .map_err(|error| ProtocolError::Tokenizer(error.to_string()))?;
                self.policy.validate_payload(&ids)?;
                ids
            }
            (None, Some(id)) => vec![id],
            _ => {
                return Err(ProtocolError::Tokenizer(
                    "invalid trace append boundary".to_owned(),
                ));
            }
        };
        let end = start
            .checked_add(token_ids.len())
            .ok_or(ProtocolError::TokenLengthOverflow)?;
        let prediction_positions = if ownership == TraceOwnership::Learned {
            (0..token_ids.len())
                .map(|offset| {
                    start
                        .checked_add(offset)
                        .and_then(|position| position.checked_sub(1))
                        .ok_or(ProtocolError::TokenLengthOverflow)
                })
                .collect::<crate::Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        trace.token_ids.extend_from_slice(&token_ids);
        trace.appends.push(MainAppend {
            source,
            special_id,
            token_ids,
            ownership,
            label: label.to_owned(),
            main_start: start,
            main_end: end,
            prediction_positions,
        });
        Ok(())
    }
}
