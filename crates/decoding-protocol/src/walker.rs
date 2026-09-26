//! Pure schema-driven teacher traces. This module owns no tokenizer assets,
//! model state, filesystem, or product execution.
use crate::{
    MainAppend, ProtocolError, RawJson, RawJsonLimits, SchemaPlan, SegmentTokenizer,
    TeacherTraceInput, TeacherUnionChoice, TokenByteMap, TokenId, TokenPolicy, TraceOwnership,
    TracePrefix, TracePrefixBuilder, ValidationFailureKind, plan_teacher, safe_json,
    semantic_equal,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeOperation {
    Presence,
    Array,
    Union,
}
impl ProbeOperation {
    #[must_use]
    pub const fn wire(self) -> &'static str {
        match self {
            Self::Presence => "presence",
            Self::Array => "array",
            Self::Union => "union",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchAppend {
    pub source: Option<Vec<u8>>,
    pub special_id: Option<TokenId>,
    pub token_ids: Vec<TokenId>,
    pub ownership: TraceOwnership,
    pub branch_start: usize,
    pub branch_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbeCandidate {
    pub label: String,
    pub segments: Vec<BranchAppend>,
    pub token_ids: Vec<TokenId>,
    pub prediction_positions: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbeTrace {
    pub operation: ProbeOperation,
    pub path: String,
    pub fork_main_length: usize,
    pub fork_main_sha256_u32le: String,
    pub prefix_token_ids: Vec<TokenId>,
    pub suffix_segments: Vec<BranchAppend>,
    pub candidates: Vec<ProbeCandidate>,
    pub selected_index: usize,
    /// Index of this probe in the authoritative interleaved operation log.
    pub global_operation_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteCandidate {
    pub value: RawJson,
    pub segments: Vec<BranchAppend>,
    pub token_ids: Vec<TokenId>,
    pub prediction_positions: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteChoice {
    pub path: String,
    pub fork_main_length: usize,
    pub fork_main_sha256_u32le: String,
    pub candidates: Vec<FiniteCandidate>,
    pub selected_index: usize,
}

/// One structural array transition that has no model target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcedArrayLabel {
    Continue,
    Stop,
}
impl ForcedArrayLabel {
    #[must_use]
    pub const fn wire(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Stop => "stop",
        }
    }
}

/// Why an array transition was fixed by its structural bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcedArrayReason {
    MinItems,
    MaxItems,
}
impl ForcedArrayReason {
    #[must_use]
    pub const fn wire(self) -> &'static str {
        match self {
            Self::MinItems => "minItems",
            Self::MaxItems => "maxItems",
        }
    }
}

/// One zero-loss array transition fixed by a minimum or maximum item bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedArray {
    pub path: String,
    pub label: ForcedArrayLabel,
    pub reason: ForcedArrayReason,
    pub main_length: usize,
    pub direct_loss_tokens: usize,
}

/// One action in the trace's authoritative temporal order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TeacherOperation {
    MainAppend { main_append_index: usize },
    Probe { probe_index: usize },
    FiniteChoice { finite_choice_index: usize },
    ForcedArray { forced_array_index: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeacherTrace {
    pub main: TracePrefix,
    pub probes: Vec<ProbeTrace>,
    pub finite_choices: Vec<FiniteChoice>,
    pub forced_arrays: Vec<ForcedArray>,
    /// Every main append and structural decision, recorded in occurrence order.
    pub operation_log: Vec<TeacherOperation>,
    pub learned_token_count: usize,
    recorded_main_append_count: usize,
}

/// A runtime-generated primitive that must retain its committed lexical
/// bytes and token IDs. The semantic value is still checked against the typed
/// teacher arguments before it enters the main sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TeacherLexicalValue {
    pub path: String,
    pub source: Vec<u8>,
    pub token_ids: Vec<TokenId>,
}

pub struct TeacherTraceBuilder<'a, T> {
    tokenizer: &'a T,
    policy: &'a TokenPolicy,
    token_bytes: Option<&'a TokenByteMap>,
}

impl<'a, T> TeacherTraceBuilder<'a, T>
where
    T: SegmentTokenizer,
{
    pub const fn new(tokenizer: &'a T, policy: &'a TokenPolicy) -> Self {
        Self {
            tokenizer,
            policy,
            token_bytes: None,
        }
    }

    /// Bind the caller's pinned token-byte inventory. Required whenever a
    /// teacher trace retains committed runtime lexical token IDs.
    pub const fn with_token_bytes(
        tokenizer: &'a T,
        policy: &'a TokenPolicy,
        token_bytes: &'a TokenByteMap,
    ) -> Self {
        Self {
            tokenizer,
            policy,
            token_bytes: Some(token_bytes),
        }
    }

    pub fn build(
        &self,
        input: &TeacherTraceInput<'_>,
        arguments: &RawJson,
    ) -> crate::Result<TeacherTrace> {
        self.build_with_lexical(input, arguments, &[])
    }

    /// Build a teacher trace while retaining caller-supplied committed runtime
    /// lexical primitive segments. These are never canonicalized or retokenized.
    pub fn build_with_lexical(
        &self,
        input: &TeacherTraceInput<'_>,
        arguments: &RawJson,
        lexical_values: &[TeacherLexicalValue],
    ) -> crate::Result<TeacherTrace> {
        let prefix_builder = TracePrefixBuilder::new(self.tokenizer, self.policy);
        let main = prefix_builder.build(input)?;
        let teacher = plan_teacher(input.schema_plan, arguments)?;
        let mut unions = planned_unions(&teacher);
        let mut lexical = self.prepare_lexical(arguments, lexical_values)?;
        let mut trace = TeacherTrace {
            main,
            probes: Vec::new(),
            finite_choices: Vec::new(),
            forced_arrays: Vec::new(),
            operation_log: Vec::new(),
            learned_token_count: 0,
            recorded_main_append_count: 0,
        };
        record_main_appends(&mut trace);
        let root = flatten_constraints(&teacher.effective, &teacher.effective.schema, "");
        let mut walk = Walk {
            prefix_builder: &prefix_builder,
            tokenizer: self.tokenizer,
            policy: self.policy,
            plan: input.schema_plan,
            arguments,
            effective: &teacher.effective,
            trace: &mut trace,
            unions: &mut unions,
            lexical: &mut lexical,
        };
        walk.node("", &root, arguments)?;
        prefix_builder.append_fixed(&mut trace.main, "}", "envelope-close")?;
        record_main_appends(&mut trace);
        finish_trace(&mut trace, unions, lexical)?;
        Ok(trace)
    }

    fn prepare_lexical(
        &self,
        arguments: &RawJson,
        lexical_values: &[TeacherLexicalValue],
    ) -> crate::Result<BTreeMap<String, TeacherLexicalValue>> {
        let token_bytes = lexical_values
            .first()
            .map(|_| {
                self.token_bytes.ok_or_else(|| {
                    ProtocolError::Schema(
                        "committed lexical values require an injected pinned token-byte map"
                            .to_owned(),
                    )
                })
            })
            .transpose()?;
        let mut lexical = BTreeMap::new();
        for lexical_value in lexical_values {
            verify_committed_lexical_bytes(
                token_bytes.expect("nonempty lexical values require token bytes"),
                lexical_value,
            )?;
            let semantic = crate::parse_runtime_value(&lexical_value.source)?;
            let target = value_at(arguments, &lexical_value.path)?;
            if !semantic_equal(&semantic, target)
                .map_err(|error| ProtocolError::Schema(error.to_string()))?
            {
                return Err(ProtocolError::Schema(format!(
                    "committed lexical value at {} differs from typed teacher arguments",
                    lexical_value.path
                )));
            }
            self.policy.validate_payload(&lexical_value.token_ids)?;
            if lexical
                .insert(lexical_value.path.clone(), lexical_value.clone())
                .is_some()
            {
                return Err(ProtocolError::Schema(format!(
                    "duplicate committed lexical value at {}",
                    lexical_value.path
                )));
            }
        }
        Ok(lexical)
    }
}

fn planned_unions(teacher: &crate::TeacherPlan) -> BTreeMap<String, Vec<PlannedUnion>> {
    let instance_union_ordinals = teacher
        .instance_union_ordinals
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut unions = BTreeMap::<String, Vec<PlannedUnion>>::new();
    for (ordinal, choice) in teacher.unions.iter().cloned().enumerate() {
        unions
            .entry(choice.argument_path.clone())
            .or_default()
            .push(PlannedUnion {
                choice,
                instance_local: instance_union_ordinals.contains(&ordinal),
            });
    }
    unions
}

/// Add all main appends created since the last structural action. Callers invoke
/// this at each action boundary, so the log preserves creation order without
/// deriving order from offsets after trace construction.
fn record_main_appends(trace: &mut TeacherTrace) {
    for main_append_index in trace.recorded_main_append_count..trace.main.appends.len() {
        trace
            .operation_log
            .push(TeacherOperation::MainAppend { main_append_index });
    }
    trace.recorded_main_append_count = trace.main.appends.len();
}

fn finish_trace(
    trace: &mut TeacherTrace,
    unions: BTreeMap<String, Vec<PlannedUnion>>,
    lexical: BTreeMap<String, TeacherLexicalValue>,
) -> crate::Result<()> {
    if let Some((path, _)) = unions.into_iter().next() {
        return Err(ProtocolError::Schema(format!(
            "schema choice at argument path {path} was not reached by structural planning"
        )));
    }
    if let Some((path, _)) = lexical.into_iter().next() {
        return Err(ProtocolError::Schema(format!(
            "committed lexical value at {path} was not reached by structural planning"
        )));
    }
    validate_operation_log(trace)?;
    let main_learned = trace
        .main
        .appends
        .iter()
        .map(|append| append.prediction_positions.len())
        .sum::<usize>();
    let probe_learned = trace
        .probes
        .iter()
        .map(|probe| {
            probe.candidates[probe.selected_index]
                .prediction_positions
                .len()
        })
        .sum::<usize>();
    trace.learned_token_count = main_learned
        .checked_add(probe_learned)
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    Ok(())
}

fn validate_operation_log(trace: &TeacherTrace) -> crate::Result<()> {
    let mut expected_main = 0usize;
    let mut expected_probe = 0usize;
    let mut expected_finite = 0usize;
    let mut expected_forced = 0usize;
    for (global_operation_index, operation) in trace.operation_log.iter().enumerate() {
        match *operation {
            TeacherOperation::MainAppend { main_append_index } => {
                if main_append_index != expected_main
                    || trace.main.appends.get(main_append_index).is_none()
                {
                    return Err(ProtocolError::Schema(
                        "operation log has an invalid main append reference".to_owned(),
                    ));
                }
                expected_main += 1;
            }
            TeacherOperation::Probe { probe_index } => {
                let Some(probe) = trace.probes.get(probe_index) else {
                    return Err(ProtocolError::Schema(
                        "operation log has an invalid probe reference".to_owned(),
                    ));
                };
                if probe_index != expected_probe
                    || probe.global_operation_index != global_operation_index
                {
                    return Err(ProtocolError::Schema(
                        "operation log has an out-of-order probe reference".to_owned(),
                    ));
                }
                expected_probe += 1;
            }
            TeacherOperation::FiniteChoice {
                finite_choice_index,
            } => {
                if finite_choice_index != expected_finite
                    || trace.finite_choices.get(finite_choice_index).is_none()
                {
                    return Err(ProtocolError::Schema(
                        "operation log has an invalid finite-choice reference".to_owned(),
                    ));
                }
                expected_finite += 1;
            }
            TeacherOperation::ForcedArray { forced_array_index } => {
                if forced_array_index != expected_forced
                    || trace.forced_arrays.get(forced_array_index).is_none()
                {
                    return Err(ProtocolError::Schema(
                        "operation log has an invalid forced-array reference".to_owned(),
                    ));
                }
                expected_forced += 1;
            }
        }
    }
    if expected_main != trace.main.appends.len()
        || expected_probe != trace.probes.len()
        || expected_finite != trace.finite_choices.len()
        || expected_forced != trace.forced_arrays.len()
        || trace.recorded_main_append_count != trace.main.appends.len()
    {
        return Err(ProtocolError::Schema(
            "operation log does not cover every trace record exactly once".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone)]
struct NodeConstraint {
    schema: RawJson,
    current_path: String,
    source_path: String,
}

/// Apply the same fixed source bound used by strict runtime-value parsing.
fn ensure_committed_lexical_source_limit(lexical_value: &TeacherLexicalValue) -> crate::Result<()> {
    let max_bytes = RawJsonLimits::draft5_value().max_bytes;
    if lexical_value.source.len() > max_bytes {
        return Err(ProtocolError::InputLimit(format!(
            "committed lexical source at {} exceeds the {max_bytes}-byte runtime value limit",
            lexical_value.path
        )));
    }
    Ok(())
}
/// Compare committed token pieces directly against the caller-owned lexical
/// source. This enforces byte ownership without allocating a reconstructed
/// stream whose size is controlled by the token inventory.
fn verify_committed_lexical_bytes(
    token_bytes: &TokenByteMap,
    lexical_value: &TeacherLexicalValue,
) -> crate::Result<()> {
    ensure_committed_lexical_source_limit(lexical_value)?;
    let mut offset = 0usize;
    for token_id in &lexical_value.token_ids {
        let piece = token_bytes.token_bytes(*token_id)?;
        let end = offset.checked_add(piece.len()).ok_or_else(|| {
            ProtocolError::InputLimit(format!(
                "committed lexical token length overflows at {}",
                lexical_value.path
            ))
        })?;
        if end > lexical_value.source.len() || lexical_value.source.get(offset..end) != Some(piece)
        {
            return Err(ProtocolError::Schema(format!(
                "committed token IDs at {} do not decode to their supplied lexical bytes",
                lexical_value.path
            )));
        }
        offset = end;
    }
    if offset != lexical_value.source.len() {
        return Err(ProtocolError::Schema(format!(
            "committed token IDs at {} do not decode to their supplied lexical bytes",
            lexical_value.path
        )));
    }
    Ok(())
}
struct PlannedUnion {
    choice: TeacherUnionChoice,
    instance_local: bool,
}

struct Walk<'a, 'b, T> {
    prefix_builder: &'a TracePrefixBuilder<'a, T>,
    tokenizer: &'a T,
    policy: &'a TokenPolicy,
    plan: &'a SchemaPlan,
    arguments: &'a RawJson,
    effective: &'a crate::EffectiveTree,
    trace: &'b mut TeacherTrace,
    unions: &'b mut BTreeMap<String, Vec<PlannedUnion>>,
    lexical: &'b mut BTreeMap<String, TeacherLexicalValue>,
}

impl<T> Walk<'_, '_, T>
where
    T: SegmentTokenizer,
{
    fn node(
        &mut self,
        path: &str,
        constraints: &[NodeConstraint],
        value: &RawJson,
    ) -> crate::Result<()> {
        let mut selected_constraints = constraints.to_vec();
        if let Some(choices) = self.unions.remove(path) {
            // The planner only retains same-node choices when no source finite
            // domain consumed the node. Emit those choices before a selected
            // branch's finite domain, matching the two-stage Python dispatch.
            for planned in choices {
                self.emit_union(path, &planned.choice)?;
                if planned.instance_local {
                    selected_constraints =
                        selected_instance_constraints(&selected_constraints, &planned.choice)?;
                }
            }
            if let Some(domain) = self.finite_domain(path, &selected_constraints, value)? {
                return self.consume_finite(path, domain, value);
            }
        }
        if let Some(domain) = self.finite_domain(path, constraints, value)? {
            return self.consume_finite(path, domain, value);
        }
        if wholly_dynamic_constraints(&selected_constraints)? {
            return self.generated_value(path, value, "dynamic-value");
        }
        match value {
            RawJson::Object(entries) => self.object(path, &selected_constraints, entries),
            RawJson::Array(values) => self.array(path, &selected_constraints, values),
            RawJson::Bool(value) => self.boolean(path, &selected_constraints, *value),
            RawJson::Null if is_null_only(&selected_constraints) => {
                self.prefix_builder.append_value(
                    &mut self.trace.main,
                    value,
                    TraceOwnership::Fixed,
                    "fixed-null",
                )
            }
            RawJson::Null | RawJson::Number(_) | RawJson::String(_) => {
                self.generated_value(path, value, "value")
            }
        }
    }

    fn consume_finite(
        &mut self,
        path: &str,
        domain: Vec<RawJson>,
        selected_value: &RawJson,
    ) -> crate::Result<()> {
        if domain.len() == 1 {
            return self.prefix_builder.append_value(
                &mut self.trace.main,
                &domain[0],
                TraceOwnership::Fixed,
                "fixed-finite",
            );
        }
        self.emit_finite(path, domain, selected_value)
    }

    fn generated_value(&mut self, path: &str, value: &RawJson, label: &str) -> crate::Result<()> {
        if let Some(lexical) = self.lexical.remove(path) {
            return append_committed_lexical(&mut self.trace.main, self.policy, lexical, label);
        }
        self.prefix_builder.append_value(
            &mut self.trace.main,
            value,
            TraceOwnership::Learned,
            label,
        )
    }

    fn emit_union(&mut self, path: &str, choice: &TeacherUnionChoice) -> crate::Result<()> {
        let alternatives = choice
            .alternatives
            .iter()
            .map(|alternative| {
                let schema = alternative.schema.clone().into_value()?;
                Ok(serde_json::json!({
                    "label": alternative.label,
                    "schema_path": alternative.schema_path,
                    "schema": schema,
                }))
            })
            .collect::<crate::Result<Vec<Value>>>()?;
        self.emit_probe(
            ProbeOperation::Union,
            path,
            alternatives,
            choice
                .alternatives
                .iter()
                .map(|alternative| alternative.label.clone())
                .collect(),
            choice.selected_index,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn emit_probe(
        &mut self,
        operation: ProbeOperation,
        path: &str,
        alternatives: Vec<Value>,
        labels: Vec<String>,
        selected_index: usize,
    ) -> crate::Result<()> {
        if selected_index >= labels.len() || labels.len() != alternatives.len() {
            return Err(ProtocolError::Schema(
                "probe selection is outside its alternatives".to_owned(),
            ));
        }
        record_main_appends(self.trace);
        let global_operation_index = self.trace.operation_log.len();
        let fork_main_length = self.trace.main.token_ids.len();
        let prefix_token_ids = self.trace.main.token_ids.clone();
        let fork_main_sha256_u32le = token_hash(&prefix_token_ids);
        let metadata = serde_json::json!({
            "operation": operation.wire(),
            "path": path,
            "alternatives": alternatives,
        });
        let mut suffix_segments = Vec::new();
        let mut branch_length = fork_main_length;
        append_branch_text(
            self.tokenizer,
            self.policy,
            &mut suffix_segments,
            &mut branch_length,
            "\nminifield.choice/1\n",
        )?;
        append_branch_bytes(
            self.tokenizer,
            self.policy,
            &mut suffix_segments,
            &mut branch_length,
            safe_json(&metadata)?,
        )?;
        append_branch_text(
            self.tokenizer,
            self.policy,
            &mut suffix_segments,
            &mut branch_length,
            "\nAnswer:\n",
        )?;
        let candidates = labels
            .into_iter()
            .map(|label| candidate_text(self.tokenizer, self.policy, branch_length, label))
            .collect::<crate::Result<Vec<_>>>()?;
        self.trace.probes.push(ProbeTrace {
            operation,
            path: path.to_owned(),
            fork_main_length,
            fork_main_sha256_u32le,
            prefix_token_ids,
            suffix_segments,
            candidates,
            selected_index,
            global_operation_index,
        });
        let probe_index = self.trace.probes.len() - 1;
        self.trace
            .operation_log
            .push(TeacherOperation::Probe { probe_index });
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn emit_finite(
        &mut self,
        path: &str,
        values: Vec<RawJson>,
        selected_value: &RawJson,
    ) -> crate::Result<()> {
        record_main_appends(self.trace);
        let fork_main_length = self.trace.main.token_ids.len();
        let fork_main_sha256_u32le = token_hash(&self.trace.main.token_ids);
        let candidates = values
            .iter()
            .cloned()
            .map(|value| candidate_value(self.tokenizer, self.policy, fork_main_length, value))
            .collect::<crate::Result<Vec<_>>>()?;
        let mut selected_index = None;
        for (index, value) in values.iter().enumerate() {
            if semantic(value, selected_value)? {
                selected_index = Some(index);
                break;
            }
        }
        let selected_index = selected_index.ok_or_else(|| {
            ProtocolError::Schema("teacher finite value is outside its domain".to_owned())
        })?;
        self.trace.finite_choices.push(FiniteChoice {
            path: path.to_owned(),
            fork_main_length,
            fork_main_sha256_u32le,
            candidates,
            selected_index,
        });
        let finite_choice_index = self.trace.finite_choices.len() - 1;
        self.trace
            .operation_log
            .push(TeacherOperation::FiniteChoice {
                finite_choice_index,
            });
        self.prefix_builder.append_value(
            &mut self.trace.main,
            selected_value,
            TraceOwnership::Learned,
            "finite-value",
        )
    }

    fn emit_forced_array(
        &mut self,
        path: &str,
        label: ForcedArrayLabel,
        reason: ForcedArrayReason,
    ) {
        record_main_appends(self.trace);
        self.trace.forced_arrays.push(ForcedArray {
            path: path.to_owned(),
            label,
            reason,
            main_length: self.trace.main.token_ids.len(),
            direct_loss_tokens: 0,
        });
        let forced_array_index = self.trace.forced_arrays.len() - 1;
        self.trace
            .operation_log
            .push(TeacherOperation::ForcedArray { forced_array_index });
    }

    fn finite_domain(
        &self,
        path: &str,
        constraints: &[NodeConstraint],
        value: &RawJson,
    ) -> crate::Result<Option<Vec<RawJson>>> {
        let mut domain: Option<Vec<RawJson>> = None;
        for constraint in ordered(constraints) {
            let entries = object_entries(&constraint.schema)?;
            let candidates = match field(entries, "const") {
                Some(value) => Some(vec![value.clone()]),
                None => match field(entries, "enum") {
                    Some(RawJson::Array(values)) => Some(values.clone()),
                    Some(_) => {
                        return Err(ProtocolError::Schema(format!(
                            "enum at {} is not an array",
                            constraint.source_path
                        )));
                    }
                    None => None,
                },
            };
            let Some(candidates) = candidates else {
                continue;
            };
            domain = Some(match domain {
                None => dedupe(candidates)?,
                Some(existing) => {
                    let mut intersection = Vec::new();
                    for candidate in existing {
                        if candidates.iter().try_fold(false, |matched, other| {
                            Ok::<bool, ProtocolError>(matched || semantic(&candidate, other)?)
                        })? {
                            intersection.push(candidate);
                        }
                    }
                    intersection
                }
            });
        }
        let Some(domain) = domain else {
            return Ok(None);
        };
        let mut allowed = Vec::new();
        for candidate in domain {
            if self.candidate_valid(path, &candidate)? {
                allowed.push(candidate);
            }
        }
        if allowed.is_empty() {
            return Err(ProtocolError::Schema(format!(
                "finite domain at {path} has no value valid under the complete original schema"
            )));
        }
        let mut teacher_in_domain = false;
        for candidate in &allowed {
            if semantic(candidate, value)? {
                teacher_in_domain = true;
                break;
            }
        }
        if !teacher_in_domain {
            return Err(ProtocolError::Schema(format!(
                "teacher value at {path} is outside its finite domain"
            )));
        }
        Ok(Some(allowed))
    }

    fn candidate_valid(&self, path: &str, candidate: &RawJson) -> crate::Result<bool> {
        let mut instance = self.arguments.clone();
        replace_at(&mut instance, path, candidate.clone())?;
        match self.plan.validate_original_instance(&instance) {
            Ok(()) => Ok(true),
            Err(failure) if failure.class == ValidationFailureKind::InstanceInvalid => Ok(false),
            Err(failure) => Err(ProtocolError::Schema(format!(
                "candidate validation at {path} failed operationally: {failure}"
            ))),
        }
    }

    fn boolean(
        &mut self,
        path: &str,
        constraints: &[NodeConstraint],
        value: bool,
    ) -> crate::Result<()> {
        if !boolean_restricted(constraints) {
            return self.prefix_builder.append_value(
                &mut self.trace.main,
                &RawJson::Bool(value),
                TraceOwnership::Learned,
                "dynamic-boolean",
            );
        }
        let mut values = Vec::new();
        for candidate in [false, true] {
            let candidate = RawJson::Bool(candidate);
            if self.candidate_valid(path, &candidate)? {
                values.push(candidate);
            }
        }
        if values.is_empty() {
            return Err(ProtocolError::Schema(format!(
                "Boolean domain at {path} has no valid values"
            )));
        }
        if values.len() == 1 {
            return self.prefix_builder.append_value(
                &mut self.trace.main,
                &values[0],
                TraceOwnership::Fixed,
                "fixed-boolean",
            );
        }
        self.emit_finite(path, values, &RawJson::Bool(value))
    }

    fn object(
        &mut self,
        path: &str,
        constraints: &[NodeConstraint],
        values: &[(String, RawJson)],
    ) -> crate::Result<()> {
        let declarations = property_names(constraints)?;
        let open = allows_unknown_properties(constraints)?;
        if declarations.is_empty() {
            if open {
                return self.prefix_builder.append_value(
                    &mut self.trace.main,
                    &RawJson::Object(values.to_vec()),
                    TraceOwnership::Learned,
                    "dynamic-object",
                );
            }
            if !values.is_empty() {
                return Err(ProtocolError::Schema(format!(
                    "closed object at {path} has an undeclared property"
                )));
            }
            self.prefix_builder
                .append_fixed(&mut self.trace.main, "{", "object-open")?;
            return self
                .prefix_builder
                .append_fixed(&mut self.trace.main, "}", "object-close");
        }
        if open {
            return Err(ProtocolError::Schema(format!(
                "mixed_declared_and_open_properties at {path}"
            )));
        }
        self.prefix_builder
            .append_fixed(&mut self.trace.main, "{", "object-open")?;
        let required = required_names(constraints)?;
        let mut emitted = false;
        for name in &declarations {
            let present = values.iter().find(|(key, _)| key == name);
            let allowed = property_allowed(constraints, name)?;
            if required.contains(name) && !allowed {
                return Err(ProtocolError::Schema(format!(
                    "required property {name:?} is forbidden at {path}"
                )));
            }
            if !allowed {
                if present.is_some() {
                    return Err(ProtocolError::Schema(format!(
                        "forbidden property {name:?} is present at {path}"
                    )));
                }
                continue;
            }
            if !required.contains(name) {
                let selected = usize::from(present.is_none());
                self.emit_probe(
                    ProbeOperation::Presence,
                    &append(path, name),
                    vec![
                        serde_json::json!({"label": "present"}),
                        serde_json::json!({"label": "absent"}),
                    ],
                    vec!["present".to_owned(), "absent".to_owned()],
                    selected,
                )?;
            }
            let Some((_, child)) = present else {
                continue;
            };
            if emitted {
                self.prefix_builder
                    .append_fixed(&mut self.trace.main, ",", "object-comma")?;
            }
            emitted = true;
            self.prefix_builder.append_json(
                &mut self.trace.main,
                &RawJson::String(name.clone()),
                TraceOwnership::Fixed,
                "object-key",
            )?;
            self.prefix_builder
                .append_fixed(&mut self.trace.main, ":", "object-colon")?;
            let child_constraints = child_constraints(constraints, name, self.effective);
            self.node(&append(path, name), &child_constraints, child)?;
        }
        if values.iter().any(|(name, _)| !declarations.contains(name)) {
            return Err(ProtocolError::Schema(format!(
                "object at {path} has an undeclared property after closed planning"
            )));
        }
        self.prefix_builder
            .append_fixed(&mut self.trace.main, "}", "object-close")
    }

    fn array(
        &mut self,
        path: &str,
        constraints: &[NodeConstraint],
        values: &[RawJson],
    ) -> crate::Result<()> {
        let (min_items, max_items) = array_bounds(constraints)?;
        if values.len() < min_items || max_items.is_some_and(|maximum| values.len() > maximum) {
            return Err(ProtocolError::Schema(format!(
                "teacher array at {path} is outside effective item bounds"
            )));
        }
        self.prefix_builder
            .append_fixed(&mut self.trace.main, "[", "array-open")?;
        for (index, value) in values.iter().enumerate() {
            let item_path = append(path, &index.to_string());
            if index < min_items {
                self.emit_forced_array(
                    &item_path,
                    ForcedArrayLabel::Continue,
                    ForcedArrayReason::MinItems,
                );
            } else {
                self.emit_probe(
                    ProbeOperation::Array,
                    &item_path,
                    vec![
                        serde_json::json!({"label": "continue"}),
                        serde_json::json!({"label": "stop"}),
                    ],
                    vec!["continue".to_owned(), "stop".to_owned()],
                    0,
                )?;
            }
            if index > 0 {
                self.prefix_builder
                    .append_fixed(&mut self.trace.main, ",", "array-comma")?;
            }
            let child = item_constraints(constraints, self.effective);
            self.node(&item_path, &child, value)?;
        }
        let stop_path = append(path, &values.len().to_string());
        if max_items == Some(values.len()) {
            self.emit_forced_array(
                &stop_path,
                ForcedArrayLabel::Stop,
                ForcedArrayReason::MaxItems,
            );
        } else if values.len() >= min_items {
            self.emit_probe(
                ProbeOperation::Array,
                &stop_path,
                vec![
                    serde_json::json!({"label": "continue"}),
                    serde_json::json!({"label": "stop"}),
                ],
                vec!["continue".to_owned(), "stop".to_owned()],
                1,
            )?;
        }
        self.prefix_builder
            .append_fixed(&mut self.trace.main, "]", "array-close")
    }
}

fn selected_instance_constraints(
    constraints: &[NodeConstraint],
    choice: &TeacherUnionChoice,
) -> crate::Result<Vec<NodeConstraint>> {
    let (scope, _) = choice
        .current_keyword_path
        .rsplit_once('/')
        .ok_or_else(|| {
            ProtocolError::Schema(format!(
                "choice {} has no parent scope",
                choice.current_keyword_path
            ))
        })?;
    let (source_scope, _) = choice.source_keyword_path.rsplit_once('/').ok_or_else(|| {
        ProtocolError::Schema(format!(
            "choice {} has no source parent",
            choice.source_keyword_path
        ))
    })?;
    let alternative = choice
        .alternatives
        .get(choice.selected_index)
        .ok_or_else(|| {
            ProtocolError::Schema(format!(
                "choice {} selected branch is absent",
                choice.argument_path
            ))
        })?;
    let RawJson::Object(entries) = &alternative.schema else {
        return Err(ProtocolError::Schema(
            "effective alternative must be an object".to_owned(),
        ));
    };
    let Some(RawJson::Array(branches)) = field(entries, "allOf") else {
        return Err(ProtocolError::Schema(
            "effective alternative must retain allOf wrapper".to_owned(),
        ));
    };
    if branches.len() != 2 {
        return Err(ProtocolError::Schema(
            "effective alternative wrapper must have base and selected branch".to_owned(),
        ));
    }
    let mut output = Vec::new();
    let mut replaced = false;
    for constraint in constraints {
        if constraint.current_path == scope {
            if !replaced {
                flatten_with_origin(
                    &branches[0],
                    &append2(scope, "allOf", "0"),
                    source_scope,
                    &mut output,
                );
                flatten_with_origin(
                    &branches[1],
                    &append2(scope, "allOf", "1"),
                    &alternative.schema_path,
                    &mut output,
                );
                replaced = true;
            }
        } else {
            output.push(constraint.clone());
        }
    }
    if !replaced {
        return Err(ProtocolError::Schema(format!(
            "no structural constraint for selected array-item choice at {}",
            choice.argument_path
        )));
    }
    Ok(output)
}

fn flatten_with_origin(
    schema: &RawJson,
    current_path: &str,
    source_path: &str,
    output: &mut Vec<NodeConstraint>,
) {
    let Ok(entries) = object_entries(schema) else {
        output.push(NodeConstraint {
            schema: schema.clone(),
            current_path: current_path.to_owned(),
            source_path: source_path.to_owned(),
        });
        return;
    };
    if let Some(RawJson::Array(branches)) = field(entries, "allOf") {
        let base = RawJson::Object(
            entries
                .iter()
                .filter(|(key, _)| key != "allOf")
                .cloned()
                .collect(),
        );
        if !matches!(&base, RawJson::Object(entries) if entries.is_empty()) {
            output.push(NodeConstraint {
                schema: base,
                current_path: current_path.to_owned(),
                source_path: source_path.to_owned(),
            });
        }
        for (index, branch) in branches.iter().enumerate() {
            flatten_with_origin(
                branch,
                &append2(current_path, "allOf", &index.to_string()),
                &append2(source_path, "allOf", &index.to_string()),
                output,
            );
        }
    } else {
        output.push(NodeConstraint {
            schema: schema.clone(),
            current_path: current_path.to_owned(),
            source_path: source_path.to_owned(),
        });
    }
}

fn flatten_constraints(
    effective: &crate::EffectiveTree,
    schema: &RawJson,
    current_path: &str,
) -> Vec<NodeConstraint> {
    let mut output = Vec::new();
    flatten(effective, schema, current_path, &mut output);
    output
}

fn flatten(
    effective: &crate::EffectiveTree,
    schema: &RawJson,
    current_path: &str,
    output: &mut Vec<NodeConstraint>,
) {
    let Ok(entries) = object_entries(schema) else {
        output.push(NodeConstraint {
            schema: schema.clone(),
            current_path: current_path.to_owned(),
            source_path: effective
                .source_path_for(current_path)
                .unwrap_or(current_path)
                .to_owned(),
        });
        return;
    };
    if let Some(RawJson::Array(branches)) = field(entries, "allOf") {
        output.push(NodeConstraint {
            schema: RawJson::Object(
                entries
                    .iter()
                    .filter(|(key, _)| key != "allOf")
                    .cloned()
                    .collect(),
            ),
            current_path: current_path.to_owned(),
            source_path: effective
                .source_path_for(current_path)
                .unwrap_or(current_path)
                .to_owned(),
        });
        for (index, branch) in branches.iter().enumerate() {
            flatten(
                effective,
                branch,
                &append2(current_path, "allOf", &index.to_string()),
                output,
            );
        }
    } else {
        output.push(NodeConstraint {
            schema: schema.clone(),
            current_path: current_path.to_owned(),
            source_path: effective
                .source_path_for(current_path)
                .unwrap_or(current_path)
                .to_owned(),
        });
    }
}

fn object_entries(value: &RawJson) -> crate::Result<&[(String, RawJson)]> {
    value.object_entries().ok_or_else(|| {
        ProtocolError::Schema("effective schema must be an object or Boolean".to_owned())
    })
}
fn field<'a>(entries: &'a [(String, RawJson)], name: &str) -> Option<&'a RawJson> {
    entries
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}
fn ordered(constraints: &[NodeConstraint]) -> Vec<&NodeConstraint> {
    let mut ordered = constraints.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.source_path
            .encode_utf16()
            .cmp(right.source_path.encode_utf16())
            .then_with(|| {
                left.current_path
                    .encode_utf16()
                    .cmp(right.current_path.encode_utf16())
            })
    });
    ordered
}
fn semantic(left: &RawJson, right: &RawJson) -> crate::Result<bool> {
    semantic_equal(left, right).map_err(|error| ProtocolError::Schema(error.to_string()))
}
fn dedupe(values: Vec<RawJson>) -> crate::Result<Vec<RawJson>> {
    let mut output = Vec::new();
    for value in values {
        let mut seen = false;
        for existing in &output {
            if semantic(existing, &value)? {
                seen = true;
                break;
            }
        }
        if !seen {
            output.push(value);
        }
    }
    Ok(output)
}

fn is_null_only(constraints: &[NodeConstraint]) -> bool {
    constraints.iter().any(|constraint| {
        object_entries(&constraint.schema).ok().is_some_and(|entries| {
            matches!(field(entries, "type"), Some(RawJson::String(kind)) if kind == "null")
        })
    })
}
fn boolean_restricted(constraints: &[NodeConstraint]) -> bool {
    constraints.iter().any(|constraint| {
        object_entries(&constraint.schema).ok().is_some_and(|entries| {
            matches!(field(entries, "type"), Some(RawJson::String(kind)) if kind == "boolean")
        })
    })
}
fn property_names(constraints: &[NodeConstraint]) -> crate::Result<Vec<String>> {
    let mut names = Vec::new();
    for constraint in ordered(constraints) {
        let entries = object_entries(&constraint.schema)?;
        let Some(properties) = field(entries, "properties") else {
            continue;
        };
        let properties = object_entries(properties)?;
        for (name, _) in properties {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    for name in required_names(constraints)? {
        if !names.contains(&name) {
            if !property_allowed(constraints, &name)? {
                return Err(ProtocolError::Schema(format!(
                    "required property {name:?} is forbidden by an object conjunct"
                )));
            }
            names.push(name);
        }
    }
    Ok(names)
}
fn required_names(constraints: &[NodeConstraint]) -> crate::Result<Vec<String>> {
    let mut names = Vec::new();
    for constraint in ordered(constraints) {
        let entries = object_entries(&constraint.schema)?;
        let Some(RawJson::Array(required)) = field(entries, "required") else {
            continue;
        };
        for name in required {
            let RawJson::String(name) = name else {
                return Err(ProtocolError::Schema(format!(
                    "required at {} has a non-string name",
                    constraint.source_path
                )));
            };
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    Ok(names)
}
fn property_allowed(constraints: &[NodeConstraint], name: &str) -> crate::Result<bool> {
    for constraint in constraints {
        let entries = object_entries(&constraint.schema)?;
        if field(entries, "properties")
            .and_then(RawJson::object_entries)
            .is_some_and(|properties| properties.iter().any(|(key, _)| key == name))
        {
            continue;
        }
        match field(entries, "additionalProperties") {
            Some(RawJson::Bool(false)) => return Ok(false),
            Some(RawJson::Bool(true)) | None => {}
            Some(schema @ RawJson::Object(_)) if is_unconstrained_schema(schema)? => {}
            Some(RawJson::Object(_)) => {
                return Err(ProtocolError::Schema(format!(
                    "constrained additionalProperties at {} requires a reviewed extension",
                    constraint.source_path
                )));
            }
            Some(_) => {
                return Err(ProtocolError::Schema(format!(
                    "additionalProperties at {} is not a Boolean or schema",
                    constraint.source_path
                )));
            }
        }
    }
    Ok(true)
}
fn allows_unknown_properties(constraints: &[NodeConstraint]) -> crate::Result<bool> {
    let mut constrained_open_schema = false;
    for constraint in constraints {
        let entries = object_entries(&constraint.schema)?;
        match field(entries, "additionalProperties") {
            Some(RawJson::Bool(false)) => return Ok(false),
            Some(RawJson::Bool(true)) | None => {}
            Some(schema @ RawJson::Object(_)) if is_unconstrained_schema(schema)? => {}
            Some(RawJson::Object(_)) => constrained_open_schema = true,
            Some(_) => {
                return Err(ProtocolError::Schema(format!(
                    "additionalProperties at {} is not a Boolean or schema",
                    constraint.source_path
                )));
            }
        }
    }
    if constrained_open_schema {
        return Err(ProtocolError::Schema(
            "constrained additionalProperties requires a reviewed dynamic-object extension"
                .to_owned(),
        ));
    }
    Ok(true)
}
fn wholly_dynamic_constraints(constraints: &[NodeConstraint]) -> crate::Result<bool> {
    if constraints.is_empty() {
        return Ok(false);
    }
    constraints
        .iter()
        .try_fold(true, |all_dynamic, constraint| {
            Ok(all_dynamic && is_unconstrained_schema(&constraint.schema)?)
        })
}

fn is_unconstrained_schema(schema: &RawJson) -> crate::Result<bool> {
    match schema {
        RawJson::Bool(value) => Ok(*value),
        RawJson::Object(entries) => {
            for (key, value) in entries {
                match key.as_str() {
                    "title" | "description" | "default" | "examples" | "$comment"
                    | "deprecated" | "readOnly" | "writeOnly" => {}
                    "allOf" => {
                        let RawJson::Array(branches) = value else {
                            return Ok(false);
                        };
                        for branch in branches {
                            if !is_unconstrained_schema(branch)? {
                                return Ok(false);
                            }
                        }
                    }
                    _ => return Ok(false),
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn child_constraints(
    parents: &[NodeConstraint],
    name: &str,
    effective: &crate::EffectiveTree,
) -> Vec<NodeConstraint> {
    let mut output = Vec::new();
    for parent in parents {
        let Ok(entries) = object_entries(&parent.schema) else {
            continue;
        };
        let Some(properties) = field(entries, "properties").and_then(RawJson::object_entries)
        else {
            continue;
        };
        if let Some((_, child)) = properties.iter().find(|(key, _)| key == name) {
            flatten(
                effective,
                child,
                &append2(&parent.current_path, "properties", name),
                &mut output,
            );
        }
    }
    output
}
fn item_constraints(
    parents: &[NodeConstraint],
    effective: &crate::EffectiveTree,
) -> Vec<NodeConstraint> {
    let mut output = Vec::new();
    for parent in parents {
        let Ok(entries) = object_entries(&parent.schema) else {
            continue;
        };
        if let Some(items) = field(entries, "items") {
            flatten(
                effective,
                items,
                &append(&parent.current_path, "items"),
                &mut output,
            );
        }
    }
    output
}
fn array_bounds(constraints: &[NodeConstraint]) -> crate::Result<(usize, Option<usize>)> {
    let mut min_items = 0usize;
    let mut max_items = None;
    for constraint in constraints {
        let entries = object_entries(&constraint.schema)?;
        if let Some(value) = field(entries, "minItems") {
            min_items = min_items.max(schema_usize(value, "minItems")?);
        }
        if let Some(value) = field(entries, "maxItems") {
            let maximum = schema_usize(value, "maxItems")?;
            max_items = Some(max_items.map_or(maximum, |current: usize| current.min(maximum)));
        }
    }
    if max_items.is_some_and(|maximum| maximum < min_items) {
        return Err(ProtocolError::Schema(
            "effective array bounds are contradictory".to_owned(),
        ));
    }
    Ok((min_items, max_items))
}
fn schema_usize(value: &RawJson, keyword: &str) -> crate::Result<usize> {
    let RawJson::Number(number) = value else {
        return Err(ProtocolError::Schema(format!("{keyword} is not a number")));
    };
    let admitted = crate::admit_number(number, crate::NumericKind::Integer)
        .map_err(|error| ProtocolError::Schema(format!("{keyword}: {error}")))?;
    if admitted.value < 0.0 {
        return Err(ProtocolError::Schema(format!("{keyword} is negative")));
    }
    // NumericKind::Integer already rejects nonintegral values and the strict
    // admission safe-integer domain is within u64, so this conversion cannot
    // truncate or lose sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let integer = admitted.value as u64;
    usize::try_from(integer)
        .map_err(|_| ProtocolError::Schema(format!("{keyword} exceeds host usize")))
}
fn append(parent: &str, token: &str) -> String {
    format!("{parent}/{}", token.replace('~', "~0").replace('/', "~1"))
}
fn append2(parent: &str, first: &str, second: &str) -> String {
    append(&append(parent, first), second)
}
fn value_at<'a>(root: &'a RawJson, path: &str) -> crate::Result<&'a RawJson> {
    if path.is_empty() {
        return Ok(root);
    }
    if !path.starts_with('/') {
        return Err(ProtocolError::Schema(format!(
            "argument path {path:?} is not RFC6901"
        )));
    }
    let tokens = path[1..]
        .split('/')
        .map(decode_token)
        .collect::<crate::Result<Vec<_>>>()?;
    let mut current = root;
    for token in tokens {
        current = match current {
            RawJson::Object(entries) => entries
                .iter()
                .find(|(key, _)| key == &token)
                .map(|(_, value)| value),
            RawJson::Array(values) => pointer_index(&token).and_then(|index| values.get(index)),
            _ => None,
        }
        .ok_or_else(|| ProtocolError::Schema(format!("argument path {path:?} is absent")))?;
    }
    Ok(current)
}

fn append_committed_lexical(
    trace: &mut TracePrefix,
    policy: &TokenPolicy,
    lexical: TeacherLexicalValue,
    label: &str,
) -> crate::Result<()> {
    let start = trace.token_ids.len();
    let end = start
        .checked_add(lexical.token_ids.len())
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    let positions = (0..lexical.token_ids.len())
        .map(|offset| {
            start
                .checked_add(offset)
                .and_then(|position| position.checked_sub(1))
                .ok_or(ProtocolError::TokenLengthOverflow)
        })
        .collect::<crate::Result<Vec<_>>>()?;
    trace.token_ids.extend_from_slice(&lexical.token_ids);
    trace.appends.push(MainAppend {
        source: Some(lexical.source),
        special_id: None,
        token_ids: lexical.token_ids,
        ownership: TraceOwnership::Learned,
        label: label.to_owned(),
        main_start: start,
        main_end: end,
        prediction_positions: positions,
    });
    let framing = policy.framing_ids().value_end;
    policy.validate_framing(framing)?;
    let marker_start = trace.token_ids.len();
    let marker_end = marker_start
        .checked_add(1)
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    trace.token_ids.push(framing);
    trace.appends.push(MainAppend {
        source: None,
        special_id: Some(framing),
        token_ids: vec![framing],
        ownership: TraceOwnership::Learned,
        label: format!("{label}-end"),
        main_start: marker_start,
        main_end: marker_end,
        prediction_positions: vec![
            marker_start
                .checked_sub(1)
                .ok_or(ProtocolError::TokenLengthOverflow)?,
        ],
    });
    Ok(())
}

fn replace_at(root: &mut RawJson, path: &str, replacement: RawJson) -> crate::Result<()> {
    if path.is_empty() {
        *root = replacement;
        return Ok(());
    }
    if !path.starts_with('/') {
        return Err(ProtocolError::Schema(format!(
            "argument path {path:?} is not RFC6901"
        )));
    }
    let tokens = path[1..]
        .split('/')
        .map(decode_token)
        .collect::<crate::Result<Vec<_>>>()?;
    replace_tokens(root, &tokens, replacement)
}
fn replace_tokens(
    value: &mut RawJson,
    tokens: &[String],
    replacement: RawJson,
) -> crate::Result<()> {
    let Some((first, rest)) = tokens.split_first() else {
        *value = replacement;
        return Ok(());
    };
    match value {
        RawJson::Object(entries) => {
            let (_, child) = entries
                .iter_mut()
                .find(|(key, _)| key == first)
                .ok_or_else(|| {
                    ProtocolError::Schema(format!("argument key {first:?} is absent"))
                })?;
            replace_tokens(child, rest, replacement)
        }
        RawJson::Array(values) => {
            let index = pointer_index(first).ok_or_else(|| {
                ProtocolError::Schema(format!("argument array token {first:?} is invalid"))
            })?;
            let child = values.get_mut(index).ok_or_else(|| {
                ProtocolError::Schema(format!("argument array index {index} is absent"))
            })?;
            replace_tokens(child, rest, replacement)
        }
        _ => Err(ProtocolError::Schema(
            "argument path crosses a scalar".to_owned(),
        )),
    }
}
fn decode_token(token: &str) -> crate::Result<String> {
    let mut output = String::with_capacity(token.len());
    let mut characters = token.chars();
    while let Some(character) = characters.next() {
        if character != '~' {
            output.push(character);
            continue;
        }
        match characters.next() {
            Some('0') => output.push('~'),
            Some('1') => output.push('/'),
            _ => {
                return Err(ProtocolError::Schema(
                    "argument path has an invalid RFC6901 escape".to_owned(),
                ));
            }
        }
    }
    Ok(output)
}
fn pointer_index(token: &str) -> Option<usize> {
    if token == "0" {
        return Some(0);
    }
    if token.is_empty()
        || token.starts_with('0')
        || !token.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    token.parse().ok()
}
fn token_hash(tokens: &[TokenId]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len().saturating_mul(4));
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    crate::sha256_hex(&bytes)
}
fn append_branch_text<T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    output: &mut Vec<BranchAppend>,
    length: &mut usize,
    text: &str,
) -> crate::Result<()>
where
    T: SegmentTokenizer,
{
    append_branch_bytes(tokenizer, policy, output, length, text.as_bytes().to_vec())
}
fn append_branch_bytes<T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    output: &mut Vec<BranchAppend>,
    length: &mut usize,
    source: Vec<u8>,
) -> crate::Result<()>
where
    T: SegmentTokenizer,
{
    let text = std::str::from_utf8(&source).map_err(|_| ProtocolError::InvalidCanonicalUtf8)?;
    let token_ids = tokenizer
        .encode_without_special_tokens(text)
        .map_err(|error| ProtocolError::Tokenizer(error.to_string()))?;
    policy.validate_payload(&token_ids)?;
    let start = *length;
    *length = length
        .checked_add(token_ids.len())
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    output.push(BranchAppend {
        source: Some(source),
        special_id: None,
        token_ids,
        ownership: TraceOwnership::Fixed,
        branch_start: start,
        branch_end: *length,
    });
    Ok(())
}
fn candidate_text<T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    prefix_length: usize,
    label: String,
) -> crate::Result<ProbeCandidate>
where
    T: SegmentTokenizer,
{
    let (segments, token_ids, prediction_positions) =
        learned_segments(tokenizer, policy, prefix_length, label.as_bytes().to_vec())?;
    Ok(ProbeCandidate {
        label,
        segments,
        token_ids,
        prediction_positions,
    })
}
fn candidate_value<T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    prefix_length: usize,
    value: RawJson,
) -> crate::Result<FiniteCandidate>
where
    T: SegmentTokenizer,
{
    let source = safe_json(&value.clone().into_value()?)?;
    let (segments, token_ids, prediction_positions) =
        learned_segments(tokenizer, policy, prefix_length, source)?;
    Ok(FiniteCandidate {
        value,
        segments,
        token_ids,
        prediction_positions,
    })
}
fn learned_segments<T>(
    tokenizer: &T,
    policy: &TokenPolicy,
    prefix_length: usize,
    source: Vec<u8>,
) -> crate::Result<(Vec<BranchAppend>, Vec<TokenId>, Vec<usize>)>
where
    T: SegmentTokenizer,
{
    let text = std::str::from_utf8(&source).map_err(|_| ProtocolError::InvalidCanonicalUtf8)?;
    let mut token_ids = tokenizer
        .encode_without_special_tokens(text)
        .map_err(|error| ProtocolError::Tokenizer(error.to_string()))?;
    policy.validate_payload(&token_ids)?;
    let text_end = prefix_length
        .checked_add(token_ids.len())
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    let framing = policy.framing_ids().value_end;
    policy.validate_framing(framing)?;
    let mut segments = vec![BranchAppend {
        source: Some(source),
        special_id: None,
        token_ids: token_ids.clone(),
        ownership: TraceOwnership::Learned,
        branch_start: prefix_length,
        branch_end: text_end,
    }];
    token_ids.push(framing);
    let end = text_end
        .checked_add(1)
        .ok_or(ProtocolError::TokenLengthOverflow)?;
    segments.push(BranchAppend {
        source: None,
        special_id: Some(framing),
        token_ids: vec![framing],
        ownership: TraceOwnership::Learned,
        branch_start: text_end,
        branch_end: end,
    });
    let prediction_positions = (0..token_ids.len())
        .map(|offset| {
            prefix_length
                .checked_add(offset)
                .and_then(|position| position.checked_sub(1))
                .ok_or(ProtocolError::TokenLengthOverflow)
        })
        .collect::<crate::Result<Vec<_>>>()?;
    Ok((segments, token_ids, prediction_positions))
}
