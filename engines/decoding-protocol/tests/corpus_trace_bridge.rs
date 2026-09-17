#![allow(
    clippy::collapsible_if,
    clippy::double_ended_iterator_last,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
// This private test-only transport comparator intentionally uses exact f64 equality
// and compact local names while preserving every independently serialized field.
//! Opt-in, private cross-language corpus trace comparison. It owns no production behavior.
use minifield_decoding_protocol::{
    BranchAppend, FiniteChoice, PublicEvent, RawJson, RawJsonLimits, RoutingTrace,
    RoutingTraceInput, SchemaLimits, SegmentTokenizer, TeacherOperation, TeacherTrace,
    TeacherTraceBuilder, TeacherTraceInput, TokenId, TokenPolicy, TraceOwnership,
    TracePrefixBuilder, normalize_schema_document, parse_json_document, parse_runtime_value,
    plan_teacher, sha256_hex,
};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};
const ROOT: &str = "MINIFIELD_CORPUS_TRACE_BRIDGE";
const SCOPE: &str = "MINIFIELD_CORPUS_TRACE_SCOPE";
const REPORT: &str = "MINIFIELD_CORPUS_TRACE_REPORT_DIR";
const MAX_DETAIL: usize = 25;
type R<T> = Result<T, String>;

struct PinnedTokenizer<'a>(&'a HashMap<String, Vec<TokenId>>);
impl SegmentTokenizer for PinnedTokenizer<'_> {
    type Error = String;
    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<TokenId>, Self::Error> {
        self.0.get(text).cloned().ok_or_else(|| {
            format!(
                "unknown pinned segment bytes={} sha256_utf8={} escaped={text:?}",
                text.len(),
                sha256_hex(text.as_bytes())
            )
        })
    }
}
#[derive(Default)]
struct Row {
    checks: BTreeMap<String, u64>,
    errors: Vec<(String, String)>,
    labels: Vec<(String, String)>,
}
impl Row {
    fn check(&mut self, k: &str, f: impl FnOnce() -> R<()>) {
        *self.checks.entry(k.into()).or_default() += 1;
        if let Err(e) = f() {
            self.errors.push((k.into(), e));
        }
    }
    fn eq<T: std::fmt::Debug + PartialEq>(&mut self, k: &str, a: T, b: T) {
        self.check(k, || {
            if a == b {
                Ok(())
            } else {
                Err(format!("actual={a:?}; expected={b:?}"))
            }
        });
    }
}
#[derive(Default)]
struct Total {
    rows: u64,
    ok: u64,
    bad: u64,
    malformed: u64,
    build: u64,
    fields: BTreeMap<String, u64>,
    checks: BTreeMap<String, u64>,
    labels: u64,
    detail: Vec<Value>,
}
fn merge(a: &mut BTreeMap<String, u64>, b: BTreeMap<String, u64>) {
    for (k, v) in b {
        *a.entry(k).or_default() += v
    }
}
impl Total {
    fn add(&mut self, split: &str, id: &str, r: R<Row>) {
        self.rows += 1;
        match r {
            Ok(x) => {
                merge(&mut self.checks, x.checks);
                self.labels += x.labels.len() as u64;
                if x.errors.is_empty() {
                    self.ok += 1
                } else {
                    self.bad += 1;
                    for (k, _) in &x.errors {
                        *self.fields.entry(k.clone()).or_default() += 1
                    }
                    if self.detail.len() < MAX_DETAIL {
                        self.detail.push(json!({"split":split,"decision_id":id,"kind":"semantic_mismatch","issues":x.errors,"debug_label_differences":x.labels}));
                    }
                }
            }
            Err(e) => {
                self.bad += 1;
                self.malformed += 1;
                if e.starts_with("rust_build:") {
                    self.build += 1
                }
                *self.fields.entry("preflight_or_build".into()).or_default() += 1;
                if self.detail.len() < MAX_DETAIL {
                    self.detail.push(json!({"split":split,"decision_id":id,"kind":"preflight_or_build","detail":e}));
                }
            }
        }
    }
}
fn o(v: &Value) -> R<&Map<String, Value>> {
    v.as_object().ok_or_else(|| "expected object".into())
}
fn f<'a>(v: &'a Value, k: &str) -> R<&'a Value> {
    o(v)?.get(k).ok_or_else(|| format!("missing {k}"))
}
fn a<'a>(v: &'a Value, k: &str) -> R<&'a Vec<Value>> {
    v.as_array().ok_or_else(|| format!("{k} must be array"))
}
fn s(v: &Value, k: &str) -> R<String> {
    f(v, k)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{k} must be string"))
}
fn n(v: &Value, k: &str) -> R<usize> {
    usize::try_from(
        f(v, k)?
            .as_u64()
            .ok_or_else(|| format!("{k} must be u64"))?,
    )
    .map_err(|_| format!("{k} too large"))
}
fn n32(v: &Value, k: &str) -> R<u32> {
    u32::try_from(
        f(v, k)?
            .as_u64()
            .ok_or_else(|| format!("{k} must be u64"))?,
    )
    .map_err(|_| format!("{k} too large"))
}
fn ids_hash(ids: &[u32]) -> String {
    let mut b = Vec::with_capacity(ids.len() * 4);
    for x in ids {
        b.extend_from_slice(&x.to_le_bytes())
    }
    sha256_hex(&b)
}
fn summary(v: &Value, ids: &[u32]) -> R<()> {
    let l = n(v, "length")?;
    let h = s(v, "sha256_u32le")?;
    if l == ids.len() && h == ids_hash(ids) {
        Ok(())
    } else {
        Err(format!("summary got {} {}", ids.len(), ids_hash(ids)))
    }
}
fn pos(v: &Value) -> R<Vec<usize>> {
    a(f(v, "prediction_positions")?, "prediction_positions")?
        .iter()
        .map(|x| {
            usize::try_from(x.as_u64().ok_or_else(|| "position not u64".to_owned())?)
                .map_err(|_| "position too large".into())
        })
        .collect()
}
fn semantic_value(left: &Value, right: &Value) -> R<bool> {
    match (left, right) {
        (Value::Null, Value::Null) => Ok(true),
        (Value::Bool(left), Value::Bool(right)) => Ok(left == right),
        (Value::Number(left), Value::Number(right)) => {
            let left = left
                .as_f64()
                .ok_or_else(|| format!("non-binary64 left number {left}"))?;
            let right = right
                .as_f64()
                .ok_or_else(|| format!("non-binary64 right number {right}"))?;
            if !left.is_finite() || !right.is_finite() {
                return Err("non-finite semantic comparison".into());
            }
            Ok(left == right)
        }
        (Value::String(left), Value::String(right)) => Ok(left == right),
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (left, right) in left.iter().zip(right) {
                if !semantic_value(left, right)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Object(left), Value::Object(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (key, value) in left {
                let Some(other) = right.get(key) else {
                    return Ok(false);
                };
                if !semantic_value(value, other)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
fn json_pointer(parent: &str, token: &str) -> String {
    format!("{parent}/{}", token.replace('~', "~0").replace('/', "~1"))
}
fn first_semantic_difference(left: &Value, right: &Value, path: &str) -> R<String> {
    if semantic_value(left, right)? {
        return Ok("no semantic difference".into());
    }
    match (left, right) {
        (Value::Array(left), Value::Array(right)) => {
            for (index, (left, right)) in left.iter().zip(right).enumerate() {
                if !semantic_value(left, right)? {
                    return first_semantic_difference(
                        left,
                        right,
                        &json_pointer(path, &index.to_string()),
                    );
                }
            }
            Ok(format!(
                "{path}: array lengths actual={} expected={}",
                left.len(),
                right.len()
            ))
        }
        (Value::Object(left), Value::Object(right)) => {
            for (key, value) in left {
                match right.get(key) {
                    Some(other) if !semantic_value(value, other)? => {
                        return first_semantic_difference(value, other, &json_pointer(path, key));
                    }
                    Some(_) => {}
                    None => {
                        return Ok(format!(
                            "{}: actual has key {key:?}; expected does not",
                            path
                        ));
                    }
                }
            }
            for key in right.keys() {
                if !left.contains_key(key) {
                    return Ok(format!(
                        "{}: expected has key {key:?}; actual does not",
                        path
                    ));
                }
            }
            Ok(format!("{path}: object difference"))
        }
        _ => Ok(format!("{path}: actual={} expected={}", left, right)),
    }
}

fn rf<'a>(v: &'a RawJson, k: &str) -> R<&'a RawJson> {
    v.object_entries()
        .and_then(|x| x.iter().find(|(key, _)| key == k).map(|(_, x)| x))
        .ok_or_else(|| format!("event missing {k}"))
}
fn rs(v: &RawJson, k: &str) -> R<String> {
    match rf(v, k)? {
        RawJson::String(x) => Ok(x.clone()),
        _ => Err(format!("event {k} not string")),
    }
}
fn event(text: &str) -> R<PublicEvent> {
    let v = parse_json_document(text.as_bytes(), RawJsonLimits::default())
        .map_err(|e| e.to_string())?;
    match rs(&v, "type")?.as_str() {
        "system" => Ok(PublicEvent::System {
            policy: rs(&v, "policy")?,
            observation: v.object_entries().and_then(|entries| {
                entries
                    .iter()
                    .find(|(key, _)| key == "observation")
                    .map(|(_, value)| value.clone())
            }),
        }),
        "user" => Ok(PublicEvent::User {
            content: rs(&v, "content")?,
        }),
        "tool_call" => Ok(PublicEvent::ToolCall {
            tool_call_id: rs(&v, "tool_call_id")?,
            tool: rs(&v, "tool")?,
            arguments: rf(&v, "arguments")?.clone(),
        }),
        "tool_result" => Ok(PublicEvent::ToolResult {
            tool_call_id: rs(&v, "tool_call_id")?,
            result: rf(&v, "result")?.clone(),
        }),
        "assistant_text" => Ok(PublicEvent::AssistantText {
            content: rs(&v, "content")?,
        }),
        x => Err(format!("unknown event {x}")),
    }
}
fn ownership(v: &Value) -> R<TraceOwnership> {
    match s(v, "ownership")?.as_str() {
        "fixed" => Ok(TraceOwnership::Fixed),
        "learned" => Ok(TraceOwnership::Learned),
        x => Err(format!("bad ownership {x}")),
    }
}
fn text(v: &Value) -> R<Option<Vec<u8>>> {
    match f(v, "text")?.as_str() {
        Some(x) => Ok(Some(x.as_bytes().to_vec())),
        None if f(v, "text")?.is_null() => Ok(None),
        None => Err("text invalid".into()),
    }
}
fn special(v: &Value) -> R<Option<u32>> {
    match f(v, "special_id")?.as_u64() {
        Some(x) => Ok(Some(u32::try_from(x).map_err(|_| "special too large")?)),
        None if f(v, "special_id")?.is_null() => Ok(None),
        None => Err("special invalid".into()),
    }
}
fn ids(v: &Value, m: &HashMap<String, Vec<u32>>, p: &TokenPolicy) -> R<Vec<u32>> {
    let out = match (text(v)?, special(v)?) {
        (Some(x), None) => m
            .get(std::str::from_utf8(&x).map_err(|e| e.to_string())?)
            .cloned()
            .ok_or_else(|| format!("unknown segment {} bytes", x.len()))?,
        (None, Some(x)) => {
            p.validate_framing(x).map_err(|e| e.to_string())?;
            vec![x]
        }
        _ => return Err("text/special contract".into()),
    };
    summary(f(v, "token_ids")?, &out)?;
    Ok(out)
}
fn load_segments(path: &Path, p: &TokenPolicy) -> R<HashMap<String, Vec<u32>>> {
    let mut out = HashMap::new();
    for (i, line) in BufReader::new(File::open(path).map_err(|e| e.to_string())?)
        .lines()
        .enumerate()
    {
        let v: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
            .map_err(|e| format!("segments {}: {e}", i + 1))?;
        let source = s(&v, "source")?;
        if s(&v, "sha256_utf8")? != sha256_hex(source.as_bytes()) {
            return Err(format!("segments {} hash", i + 1));
        }
        let z = a(f(&v, "token_ids")?, "token_ids")?
            .iter()
            .map(|x| {
                u32::try_from(x.as_u64().ok_or_else(|| "segment ID invalid".to_owned())?)
                    .map_err(|_| "segment ID too large".into())
            })
            .collect::<R<Vec<_>>>()?;
        p.validate_payload(&z).map_err(|e| e.to_string())?;
        if out.insert(source, z).is_some() {
            return Err(format!("segments {} duplicate", i + 1));
        }
    }
    if out.is_empty() {
        Err("empty segment map".into())
    } else {
        Ok(out)
    }
}
fn branch(
    r: &mut Row,
    k: &str,
    x: &BranchAppend,
    v: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let z = match ids(v, m, p) {
        Ok(z) => z,
        Err(e) => {
            r.check(k, || Err(e));
            return;
        }
    };
    r.check(&format!("{k}.source"), || {
        if x.source == text(v)? {
            Ok(())
        } else {
            Err("source differs".into())
        }
    });
    r.check(&format!("{k}.special"), || {
        if x.special_id == special(v)? {
            Ok(())
        } else {
            Err("special differs".into())
        }
    });
    r.eq(&format!("{k}.ids"), x.token_ids.clone(), z);
    r.check(&format!("{k}.ownership"), || {
        if x.ownership == ownership(v)? {
            Ok(())
        } else {
            Err("ownership differs".into())
        }
    });
}
fn expected_ops<'a>(oracle: &'a Value, kind: &str) -> R<Vec<&'a Value>> {
    Ok(a(f(oracle, "expected_operations")?, "ops")?
        .iter()
        .filter(|x| f(x, "kind").ok().and_then(Value::as_str) == Some(kind))
        .collect())
}
fn compare_main(
    r: &mut Row,
    t: &TeacherTrace,
    o: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let wanted = match expected_ops(o, "main_append") {
        Ok(x) => x,
        Err(e) => {
            r.check("main.ops", || Err(e));
            return;
        }
    };
    r.eq("main.append_count", t.main.appends.len(), wanted.len());
    for (i, (x, v)) in t.main.appends.iter().zip(wanted.iter()).enumerate() {
        let k = format!("main[{i}]");
        let b = BranchAppend {
            source: x.source.clone(),
            special_id: x.special_id,
            token_ids: x.token_ids.clone(),
            ownership: x.ownership,
            branch_start: x.main_start,
            branch_end: x.main_end,
        };
        branch(r, &k, &b, v, m, p);
        r.check(&format!("{k}.offset"), || {
            if x.main_start == n(v, "main_start")? && x.main_end == n(v, "main_end")? {
                Ok(())
            } else {
                Err("main offsets differ".into())
            }
        });
        r.check(&format!("{k}.positions"), || {
            if x.prediction_positions == pos(v)? {
                Ok(())
            } else {
                Err("positions differ".into())
            }
        });
        if let Ok(label) = s(v, "label") {
            if label != x.label {
                r.labels
                    .push((k, format!("rust={:?}; python={label:?}", x.label)))
            }
        }
    }
    r.check("main.ids", || {
        summary(f(o, "main_token_ids")?, &t.main.token_ids)
    });
    r.check("main.hash", || {
        if ids_hash(&t.main.token_ids) == s(o, "main_sha256_u32le")? {
            Ok(())
        } else {
            Err("main hash differs".into())
        }
    });
    r.check("main.context", || {
        summary(
            f(o, "context_token_ids")?,
            t.main
                .token_ids
                .get(..t.main.argument_start)
                .ok_or_else(|| "argument start bounds".to_owned())?,
        )
    });
    r.check("main.argument_range", || {
        let q = a(f(o, "argument_token_range")?, "range")?;
        if q.len() != 2 {
            return Err("argument range length".into());
        }
        let start = usize::try_from(q[0].as_u64().ok_or_else(|| "range start".to_owned())?)
            .map_err(|_| "range start")?;
        let end = usize::try_from(q[1].as_u64().ok_or_else(|| "range end".to_owned())?)
            .map_err(|_| "range end")?;
        let a = t
            .main
            .appends
            .iter()
            .filter(|x| x.label == "envelope")
            .last()
            .ok_or_else(|| "missing envelope".to_owned())?
            .main_end;
        let b = t
            .main
            .appends
            .iter()
            .find(|x| x.label == "envelope-close")
            .ok_or_else(|| "missing envelope-close".to_owned())?
            .main_start;
        if a == start && b == end {
            Ok(())
        } else {
            Err(format!("rust {a}..{b}; python {start}..{end}"))
        }
    });
    r.check("main.assembly", || {
        let a = t
            .main
            .appends
            .iter()
            .filter(|x| x.label == "envelope")
            .last()
            .ok_or_else(|| "missing envelope".to_owned())?
            .main_end;
        let b = t
            .main
            .appends
            .iter()
            .find(|x| x.label == "envelope-close")
            .ok_or_else(|| "missing envelope-close".to_owned())?
            .main_start;
        let mut z = Vec::new();
        for x in &t.main.appends {
            if x.main_start >= a && x.main_end <= b {
                if let Some(q) = &x.source {
                    z.extend_from_slice(q)
                }
            }
        }
        if z == s(o, "argument_json_for_assembly_check_only")?.as_bytes() {
            Ok(())
        } else {
            Err("argument assembly differs".into())
        }
    });
}
fn cand_ids(v: &Value, m: &HashMap<String, Vec<u32>>, p: &TokenPolicy) -> R<Vec<u32>> {
    let mut out = Vec::new();
    for x in a(f(v, "segments")?, "segments")? {
        out.extend(ids(x, m, p)?)
    }
    summary(f(v, "token_ids")?, &out)?;
    Ok(out)
}
fn candidate(
    r: &mut Row,
    k: &str,
    x: &minifield_decoding_protocol::ProbeCandidate,
    v: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let q = match a(
        f(v, "segments").unwrap_or(&Value::Null),
        "candidate segments",
    ) {
        Ok(x) => x,
        Err(e) => {
            r.check(k, || Err(e));
            return;
        }
    };
    r.eq(&format!("{k}.segment_count"), x.segments.len(), q.len());
    for (i, (a, b)) in x.segments.iter().zip(q.iter()).enumerate() {
        branch(r, &format!("{k}.segment[{i}]"), a, b, m, p)
    }
    r.check(&format!("{k}.ids"), || {
        if x.token_ids == cand_ids(v, m, p)? {
            Ok(())
        } else {
            Err("candidate IDs differ".into())
        }
    });
    r.check(&format!("{k}.positions"), || {
        if x.prediction_positions == pos(v)? {
            Ok(())
        } else {
            Err("candidate positions differ".into())
        }
    });
    if let Ok(q) = s(v, "label") {
        if x.label != q {
            r.labels
                .push((k.into(), format!("rust={:?}; python={q:?}", x.label)))
        }
    }
}
fn probe(
    r: &mut Row,
    i: usize,
    x: &minifield_decoding_protocol::ProbeTrace,
    v: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let k = format!("probe[{i}]");
    r.check(&format!("{k}.operation"), || {
        if x.operation.wire() == s(v, "operation")? {
            Ok(())
        } else {
            Err("operation differs".into())
        }
    });
    r.check(&format!("{k}.path"), || {
        if x.path == s(v, "path")? {
            Ok(())
        } else {
            Err("path differs".into())
        }
    });
    r.check(&format!("{k}.fork"), || {
        if x.fork_main_length == n(v, "fork_main_length")?
            && x.fork_main_sha256_u32le == s(v, "fork_main_sha256_u32le")?
        {
            Ok(())
        } else {
            Err("fork differs".into())
        }
    });
    let q = match a(f(v, "suffix_segments").unwrap_or(&Value::Null), "suffix") {
        Ok(x) => x,
        Err(e) => {
            r.check(&k, || Err(e));
            return;
        }
    };
    r.eq(
        &format!("{k}.suffix_count"),
        x.suffix_segments.len(),
        q.len(),
    );
    let mut full = x.prefix_token_ids.clone();
    for (j, (a, b)) in x.suffix_segments.iter().zip(q.iter()).enumerate() {
        branch(r, &format!("{k}.suffix[{j}]"), a, b, m, p);
        r.check(&format!("{k}.suffix[{j}].offset"), || {
            if a.branch_start == n(b, "branch_start")? && a.branch_end == n(b, "branch_end")? {
                Ok(())
            } else {
                Err("suffix offset differs".into())
            }
        });
        full.extend_from_slice(&a.token_ids)
    }
    r.check(&format!("{k}.prefix"), || {
        summary(f(v, "prefix_token_ids")?, &full)
    });
    r.check(&format!("{k}.discard"), || {
        if x.fork_main_length == n(v, "main_length_after_discard")?
            && x.fork_main_sha256_u32le == s(v, "main_sha256_after_discard_u32le")?
        {
            Ok(())
        } else {
            Err("discard differs".into())
        }
    });
    let c = match a(f(v, "candidates").unwrap_or(&Value::Null), "candidates") {
        Ok(x) => x,
        Err(e) => {
            r.check(&k, || Err(e));
            return;
        }
    };
    r.eq(&format!("{k}.candidate_count"), x.candidates.len(), c.len());
    for (j, (a, b)) in x.candidates.iter().zip(c.iter()).enumerate() {
        candidate(r, &format!("{k}.candidate[{j}]"), a, b, m, p)
    }
    r.check(&format!("{k}.selected"), || {
        if x.selected_index == n(v, "selected_index")? {
            Ok(())
        } else {
            Err("selected differs".into())
        }
    });
}
fn finite(
    r: &mut Row,
    i: usize,
    x: &FiniteChoice,
    v: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let k = format!("finite[{i}]");
    r.check(&format!("{k}.path"), || {
        if x.path == s(v, "path")? {
            Ok(())
        } else {
            Err("path differs".into())
        }
    });
    r.check(&format!("{k}.fork"), || {
        if x.fork_main_length == n(v, "fork_main_length")?
            && x.fork_main_sha256_u32le == s(v, "fork_main_sha256_u32le")?
        {
            Ok(())
        } else {
            Err("fork differs".into())
        }
    });
    let q = match a(f(v, "candidates").unwrap_or(&Value::Null), "candidates") {
        Ok(x) => x,
        Err(e) => {
            r.check(&k, || Err(e));
            return;
        }
    };
    r.eq(&format!("{k}.candidate_count"), x.candidates.len(), q.len());
    for (j, (candidate, b)) in x.candidates.iter().zip(q.iter()).enumerate() {
        let h = format!("{k}.candidate[{j}]");
        let bs = match a(f(b, "segments").unwrap_or(&Value::Null), "segments") {
            Ok(x) => x,
            Err(e) => {
                r.check(&h, || Err(e));
                continue;
            }
        };
        r.eq(
            &format!("{h}.segment_count"),
            candidate.segments.len(),
            bs.len(),
        );
        for (z, (aa, bb)) in candidate.segments.iter().zip(bs.iter()).enumerate() {
            branch(r, &format!("{h}.segment[{z}]"), aa, bb, m, p)
        }
        r.check(&format!("{h}.ids"), || {
            if candidate.token_ids == cand_ids(b, m, p)? {
                Ok(())
            } else {
                Err("IDs differ".into())
            }
        });
        r.check(&format!("{h}.positions"), || {
            if candidate.prediction_positions == pos(b)? {
                Ok(())
            } else {
                Err("positions differ".into())
            }
        });
        r.check(&format!("{h}.value"), || {
            let actual = candidate
                .value
                .clone()
                .into_value()
                .map_err(|error| error.to_string())?;
            let expected = f(b, "value")?;
            if semantic_value(&actual, expected)? {
                Ok(())
            } else {
                Err(first_semantic_difference(&actual, expected, "")?)
            }
        });
    }
    r.check(&format!("{k}.selected"), || {
        if x.selected_index == n(v, "selected_index")? {
            Ok(())
        } else {
            Err("selected differs".into())
        }
    });
}
fn main_branch(append: &minifield_decoding_protocol::MainAppend) -> BranchAppend {
    BranchAppend {
        source: append.source.clone(),
        special_id: append.special_id,
        token_ids: append.token_ids.clone(),
        ownership: append.ownership,
        branch_start: append.main_start,
        branch_end: append.main_end,
    }
}

fn routing(
    r: &mut Row,
    trace: &RoutingTrace,
    expected: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) {
    let suffix = match a(
        f(expected, "suffix_segments").unwrap_or(&Value::Null),
        "routing suffix",
    ) {
        Ok(value) => value,
        Err(error) => {
            r.check("routing", || Err(error));
            return;
        }
    };
    r.eq(
        "routing.suffix_count",
        trace.suffix_segments.len(),
        suffix.len(),
    );
    for (index, (actual, wanted)) in trace.suffix_segments.iter().zip(suffix).enumerate() {
        let key = format!("routing.suffix[{index}]");
        branch(r, &key, &main_branch(actual), wanted, m, p);
        if let Ok(label) = s(wanted, "label") {
            if actual.label != label {
                r.labels
                    .push((key, format!("rust={:?}; python={label:?}", actual.label)));
            }
        }
    }
    r.check("routing.prefix_ids", || {
        summary(f(expected, "prefix_token_ids")?, &trace.prefix_token_ids)
    });
    let candidates = match a(
        f(expected, "candidates").unwrap_or(&Value::Null),
        "routing candidates",
    ) {
        Ok(value) => value,
        Err(error) => {
            r.check("routing", || Err(error));
            return;
        }
    };
    r.eq(
        "routing.candidate_count",
        trace.candidates.len(),
        candidates.len(),
    );
    for (index, (actual, wanted)) in trace.candidates.iter().zip(candidates).enumerate() {
        let key = format!("routing.candidate[{index}]");
        r.check(&format!("{key}.label"), || {
            if actual.label == s(wanted, "label")? {
                Ok(())
            } else {
                Err("label differs".into())
            }
        });
        r.check(&format!("{key}.ids"), || {
            summary(f(wanted, "token_ids")?, &actual.token_ids)
        });
        r.check(&format!("{key}.positions"), || {
            if actual.prediction_positions == pos(wanted)? {
                Ok(())
            } else {
                Err("positions differ".into())
            }
        });
    }
}

fn forced(
    r: &mut Row,
    index: usize,
    actual: &minifield_decoding_protocol::ForcedArray,
    wanted: &Value,
) {
    let key = format!("forced[{index}]");
    r.check(&format!("{key}.operation"), || {
        if s(wanted, "operation")? == "array" {
            Ok(())
        } else {
            Err("operation differs".into())
        }
    });
    r.check(&format!("{key}.path"), || {
        let expected = s(wanted, "path")?;
        if actual.path == expected {
            Ok(())
        } else {
            Err(format!("actual={:?}; expected={expected:?}", actual.path))
        }
    });
    r.check(&format!("{key}.label"), || {
        if actual.label.wire() == s(wanted, "label")? {
            Ok(())
        } else {
            Err("label differs".into())
        }
    });
    r.check(&format!("{key}.reason"), || {
        if actual.reason.wire() == s(wanted, "reason")? {
            Ok(())
        } else {
            Err("reason differs".into())
        }
    });
    r.check(&format!("{key}.main_length"), || {
        if actual.main_length == n(wanted, "main_length")? {
            Ok(())
        } else {
            Err("main length differs".into())
        }
    });
    r.check(&format!("{key}.zero_loss"), || {
        if actual.direct_loss_tokens == n(wanted, "direct_loss_tokens")? {
            summary(f(wanted, "probe_token_ids")?, &[])?;
            Ok(())
        } else {
            Err("direct loss differs".into())
        }
    });
}

fn compare_operations(r: &mut Row, trace: &TeacherTrace, oracle: &Value) {
    let wanted = match a(
        f(oracle, "expected_operations").unwrap_or(&Value::Null),
        "operations",
    ) {
        Ok(value) => value,
        Err(error) => {
            r.check("operations", || Err(error));
            return;
        }
    };
    r.eq("operations.count", trace.operation_log.len(), wanted.len());
    let mut next_main = 0usize;
    let mut next_probe = 0usize;
    let mut next_finite = 0usize;
    let mut next_forced = 0usize;
    for (global, (actual, expected)) in trace.operation_log.iter().zip(wanted).enumerate() {
        let key = format!("operations[{global}]");
        match *actual {
            TeacherOperation::MainAppend { main_append_index } => {
                r.check(&format!("{key}.main"), || {
                    if s(expected, "kind")? == "main_append" && main_append_index == next_main {
                        Ok(())
                    } else {
                        Err("main operation differs".into())
                    }
                });
                next_main += 1;
            }
            TeacherOperation::Probe { probe_index } => {
                r.check(&format!("{key}.probe"), || {
                    let probe = trace.probes.get(probe_index).ok_or("probe index")?;
                    if s(expected, "kind")? == "probe"
                        && probe_index == next_probe
                        && probe.global_operation_index == global
                    {
                        Ok(())
                    } else {
                        Err("probe operation differs".into())
                    }
                });
                next_probe += 1;
            }
            TeacherOperation::FiniteChoice {
                finite_choice_index,
            } => {
                r.check(&format!("{key}.finite"), || {
                    if s(expected, "kind")? == "finite_choice" && finite_choice_index == next_finite
                    {
                        Ok(())
                    } else {
                        Err("finite operation differs".into())
                    }
                });
                next_finite += 1;
            }
            TeacherOperation::ForcedArray { forced_array_index } => {
                r.check(&format!("{key}.forced"), || {
                    if s(expected, "kind")? == "forced" && forced_array_index == next_forced {
                        Ok(())
                    } else {
                        Err("forced operation differs".into())
                    }
                });
                next_forced += 1;
            }
        }
    }
    r.eq(
        "operations.main_coverage",
        next_main,
        trace.main.appends.len(),
    );
    r.eq("operations.probe_coverage", next_probe, trace.probes.len());
    r.eq(
        "operations.finite_coverage",
        next_finite,
        trace.finite_choices.len(),
    );
    r.eq(
        "operations.forced_coverage",
        next_forced,
        trace.forced_arrays.len(),
    );
}

fn compare_probe_targets(r: &mut Row, trace: &TeacherTrace, oracle: &Value) {
    let wanted = match a(
        f(oracle, "learned_probe_targets").unwrap_or(&Value::Null),
        "probe targets",
    ) {
        Ok(value) => value,
        Err(error) => {
            r.check("targets.learned_probe", || Err(error));
            return;
        }
    };
    r.eq(
        "targets.learned_probe_count",
        trace.probes.len(),
        wanted.len(),
    );
    for (index, (probe, expected)) in trace.probes.iter().zip(wanted).enumerate() {
        let key = format!("targets.learned_probe[{index}]");
        r.check(&format!("{key}.operation_index"), || {
            if probe.global_operation_index == n(expected, "operation_index")? {
                Ok(())
            } else {
                Err("global operation index differs".into())
            }
        });
        let Some(selected) = probe.candidates.get(probe.selected_index) else {
            r.check(&format!("{key}.candidate"), || {
                Err("selected probe candidate missing".into())
            });
            continue;
        };
        r.check(&format!("{key}.ids"), || {
            summary(f(expected, "selected_token_ids")?, &selected.token_ids)
        });
        r.check(&format!("{key}.positions"), || {
            if selected.prediction_positions == pos(expected)? {
                Ok(())
            } else {
                Err("positions differ".into())
            }
        });
    }
}

fn compare(
    input: &Value,
    expected: &Value,
    m: &HashMap<String, Vec<u32>>,
    p: &TokenPolicy,
) -> R<Row> {
    let oracle = f(expected, "oracle")?;
    let split = s(input, "split")?;
    let id = s(input, "decision_id")?;
    if split != s(expected, "split")? || id != s(expected, "decision_id")? {
        return Err("input/expected identity mismatch".into());
    }
    let events = a(f(input, "event_jsons")?, "events")?
        .iter()
        .map(|x| {
            x.as_str()
                .ok_or_else(|| "event string".into())
                .and_then(event)
        })
        .collect::<R<Vec<_>>>()?;
    let route = s(input, "selected_route")?;
    let plan = normalize_schema_document(
        &route,
        s(input, "schema_json")?.as_bytes(),
        SchemaLimits::default(),
    )
    .map_err(|e| format!("rust_build: schema: {e}"))?;
    let args = parse_runtime_value(s(input, "argument_json")?.as_bytes())
        .map_err(|e| format!("rust_build: arguments: {e}"))?;
    let tokenizer = PinnedTokenizer(m);
    let trace = TeacherTraceBuilder::new(&tokenizer, p)
        .build(
            &TeacherTraceInput {
                public_events: &events,
                selected_route: &route,
                schema_plan: &plan,
            },
            &args,
        )
        .map_err(|e| format!("rust_build: trace: {e}"))?;
    let route_trace = TracePrefixBuilder::new(&tokenizer, p)
        .build_routing(&RoutingTraceInput {
            public_events: &events,
            candidate_name: &route,
            candidate_description: &s(input, "route_description")?,
        })
        .map_err(|e| format!("rust_build: routing: {e}"))?;
    let teacher = plan_teacher(&plan, &args).map_err(|e| format!("rust_build: teacher: {e}"))?;
    let mut r = Row::default();
    r.eq("oracle.mode", s(oracle, "mode")?, "teacher".into());
    r.eq("input.route", route.clone(), s(oracle, "selected_route")?);
    r.check("public_events.semantic", || {
        let actual = events
            .iter()
            .map(|event| {
                event
                    .as_raw()
                    .into_value()
                    .map_err(|error| error.to_string())
            })
            .collect::<R<Vec<_>>>()?;
        let expected = a(f(oracle, "public_events")?, "events")?;
        if actual.len() != expected.len() {
            return Err(format!(
                "/public_events: actual length {} expected {}",
                actual.len(),
                expected.len()
            ));
        }
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            if !semantic_value(actual, expected)? {
                return Err(first_semantic_difference(
                    actual,
                    expected,
                    &format!("/public_events/{index}"),
                )?);
            }
        }
        Ok(())
    });
    r.check("schema.original", || {
        let actual = plan
            .original
            .clone()
            .into_value()
            .map_err(|error| error.to_string())?;
        let expected = f(oracle, "original_schema")?;
        if semantic_value(&actual, expected)? {
            Ok(())
        } else {
            Err(first_semantic_difference(&actual, expected, "")?)
        }
    });
    r.check("schema.resolved", || {
        let actual = plan
            .resolved
            .clone()
            .into_value()
            .map_err(|error| error.to_string())?;
        let expected = f(oracle, "resolved_schema")?;
        if semantic_value(&actual, expected)? {
            Ok(())
        } else {
            Err(first_semantic_difference(&actual, expected, "")?)
        }
    });
    r.check("schema.property_order", || {
        if serde_json::to_value(&plan.property_order).map_err(|e| e.to_string())?
            == *f(oracle, "property_order")?
        {
            Ok(())
        } else {
            Err("order differs".into())
        }
    });
    r.check("arguments.semantic", || {
        let actual = args
            .clone()
            .into_value()
            .map_err(|error| error.to_string())?;
        let expected = f(oracle, "expected_arguments")?;
        if semantic_value(&actual, expected)? {
            Ok(())
        } else {
            Err(first_semantic_difference(&actual, expected, "")?)
        }
    });
    r.check("schema.original_document_preserved", || {
        if plan.original_document.as_deref() == Some(s(input, "schema_json")?.as_bytes()) {
            Ok(())
        } else {
            Err("original schema bytes not retained".into())
        }
    });
    compare_main(&mut r, &trace, oracle, m, p);
    routing(&mut r, &route_trace, f(oracle, "routing")?, m, p);
    compare_operations(&mut r, &trace, oracle);
    let probes = expected_ops(oracle, "probe")?;
    r.eq("probe.count", trace.probes.len(), probes.len());
    for (i, (x, v)) in trace.probes.iter().zip(probes.iter()).enumerate() {
        probe(&mut r, i, x, v, m, p)
    }
    let choices = expected_ops(oracle, "finite_choice")?;
    r.eq("finite.count", trace.finite_choices.len(), choices.len());
    for (i, (x, v)) in trace.finite_choices.iter().zip(choices.iter()).enumerate() {
        finite(&mut r, i, x, v, m, p)
    }
    let unions = probes
        .iter()
        .copied()
        .filter(|x| f(x, "operation").ok().and_then(Value::as_str) == Some("union"))
        .collect::<Vec<_>>();
    r.eq("teacher.union_count", teacher.unions.len(), unions.len());
    for (i, (x, v)) in teacher.unions.iter().zip(unions.iter()).enumerate() {
        let k = format!("teacher.union[{i}]");
        r.check(&format!("{k}.path"), || {
            if x.argument_path == s(v, "path")? {
                Ok(())
            } else {
                Err("union path differs".into())
            }
        });
        r.check(&format!("{k}.selected"), || {
            if x.selected_index == n(v, "selected_index")? {
                Ok(())
            } else {
                Err("union selected differs".into())
            }
        });
        r.check(&format!("{k}.matching"), || {
            let q = f(v, "private_matching_indices")?;
            let z = if q.is_null() {
                Vec::new()
            } else {
                a(q, "matching")?
                    .iter()
                    .map(|x| {
                        usize::try_from(x.as_u64().ok_or_else(|| "matching invalid".to_owned())?)
                            .map_err(|_| "matching too large".into())
                    })
                    .collect::<R<Vec<_>>>()?
            };
            if x.matching_indices == z {
                Ok(())
            } else {
                Err("matching differs".into())
            }
        })
    }
    let main = trace
        .main
        .appends
        .iter()
        .filter(|x| x.ownership == TraceOwnership::Learned)
        .flat_map(|x| {
            x.prediction_positions
                .iter()
                .copied()
                .zip(x.token_ids.iter().copied())
        })
        .collect::<Vec<_>>();
    r.check("targets.learned_main", || {
        let z = a(f(oracle, "learned_main_targets")?, "main targets")?
            .iter()
            .map(|x| Ok((n(x, "position")?, n32(x, "token_id")?)))
            .collect::<R<Vec<_>>>()?;
        if main == z {
            Ok(())
        } else {
            Err("main targets differ".into())
        }
    });
    compare_probe_targets(&mut r, &trace, oracle);
    let forced_values = expected_ops(oracle, "forced")?;
    r.eq(
        "forced.count",
        trace.forced_arrays.len(),
        forced_values.len(),
    );
    for (index, (actual, wanted)) in trace.forced_arrays.iter().zip(forced_values).enumerate() {
        forced(&mut r, index, actual, wanted);
    }
    r.check("targets.denominator", || {
        if trace.learned_token_count == n(oracle, "learned_argument_token_count")? {
            Ok(())
        } else {
            Err("denominator differs".into())
        }
    });
    Ok(r)
}
fn lines(path: &Path) -> R<Vec<Value>> {
    BufReader::new(File::open(path).map_err(|e| e.to_string())?)
        .lines()
        .enumerate()
        .map(|(i, x)| {
            serde_json::from_str(&x.map_err(|e| e.to_string())?)
                .map_err(|e| format!("{}:{} {e}", path.display(), i + 1))
        })
        .collect()
}
fn selected(path: &Path, want: &BTreeSet<String>) -> R<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (i, x) in BufReader::new(File::open(path).map_err(|e| e.to_string())?)
        .lines()
        .enumerate()
    {
        let v: Value = serde_json::from_str(&x.map_err(|e| e.to_string())?)
            .map_err(|e| format!("expected {}: {e}", i + 1))?;
        let id = s(&v, "decision_id")?;
        if want.contains(&id) && out.insert(id.clone(), v).is_some() {
            return Err(format!("duplicate expected {id}"));
        }
    }
    if out.len() == want.len() {
        Ok(out)
    } else {
        Err(format!("missing {} expected rows", want.len() - out.len()))
    }
}
fn report(v: &Value) -> R<()> {
    if let Some(d) = env::var_os(REPORT).map(PathBuf::from) {
        fs::create_dir_all(&d).map_err(|e| e.to_string())?;
        let mut x = File::create(d.join("rust-corpus-trace-bridge-report.json"))
            .map_err(|e| e.to_string())?;
        x.write_all(
            serde_json::to_string_pretty(v)
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
        x.write_all(b"\n").map_err(|e| e.to_string())?
    }
    Ok(())
}
#[test]
fn compare_private_python_teacher_bridge_when_explicitly_requested()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(root) = env::var_os(ROOT).map(PathBuf::from) else {
        eprintln!("skipping private bridge: {ROOT} unset");
        return Ok(());
    };
    let scope = env::var(SCOPE).unwrap_or_else(|_| "representatives".into());
    if scope != "representatives" && scope != "full" {
        return Err(format!("{SCOPE} must be representatives or full").into());
    }
    let began = Instant::now();
    let manifest: Value = serde_json::from_reader(File::open(root.join("manifest.json"))?)?;
    if manifest["schema_version"] != 1
        || manifest["protocol_sha256"] != minifield_decoding_protocol::DRAFT5_ARTIFACT_SHA256
    {
        return Err("bridge manifest schema/protocol mismatch".into());
    }
    let policy = TokenPolicy::draft5()?;
    let map = load_segments(&root.join("segments.jsonl"), &policy)?;
    let mut total = Total::default();
    if scope == "representatives" {
        let input = lines(&root.join("representatives.jsonl"))?;
        let ids = input
            .iter()
            .map(|x| s(x, "decision_id"))
            .collect::<R<BTreeSet<_>>>()?;
        let expected = selected(&root.join("expected.jsonl"), &ids)?;
        for x in &input {
            let id = s(x, "decision_id")?;
            let split = s(x, "split")?;
            total.add(
                &split,
                &id,
                compare(
                    x,
                    expected.get(&id).ok_or("missing expected rep")?,
                    &map,
                    &policy,
                ),
            );
        }
    } else {
        let mut inputs = BufReader::new(File::open(root.join("inputs.jsonl"))?).lines();
        let mut expected = BufReader::new(File::open(root.join("expected.jsonl"))?).lines();
        loop {
            match (inputs.next(), expected.next()) {
                (None, None) => break,
                (Some(Ok(i)), Some(Ok(e))) => {
                    let i: Value = match serde_json::from_str(&i) {
                        Ok(x) => x,
                        Err(e) => {
                            total.add("unknown", "unknown", Err(format!("input JSON {e}")));
                            continue;
                        }
                    };
                    let e: Value = match serde_json::from_str(&e) {
                        Ok(x) => x,
                        Err(e) => {
                            total.add("unknown", "unknown", Err(format!("expected JSON {e}")));
                            continue;
                        }
                    };
                    let id = s(&i, "decision_id").unwrap_or_else(|_| "unknown".into());
                    let split = s(&i, "split").unwrap_or_else(|_| "unknown".into());
                    total.add(&split, &id, compare(&i, &e, &map, &policy));
                }
                (Some(Err(e)), _) | (_, Some(Err(e))) => return Err(e.into()),
                _ => {
                    total.add(
                        "unknown",
                        "unknown",
                        Err("input/expected stream count mismatch".into()),
                    );
                    break;
                }
            }
        }
    }
    let want = if scope == "representatives" {
        manifest["representatives"].as_u64()
    } else {
        manifest["total"].as_u64()
    }
    .ok_or("missing manifest count")?;
    let out = json!({"schema_version":1,"scope":scope,"bridge":root,"bridge_manifest_sha256":"7475c2bc8c11b4b4fea1b4f222dad047ce1f07b91bc21a268c80dacd7053a5ea","expected_rows":want,"rows":total.rows,"rows_ok":total.ok,"rows_failed":total.bad,"malformed_or_preflight_rows":total.malformed,"rust_build_failures":total.build,"semantic_issue_rows_by_field":total.fields,"executed_check_counts":total.checks,"debug_label_differences":total.labels,"unqualified_gates":[],"detailed_failures_first_25":total.detail,"elapsed_millis":began.elapsed().as_millis()});
    report(&out)?;
    if total.rows != want {
        return Err(format!("bridge rows {} != manifest {want}", total.rows).into());
    }
    if total.bad != 0 {
        return Err(format!(
            "{} bridge rows mismatched; report written if {REPORT} set",
            total.bad
        )
        .into());
    }
    Ok(())
}

#[test]
fn bridge_semantic_value_controls_binary64_without_type_coercion()
-> Result<(), Box<dyn std::error::Error>> {
    let one: Value = serde_json::from_str("1")?;
    let one_point_zero: Value = serde_json::from_str("1.0")?;
    let boolean: Value = serde_json::from_str("true")?;
    let changed: Value = serde_json::from_str("2")?;
    let first: Value = serde_json::from_str("[1,2]")?;
    let reversed: Value = serde_json::from_str("[2,1]")?;
    assert!(semantic_value(&one, &one_point_zero)?);
    assert!(!semantic_value(&one, &boolean)?);
    assert!(!semantic_value(&one, &changed)?);
    assert!(!semantic_value(&first, &reversed)?);
    Ok(())
}
