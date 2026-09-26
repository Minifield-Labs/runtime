//! Immutable effective-schema choice wrapping with source-path provenance.
use crate::{ProtocolError, RawJson, Result};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChoiceKind {
    AnyOf,
    OneOf,
    TypeArray,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectiveLimits {
    pub max_bytes: usize,
}
impl Default for EffectiveLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaPathOrigin {
    /// Pointer in the current effective tree.
    pub current_path: String,
    /// Pointer in the immutable resolved source tree.
    pub source_path: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveAlternative {
    pub label: String,
    pub schema_path: String,
    /// The complete wrapped effective schema, never flattened.
    pub schema: RawJson,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveSelection {
    pub kind: ChoiceKind,
    /// Immutable resolved-source path of the choice keyword.
    pub source_keyword_path: String,
    /// Current-tree location at the moment this choice was selected.
    pub current_keyword_path: String,
    pub selected_branch_index: usize,
    pub alternative: EffectiveAlternative,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveProvenance {
    /// Immutable full resolved source. This is independent of the derived tree.
    pub source_resolved_schema: RawJson,
    /// All surviving schema nodes map to their immutable source locations.
    pub current_to_source: Vec<SchemaPathOrigin>,
    /// Prior selections retain their original current and source locations.
    pub selections: Vec<EffectiveSelection>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveTree {
    /// Complete current tree. Every selection wraps, it never flattens.
    pub schema: RawJson,
    pub provenance: EffectiveProvenance,
    pub limits: EffectiveLimits,
    /// Conservative byte accounting for all retained tree and provenance data.
    pub charged_bytes: usize,
}

impl EffectiveTree {
    /// Start with an immutable resolved schema and its identity path map.
    pub fn from_resolved(resolved: &RawJson) -> Result<Self> {
        Self::from_resolved_with_limits(resolved, EffectiveLimits::default())
    }

    /// Start with an explicit bound before cloning source schema or provenance.
    pub fn from_resolved_with_limits(resolved: &RawJson, limits: EffectiveLimits) -> Result<Self> {
        let mut origins = Vec::new();
        map_schema_nodes(resolved, "", "", &mut origins);
        sort_origins(&mut origins);
        let source_bytes = raw_bytes(resolved)?;
        let origin_bytes = origins_bytes(&origins)?;
        let charged_bytes = checked_add(checked_add(source_bytes, source_bytes)?, origin_bytes)?;
        if charged_bytes > limits.max_bytes {
            return Err(ProtocolError::InputLimit(format!(
                "effective schema initial state requires {charged_bytes} bytes, limit is {}",
                limits.max_bytes
            )));
        }
        Ok(Self {
            schema: resolved.clone(),
            provenance: EffectiveProvenance {
                source_resolved_schema: resolved.clone(),
                current_to_source: origins,
                selections: Vec::new(),
            },
            limits,
            charged_bytes,
        })
    }

    /// Return the immutable source path associated with a current schema node.
    #[must_use]
    pub fn source_path_for(&self, current_path: &str) -> Option<&str> {
        self.provenance
            .current_to_source
            .iter()
            .find(|origin| origin.current_path == current_path)
            .map(|origin| origin.source_path.as_str())
    }

    /// Select exactly one current union/type branch and return a new immutable
    /// effective tree. The receiver, source tree, and old provenance remain
    /// unchanged.
    #[allow(clippy::too_many_lines)]
    pub fn select_choice(
        &self,
        current_keyword_path: &str,
        selected_branch_index: usize,
    ) -> Result<Self> {
        let (parent_path, keyword) = split_keyword_path(current_keyword_path)?;
        let source_parent = self.source_path_for(&parent_path).ok_or_else(|| {
            ProtocolError::Schema(format!(
                "effective choice parent {parent_path} has no immutable source path"
            ))
        })?;
        let source_keyword_path = append(source_parent, &keyword);
        let scope_path = choice_scope_path(current_keyword_path)?;
        // The derived tree retains a new source clone, an effective wrapper,
        // an alternative copy, and copied provenance. Charge a conservative
        // complete-state allowance before any schema/provenance clone.
        let preflight = checked_add(
            self.charged_bytes,
            checked_add(
                raw_bytes(&self.schema)?.checked_mul(3).ok_or_else(|| {
                    ProtocolError::InputLimit("effective byte multiplier overflow".to_owned())
                })?,
                4096,
            )?,
        )?;
        if preflight > self.limits.max_bytes {
            return Err(ProtocolError::InputLimit(format!(
                "effective choice would require at least {preflight} bytes, limit is {}",
                self.limits.max_bytes
            )));
        }

        let mut base_tree = self.schema.clone();
        let parent = pointer_mut(&mut base_tree, &decode_pointer(&parent_path)?)?;
        let RawJson::Object(entries) = parent else {
            return Err(ProtocolError::Schema(format!(
                "effective choice parent {parent_path} is not an object"
            )));
        };
        let position = entries
            .iter()
            .position(|(name, _)| name == &keyword)
            .ok_or_else(|| {
                ProtocolError::Schema(format!("effective choice {current_keyword_path} is absent"))
            })?;
        let (kind, branch) = match &entries[position].1 {
            RawJson::Array(branches) if keyword == "anyOf" => (
                ChoiceKind::AnyOf,
                branches.get(selected_branch_index).cloned().ok_or_else(|| {
                    ProtocolError::Schema(format!(
                        "effective anyOf branch {selected_branch_index} is absent at {current_keyword_path}"
                    ))
                })?,
            ),
            RawJson::Array(branches) if keyword == "oneOf" => (
                ChoiceKind::OneOf,
                branches.get(selected_branch_index).cloned().ok_or_else(|| {
                    ProtocolError::Schema(format!(
                        "effective oneOf branch {selected_branch_index} is absent at {current_keyword_path}"
                    ))
                })?,
            ),
            RawJson::Array(types) if keyword == "type" => {
                let RawJson::String(name) = types.get(selected_branch_index).ok_or_else(|| {
                    ProtocolError::Schema(format!(
                        "effective type branch {selected_branch_index} is absent at {current_keyword_path}"
                    ))
                })? else {
                    return Err(ProtocolError::Schema(format!(
                        "effective type choice {current_keyword_path} contains a non-string"
                    )));
                };
                (
                    ChoiceKind::TypeArray,
                    RawJson::Object(vec![("type".to_owned(), RawJson::String(name.clone()))]),
                )
            }
            _ => {
                return Err(ProtocolError::Schema(format!(
                    "effective path {current_keyword_path} is not an anyOf, oneOf, or type-array choice"
                )));
            }
        };
        entries.remove(position);

        let scope_tokens = decode_pointer(&scope_path)?;
        let base = pointer(&base_tree, &scope_tokens)?.clone();
        let effective_branch = RawJson::Object(vec![(
            "allOf".to_owned(),
            RawJson::Array(vec![base, branch.clone()]),
        )]);
        let schema = if scope_path.is_empty() {
            effective_branch.clone()
        } else {
            *pointer_mut(&mut base_tree, &scope_tokens)? = effective_branch.clone();
            base_tree
        };

        let branch_source_path = append(&source_keyword_path, &selected_branch_index.to_string());
        let base_prefix = append_two(&scope_path, "allOf", "0");
        let branch_prefix = append_two(&scope_path, "allOf", "1");
        let mut origins = Vec::new();
        for origin in &self.provenance.current_to_source {
            if is_at_or_below(&origin.current_path, current_keyword_path) {
                continue;
            }
            let current_path = if scope_path.is_empty() {
                format!("/allOf/0{}", origin.current_path)
            } else if is_at_or_below(&origin.current_path, &scope_path) {
                let suffix = &origin.current_path[scope_path.len()..];
                format!("{base_prefix}{suffix}")
            } else {
                origin.current_path.clone()
            };
            origins.push(SchemaPathOrigin {
                current_path,
                source_path: origin.source_path.clone(),
            });
        }
        map_schema_nodes(&branch, &branch_prefix, &branch_source_path, &mut origins);
        sort_origins(&mut origins);

        let alternative = EffectiveAlternative {
            label: format!("branch:{selected_branch_index}"),
            schema_path: branch_source_path,
            schema: effective_branch,
        };
        let mut selections = self.provenance.selections.clone();
        selections.push(EffectiveSelection {
            kind,
            source_keyword_path,
            current_keyword_path: current_keyword_path.to_owned(),
            selected_branch_index,
            alternative: alternative.clone(),
        });
        let provenance = EffectiveProvenance {
            source_resolved_schema: self.provenance.source_resolved_schema.clone(),
            current_to_source: origins,
            selections,
        };
        let charged_bytes = state_bytes(&schema, &provenance)?;
        if charged_bytes > self.limits.max_bytes {
            return Err(ProtocolError::InputLimit(format!(
                "effective choice retained state requires {charged_bytes} bytes, limit is {}",
                self.limits.max_bytes
            )));
        }
        Ok(Self {
            schema,
            provenance,
            limits: self.limits,
            charged_bytes,
        })
    }
}

fn map_schema_nodes(
    schema: &RawJson,
    current: &str,
    source: &str,
    output: &mut Vec<SchemaPathOrigin>,
) {
    output.push(SchemaPathOrigin {
        current_path: current.to_owned(),
        source_path: source.to_owned(),
    });
    let RawJson::Object(entries) = schema else {
        return;
    };
    for (key, value) in entries {
        match key.as_str() {
            "properties" => {
                let RawJson::Object(properties) = value else {
                    continue;
                };
                for (name, child) in properties {
                    map_schema_nodes(
                        child,
                        &append_two(current, "properties", name),
                        &append_two(source, "properties", name),
                        output,
                    );
                }
            }
            "additionalProperties" | "items" => {
                map_schema_nodes(value, &append(current, key), &append(source, key), output);
            }
            "allOf" | "anyOf" | "oneOf" => {
                let RawJson::Array(branches) = value else {
                    continue;
                };
                for (index, child) in branches.iter().enumerate() {
                    map_schema_nodes(
                        child,
                        &append_two(current, key, &index.to_string()),
                        &append_two(source, key, &index.to_string()),
                        output,
                    );
                }
            }
            _ => {}
        }
    }
}
fn sort_origins(origins: &mut Vec<SchemaPathOrigin>) {
    origins.sort_by(|left, right| utf16_compare(&left.current_path, &right.current_path));
    origins.dedup_by(|left, right| left.current_path == right.current_path);
}

fn is_at_or_below(current: &str, parent: &str) -> bool {
    current == parent
        || current
            .strip_prefix(parent)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

/// Return the current-tree schema path whose instance value the choice
/// constrains. Root-level conjunction branches constrain the root; a property,
/// item, or additional-properties schema starts a new value scope.
fn choice_scope_path(current_keyword_path: &str) -> Result<String> {
    let tokens = decode_pointer(current_keyword_path)?;
    let mut scope_len = 0usize;
    let mut index = 0usize;
    while index < tokens.len() {
        match tokens[index].as_str() {
            "properties" if index + 1 < tokens.len() => {
                scope_len = index + 2;
                index += 2;
            }
            "items" | "additionalProperties" if index + 1 < tokens.len() => {
                scope_len = index + 1;
                index += 1;
            }
            _ => index += 1,
        }
    }
    Ok(tokens[..scope_len]
        .iter()
        .fold(String::new(), |path, token| append(&path, token)))
}

fn split_keyword_path(path: &str) -> Result<(String, String)> {
    let index = path.rfind('/').ok_or_else(|| {
        ProtocolError::Schema("effective choice path must be a non-root RFC6901 pointer".to_owned())
    })?;
    if index == path.len() - 1 {
        return Err(ProtocolError::Schema(
            "effective choice path may not end with an empty keyword".to_owned(),
        ));
    }
    let parent = path[..index].to_owned();
    let keyword = decode_token(&path[index + 1..])?;
    Ok((parent, keyword))
}

fn pointer<'a>(value: &'a RawJson, tokens: &[String]) -> Result<&'a RawJson> {
    if tokens.is_empty() {
        return Ok(value);
    }
    match value {
        RawJson::Object(entries) => {
            let (_, child) = entries
                .iter()
                .find(|(name, _)| name == &tokens[0])
                .ok_or_else(|| {
                    ProtocolError::Schema(format!(
                        "effective pointer component {:?} is absent",
                        tokens[0]
                    ))
                })?;
            pointer(child, &tokens[1..])
        }
        RawJson::Array(values) => {
            let index = pointer_index(&tokens[0]).ok_or_else(|| {
                ProtocolError::Schema(format!(
                    "effective pointer component {:?} is not an array index",
                    tokens[0]
                ))
            })?;
            let child = values.get(index).ok_or_else(|| {
                ProtocolError::Schema(format!("effective pointer array index {index} is absent"))
            })?;
            pointer(child, &tokens[1..])
        }
        _ => Err(ProtocolError::Schema(
            "effective pointer crosses a scalar".to_owned(),
        )),
    }
}

fn pointer_mut<'a>(value: &'a mut RawJson, tokens: &[String]) -> Result<&'a mut RawJson> {
    if tokens.is_empty() {
        return Ok(value);
    }
    match value {
        RawJson::Object(entries) => {
            let (_, child) = entries
                .iter_mut()
                .find(|(name, _)| name == &tokens[0])
                .ok_or_else(|| {
                    ProtocolError::Schema(format!(
                        "effective pointer component {:?} is absent",
                        tokens[0]
                    ))
                })?;
            pointer_mut(child, &tokens[1..])
        }
        RawJson::Array(values) => {
            let index = pointer_index(&tokens[0]).ok_or_else(|| {
                ProtocolError::Schema(format!(
                    "effective pointer component {:?} is not an array index",
                    tokens[0]
                ))
            })?;
            let child = values.get_mut(index).ok_or_else(|| {
                ProtocolError::Schema(format!("effective pointer array index {index} is absent"))
            })?;
            pointer_mut(child, &tokens[1..])
        }
        _ => Err(ProtocolError::Schema(
            "effective pointer crosses a scalar".to_owned(),
        )),
    }
}

fn decode_pointer(pointer: &str) -> Result<Vec<String>> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    if !pointer.starts_with('/') {
        return Err(ProtocolError::Schema(
            "effective pointer is not RFC6901".to_owned(),
        ));
    }
    pointer[1..].split('/').map(decode_token).collect()
}

fn decode_token(token: &str) -> Result<String> {
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
                    "invalid RFC6901 escape in effective pointer".to_owned(),
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

fn append(parent: &str, token: &str) -> String {
    format!("{parent}/{}", token.replace('~', "~0").replace('/', "~1"))
}

fn append_two(parent: &str, first: &str, second: &str) -> String {
    append(&append(parent, first), second)
}

fn utf16_compare(left: &str, right: &str) -> Ordering {
    left.encode_utf16().cmp(right.encode_utf16())
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| ProtocolError::InputLimit("effective byte counter overflow".to_owned()))
}

fn origins_bytes(origins: &[SchemaPathOrigin]) -> Result<usize> {
    origins.iter().try_fold(0usize, |total, origin| {
        checked_add(
            total,
            checked_add(origin.current_path.len(), origin.source_path.len())?,
        )
    })
}

fn state_bytes(schema: &RawJson, provenance: &EffectiveProvenance) -> Result<usize> {
    let mut total = checked_add(
        raw_bytes(schema)?,
        raw_bytes(&provenance.source_resolved_schema)?,
    )?;
    total = checked_add(total, origins_bytes(&provenance.current_to_source)?)?;
    for selection in &provenance.selections {
        total = checked_add(total, selection.source_keyword_path.len())?;
        total = checked_add(total, selection.current_keyword_path.len())?;
        total = checked_add(total, selection.alternative.label.len())?;
        total = checked_add(total, selection.alternative.schema_path.len())?;
        total = checked_add(total, raw_bytes(&selection.alternative.schema)?)?;
    }
    Ok(total)
}

fn raw_bytes(value: &RawJson) -> Result<usize> {
    match value {
        RawJson::Null | RawJson::Bool(true) => Ok(4),
        RawJson::Bool(false) => Ok(5),
        RawJson::Number(number) => Ok(number.as_str().len()),
        RawJson::String(value) => checked_add(
            2,
            value.len().checked_mul(6).ok_or_else(|| {
                ProtocolError::InputLimit("effective string byte count overflow".to_owned())
            })?,
        ),
        RawJson::Array(values) => {
            values
                .iter()
                .enumerate()
                .try_fold(2usize, |total, (index, value)| {
                    let total = if index == 0 {
                        total
                    } else {
                        checked_add(total, 1)?
                    };
                    checked_add(total, raw_bytes(value)?)
                })
        }
        RawJson::Object(entries) => {
            entries
                .iter()
                .enumerate()
                .try_fold(2usize, |total, (index, (key, value))| {
                    let total = if index == 0 {
                        total
                    } else {
                        checked_add(total, 1)?
                    };
                    checked_add(
                        checked_add(checked_add(total, key.len())?, 3)?,
                        raw_bytes(value)?,
                    )
                })
        }
    }
}
