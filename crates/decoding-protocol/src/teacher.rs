//! Schema-driven teacher union discovery, independent of fixture operations.
use crate::{
    EffectiveAlternative, EffectiveTree, RawJson, Result, SchemaPlan, ValidationFailureKind,
    ValidationLimits,
};

/// Immutable selected schema tree and schema-derived union decisions for one
/// admitted teacher target. The tree is retained so structural planning can use
/// selected root and nested conjunctions without consulting fixture operations.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherPlan {
    /// Root and non-repeated descendants with their selected immutable choices.
    pub effective: EffectiveTree,
    /// Every selected choice in deterministic depth-first argument order.
    pub unions: Vec<TeacherUnionChoice>,
    /// Ordinals in `unions` selected under independently cloned array-item cursors.
    /// They must be applied only at their matching argument pointer during walking.
    pub instance_union_ordinals: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeacherUnionChoice {
    pub argument_path: String,
    pub source_keyword_path: String,
    pub current_keyword_path: String,
    pub selected_index: usize,
    pub matching_indices: Vec<usize>,
    pub alternatives: Vec<EffectiveAlternative>,
}

/// Plan all schema choices for an admitted teacher target.
pub fn plan_teacher(plan: &SchemaPlan, arguments: &RawJson) -> Result<TeacherPlan> {
    plan.validate_original_instance(arguments)
        .map_err(|failure| crate::ProtocolError::Schema(failure.to_string()))?;
    let mut tree = plan.effective_tree()?;
    let mut unions = Vec::new();
    let mut instance_union_ordinals = Vec::new();
    walk(
        &mut tree,
        String::new(),
        String::new(),
        arguments,
        &mut unions,
        &mut instance_union_ordinals,
        false,
    )?;
    Ok(TeacherPlan {
        effective: tree,
        unions,
        instance_union_ordinals,
    })
}

/// Discover union choices only. Prefer plan_teacher when subsequent structural
/// planning also needs the immutable selected effective tree.
pub fn discover_teacher_unions(
    plan: &SchemaPlan,
    arguments: &RawJson,
) -> Result<Vec<TeacherUnionChoice>> {
    Ok(plan_teacher(plan, arguments)?.unions)
}

#[allow(clippy::needless_pass_by_value)]
fn walk(
    tree: &mut EffectiveTree,
    schema_path: String,
    argument_path: String,
    value: &RawJson,
    output: &mut Vec<TeacherUnionChoice>,
    instance_union_ordinals: &mut Vec<usize>,
    isolated_array_item: bool,
) -> Result<()> {
    // Match trace emission precedence: an already-applicable finite value
    // consumes this complete node before any same-node or descendant choice.
    if has_finite_domain(tree, &schema_path)? {
        return Ok(());
    }
    select_choices(
        tree,
        &schema_path,
        &argument_path,
        value,
        output,
        instance_union_ordinals,
        isolated_array_item,
    )?;
    // A selected same-node branch can introduce a finite domain. It likewise
    // consumes the full value, so no child choice may be planned afterwards.
    if has_finite_domain(tree, &schema_path)? {
        return Ok(());
    }
    walk_children(
        tree,
        &schema_path,
        &argument_path,
        value,
        output,
        instance_union_ordinals,
        isolated_array_item,
    )
}

fn select_choices(
    tree: &mut EffectiveTree,
    schema_path: &str,
    argument_path: &str,
    value: &RawJson,
    output: &mut Vec<TeacherUnionChoice>,
    instance_union_ordinals: &mut Vec<usize>,
    isolated_array_item: bool,
) -> Result<()> {
    loop {
        let Some(choice) = next_choice(tree, schema_path)? else {
            break;
        };
        let selected = select_choice(tree, schema_path, value, &choice)?;
        *tree = tree.select_choice(&choice.current_keyword_path, selected.selected_index)?;
        let ordinal = output.len();
        output.push(TeacherUnionChoice {
            argument_path: argument_path.to_owned(),
            source_keyword_path: choice.source_keyword_path,
            current_keyword_path: choice.current_keyword_path,
            selected_index: selected.selected_index,
            matching_indices: selected.matching_indices,
            alternatives: selected.alternatives,
        });
        if isolated_array_item {
            instance_union_ordinals.push(ordinal);
        }
    }
    Ok(())
}

fn has_finite_domain(tree: &EffectiveTree, schema_path: &str) -> Result<bool> {
    Ok(schema_has_finite(schema_at(&tree.schema, schema_path)?))
}

fn schema_has_finite(schema: &RawJson) -> bool {
    let RawJson::Object(entries) = schema else {
        return false;
    };
    entries.iter().any(|(key, value)| match key.as_str() {
        "const" | "enum" => true,
        "allOf" => {
            matches!(value, RawJson::Array(branches) if branches.iter().any(schema_has_finite))
        }
        _ => false,
    })
}

fn walk_children(
    tree: &mut EffectiveTree,
    schema_path: &str,
    argument_path: &str,
    value: &RawJson,
    output: &mut Vec<TeacherUnionChoice>,
    instance_union_ordinals: &mut Vec<usize>,
    isolated_array_item: bool,
) -> Result<()> {
    let schema = schema_at(&tree.schema, schema_path)?;
    match value {
        RawJson::Object(arguments) => {
            let mut properties = Vec::new();
            collect_properties(schema, schema_path, &mut properties);
            properties.sort_by(|left, right| {
                left.0
                    .encode_utf16()
                    .cmp(right.0.encode_utf16())
                    .then_with(|| left.1.encode_utf16().cmp(right.1.encode_utf16()))
            });
            properties.dedup_by(|left, right| left.1 == right.1);
            for (name, child_schema_path) in properties {
                if let Some((_, child)) = arguments.iter().find(|(key, _)| key == &name) {
                    walk(
                        tree,
                        child_schema_path,
                        append(argument_path, &name),
                        child,
                        output,
                        instance_union_ordinals,
                        isolated_array_item,
                    )?;
                }
            }
        }
        RawJson::Array(values) => {
            if is_wholly_dynamic_schema(schema) {
                return Ok(());
            }
            let mut items = Vec::new();
            collect_items(schema, schema_path, &mut items);
            items.sort_by(|left, right| left.encode_utf16().cmp(right.encode_utf16()));
            items.dedup();
            for (index, child) in values.iter().enumerate() {
                let item_path = append(argument_path, &index.to_string());
                let mut item_tree = tree.clone();
                for item_schema_path in &items {
                    walk(
                        &mut item_tree,
                        item_schema_path.clone(),
                        item_path.clone(),
                        child,
                        output,
                        instance_union_ordinals,
                        true,
                    )?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

struct SelectedChoice {
    selected_index: usize,
    matching_indices: Vec<usize>,
    alternatives: Vec<EffectiveAlternative>,
}

fn select_choice(
    tree: &EffectiveTree,
    schema_path: &str,
    value: &RawJson,
    choice: &ChoiceLocation,
) -> Result<SelectedChoice> {
    let mut alternatives = Vec::new();
    let mut matching_indices = Vec::new();
    for index in 0..choice.count {
        let alternative_tree = tree.select_choice(&choice.current_keyword_path, index)?;
        let alternative = alternative_tree
            .provenance
            .selections
            .last()
            .expect("selection is retained")
            .alternative
            .clone();
        let local_schema = schema_at(&alternative_tree.schema, schema_path)?;
        match crate::validation::validate_admitted_original_instance_with_limits(
            local_schema,
            value,
            ValidationLimits::default(),
        ) {
            Ok(()) => matching_indices.push(index),
            Err(failure) if failure.class == ValidationFailureKind::InstanceInvalid => {}
            Err(failure) => {
                return Err(crate::ProtocolError::Schema(format!(
                    "teacher union branch {index} at {} failed operationally: {failure}",
                    choice.source_keyword_path
                )));
            }
        }
        alternatives.push(alternative);
    }
    let selected_index = if choice.one_of {
        if matching_indices.len() == 1 {
            matching_indices[0]
        } else {
            return Err(crate::ProtocolError::Schema(format!(
                "teacher oneOf at {} matches {} branches",
                choice.source_keyword_path,
                matching_indices.len()
            )));
        }
    } else {
        *matching_indices.first().ok_or_else(|| {
            crate::ProtocolError::Schema(format!(
                "teacher union at {} has no matching branch",
                choice.source_keyword_path
            ))
        })?
    };
    Ok(SelectedChoice {
        selected_index,
        matching_indices,
        alternatives,
    })
}

fn is_wholly_dynamic_schema(schema: &RawJson) -> bool {
    match schema {
        RawJson::Bool(value) => *value,
        RawJson::Object(entries) => entries.iter().all(|(key, value)| match key.as_str() {
            "title" | "description" | "default" | "examples" | "$comment" | "deprecated"
            | "readOnly" | "writeOnly" => true,
            "allOf" => matches!(value, RawJson::Array(branches) if branches.iter().all(is_wholly_dynamic_schema)),
            _ => false,
        }),
        _ => false,
    }
}

fn collect_items(schema: &RawJson, path: &str, output: &mut Vec<String>) {
    let RawJson::Object(entries) = schema else {
        return;
    };
    for (key, value) in entries {
        match key.as_str() {
            "allOf" => {
                if let RawJson::Array(branches) = value {
                    for (index, branch) in branches.iter().enumerate() {
                        collect_items(branch, &append2(path, "allOf", &index.to_string()), output);
                    }
                }
            }
            "items" => output.push(append(path, "items")),
            _ => {}
        }
    }
}

struct ChoiceLocation {
    current_keyword_path: String,
    source_keyword_path: String,
    count: usize,
    one_of: bool,
}

fn next_choice(tree: &EffectiveTree, path: &str) -> Result<Option<ChoiceLocation>> {
    let mut candidates = Vec::new();
    collect_choices(schema_at(&tree.schema, path)?, path, tree, &mut candidates);
    candidates.sort_by(|left, right| {
        left.source_keyword_path
            .encode_utf16()
            .cmp(right.source_keyword_path.encode_utf16())
    });
    Ok(candidates.into_iter().next())
}

fn collect_choices(
    schema: &RawJson,
    path: &str,
    tree: &EffectiveTree,
    output: &mut Vec<ChoiceLocation>,
) {
    let RawJson::Object(entries) = schema else {
        return;
    };
    for (key, value) in entries {
        match key.as_str() {
            "allOf" => {
                if let RawJson::Array(branches) = value {
                    for (index, branch) in branches.iter().enumerate() {
                        collect_choices(
                            branch,
                            &append2(path, "allOf", &index.to_string()),
                            tree,
                            output,
                        );
                    }
                }
            }
            "anyOf" | "oneOf" => {
                if let RawJson::Array(branches) = value {
                    let parent_source = tree.source_path_for(path).unwrap_or("");
                    output.push(ChoiceLocation {
                        current_keyword_path: append(path, key),
                        source_keyword_path: append(parent_source, key),
                        count: branches.len(),
                        one_of: key == "oneOf",
                    });
                }
            }
            "type" => {
                if let RawJson::Array(types) = value {
                    let parent_source = tree.source_path_for(path).unwrap_or("");
                    output.push(ChoiceLocation {
                        current_keyword_path: append(path, key),
                        source_keyword_path: append(parent_source, key),
                        count: types.len(),
                        one_of: false,
                    });
                }
            }
            _ => {}
        }
    }
}

fn collect_properties(schema: &RawJson, path: &str, output: &mut Vec<(String, String)>) {
    let RawJson::Object(entries) = schema else {
        return;
    };
    for (key, value) in entries {
        match key.as_str() {
            "allOf" => {
                if let RawJson::Array(branches) = value {
                    for (index, branch) in branches.iter().enumerate() {
                        collect_properties(
                            branch,
                            &append2(path, "allOf", &index.to_string()),
                            output,
                        );
                    }
                }
            }
            "properties" => {
                if let RawJson::Object(properties) = value {
                    for (name, _) in properties {
                        output.push((name.clone(), append2(path, "properties", name)));
                    }
                }
            }
            _ => {}
        }
    }
}

fn schema_at<'a>(mut schema: &'a RawJson, pointer: &str) -> Result<&'a RawJson> {
    if pointer.is_empty() {
        return Ok(schema);
    }
    for token in pointer[1..].split('/') {
        let token = token.replace("~1", "/").replace("~0", "~");
        schema = match schema {
            RawJson::Object(entries) => entries
                .iter()
                .find(|(key, _)| key == &token)
                .map(|(_, value)| value),
            RawJson::Array(values) => token
                .parse::<usize>()
                .ok()
                .and_then(|index| values.get(index)),
            _ => None,
        }
        .ok_or_else(|| {
            crate::ProtocolError::Schema(format!("missing effective schema pointer {pointer}"))
        })?;
    }
    Ok(schema)
}

fn append(parent: &str, token: &str) -> String {
    format!("{parent}/{}", token.replace('~', "~0").replace('/', "~1"))
}
fn append2(parent: &str, one: &str, two: &str) -> String {
    append(&append(parent, one), two)
}
