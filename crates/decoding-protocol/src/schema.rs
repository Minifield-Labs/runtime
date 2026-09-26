//! Deterministic draft-5 local-schema resolution and property-order planning.
//!
//! The resolved tree remains distinct from immutable input. It expands local
//! references through explicit two-element allOf wrappers, removes definitions,
//! and keeps declared property vectors out of JCS object sorting.

use crate::{
    CanonicalDecimal, EcmaPattern, NumericKind, ProtocolError, RawJson, RawJsonLimits, Result,
    admit_number, parse_json_document, safe_json, sha256_hex,
};
use serde::Serialize;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashSet;

const DRAFT202012: &str = "https://json-schema.org/draft/2020-12/schema";
const SUPPORTED: &[&str] = &[
    "$schema",
    "$ref",
    "$defs",
    "type",
    "const",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "allOf",
    "anyOf",
    "oneOf",
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Explicit schema resolution limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchemaLimits {
    /// Maximum schema nodes visited, including discarded definitions.
    pub max_nodes: usize,
    /// Maximum schema nesting depth.
    pub max_depth: usize,
    /// Maximum active local-reference chain.
    pub max_reference_depth: usize,
    /// Maximum public property-order declarations.
    pub max_property_declarations: usize,
    /// Maximum bytes charged to expanded resolved schema plus property-order metadata.
    pub max_resolved_bytes: usize,
}

impl Default for SchemaLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_depth: 64,
            max_reference_depth: 64,
            max_property_declarations: 100_000,
            max_resolved_bytes: 16 * 1024 * 1024,
        }
    }
}

/// A public declared-property vector. The schema path is RFC6901 in resolved
/// schema space, where the root is the empty string.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PropertyOrder {
    /// Resolved-schema pointer.
    pub schema_path: String,
    /// Original declaration insertion order.
    pub properties: Vec<String>,
}

/// Original and resolved schema artifacts plus independent property order.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaPlan {
    /// Immutable input schema, with raw numbers and property order preserved.
    pub original: RawJson,
    /// Exact source bytes when this plan came from strict document ingestion.
    pub original_document: Option<Vec<u8>>,
    /// SHA-256 of exact source bytes, when a source document was supplied.
    pub original_document_sha256: Option<String>,
    /// SHA-256 of the admitted RFC8785 semantic schema before SafeJSON escaping.
    pub interpreted_schema_sha256: String,
    /// Resolved tree without definition registries.
    pub resolved: RawJson,
    /// UTF-16 sorted resolved declaration order entries.
    pub property_order: Vec<PropertyOrder>,
}

impl SchemaPlan {
    /// Safe canonical bytes for the resolved public schema.
    pub fn resolved_safe_json(&self) -> Result<Vec<u8>> {
        safe_json(&self.resolved.clone().into_value()?)
    }

    /// RFC8785 bytes of the admitted original schema before SafeJSON escaping.
    pub fn interpreted_canonical_json(&self) -> Result<Vec<u8>> {
        serde_jcs::to_vec(&self.original.clone().into_value()?).map_err(ProtocolError::from)
    }

    /// Safe canonical bytes for the immutable original schema.
    pub fn original_safe_json(&self) -> Result<Vec<u8>> {
        safe_json(&self.original.clone().into_value()?)
    }

    /// Safe canonical bytes for the separate property-order vector.
    pub fn property_order_safe_json(&self) -> Result<Vec<u8>> {
        let value: Value = serde_json::to_value(&self.property_order)
            .map_err(|error| ProtocolError::InvalidJson(error.to_string()))?;
        safe_json(&value)
    }
}

/// Strictly parse and normalize one immutable JSON Schema draft2020-12 source document.
pub fn normalize_schema_document(
    tool: &str,
    source: &[u8],
    limits: SchemaLimits,
) -> Result<SchemaPlan> {
    let original = parse_json_document(source, RawJsonLimits::default())?;
    let mut plan = normalize_schema(tool, original, limits)?;
    plan.original_document_sha256 = Some(sha256_hex(source));
    plan.original_document = Some(source.to_vec());
    Ok(plan)
}

/// Normalize one local immutable JSON Schema draft2020-12 document.
pub fn normalize_schema(tool: &str, original: RawJson, limits: SchemaLimits) -> Result<SchemaPlan> {
    check_dialect(tool, &original)?;
    let mut normalizer = Normalizer {
        tool,
        original: &original,
        limits,
        visited: 0,
        emitted_bytes: 0,
        property_order: Vec::new(),
    };
    let mut references = Vec::new();
    let resolved = normalizer.resolve(&original, "", true, 0, &mut references)?;
    normalizer
        .property_order
        .sort_by(|left, right| utf16_compare(&left.schema_path, &right.schema_path));
    let property_order = std::mem::take(&mut normalizer.property_order);
    drop(normalizer);
    let interpreted_schema_sha256 = sha256_hex(
        &serde_jcs::to_vec(&original.clone().into_value()?).map_err(ProtocolError::from)?,
    );
    Ok(SchemaPlan {
        original,
        original_document: None,
        original_document_sha256: None,
        interpreted_schema_sha256,
        resolved,
        property_order,
    })
}

struct Normalizer<'a> {
    tool: &'a str,
    original: &'a RawJson,
    limits: SchemaLimits,
    visited: usize,
    emitted_bytes: usize,
    property_order: Vec<PropertyOrder>,
}

impl Normalizer<'_> {
    fn resolve(
        &mut self,
        schema: &RawJson,
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<RawJson> {
        if depth > self.limits.max_depth {
            return Err(self.error(path, "schema depth limit exceeded"));
        }
        self.visited = self
            .visited
            .checked_add(1)
            .ok_or_else(|| self.error(path, "schema node count overflow"))?;
        if self.visited > self.limits.max_nodes {
            return Err(self.error(path, "schema node limit exceeded"));
        }
        match schema {
            RawJson::Bool(_) => {
                self.charge_raw_clone(schema, path)?;
                Ok(schema.clone())
            }
            RawJson::Object(entries) => {
                self.check_keywords(entries, path)?;
                self.validate_keyword_shapes(entries, path)?;
                self.validate_definitions(entries, path, depth, references)?;
                if let Some(reference) = field(entries, "$ref") {
                    self.resolve_reference(
                        entries,
                        reference,
                        path,
                        record_properties,
                        depth,
                        references,
                    )
                } else {
                    self.resolve_entries(entries, path, record_properties, depth, references, false)
                }
            }
            _ => Err(self.error(path, "a schema must be an object or Boolean")),
        }
    }

    fn validate_definitions(
        &mut self,
        entries: &[(String, RawJson)],
        path: &str,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<()> {
        let Some(definitions) = field(entries, "$defs") else {
            return Ok(());
        };
        let RawJson::Object(definitions) = definitions else {
            return Err(self.error(path, "$defs must be an object"));
        };
        for (name, schema) in definitions {
            let definition_path = append_two_pointers(path, "$defs", name);
            self.resolve(schema, &definition_path, false, depth + 1, references)?;
        }
        Ok(())
    }

    fn resolve_reference(
        &mut self,
        entries: &[(String, RawJson)],
        reference: &RawJson,
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<RawJson> {
        let RawJson::String(reference) = reference else {
            return Err(self.error(path, "$ref must be a string"));
        };
        let (target, target_pointer) = self.reference_target(reference)?;
        if references.iter().any(|active| active == &target_pointer) {
            return Err(self.error(path, "cyclic local $ref"));
        }
        if references.len() >= self.limits.max_reference_depth {
            return Err(self.error(path, "local $ref depth limit exceeded"));
        }
        references.push(target_pointer);
        let target_path = append_two_pointers(path, "allOf", "0");
        self.ensure_raw_clone_fits(target, path)?;
        let target = target.clone();
        let target = self.resolve(
            &target,
            &target_path,
            record_properties,
            depth + 1,
            references,
        )?;
        references.pop();

        let sibling_path = append_two_pointers(path, "allOf", "1");
        let siblings = self.resolve_entries(
            entries,
            &sibling_path,
            record_properties,
            depth + 1,
            references,
            true,
        )?;
        self.charge_bytes(path, 12)?;
        Ok(RawJson::Object(vec![(
            "allOf".to_owned(),
            RawJson::Array(vec![target, siblings]),
        )]))
    }

    fn resolve_entries(
        &mut self,
        entries: &[(String, RawJson)],
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
        skip_reference: bool,
    ) -> Result<RawJson> {
        self.charge_bytes(path, 2)?;
        let mut output = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if key == "$defs" || (skip_reference && key == "$ref") {
                continue;
            }
            self.charge_object_key(path, key)?;
            let resolved = match key.as_str() {
                "properties" => {
                    self.resolve_properties(value, path, record_properties, depth, references)?
                }
                "additionalProperties" | "items" => self.resolve_single(
                    value,
                    &append_pointer(path, key),
                    record_properties,
                    depth,
                    references,
                )?,
                "allOf" | "anyOf" | "oneOf" => self.resolve_array(
                    value,
                    &append_pointer(path, key),
                    record_properties,
                    depth,
                    references,
                )?,
                "pattern" => {
                    let RawJson::String(pattern) = value else {
                        return Err(self.error(path, "pattern must be a string"));
                    };
                    EcmaPattern::compile(pattern).map_err(|error| {
                        self.error(path, &format!("unsupported pattern: {error}"))
                    })?;
                    self.charge_raw_clone(value, path)?;
                    value.clone()
                }
                "$ref" => return Err(self.error(path, "internal unresolved $ref")),
                _ => {
                    self.validate_data_value(value, &append_pointer(path, key))?;
                    self.charge_raw_clone(value, path)?;
                    value.clone()
                }
            };
            output.push((key.clone(), resolved));
        }
        Ok(RawJson::Object(output))
    }

    fn resolve_properties(
        &mut self,
        properties: &RawJson,
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<RawJson> {
        let RawJson::Object(properties) = properties else {
            return Err(self.error(path, "properties must be an object"));
        };
        self.charge_bytes(path, 2)?;
        if record_properties {
            if self.property_order.len() >= self.limits.max_property_declarations {
                return Err(self.error(path, "property declaration limit exceeded"));
            }
            self.charge_property_order(path, properties)?;
            self.property_order.push(PropertyOrder {
                schema_path: path.to_owned(),
                properties: properties.iter().map(|(name, _)| name.clone()).collect(),
            });
        }
        let mut output = Vec::with_capacity(properties.len());
        for (name, schema) in properties {
            self.charge_object_key(path, name)?;
            let child_path = append_two_pointers(path, "properties", name);
            output.push((
                name.clone(),
                self.resolve(
                    schema,
                    &child_path,
                    record_properties,
                    depth + 1,
                    references,
                )?,
            ));
        }
        Ok(RawJson::Object(output))
    }

    fn resolve_single(
        &mut self,
        schema: &RawJson,
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<RawJson> {
        match schema {
            RawJson::Bool(_) | RawJson::Object(_) => {
                self.resolve(schema, path, record_properties, depth + 1, references)
            }
            _ => Err(self.error(path, "schema-valued keyword requires an object or Boolean")),
        }
    }

    fn resolve_array(
        &mut self,
        schemas: &RawJson,
        path: &str,
        record_properties: bool,
        depth: usize,
        references: &mut Vec<String>,
    ) -> Result<RawJson> {
        let RawJson::Array(schemas) = schemas else {
            return Err(self.error(path, "schema array keyword requires an array"));
        };
        self.charge_bytes(path, 2)?;
        let mut output = Vec::with_capacity(schemas.len());
        for (index, schema) in schemas.iter().enumerate() {
            if index > 0 {
                self.charge_bytes(path, 1)?;
            }
            output.push(self.resolve(
                schema,
                &append_pointer(path, &index.to_string()),
                record_properties,
                depth + 1,
                references,
            )?);
        }
        Ok(RawJson::Array(output))
    }

    fn check_keywords(&self, entries: &[(String, RawJson)], path: &str) -> Result<()> {
        for (key, value) in entries {
            if !SUPPORTED.contains(&key.as_str()) {
                return Err(self.error(path, &format!("unsupported keyword {key}")));
            }
            if key == "$schema" {
                let RawJson::String(dialect) = value else {
                    return Err(self.error(path, "$schema must be a string"));
                };
                if dialect != DRAFT202012 {
                    return Err(self.error(path, "declared $schema is not draft2020-12"));
                }
            }
        }
        Ok(())
    }

    fn charge_raw_clone(&mut self, value: &RawJson, path: &str) -> Result<()> {
        self.charge_bytes(path, raw_json_wire_bytes(value)?)
    }

    fn ensure_raw_clone_fits(&self, value: &RawJson, path: &str) -> Result<()> {
        let required = raw_json_wire_bytes(value)?;
        let total = self
            .emitted_bytes
            .checked_add(required)
            .ok_or_else(|| self.error(path, "resolved artifact byte counter overflow"))?;
        if total > self.limits.max_resolved_bytes {
            return Err(self.error(
                path,
                &format!(
                    "resolved artifact exceeds {} byte budget before clone",
                    self.limits.max_resolved_bytes
                ),
            ));
        }
        Ok(())
    }

    fn charge_object_key(&mut self, path: &str, key: &str) -> Result<()> {
        let bytes = json_string_wire_bytes(key)?
            .checked_add(2)
            .ok_or_else(|| self.error(path, "resolved key byte counter overflow"))?;
        self.charge_bytes(path, bytes)
    }

    fn charge_property_order(
        &mut self,
        path: &str,
        properties: &[(String, RawJson)],
    ) -> Result<()> {
        let mut bytes = json_string_wire_bytes("schema_path")?
            .checked_add(1)
            .and_then(|value| value.checked_add(json_string_wire_bytes(path).ok()?))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(json_string_wire_bytes("properties").ok()?))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(2))
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| self.error(path, "property-order byte counter overflow"))?;
        for (index, (name, _)) in properties.iter().enumerate() {
            if index > 0 {
                bytes = bytes
                    .checked_add(1)
                    .ok_or_else(|| self.error(path, "property-order byte counter overflow"))?;
            }
            bytes = bytes
                .checked_add(json_string_wire_bytes(name)?)
                .ok_or_else(|| self.error(path, "property-order byte counter overflow"))?;
        }
        self.charge_bytes(path, bytes)
    }

    fn charge_bytes(&mut self, path: &str, bytes: usize) -> Result<()> {
        self.emitted_bytes = self
            .emitted_bytes
            .checked_add(bytes)
            .ok_or_else(|| self.error(path, "resolved artifact byte counter overflow"))?;
        if self.emitted_bytes > self.limits.max_resolved_bytes {
            return Err(self.error(
                path,
                &format!(
                    "resolved artifact exceeds {} byte budget",
                    self.limits.max_resolved_bytes
                ),
            ));
        }
        Ok(())
    }

    fn validate_keyword_shapes(&self, entries: &[(String, RawJson)], path: &str) -> Result<()> {
        for (key, value) in entries {
            let keyword_path = append_pointer(path, key);
            match key.as_str() {
                "$schema" | "$ref" | "title" | "description" | "$comment" => {
                    require_string(value, self, &keyword_path, key)?;
                }
                "$defs" | "properties" => {
                    require_object(value, self, &keyword_path, key)?;
                }
                "type" => validate_type(value, self, &keyword_path)?,
                "enum" => {
                    let values = require_array(value, self, &keyword_path, key)?;
                    if values.is_empty() {
                        return Err(self.error(&keyword_path, "enum must be nonempty"));
                    }
                }
                "required" => validate_required(value, self, &keyword_path)?,
                "additionalProperties" | "items" => {
                    if !matches!(value, RawJson::Bool(_) | RawJson::Object(_)) {
                        return Err(self.error(
                            &keyword_path,
                            &format!("{key} must be a schema object or Boolean"),
                        ));
                    }
                }
                "minItems" | "maxItems" | "minLength" | "maxLength" => {
                    self.nonnegative_safe_integer(value, &keyword_path, key)?;
                }
                "uniqueItems" | "deprecated" | "readOnly" | "writeOnly" => {
                    require_boolean(value, self, &keyword_path, key)?;
                }
                "pattern" => {
                    let pattern = require_string(value, self, &keyword_path, key)?;
                    EcmaPattern::compile(pattern).map_err(|error| {
                        self.error(&keyword_path, &format!("unsupported pattern: {error}"))
                    })?;
                }
                "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
                    self.admitted_decimal(value, &keyword_path, key)?;
                }
                "multipleOf" => {
                    let decimal = self.admitted_decimal(value, &keyword_path, key)?;
                    if decimal.compare(&CanonicalDecimal::from_wire("0")?) != Ordering::Greater {
                        return Err(
                            self.error(&keyword_path, "multipleOf must be strictly positive")
                        );
                    }
                }
                "allOf" | "anyOf" | "oneOf" => {
                    let schemas = require_array(value, self, &keyword_path, key)?;
                    if schemas.is_empty() {
                        return Err(self.error(
                            &keyword_path,
                            &format!("{key} must contain at least one schema"),
                        ));
                    }
                }
                "examples" => {
                    require_array(value, self, &keyword_path, key)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn nonnegative_safe_integer(&self, value: &RawJson, path: &str, keyword: &str) -> Result<()> {
        let RawJson::Number(number) = value else {
            return Err(self.error(path, &format!("{keyword} must be a nonnegative integer")));
        };
        let admitted = admit_number(number, NumericKind::Integer)
            .map_err(|error| self.error(path, &error.to_string()))?;
        if admitted.value < 0.0 {
            return Err(self.error(path, &format!("{keyword} must be nonnegative")));
        }
        Ok(())
    }

    fn admitted_decimal(
        &self,
        value: &RawJson,
        path: &str,
        keyword: &str,
    ) -> Result<CanonicalDecimal> {
        let RawJson::Number(number) = value else {
            return Err(self.error(path, &format!("{keyword} must be a finite number")));
        };
        let admitted = admit_number(number, NumericKind::Number)
            .map_err(|error| self.error(path, &error.to_string()))?;
        CanonicalDecimal::from_admitted(&admitted)
            .map_err(|error| self.error(path, &error.to_string()))
    }

    /// Recursively admit every numeric semantic data value without treating
    /// its object keys as schema keywords.
    fn validate_data_value(&self, value: &RawJson, path: &str) -> Result<()> {
        match value {
            RawJson::Number(number) => {
                admit_number(number, NumericKind::Number)
                    .map_err(|error| self.error(path, &error.to_string()))?;
            }
            RawJson::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    self.validate_data_value(value, &append_pointer(path, &index.to_string()))?;
                }
            }
            RawJson::Object(entries) => {
                for (key, value) in entries {
                    self.validate_data_value(value, &append_pointer(path, key))?;
                }
            }
            RawJson::Null | RawJson::Bool(_) | RawJson::String(_) => {}
        }
        Ok(())
    }

    fn reference_target<'a>(&'a self, reference: &str) -> Result<(&'a RawJson, String)> {
        let Some(fragment) = reference.strip_prefix('#') else {
            return Err(self.error("", "external $ref is unsupported"));
        };
        let decoded = percent_decode(fragment)?;
        if decoded.is_empty() {
            return Ok((self.original, String::new()));
        }
        if !decoded.starts_with('/') {
            return Err(self.error("", "local $ref anchors are unsupported"));
        }
        let tokens = decode_pointer(&decoded)?;
        let mut current = self.original;
        for token in &tokens {
            current = match current {
                RawJson::Object(entries) => entries
                    .iter()
                    .find(|(key, _)| key == token)
                    .map(|(_, value)| value)
                    .ok_or_else(|| self.error("", "unresolved local $ref object token"))?,
                RawJson::Array(values) => {
                    let index = pointer_index(token)
                        .ok_or_else(|| self.error("", "invalid local $ref array index"))?;
                    values
                        .get(index)
                        .ok_or_else(|| self.error("", "unresolved local $ref array index"))?
                }
                _ => return Err(self.error("", "local $ref crosses a scalar")),
            };
        }
        Ok((current, decoded))
    }

    fn error(&self, path: &str, message: &str) -> ProtocolError {
        ProtocolError::Schema(format!("tool {} schema {path}: {message}", self.tool))
    }
}

fn check_dialect(tool: &str, schema: &RawJson) -> Result<()> {
    let RawJson::Object(entries) = schema else {
        return Ok(());
    };
    let Some(value) = field(entries, "$schema") else {
        return Ok(());
    };
    let RawJson::String(dialect) = value else {
        return Err(ProtocolError::Schema(format!(
            "tool {tool} schema root: $schema must be a string"
        )));
    };
    if dialect == DRAFT202012 {
        Ok(())
    } else {
        Err(ProtocolError::Schema(format!(
            "tool {tool} schema root: declared $schema is not draft2020-12"
        )))
    }
}

fn require_string<'a>(
    value: &'a RawJson,
    normalizer: &Normalizer<'_>,
    path: &str,
    keyword: &str,
) -> Result<&'a str> {
    let RawJson::String(value) = value else {
        return Err(normalizer.error(path, &format!("{keyword} must be a string")));
    };
    Ok(value)
}

fn require_boolean(
    value: &RawJson,
    normalizer: &Normalizer<'_>,
    path: &str,
    keyword: &str,
) -> Result<()> {
    if matches!(value, RawJson::Bool(_)) {
        Ok(())
    } else {
        Err(normalizer.error(path, &format!("{keyword} must be a Boolean")))
    }
}

fn require_object<'a>(
    value: &'a RawJson,
    normalizer: &Normalizer<'_>,
    path: &str,
    keyword: &str,
) -> Result<&'a [(String, RawJson)]> {
    let RawJson::Object(entries) = value else {
        return Err(normalizer.error(path, &format!("{keyword} must be an object")));
    };
    Ok(entries)
}

fn require_array<'a>(
    value: &'a RawJson,
    normalizer: &Normalizer<'_>,
    path: &str,
    keyword: &str,
) -> Result<&'a [RawJson]> {
    let RawJson::Array(values) = value else {
        return Err(normalizer.error(path, &format!("{keyword} must be an array")));
    };
    Ok(values)
}

fn validate_type(value: &RawJson, normalizer: &Normalizer<'_>, path: &str) -> Result<()> {
    match value {
        RawJson::String(value) => validate_type_name(value, normalizer, path),
        RawJson::Array(values) if !values.is_empty() => {
            let mut seen = HashSet::new();
            for value in values {
                let RawJson::String(value) = value else {
                    return Err(normalizer.error(path, "type arrays may contain only strings"));
                };
                validate_type_name(value, normalizer, path)?;
                if !seen.insert(value) {
                    return Err(normalizer.error(path, "type arrays may not contain duplicates"));
                }
            }
            Ok(())
        }
        _ => Err(normalizer.error(
            path,
            "type must be a supported string or nonempty string array",
        )),
    }
}

fn validate_type_name(value: &str, normalizer: &Normalizer<'_>, path: &str) -> Result<()> {
    if matches!(
        value,
        "null" | "boolean" | "object" | "array" | "number" | "integer" | "string"
    ) {
        Ok(())
    } else {
        Err(normalizer.error(path, &format!("unsupported JSON Schema type {value}")))
    }
}

fn validate_required(value: &RawJson, normalizer: &Normalizer<'_>, path: &str) -> Result<()> {
    let values = require_array(value, normalizer, path, "required")?;
    let mut seen = HashSet::new();
    for value in values {
        let RawJson::String(value) = value else {
            return Err(normalizer.error(path, "required may contain only strings"));
        };
        if !seen.insert(value) {
            return Err(normalizer.error(path, "required may not contain duplicates"));
        }
    }
    Ok(())
}

fn field<'a>(entries: &'a [(String, RawJson)], name: &str) -> Option<&'a RawJson> {
    entries
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

fn append_pointer(parent: &str, token: &str) -> String {
    format!("{parent}/{}", escape_pointer(token))
}

fn append_two_pointers(parent: &str, first: &str, second: &str) -> String {
    append_pointer(&append_pointer(parent, first), second)
}

fn escape_pointer(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

fn percent_decode(fragment: &str) -> Result<String> {
    let bytes = fragment.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            output.push(bytes[index]);
            index += 1;
            continue;
        }
        let first = bytes
            .get(index + 1)
            .copied()
            .ok_or_else(|| ProtocolError::Schema("truncated percent escape in $ref".to_owned()))?;
        let second = bytes
            .get(index + 2)
            .copied()
            .ok_or_else(|| ProtocolError::Schema("truncated percent escape in $ref".to_owned()))?;
        output.push((hex(first)? << 4) | hex(second)?);
        index += 3;
    }
    String::from_utf8(output)
        .map_err(|error| ProtocolError::Schema(format!("invalid UTF-8 in $ref: {error}")))
}

fn hex(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(ProtocolError::Schema(
            "invalid percent escape in $ref".to_owned(),
        )),
    }
}

fn decode_pointer(pointer: &str) -> Result<Vec<String>> {
    let mut output = Vec::new();
    for token in pointer[1..].split('/') {
        let mut decoded = String::with_capacity(token.len());
        let mut chars = token.chars();
        while let Some(character) = chars.next() {
            if character != '~' {
                decoded.push(character);
                continue;
            }
            match chars.next() {
                Some('0') => decoded.push('~'),
                Some('1') => decoded.push('/'),
                _ => {
                    return Err(ProtocolError::Schema(
                        "invalid RFC6901 escape in $ref".to_owned(),
                    ));
                }
            }
        }
        output.push(decoded);
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

fn raw_json_wire_bytes(value: &RawJson) -> Result<usize> {
    match value {
        RawJson::Null | RawJson::Bool(true) => Ok(4),
        RawJson::Bool(false) => Ok(5),
        RawJson::Number(number) => Ok(number.as_str().len()),
        RawJson::String(value) => json_string_wire_bytes(value),
        RawJson::Array(values) => {
            let mut total = 2usize;
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    total = total.checked_add(1).ok_or_else(|| {
                        ProtocolError::InputLimit("resolved array byte count overflow".to_owned())
                    })?;
                }
                total = total
                    .checked_add(raw_json_wire_bytes(value)?)
                    .ok_or_else(|| {
                        ProtocolError::InputLimit("resolved array byte count overflow".to_owned())
                    })?;
            }
            Ok(total)
        }
        RawJson::Object(entries) => {
            let mut total = 2usize;
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    total = total.checked_add(1).ok_or_else(|| {
                        ProtocolError::InputLimit("resolved object byte count overflow".to_owned())
                    })?;
                }
                total = total
                    .checked_add(json_string_wire_bytes(key)?)
                    .and_then(|count| count.checked_add(1))
                    .and_then(|count| count.checked_add(raw_json_wire_bytes(value).ok()?))
                    .ok_or_else(|| {
                        ProtocolError::InputLimit("resolved object byte count overflow".to_owned())
                    })?;
            }
            Ok(total)
        }
    }
}

fn json_string_wire_bytes(value: &str) -> Result<usize> {
    let mut total = 2usize;
    for character in value.chars() {
        let width = match character {
            '"' | '\\' | '\u{0008}' | '\u{0009}' | '\u{000a}' | '\u{000c}' | '\u{000d}' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        total = total.checked_add(width).ok_or_else(|| {
            ProtocolError::InputLimit("JSON string byte count overflow".to_owned())
        })?;
    }
    Ok(total)
}

fn utf16_compare(left: &str, right: &str) -> Ordering {
    left.encode_utf16().cmp(right.encode_utf16())
}

impl SchemaPlan {
    /// Validate a raw target against the preserved immutable source schema.
    /// Effective selected branches never replace this complete contract check.
    pub fn validate_original_instance(
        &self,
        instance: &RawJson,
    ) -> std::result::Result<(), crate::ValidationFailure> {
        self.validate_original_instance_with_limits(instance, crate::ValidationLimits::default())
    }

    /// Validate after strict full-instance admission, using caller-selected
    /// finite validation limits. The immutable plan was fully admitted once at
    /// construction; this never reparses or renormalizes it.
    pub fn validate_original_instance_with_limits(
        &self,
        instance: &RawJson,
        limits: crate::ValidationLimits,
    ) -> std::result::Result<(), crate::ValidationFailure> {
        crate::validation::validate_original_instance_with_limits(&self.original, instance, limits)
    }
}

impl SchemaPlan {
    /// Start an immutable effective tree from this plan's resolved source.
    pub fn effective_tree(&self) -> Result<crate::EffectiveTree> {
        crate::EffectiveTree::from_resolved(&self.resolved)
    }
}
