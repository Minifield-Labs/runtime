//! Complete validation against an immutable source schema.
use crate::{CanonicalDecimal, EcmaPattern, NumericKind, RawJson, admit_number};
use std::{cmp::Ordering, fmt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidationLimits {
    /// Maximum admission, schema, or recursive comparison depth. Roots are zero.
    pub max_depth: usize,
    /// Shared work budget for admission nodes, schema visits, comparison nodes,
    /// and object-key searches during semantic equality.
    pub max_work: usize,
}
impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_depth: 128,
            max_work: 1_000_000,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationFailureKind {
    InstanceInvalid,
    Operational,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationFailure {
    pub class: ValidationFailureKind,
    pub schema_path: String,
    pub instance_path: String,
    pub keyword: Option<String>,
    pub message: String,
}
impl fmt::Display for ValidationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.keyword {
            Some(k) => write!(
                f,
                "schema {} instance {} keyword {}: {}",
                self.schema_path, self.instance_path, k, self.message
            ),
            None => write!(
                f,
                "schema {} instance {}: {}",
                self.schema_path, self.instance_path, self.message
            ),
        }
    }
}
impl std::error::Error for ValidationFailure {}
pub(crate) fn validate_admitted_original_instance_with_limits(
    s: &RawJson,
    v: &RawJson,
    limits: ValidationLimits,
) -> std::result::Result<(), ValidationFailure> {
    Validator {
        root: s,
        limits,
        work: 0,
        refs: Vec::new(),
    }
    .val(s, "", v, "", 0)
}
pub(crate) fn validate_original_instance_with_limits(
    schema: &RawJson,
    instance: &RawJson,
    limits: ValidationLimits,
) -> ValidationResult {
    let mut validator = Validator {
        root: schema,
        limits,
        work: 0,
        refs: Vec::new(),
    };
    validator.admit(instance, "", 0)?;
    validator.val(schema, "", instance, "", 0)
}

type ValidationResult<T = ()> = std::result::Result<T, ValidationFailure>;
struct Validator<'a> {
    root: &'a RawJson,
    limits: ValidationLimits,
    work: usize,
    refs: Vec<String>,
}
impl Validator<'_> {
    fn bad(s: &str, i: &str, k: Option<&str>, m: impl Into<String>) -> ValidationFailure {
        ValidationFailure {
            class: ValidationFailureKind::InstanceInvalid,
            schema_path: s.into(),
            instance_path: i.into(),
            keyword: k.map(str::to_owned),
            message: m.into(),
        }
    }
    fn operational(s: &str, i: &str, k: Option<&str>, m: impl Into<String>) -> ValidationFailure {
        let mut failure = Self::bad(s, i, k, m);
        failure.class = ValidationFailureKind::Operational;
        failure
    }
    fn charge(
        &mut self,
        schema_path: &str,
        instance_path: &str,
        keyword: Option<&str>,
    ) -> ValidationResult {
        self.work = self.work.checked_add(1).ok_or_else(|| {
            Self::operational(
                schema_path,
                instance_path,
                keyword,
                "validation work counter overflow",
            )
        })?;
        if self.work > self.limits.max_work {
            return Err(Self::operational(
                schema_path,
                instance_path,
                keyword,
                "validation work limit exceeded",
            ));
        }
        Ok(())
    }
    fn check_depth(
        &self,
        schema_path: &str,
        instance_path: &str,
        keyword: Option<&str>,
        depth: usize,
    ) -> ValidationResult {
        if depth > self.limits.max_depth {
            return Err(Self::operational(
                schema_path,
                instance_path,
                keyword,
                "validation depth limit exceeded",
            ));
        }
        Ok(())
    }
    fn admit(&mut self, value: &RawJson, path: &str, depth: usize) -> ValidationResult {
        self.check_depth("", path, None, depth)?;
        self.charge("", path, None)?;
        match value {
            RawJson::Number(number) => {
                admit_number(number, NumericKind::Number)
                    .map_err(|error| Self::bad("", path, None, error.to_string()))?;
            }
            RawJson::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    self.admit(value, &add(path, &index.to_string()), depth + 1)?;
                }
            }
            RawJson::Object(entries) => {
                for (key, value) in entries {
                    self.admit(value, &add(path, key), depth + 1)?;
                }
            }
            RawJson::Null | RawJson::Bool(_) | RawJson::String(_) => {}
        }
        Ok(())
    }
    fn equal(
        &mut self,
        left: &RawJson,
        right: &RawJson,
        schema_path: &str,
        instance_path: &str,
        keyword: &str,
        depth: usize,
    ) -> ValidationResult<bool> {
        self.check_depth(schema_path, instance_path, Some(keyword), depth)?;
        self.charge(schema_path, instance_path, Some(keyword))?;
        match (left, right) {
            (RawJson::Null, RawJson::Null) => Ok(true),
            (RawJson::Bool(left), RawJson::Bool(right)) => Ok(left == right),
            (RawJson::Number(left), RawJson::Number(right)) => {
                let left_bits = admit_number(left, NumericKind::Number)
                    .map_err(|error| {
                        Self::operational(
                            schema_path,
                            instance_path,
                            Some(keyword),
                            error.to_string(),
                        )
                    })?
                    .value
                    .to_bits();
                let right_bits = admit_number(right, NumericKind::Number)
                    .map_err(|error| {
                        Self::operational(
                            schema_path,
                            instance_path,
                            Some(keyword),
                            error.to_string(),
                        )
                    })?
                    .value
                    .to_bits();
                Ok(left_bits == right_bits || ((left_bits << 1) == 0 && (right_bits << 1) == 0))
            }
            (RawJson::String(left), RawJson::String(right)) => Ok(left == right),
            (RawJson::Array(left), RawJson::Array(right)) => {
                if left.len() != right.len() {
                    return Ok(false);
                }
                for (index, (left, right)) in left.iter().zip(right).enumerate() {
                    if !self.equal(
                        left,
                        right,
                        schema_path,
                        &add(instance_path, &index.to_string()),
                        keyword,
                        depth + 1,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (RawJson::Object(left), RawJson::Object(right)) => {
                if left.len() != right.len() {
                    return Ok(false);
                }
                for (key, value) in left {
                    let mut other = None;
                    for (other_key, other_value) in right {
                        self.charge(schema_path, instance_path, Some(keyword))?;
                        if other_key == key {
                            other = Some(other_value);
                            break;
                        }
                    }
                    let Some(other) = other else {
                        return Ok(false);
                    };
                    if !self.equal(
                        value,
                        other,
                        schema_path,
                        &add(instance_path, key),
                        keyword,
                        depth + 1,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    fn val(&mut self, s: &RawJson, sp: &str, v: &RawJson, ip: &str, d: usize) -> ValidationResult {
        self.check_depth(sp, ip, None, d)?;
        self.charge(sp, ip, None)?;
        match s {
            RawJson::Bool(true) => Ok(()),
            RawJson::Bool(false) => Err(Self::bad(sp, ip, None, "false schema")),
            RawJson::Object(e) => self.obj(e, sp, v, ip, d),
            _ => Err(Self::operational(
                sp,
                ip,
                None,
                "schema must be an object or Boolean",
            )),
        }
    }
    #[allow(clippy::collapsible_if)]
    fn obj(
        &mut self,
        e: &[(String, RawJson)],
        sp: &str,
        v: &RawJson,
        ip: &str,
        d: usize,
    ) -> ValidationResult {
        if let Some(RawJson::String(r)) = field(e, "$ref") {
            let (target, path) = self.reference(r, sp, ip)?;
            if self.refs.iter().any(|x| x == &path) {
                return Err(Self::bad(sp, ip, Some("$ref"), "cyclic local $ref"));
            }
            self.refs.push(path.clone());
            let answer = self.val(&target, &path, v, ip, d + 1);
            self.refs.pop();
            answer?;
        }
        if let Some(t) = field(e, "type") {
            let matches = match t {
                RawJson::String(x) => Self::kind(x, v),
                RawJson::Array(xs) => xs
                    .iter()
                    .any(|x| matches!(x, RawJson::String(x) if Self::kind(x, v))),
                _ => false,
            };
            if !matches {
                return Err(Self::bad(
                    &add(sp, "type"),
                    ip,
                    Some("type"),
                    "type does not match",
                ));
            }
        }
        if let Some(x) = field(e, "const") {
            if !self.equal(x, v, &add(sp, "const"), ip, "const", d)? {
                return Err(Self::bad(
                    &add(sp, "const"),
                    ip,
                    Some("const"),
                    "value differs from const",
                ));
            }
        }
        if let Some(RawJson::Array(xs)) = field(e, "enum") {
            let mut matched = false;
            for candidate in xs {
                if self.equal(candidate, v, &add(sp, "enum"), ip, "enum", d)? {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Err(Self::bad(
                    &add(sp, "enum"),
                    ip,
                    Some("enum"),
                    "value is not in enum",
                ));
            }
        }
        self.object(e, sp, v, ip, d)?;
        self.array(e, sp, v, ip, d)?;
        self.string(e, sp, v, ip)?;
        self.number(e, sp, v, ip)?;
        for (k, x) in e {
            match k.as_str() {
                "allOf" => self.all(x, &add(sp, k), v, ip, d)?,
                "anyOf" => self.any(x, &add(sp, k), v, ip, d)?,
                "oneOf" => self.one(x, &add(sp, k), v, ip, d)?,
                _ => {}
            }
        }
        Ok(())
    }
    fn kind(t: &str, x: &RawJson) -> bool {
        match t {
            "null" => matches!(x, RawJson::Null),
            "boolean" => matches!(x, RawJson::Bool(_)),
            "object" => matches!(x, RawJson::Object(_)),
            "array" => matches!(x, RawJson::Array(_)),
            "string" => matches!(x, RawJson::String(_)),
            "number" => {
                matches!(x, RawJson::Number(n) if admit_number(n, NumericKind::Number).is_ok())
            }
            "integer" => {
                matches!(x, RawJson::Number(n) if admit_number(n, NumericKind::Integer).is_ok())
            }
            _ => false,
        }
    }
    fn object(
        &mut self,
        e: &[(String, RawJson)],
        sp: &str,
        v: &RawJson,
        ip: &str,
        d: usize,
    ) -> ValidationResult {
        let RawJson::Object(values) = v else {
            return Ok(());
        };
        let props = match field(e, "properties") {
            None => &[][..],
            Some(RawJson::Object(x)) => x.as_slice(),
            Some(_) => {
                return Err(Self::bad(
                    sp,
                    ip,
                    Some("properties"),
                    "properties must be an object",
                ));
            }
        };
        if let Some(RawJson::Array(required)) = field(e, "required") {
            for x in required {
                let RawJson::String(name) = x else {
                    return Err(Self::bad(
                        sp,
                        ip,
                        Some("required"),
                        "required entries must be strings",
                    ));
                };
                if !values.iter().any(|(key, _)| key == name) {
                    return Err(Self::bad(
                        &add(sp, "required"),
                        ip,
                        Some("required"),
                        format!("required property {name:?} is absent"),
                    ));
                }
            }
        }
        for (name, child) in props {
            if let Some((_, x)) = values.iter().find(|(key, _)| key == name) {
                self.val(
                    child,
                    &add2(sp, "properties", name),
                    x,
                    &add(ip, name),
                    d + 1,
                )?;
            }
        }
        for (name, x) in values {
            if props.iter().any(|(key, _)| key == name) {
                continue;
            }
            match field(e, "additionalProperties") {
                None | Some(RawJson::Bool(true)) => {}
                Some(RawJson::Bool(false)) => {
                    return Err(Self::bad(
                        &add(sp, "additionalProperties"),
                        &add(ip, name),
                        Some("additionalProperties"),
                        "additional property is forbidden",
                    ));
                }
                Some(child) => self.val(
                    child,
                    &add(sp, "additionalProperties"),
                    x,
                    &add(ip, name),
                    d + 1,
                )?,
            }
        }
        Ok(())
    }
    #[allow(clippy::cast_precision_loss, clippy::collapsible_if)]
    fn array(
        &mut self,
        e: &[(String, RawJson)],
        sp: &str,
        v: &RawJson,
        ip: &str,
        d: usize,
    ) -> ValidationResult {
        let RawJson::Array(values) = v else {
            return Ok(());
        };
        let length = values.len() as f64;
        if let Some(x) = field(e, "minItems") {
            if length < integer(x, sp, ip, "minItems", self)? {
                return Err(Self::bad(
                    &add(sp, "minItems"),
                    ip,
                    Some("minItems"),
                    "array is shorter than minItems",
                ));
            }
        }
        if let Some(x) = field(e, "maxItems") {
            if length > integer(x, sp, ip, "maxItems", self)? {
                return Err(Self::bad(
                    &add(sp, "maxItems"),
                    ip,
                    Some("maxItems"),
                    "array is longer than maxItems",
                ));
            }
        }
        if matches!(field(e, "uniqueItems"), Some(RawJson::Bool(true))) {
            for a in 0..values.len() {
                for b in a + 1..values.len() {
                    if self.equal(
                        &values[a],
                        &values[b],
                        &add(sp, "uniqueItems"),
                        &add(ip, &b.to_string()),
                        "uniqueItems",
                        d + 1,
                    )? {
                        return Err(Self::bad(
                            &add(sp, "uniqueItems"),
                            ip,
                            Some("uniqueItems"),
                            "array contains semantically duplicate items",
                        ));
                    }
                }
            }
        }
        if let Some(item) = field(e, "items") {
            for (n, x) in values.iter().enumerate() {
                self.val(item, &add(sp, "items"), x, &add(ip, &n.to_string()), d + 1)?;
            }
        }
        Ok(())
    }
    #[allow(clippy::cast_precision_loss, clippy::collapsible_if)]
    fn string(&self, e: &[(String, RawJson)], sp: &str, v: &RawJson, ip: &str) -> ValidationResult {
        let RawJson::String(value) = v else {
            return Ok(());
        };
        let length = value.chars().count() as f64;
        if let Some(x) = field(e, "minLength") {
            if length < integer(x, sp, ip, "minLength", self)? {
                return Err(Self::bad(
                    &add(sp, "minLength"),
                    ip,
                    Some("minLength"),
                    "string is shorter than minLength",
                ));
            }
        }
        if let Some(x) = field(e, "maxLength") {
            if length > integer(x, sp, ip, "maxLength", self)? {
                return Err(Self::bad(
                    &add(sp, "maxLength"),
                    ip,
                    Some("maxLength"),
                    "string is longer than maxLength",
                ));
            }
        }
        if let Some(RawJson::String(p)) = field(e, "pattern") {
            let m = EcmaPattern::compile(p).map_err(|x| {
                Self::operational(&add(sp, "pattern"), ip, Some("pattern"), x.to_string())
            })?;
            if !m.is_match(value).map_err(|x| {
                Self::operational(&add(sp, "pattern"), ip, Some("pattern"), x.to_string())
            })? {
                return Err(Self::bad(
                    &add(sp, "pattern"),
                    ip,
                    Some("pattern"),
                    "string does not match pattern",
                ));
            }
        }
        Ok(())
    }
    #[allow(clippy::collapsible_if)]
    fn number(&self, e: &[(String, RawJson)], sp: &str, v: &RawJson, ip: &str) -> ValidationResult {
        let RawJson::Number(value) = v else {
            return Ok(());
        };
        let actual = decimal(value, sp, ip, "number", self)?;
        for k in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            let Some(x) = field(e, k) else { continue };
            let cmp = actual.compare(&schema_decimal(x, sp, ip, k, self)?);
            let bad = match k {
                "minimum" => cmp == Ordering::Less,
                "maximum" => cmp == Ordering::Greater,
                "exclusiveMinimum" => cmp != Ordering::Greater,
                "exclusiveMaximum" => cmp != Ordering::Less,
                _ => false,
            };
            if bad {
                return Err(Self::bad(
                    &add(sp, k),
                    ip,
                    Some(k),
                    "numeric bound is violated",
                ));
            }
        }
        if let Some(x) = field(e, "multipleOf") {
            if !actual
                .is_multiple_of(&schema_decimal(x, sp, ip, "multipleOf", self)?)
                .map_err(|x| {
                    Self::bad(
                        &add(sp, "multipleOf"),
                        ip,
                        Some("multipleOf"),
                        x.to_string(),
                    )
                })?
            {
                return Err(Self::bad(
                    &add(sp, "multipleOf"),
                    ip,
                    Some("multipleOf"),
                    "number is not an exact multiple",
                ));
            }
        }
        Ok(())
    }
    fn all(&mut self, x: &RawJson, sp: &str, v: &RawJson, ip: &str, d: usize) -> ValidationResult {
        let RawJson::Array(xs) = x else {
            return Err(Self::bad(sp, ip, Some("allOf"), "allOf must be an array"));
        };
        for (n, x) in xs.iter().enumerate() {
            self.val(x, &add(sp, &n.to_string()), v, ip, d + 1)?;
        }
        Ok(())
    }
    fn any(&mut self, x: &RawJson, sp: &str, v: &RawJson, ip: &str, d: usize) -> ValidationResult {
        let RawJson::Array(xs) = x else {
            return Err(Self::bad(sp, ip, Some("anyOf"), "anyOf must be an array"));
        };
        for (n, x) in xs.iter().enumerate() {
            match self.val(x, &add(sp, &n.to_string()), v, ip, d + 1) {
                Ok(()) => return Ok(()),
                Err(failure) if failure.class == ValidationFailureKind::InstanceInvalid => {}
                Err(failure) => return Err(failure),
            }
        }
        Err(Self::bad(
            sp,
            ip,
            Some("anyOf"),
            "no anyOf branch validates",
        ))
    }
    fn one(&mut self, x: &RawJson, sp: &str, v: &RawJson, ip: &str, d: usize) -> ValidationResult {
        let RawJson::Array(xs) = x else {
            return Err(Self::bad(sp, ip, Some("oneOf"), "oneOf must be an array"));
        };
        let mut found = 0;
        for (n, x) in xs.iter().enumerate() {
            match self.val(x, &add(sp, &n.to_string()), v, ip, d + 1) {
                Ok(()) => found += 1,
                Err(failure) if failure.class == ValidationFailureKind::InstanceInvalid => {}
                Err(failure) => return Err(failure),
            }
        }
        if found == 1 {
            Ok(())
        } else {
            Err(Self::bad(
                sp,
                ip,
                Some("oneOf"),
                format!("expected exactly one matching branch, found {found}"),
            ))
        }
    }
    fn reference(&self, r: &str, sp: &str, ip: &str) -> ValidationResult<(RawJson, String)> {
        let Some(fragment) = r.strip_prefix('#') else {
            return Err(Self::operational(
                sp,
                ip,
                Some("$ref"),
                "only local references are supported",
            ));
        };
        let path = percent(fragment).map_err(|x| Self::operational(sp, ip, Some("$ref"), x))?;
        if path.is_empty() {
            return Ok((self.root.clone(), path));
        }
        if !path.starts_with('/') {
            return Err(Self::operational(
                sp,
                ip,
                Some("$ref"),
                "local reference must be an RFC6901 pointer",
            ));
        }
        let mut current = self.root;
        for token in pointer(&path).map_err(|x| Self::operational(sp, ip, Some("$ref"), x))? {
            current = match current {
                RawJson::Object(xs) => xs
                    .iter()
                    .find(|(k, _)| k == &token)
                    .map(|(_, v)| v)
                    .ok_or_else(|| {
                        Self::operational(sp, ip, Some("$ref"), "unresolved local reference")
                    })?,
                RawJson::Array(xs) => xs
                    .get(index(&token).ok_or_else(|| {
                        Self::operational(
                            sp,
                            ip,
                            Some("$ref"),
                            "invalid array index in local reference",
                        )
                    })?)
                    .ok_or_else(|| {
                        Self::operational(sp, ip, Some("$ref"), "unresolved local reference")
                    })?,
                _ => {
                    return Err(Self::operational(
                        sp,
                        ip,
                        Some("$ref"),
                        "local reference crosses a scalar",
                    ));
                }
            };
        }
        Ok((current.clone(), path))
    }
}
fn decimal(
    v: &crate::RawNumber,
    sp: &str,
    ip: &str,
    k: &str,
    _val: &Validator<'_>,
) -> ValidationResult<CanonicalDecimal> {
    let n = admit_number(v, NumericKind::Number)
        .map_err(|x| Validator::bad(sp, ip, Some(k), x.to_string()))?;
    CanonicalDecimal::from_admitted(&n).map_err(|x| Validator::bad(sp, ip, Some(k), x.to_string()))
}
fn schema_decimal(
    v: &RawJson,
    sp: &str,
    ip: &str,
    k: &str,
    val: &Validator<'_>,
) -> ValidationResult<CanonicalDecimal> {
    match v {
        RawJson::Number(x) => decimal(x, sp, ip, k, val),
        _ => Err(Validator::bad(sp, ip, Some(k), "must be a number")),
    }
}
fn integer(
    v: &RawJson,
    sp: &str,
    ip: &str,
    k: &str,
    _val: &Validator<'_>,
) -> ValidationResult<f64> {
    match v {
        RawJson::Number(x) => admit_number(x, NumericKind::Integer)
            .map(|x| x.value)
            .map_err(|x| Validator::bad(sp, ip, Some(k), x.to_string())),
        _ => Err(Validator::bad(
            sp,
            ip,
            Some(k),
            "must be a nonnegative integer",
        )),
    }
}
fn field<'a>(x: &'a [(String, RawJson)], k: &str) -> Option<&'a RawJson> {
    x.iter().find(|(n, _)| n == k).map(|(_, v)| v)
}
fn add(p: &str, x: &str) -> String {
    format!("{p}/{}", x.replace('~', "~0").replace('/', "~1"))
}
fn add2(p: &str, x: &str, y: &str) -> String {
    add(&add(p, x), y)
}
fn percent(x: &str) -> std::result::Result<String, String> {
    let mut out = Vec::new();
    let b = x.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'%' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        let a = *b
            .get(i + 1)
            .ok_or_else(|| "truncated percent escape".to_owned())?;
        let c = *b
            .get(i + 2)
            .ok_or_else(|| "truncated percent escape".to_owned())?;
        out.push((hex(a)? << 4) | hex(c)?);
        i += 3;
    }
    String::from_utf8(out).map_err(|_| "invalid UTF-8 in local reference".to_owned())
}
fn hex(x: u8) -> std::result::Result<u8, String> {
    match x {
        b'0'..=b'9' => Ok(x - b'0'),
        b'a'..=b'f' => Ok(x - b'a' + 10),
        b'A'..=b'F' => Ok(x - b'A' + 10),
        _ => Err("invalid percent escape".to_owned()),
    }
}
fn pointer(x: &str) -> std::result::Result<Vec<String>, String> {
    x[1..]
        .split('/')
        .map(|x| {
            let mut out = String::new();
            let mut it = x.chars();
            while let Some(c) = it.next() {
                if c == '~' {
                    match it.next() {
                        Some('0') => out.push('~'),
                        Some('1') => out.push('/'),
                        _ => return Err("invalid RFC6901 escape".to_owned()),
                    }
                } else {
                    out.push(c);
                }
            }
            Ok(out)
        })
        .collect()
}
fn index(x: &str) -> Option<usize> {
    if x == "0" {
        return Some(0);
    }
    if x.is_empty() || x.starts_with('0') || !x.bytes().all(|x| x.is_ascii_digit()) {
        return None;
    }
    x.parse().ok()
}
