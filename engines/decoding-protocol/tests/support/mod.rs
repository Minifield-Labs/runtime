//! Helpers for opt-in checks over externally stored immutable oracle bundles.
//!
//! The default crate test suite deliberately has no dependency on these large
//! files. Set `MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT` to an absolute
//! directory containing the named bundles, then run the ignored bulk tests.

use minifield_decoding_protocol::sha256_hex;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Component, Path, PathBuf},
};

pub type FixtureResult<T> = Result<T, Box<dyn Error>>;

const ROOT_ENV: &str = "MINIFIELD_DECODING_PROTOCOL_BULK_FIXTURE_ROOT";

pub struct Bundle {
    root: PathBuf,
    files: BTreeMap<String, String>,
}

impl Bundle {
    pub fn read(&self, relative: &str) -> FixtureResult<Vec<u8>> {
        let expected = self
            .files
            .get(relative)
            .ok_or_else(|| format!("{relative:?} is absent from the pinned fixture manifest"))?;
        let path = checked_path(&self.root, relative)?;
        let bytes = fs::read(&path)?;
        let actual = sha256_hex(&bytes);
        if actual != *expected {
            return Err(format!(
                "fixture file {} hash mismatch: expected {expected}, got {actual}",
                path.display()
            )
            .into());
        }
        Ok(bytes)
    }
}

pub fn required_bundle(name: &str, expected_manifest_sha256: &str) -> FixtureResult<Bundle> {
    let root = env::var_os(ROOT_ENV).ok_or_else(|| {
        format!(
            "{ROOT_ENV} is required for ignored bulk-fixture tests; set it to an absolute fixture root"
        )
    })?;
    let root = PathBuf::from(root);
    if !root.is_absolute() {
        return Err(format!("{ROOT_ENV} must be an absolute path").into());
    }
    let bundle_root = checked_path(&root, name)?;
    let manifest_path = bundle_root.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let actual_manifest_sha256 = sha256_hex(&manifest_bytes);
    if actual_manifest_sha256 != expected_manifest_sha256 {
        return Err(format!(
            "fixture manifest {} hash mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha256}",
            manifest_path.display()
        )
        .into());
    }

    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let mut files = BTreeMap::new();
    for field in ["files", "artifacts"] {
        if let Some(entries) = manifest.get(field).and_then(Value::as_object) {
            for (relative, expected) in entries {
                let expected = expected
                    .as_str()
                    .ok_or_else(|| format!("{field}.{relative} must be a SHA-256 string"))?;
                require_sha256(expected, &format!("{field}.{relative}"))?;
                if files
                    .insert(relative.clone(), expected.to_owned())
                    .is_some()
                {
                    return Err(format!("duplicate fixture entry {relative:?}").into());
                }
            }
        }
    }
    if let Some(expected) = manifest.get("cases_sha256").and_then(Value::as_str) {
        require_sha256(expected, "cases_sha256")?;
        files.insert("cases.json".to_owned(), expected.to_owned());
    }
    if files.is_empty() {
        return Err(format!(
            "fixture manifest {} declares no hash-addressed payload files",
            manifest_path.display()
        )
        .into());
    }

    let bundle = Bundle {
        root: bundle_root,
        files,
    };
    for relative in bundle.files.keys() {
        let _ = bundle.read(relative)?;
    }
    Ok(bundle)
}

fn checked_path(root: &Path, relative: &str) -> FixtureResult<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe fixture relative path {relative:?}").into());
    }
    Ok(root.join(relative_path))
}

fn require_sha256(value: &str, field: &str) -> FixtureResult<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} is not a lowercase hexadecimal SHA-256 digest").into());
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(format!("{field} must use lowercase hexadecimal SHA-256").into());
    }
    Ok(())
}
