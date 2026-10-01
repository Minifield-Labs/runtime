use sha2::{Digest, Sha256};
use std::{env, fs, path::Path, process::Command};

fn hash_sources(root: &Path, path: &Path, hash: &mut Sha256) -> std::io::Result<()> {
    if path.is_dir() {
        if matches!(
            path.file_name().and_then(|n| n.to_str()),
            Some("target" | "pkg" | "node_modules" | "models" | ".git")
        ) {
            return Ok(());
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            hash_sources(root, &entry.path(), hash)?;
        }
    } else if matches!(
        path.extension().and_then(|n| n.to_str()),
        Some("rs" | "toml" | "lock" | "wgsl" | "metal" | "mjs")
    ) {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .as_bytes(),
        );
        hash.update([0]);
        hash.update(fs::read(path)?);
        hash.update([0]);
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let manifest = env::var("CARGO_MANIFEST_DIR").map_err(std::io::Error::other)?;
    let root = Path::new(&manifest).join("../..").canonicalize()?;
    let mut hash = Sha256::new();
    for part in ["Cargo.toml", "Cargo.lock", "crates", "web"] {
        hash_sources(&root, &root.join(part), &mut hash)?;
    }
    hash.update(env::var("TARGET").unwrap_or_default());
    hash.update(env::var("PROFILE").unwrap_or_default());
    println!(
        "cargo:rustc-env=MINIFIELD_BUILD_ID=sha256:{:x}",
        hash.finalize()
    );
    // Source digest distinguishes uncommitted builds; Git revision is additional provenance.
    let git = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output();
    let revision = git
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!("cargo:rustc-env=MINIFIELD_GIT_COMMIT={}", revision.trim());
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .current_dir(&root)
        .output()
        && let Ok(path) = String::from_utf8(output.stdout)
    {
        println!(
            "cargo:rerun-if-changed={}",
            root.join(path.trim()).display()
        );
    }
    if let Ok(output) = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(&root)
        .output()
        && output.status.success()
        && let Ok(reference) = String::from_utf8(output.stdout)
        && let Ok(output) = Command::new("git")
            .args(["rev-parse", "--git-path", reference.trim()])
            .current_dir(&root)
            .output()
        && let Ok(path) = String::from_utf8(output.stdout)
    {
        println!(
            "cargo:rerun-if-changed={}",
            root.join(path.trim()).display()
        );
    }
    Ok(())
}
