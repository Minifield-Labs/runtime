#![allow(clippy::expect_used, clippy::format_collect, clippy::too_many_lines)]
// This ignored external-oracle test keeps each exact artifact assertion adjacent for audit.
use std::{
    env, fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use minifield_infer::{CliOptions, run_with_io};
use minifield_text_generation::StopReason;
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerLimits};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_MANIFEST_SHA256: &str =
    "1f81feb37b0aad48c44754d938c71cc57beb3fd2433233c6ac35cf65f80ad1d2";
const CASES_SHA256: &str = "f042cb98212510cc81e7e680e08548d640315c3aadb8318e7bf49b1c1eacaf98";
const MODEL_WEIGHTS_SHA256: &str =
    "ccb6771706166106908d040ba1131fc49037c613e0357e3d0f0cb1ef128115a3";
const CONFIG_SHA256: &str = "1065cb6225b6ee3863e6a5b0bf7624db7f8b1898e7cd61a40fda13edadb9cf5f";
const TOKENIZER_SHA256: &str = "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd";

fn external_path(name: &str) -> PathBuf {
    env::var_os(name).map_or_else(
        || panic!("{name} must name the immutable external fixture or trained artifact"),
        PathBuf::from,
    )
}

fn read_hashed(path: &Path, expected: &str) -> Vec<u8> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let actual = hex(&Sha256::digest(&bytes));
    assert_eq!(actual, expected, "unexpected bytes at {}", path.display());
    bytes
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected ID array"))
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or_else(|| panic!("expected u32 token ID"))
        })
        .collect()
}

#[test]
#[ignore = "requires MINIFIELD_TRAINED_TINY_TEXT_FIXTURE_ROOT and MINIFIELD_TRAINED_TINY_ARTIFACT"]
fn exact_trained_tiny_plaintext_oracle_matches_all_six_cases() {
    let fixture = external_path("MINIFIELD_TRAINED_TINY_TEXT_FIXTURE_ROOT");
    let artifact = external_path("MINIFIELD_TRAINED_TINY_ARTIFACT");
    let manifest = read_hashed(&fixture.join("manifest.json"), FIXTURE_MANIFEST_SHA256);
    let manifest: Value = serde_json::from_slice(&manifest).expect("fixture manifest JSON");
    assert_eq!(
        manifest["schema"],
        "minifield.trained-tiny-plaintext-fixture/1"
    );
    assert_eq!(manifest["case_count"], 6);
    let cases = read_hashed(&fixture.join("cases.json"), CASES_SHA256);
    let cases: Value = serde_json::from_slice(&cases).expect("fixture cases JSON");
    assert_eq!(
        cases["schema"],
        "minifield.trained-tiny-plaintext-reference/1"
    );
    let cases = cases["cases"].as_array().expect("fixture case array");
    assert_eq!(cases.len(), 6);

    let model_dir = artifact.join("model");
    let tokenizer_bytes = read_hashed(
        &model_dir.join("tokenizer").join("tokenizer.json"),
        TOKENIZER_SHA256,
    );
    let tokenizer = Tokenizer::from_json_bytes(&tokenizer_bytes, TokenizerLimits::default())
        .expect("pinned tokenizer asset should admit");
    let weight_path = model_dir.join("model.safetensors");
    let weight_bytes = read_hashed(&weight_path, MODEL_WEIGHTS_SHA256);
    let _config = read_hashed(&model_dir.join("config.json"), CONFIG_SHA256);
    let asset_bytes = u64::try_from(weight_bytes.len()).expect("weight length fits u64");
    let runtime_bytes = asset_bytes.checked_mul(4).expect("runtime bound overflow");

    for case in cases {
        assert_eq!(case["full_vocab_size"], 65_536);
        assert_eq!(case["decode_skip_special_tokens"], true);
        let prompt = case["prompt"].as_str().expect("case prompt");
        let add_bos = case["add_bos"].as_bool().expect("case BOS mode");
        let input_ids = tokenizer
            .encode(
                prompt,
                EncodeOptions {
                    add_special_tokens: add_bos,
                },
            )
            .expect("reference prompt should encode");
        assert_eq!(
            input_ids,
            ids(&case["input_ids"]),
            "input IDs for {}",
            case["name"]
        );
        let max_output_tokens =
            usize::try_from(case["max_output_tokens"].as_u64().expect("output limit"))
                .expect("output limit fits usize");
        let max_context_tokens = input_ids
            .len()
            .checked_add(max_output_tokens)
            .expect("context length overflow");
        let options = CliOptions {
            model_dir: model_dir.clone(),
            add_bos,
            max_output_tokens,
            max_context_tokens,
            max_prompt_bytes: 4 * 1024 * 1024,
            max_asset_bytes: asset_bytes,
            max_runtime_bytes: runtime_bytes,
        };
        let mut output = Vec::new();
        let result = run_with_io(&options, Cursor::new(prompt.as_bytes()), &mut output)
            .unwrap_or_else(|error| panic!("case {} failed: {error}", case["name"]));
        assert_eq!(
            result.input_ids, input_ids,
            "result input IDs for {}",
            case["name"]
        );
        assert_eq!(
            result.generated_ids,
            ids(&case["generated_ids"]),
            "generated IDs for {}",
            case["name"]
        );
        assert_eq!(
            result.text,
            case["decoded_text"].as_str().expect("expected text"),
            "decoded text for {}",
            case["name"]
        );
        assert_eq!(
            output,
            result.text.as_bytes(),
            "plaintext stdout bytes for {}",
            case["name"]
        );
        assert_eq!(
            case["stop_reason"], "max_output_tokens",
            "fixture stop reason for {}",
            case["name"]
        );
        assert_eq!(
            result.stop_reason,
            StopReason::MaxOutputTokens,
            "stop reason for {}",
            case["name"]
        );
        let max_output_text = max_output_tokens.to_string();
        let max_context_text = max_context_tokens.to_string();
        let asset_bytes_text = asset_bytes.to_string();
        let runtime_bytes_text = runtime_bytes.to_string();
        let mut child = Command::new(env!("CARGO_BIN_EXE_minifield-infer"))
            .args([
                "--model-dir",
                model_dir.to_str().expect("UTF-8 model path"),
                "--bos",
                if add_bos { "true" } else { "false" },
                "--max-output-tokens",
                &max_output_text,
                "--max-context-tokens",
                &max_context_text,
                "--max-asset-bytes",
                &asset_bytes_text,
                "--max-runtime-bytes",
                &runtime_bytes_text,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("starting CLI for {}: {error}", case["name"]));
        child
            .stdin
            .take()
            .expect("CLI stdin handle")
            .write_all(prompt.as_bytes())
            .unwrap_or_else(|error| panic!("writing CLI stdin for {}: {error}", case["name"]));
        let binary = child
            .wait_with_output()
            .unwrap_or_else(|error| panic!("waiting for CLI {}: {error}", case["name"]));
        assert!(
            binary.status.success(),
            "CLI {} failed: {}",
            case["name"],
            String::from_utf8_lossy(&binary.stderr)
        );
        assert_eq!(
            binary.stderr,
            Vec::<u8>::new(),
            "CLI diagnostics for {}",
            case["name"]
        );
        assert_eq!(
            binary.stdout, output,
            "CLI plaintext stdout for {}",
            case["name"]
        );
    }
}
