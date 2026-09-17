use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerError, TokenizerLimits};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST_SHA256: &str = "bab69991150f78d6fbae655aa2d59b56ba1c8eb697e95f435eb62c807dec1152";
const CASES_SHA256: &str = "c8614892474e83263c62a6343240813963b066823bcac4b9e3ed5d29627e564f";
const DECODER_CASES_SHA256: &str =
    "c694cf103f14a62bc22105044eb17794cdb29ec2b39e72f76413ecf3f3c0ecf0";
const ASSET_SHA256: &str = "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd";

#[derive(Debug, Deserialize)]
struct EncodeCase {
    name: String,
    text: String,
    ids: Vec<u32>,
    ids_with_bos: Vec<u32>,
    decoded: String,
    decoded_skip_special: String,
    pretokenized: Vec<PretokenizedCase>,
}

#[derive(Debug, Deserialize)]
struct PretokenizedCase {
    byte_mapped_piece: String,
    span: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct DecoderCases {
    individual_ids: Vec<IndividualIdCase>,
    byte_tokens: BTreeMap<String, u32>,
    streaming_sequences: Vec<StreamingCase>,
}

#[derive(Debug, Deserialize)]
struct IndividualIdCase {
    id: u32,
    token: Option<String>,
    decoded: String,
    decoded_skip_special: String,
}

#[derive(Debug, Deserialize)]
struct StreamingCase {
    name: String,
    utf8_bytes: Vec<u8>,
    byte_token_ids: Vec<u32>,
    joined_decoded: String,
}

#[test]
#[ignore = "requires hash-verified text-tokenizer-001 fixture and pinned asset paths"]
#[allow(clippy::too_many_lines)]
fn text_tokenizer_001_matches_the_pinned_reference() {
    let fixture_root = required_env_path("MINIFIELD_TEXT_TOKENIZER_FIXTURE_ROOT");
    let asset_path = required_env_path("MINIFIELD_TEXT_TOKENIZER_ASSET");
    assert_hash(&fixture_root.join("manifest.json"), MANIFEST_SHA256);
    let cases_bytes = read_hashed(&fixture_root.join("cases.json"), CASES_SHA256);
    let decoder_bytes = read_hashed(
        &fixture_root.join("decoder-cases.json"),
        DECODER_CASES_SHA256,
    );
    let asset_bytes = read_hashed(&asset_path, ASSET_SHA256);
    let tokenizer = Tokenizer::from_json_bytes(&asset_bytes, TokenizerLimits::default())
        .unwrap_or_else(|error| panic!("pinned asset admission failed: {error}"));

    let cases: Vec<EncodeCase> = serde_json::from_slice(&cases_bytes)
        .unwrap_or_else(|error| panic!("cases.json parse failed: {error}"));
    assert_eq!(cases.len(), 1_145, "fixture row count changed");
    for case in cases {
        let encoded = tokenizer
            .encode(&case.text, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("{} encode failed: {error}", case.name));
        assert_eq!(encoded, case.ids, "{} IDs", case.name);
        let encoded_bos = tokenizer
            .encode(
                &case.text,
                EncodeOptions {
                    add_special_tokens: true,
                },
            )
            .unwrap_or_else(|error| panic!("{} BOS encode failed: {error}", case.name));
        assert_eq!(encoded_bos, case.ids_with_bos, "{} BOS IDs", case.name);
        assert_eq!(
            tokenizer.decode(&case.ids, false),
            Ok(case.decoded),
            "{} decoded",
            case.name
        );
        assert_eq!(
            tokenizer.decode(&case.ids, true),
            Ok(case.decoded_skip_special),
            "{} special decode",
            case.name
        );
        let pretokenized = tokenizer
            .pretokenize(&case.text)
            .unwrap_or_else(|error| panic!("{} pretokenize failed: {error}", case.name));
        assert_eq!(
            pretokenized.len(),
            case.pretokenized.len(),
            "{} pretokenized count",
            case.name
        );
        for (actual, expected) in pretokenized.iter().zip(case.pretokenized) {
            assert_eq!(
                actual.byte_mapped, expected.byte_mapped_piece,
                "{} byte-mapped piece",
                case.name
            );
            assert_eq!(
                [actual.span.start, actual.span.end],
                expected.span,
                "{} source span",
                case.name
            );
        }
    }

    let decoder_cases: DecoderCases = serde_json::from_slice(&decoder_bytes)
        .unwrap_or_else(|error| panic!("decoder-cases.json parse failed: {error}"));
    assert_eq!(
        decoder_cases.byte_tokens.len(),
        256,
        "fixture must bind every raw byte"
    );
    for (byte, id) in decoder_cases.byte_tokens {
        let byte = byte
            .parse::<u8>()
            .unwrap_or_else(|error| panic!("invalid byte key {byte}: {error}"));
        assert_eq!(
            tokenizer.token_bytes(id),
            Ok(&[byte][..]),
            "byte token {byte}"
        );
    }
    for case in decoder_cases.individual_ids {
        if case.token.is_some() {
            assert_eq!(
                tokenizer.decode(&[case.id], false),
                Ok(case.decoded),
                "individual ID {}",
                case.id
            );
            assert_eq!(
                tokenizer.decode(&[case.id], true),
                Ok(case.decoded_skip_special),
                "special individual ID {}",
                case.id
            );
        } else {
            assert_eq!(
                tokenizer.decode(&[case.id], false),
                Err(TokenizerError::UnmappedToken(case.id))
            );
        }
    }
    for case in decoder_cases.streaming_sequences {
        assert_eq!(
            case.joined_decoded.as_bytes(),
            case.utf8_bytes.as_slice(),
            "{} fixture bytes disagree with text",
            case.name
        );
        let mut decoder = tokenizer.streaming_decoder(false);
        let mut output = String::new();
        for id in case.byte_token_ids {
            output.push_str(
                &decoder
                    .push(&[id])
                    .unwrap_or_else(|error| panic!("{} stream ID {id}: {error}", case.name)),
            );
        }
        output.push_str(
            &decoder
                .finish()
                .unwrap_or_else(|error| panic!("{} stream finish: {error}", case.name)),
        );
        assert_eq!(
            output, case.joined_decoded,
            "{} streaming decode",
            case.name
        );
    }
}

fn required_env_path(name: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| path.is_dir() || path.is_file())
        .unwrap_or_else(|| panic!("{name} must name an existing fixture or asset path"))
}

fn read_hashed(path: &Path, expected: &str) -> Vec<u8> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(
        sha256(&bytes),
        expected,
        "unexpected hash for {}",
        path.display()
    );
    bytes
}

fn assert_hash(path: &Path, expected: &str) {
    let bytes = read_hashed(path, expected);
    assert!(!bytes.is_empty(), "fixture manifest is empty");
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
