#![allow(clippy::expect_used)]
use minifield_infer::{CliOptions, run_with_reporter};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Bundle(PathBuf);
impl Drop for Bundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> Bundle {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "minifield-telemetry-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(path.join("tokenizer")).expect("directory");
    let mut weights =
        include_bytes!("../../executor-core/tests/fixtures/numerical-lfm-001-weights.safetensors")
            .to_vec();
    let header_len = usize::try_from(u64::from_le_bytes(weights[..8].try_into().expect("length")))
        .expect("header size");
    // All-zero synthetic weights deterministically select token 0 ('a').
    weights[8 + header_len..].fill(0);
    fs::write(path.join("model.safetensors"), weights).expect("weights");
    fs::write(
        path.join("config.json"),
        include_bytes!("../../executor-core/tests/fixtures/numerical-lfm-001-config.json"),
    )
    .expect("config");
    fs::write(
        path.join("tokenizer/tokenizer.json"),
        include_bytes!("../../text-generation/tests/compact_tokenizer.json"),
    )
    .expect("tokenizer");
    Bundle(path)
}

#[test]
fn native_host_reports_once_for_generation_and_failure_without_changing_output() {
    let bundle = fixture();
    let options = CliOptions {
        model_dir: bundle.0.clone(),
        add_bos: true,
        max_output_tokens: 3,
        max_context_tokens: 16,
        max_prompt_bytes: 1024,
        max_asset_bytes: 1 << 20,
        max_runtime_bytes: 1 << 25,
    };
    let mut output = Vec::new();
    let mut records = Vec::new();
    let result = run_with_reporter(&options, Cursor::new("a"), &mut output, |record| {
        records.push(record);
    })
    .expect("inference");
    assert_eq!(result.text, "aaa");
    assert_eq!(output, b"aaa");
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record["execution"]["tokens"]["input"], 2);
    assert_eq!(record["execution"]["tokens"]["output"], 3);
    assert_eq!(record["execution"]["prefill"]["forward_passes"], 1);
    assert_eq!(record["execution"]["decode"]["forward_passes"], 2);
    assert_eq!(record["status"], "completed");
    assert!(record.get("text").is_none());
    output.clear();
    assert!(
        run_with_reporter(&options, Cursor::new("🦕"), &mut output, |record| records
            .push(record))
        .is_err()
    );
    assert!(output.is_empty());
    assert_eq!(records.len(), 2);
    assert_eq!(records[1]["status"], "failed");
    assert_eq!(records[1]["execution"]["tokens"]["output"], 0);
    assert!(!records[1].to_string().contains('🦕'));
    let mut zero = options;
    zero.max_output_tokens = 0;
    run_with_reporter(&zero, Cursor::new("a"), &mut output, |record| {
        records.push(record);
    })
    .expect("zero budget");
    assert_eq!(records.len(), 3);
    assert_eq!(records[2]["execution"]["estimated_flops"], "0");
    assert_eq!(records[2]["execution"]["decode"]["forward_passes"], 0);
}
