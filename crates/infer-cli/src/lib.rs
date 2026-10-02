#![forbid(unsafe_code)]
//! Native plaintext adapter for a caller-owned local LFM2 bundle.
//!
//! The binary layer owns bounded local file and stdin/stdout I/O. The generation loop, tokenizer,
//! loader, and executor remain separate reusable Rust crates.

use std::{
    ffi::OsString,
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use minifield_backend_cpu::CpuBackend;
use minifield_engine_api::{MemoryAssetProvider, ResourceLimits};
use minifield_executor_core::{
    Lfm2ExecutionLimits, Lfm2Executor, Lfm2LoadRequest, Lfm2WeightLoadTask, LoaderLimits,
    LoaderPoll,
};
use minifield_runtime_telemetry::{Measurement, Mode, Model};
use minifield_text_generation::{
    GenerationPolicy, GenerationProgress, GenerationRequest, GenerationResult, NeverCancel,
    StopReason, generate_observed,
};
use minifield_text_tokenizer::{Tokenizer, TokenizerLimits};
use sha2::{Digest, Sha256};

const DEFAULT_MAX_PROMPT_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_MAX_ASSET_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_MAX_RUNTIME_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_LOADER_STEPS: usize = 16 * 1024;
const CPU_BACKEND_OWNER: u64 = 0x4D49_4E49_4649_454C;
const EOS_TOKEN_ID: u32 = 7;

/// Exact bounded options accepted by the native plaintext runner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliOptions {
    pub model_dir: PathBuf,
    pub add_bos: bool,
    pub max_output_tokens: usize,
    pub max_context_tokens: usize,
    pub max_prompt_bytes: u64,
    pub max_asset_bytes: u64,
    pub max_runtime_bytes: u64,
}

/// CLI input, loading, and generation failures.
#[derive(Debug)]
pub enum CliError {
    Usage(String),
    Io { context: String, source: io::Error },
    InputTooLarge { actual: u64, limit: u64 },
    InvalidUtf8Prompt,
    Limit(String),
    Tokenizer(minifield_text_tokenizer::TokenizerError),
    Loader(minifield_executor_core::LoaderError),
    Executor(minifield_engine_api::ExecutorError),
    Generation(minifield_text_generation::GenerationError),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) | Self::Limit(message) => formatter.write_str(message),
            Self::Io { context, source } => write!(formatter, "{context}: {source}"),
            Self::InputTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "input is {actual} bytes, above configured limit {limit}"
                )
            }
            Self::InvalidUtf8Prompt => formatter.write_str("stdin prompt is not valid UTF-8"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer: {error}"),
            Self::Loader(error) => write!(formatter, "model loader: {error}"),
            Self::Executor(error) => write!(formatter, "executor: {error}"),
            Self::Generation(error) => write!(formatter, "generation: {error}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Tokenizer(error) => Some(error),
            Self::Loader(error) => Some(error),
            Self::Executor(error) => Some(error),
            Self::Generation(error) => Some(error),
            Self::Usage(_)
            | Self::InputTooLarge { .. }
            | Self::InvalidUtf8Prompt
            | Self::Limit(_) => None,
        }
    }
}

impl From<minifield_text_tokenizer::TokenizerError> for CliError {
    fn from(error: minifield_text_tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<minifield_executor_core::LoaderError> for CliError {
    fn from(error: minifield_executor_core::LoaderError) -> Self {
        Self::Loader(error)
    }
}

impl From<minifield_engine_api::ExecutorError> for CliError {
    fn from(error: minifield_engine_api::ExecutorError) -> Self {
        Self::Executor(error)
    }
}

impl From<minifield_text_generation::GenerationError> for CliError {
    fn from(error: minifield_text_generation::GenerationError) -> Self {
        Self::Generation(error)
    }
}

/// Parses command-line arguments without reading model or prompt bytes.
///
/// # Errors
///
/// Returns a usage error for missing, duplicate, malformed, or unsupported options.
#[allow(clippy::too_many_lines)] // Fixed explicit option surface keeps CLI behavior easy to audit.
pub fn parse_args(arguments: impl IntoIterator<Item = OsString>) -> Result<CliOptions, CliError> {
    let mut model_dir = None;
    let mut add_bos = None;
    let mut max_output_tokens = None;
    let mut max_context_tokens = None;
    let mut max_prompt_bytes = None;
    let mut max_asset_bytes = None;
    let mut max_runtime_bytes = None;

    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let flag = flag
            .to_str()
            .ok_or_else(|| CliError::Usage("option names must be UTF-8".into()))?;
        match flag {
            "--model-dir" => {
                let path = next_value(&mut arguments, flag)?;
                if path.is_empty() || model_dir.replace(PathBuf::from(path)).is_some() {
                    return Err(CliError::Usage(
                        "--model-dir must appear exactly once with a nonempty path".into(),
                    ));
                }
            }
            "--bos" => {
                let value = next_value(&mut arguments, flag)?;
                let value = value
                    .to_str()
                    .ok_or_else(|| CliError::Usage("--bos value must be UTF-8".into()))?;
                let parsed = match value {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(CliError::Usage(
                            "--bos must be exactly true or false".into(),
                        ));
                    }
                };
                if add_bos.replace(parsed).is_some() {
                    return Err(CliError::Usage("--bos must appear exactly once".into()));
                }
            }
            "--max-output-tokens" => {
                assign_once(
                    &mut max_output_tokens,
                    parse_usize_option(
                        "--max-output-tokens",
                        &next_value(&mut arguments, flag)?,
                        true,
                    )?,
                    "--max-output-tokens",
                )?;
            }
            "--max-context-tokens" => {
                assign_once(
                    &mut max_context_tokens,
                    parse_usize_option(
                        "--max-context-tokens",
                        &next_value(&mut arguments, flag)?,
                        false,
                    )?,
                    "--max-context-tokens",
                )?;
            }
            "--max-prompt-bytes" => {
                assign_once(
                    &mut max_prompt_bytes,
                    parse_u64_option(
                        "--max-prompt-bytes",
                        &next_value(&mut arguments, flag)?,
                        false,
                    )?,
                    "--max-prompt-bytes",
                )?;
            }
            "--max-asset-bytes" => {
                assign_once(
                    &mut max_asset_bytes,
                    parse_u64_option(
                        "--max-asset-bytes",
                        &next_value(&mut arguments, flag)?,
                        false,
                    )?,
                    "--max-asset-bytes",
                )?;
            }
            "--max-runtime-bytes" => {
                assign_once(
                    &mut max_runtime_bytes,
                    parse_u64_option(
                        "--max-runtime-bytes",
                        &next_value(&mut arguments, flag)?,
                        false,
                    )?,
                    "--max-runtime-bytes",
                )?;
            }
            "--help" | "-h" => return Err(CliError::Usage(help_text().into())),
            _ => {
                return Err(CliError::Usage(format!(
                    "unknown option {flag:?}\n\n{}",
                    help_text()
                )));
            }
        }
    }

    let model_dir = model_dir
        .ok_or_else(|| CliError::Usage("--model-dir is required\n\n".to_owned() + help_text()))?;
    let add_bos =
        add_bos.ok_or_else(|| CliError::Usage("--bos is required\n\n".to_owned() + help_text()))?;
    let max_output_tokens = max_output_tokens.ok_or_else(|| {
        CliError::Usage("--max-output-tokens is required\n\n".to_owned() + help_text())
    })?;
    let max_context_tokens = max_context_tokens.ok_or_else(|| {
        CliError::Usage("--max-context-tokens is required\n\n".to_owned() + help_text())
    })?;

    let max_prompt_bytes = max_prompt_bytes.unwrap_or(DEFAULT_MAX_PROMPT_BYTES);
    let max_asset_bytes = max_asset_bytes.unwrap_or(DEFAULT_MAX_ASSET_BYTES);
    let max_runtime_bytes = max_runtime_bytes.unwrap_or(DEFAULT_MAX_RUNTIME_BYTES);

    if max_runtime_bytes < max_asset_bytes {
        return Err(CliError::Usage(
            "--max-runtime-bytes must be at least --max-asset-bytes".into(),
        ));
    }

    Ok(CliOptions {
        model_dir,
        add_bos,
        max_output_tokens,
        max_context_tokens,
        max_prompt_bytes,
        max_asset_bytes,
        max_runtime_bytes,
    })
}

/// Returns the stable, plaintext-only command usage text.
#[must_use]
pub const fn help_text() -> &'static str {
    "Usage: minifield-infer --model-dir PATH --bos true|false --max-output-tokens N \\\n     --max-context-tokens N [--max-prompt-bytes N] [--max-asset-bytes N] \\\n     [--max-runtime-bytes N]\n\n\
Reads exactly one UTF-8 prompt from stdin through EOF and writes only generated plaintext to stdout.\n\
PATH must contain config.json, model.safetensors, and tokenizer/tokenizer.json. No chat template is applied.\n\
BOS insertion is controlled only by the required --bos option. EOS token ID 7 stops generation."
}

/// Reads one strictly UTF-8 prompt up to the caller-provided bound, retaining all whitespace.
///
/// # Errors
///
/// Returns an error if the stream exceeds the bound, cannot be read, or is not valid UTF-8.
pub fn read_prompt(reader: impl Read, max_bytes: u64) -> Result<String, CliError> {
    let bytes = read_bounded(reader, max_bytes, "stdin prompt")?;
    String::from_utf8(bytes).map_err(|_| CliError::InvalidUtf8Prompt)
}

/// Loads local assets, runs greedy plaintext generation, and writes no envelope or newline.
///
/// # Errors
///
/// Returns a checked error for local I/O, model/tokenizer admission, execution, or generation.
pub fn run_with_io(
    options: &CliOptions,
    input: impl Read,
    output: &mut impl Write,
) -> Result<GenerationResult, CliError> {
    run_with_reporter(options, input, output, |_| {})
}

/// Run one inference and pass its content-free terminal record to the host.
/// Model loading failures are outside the inference boundary. The callback runs before stdout I/O.
///
/// # Errors
/// Returns the same errors as [`run_with_io`].
pub fn run_with_reporter(
    options: &CliOptions,
    input: impl Read,
    output: &mut impl Write,
    mut report: impl FnMut(serde_json::Value),
) -> Result<GenerationResult, CliError> {
    let prompt = read_prompt(input, options.max_prompt_bytes)?;
    let tokenizer_bytes = read_file(
        &options.model_dir.join("tokenizer").join("tokenizer.json"),
        MAX_CONFIG_BYTES.saturating_mul(8),
        "tokenizer.json",
    )?;
    let config_bytes = read_file(
        &options.model_dir.join("config.json"),
        MAX_CONFIG_BYTES,
        "config.json",
    )?;
    let weight_bytes = read_file(
        &options.model_dir.join("model.safetensors"),
        options.max_asset_bytes,
        "model.safetensors",
    )?;
    let tokenizer = Tokenizer::from_json_bytes(&tokenizer_bytes, TokenizerLimits::default())?;
    let (mut executor, model) = load_cpu_executor(
        options,
        &config_bytes,
        weight_bytes,
        digest(&tokenizer_bytes),
    )?;
    let mut cancellation = NeverCancel;
    let request = GenerationRequest {
        prompt: &prompt,
        add_bos: options.add_bos,
        max_output_tokens: options.max_output_tokens,
        max_context_tokens: options.max_context_tokens,
        stop_token_ids: &[EOS_TOKEN_ID],
        skip_special_tokens: true,
        policy: GenerationPolicy::Greedy,
    };
    let mut measurement = Measurement::new(Mode::Autoregressive, executor.inference_work());
    measurement.max_output_tokens = request.max_output_tokens;
    let result = generate_observed(
        &mut executor,
        &tokenizer,
        &request,
        &mut cancellation,
        &mut |progress, executor| match progress {
            GenerationProgress::Tokenized(count) => measurement.tokenized(count),
            GenerationProgress::Prefilled => measurement.prefilled(executor.inference_work()),
            GenerationProgress::TokenEmitted => measurement.emitted(),
        },
    );
    if let Ok(result) = &result {
        measurement.stop_reason = match result.stop_reason {
            StopReason::MaxOutputTokens => "output_limit",
            StopReason::StopToken(_) => "end_token",
            StopReason::ConstraintComplete => "constraint_complete",
        };
    }
    report(measurement.finish(&model, executor.inference_work(), result.is_ok(), "cpu"));
    let result = result?;
    output
        .write_all(result.text.as_bytes())
        .map_err(|source| CliError::Io {
            context: "writing generated plaintext to stdout".into(),
            source,
        })?;
    output.flush().map_err(|source| CliError::Io {
        context: "flushing generated plaintext to stdout".into(),
        source,
    })?;
    Ok(result)
}

fn assign_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), CliError> {
    if slot.replace(value).is_some() {
        return Err(CliError::Usage(format!("{name} must appear at most once")));
    }
    Ok(())
}

fn next_value<I>(arguments: &mut I, flag: &str) -> Result<OsString, CliError>
where
    I: Iterator<Item = OsString>,
{
    arguments
        .next()
        .ok_or_else(|| CliError::Usage(format!("{flag} requires a following value")))
}
fn parse_usize_option(name: &str, value: &OsString, allow_zero: bool) -> Result<usize, CliError> {
    let value = parse_u64_option(name, value, allow_zero)?;
    usize::try_from(value)
        .map_err(|_| CliError::Usage(format!("{name} does not fit this platform's usize")))
}

fn parse_u64_option(name: &str, value: &OsString, allow_zero: bool) -> Result<u64, CliError> {
    let value = value
        .to_str()
        .ok_or_else(|| CliError::Usage(format!("{name} value must be UTF-8")))?;
    if value.is_empty()
        || value.starts_with('+')
        || value.starts_with('-')
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(CliError::Usage(format!(
            "{name} must be an unsigned decimal integer"
        )));
    }
    let value = value
        .parse::<u64>()
        .map_err(|_| CliError::Usage(format!("{name} exceeds unsigned 64-bit range")))?;
    if !allow_zero && value == 0 {
        return Err(CliError::Usage(format!("{name} must be greater than zero")));
    }
    Ok(value)
}

fn read_file(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>, CliError> {
    let file = File::open(path).map_err(|source| CliError::Io {
        context: format!("opening {}", path.display()),
        source,
    })?;
    read_bounded(file, max_bytes, label)
}

fn read_bounded(mut reader: impl Read, max_bytes: u64, label: &str) -> Result<Vec<u8>, CliError> {
    let overread = max_bytes
        .checked_add(1)
        .ok_or_else(|| CliError::Limit(format!("{label} limit cannot be incremented")))?;
    let capacity = usize::try_from(max_bytes.min(64 * 1024)).map_err(|_| {
        CliError::Limit(format!(
            "{label} bounded initial allocation does not fit usize"
        ))
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| CliError::Limit(format!("{label} initial allocation failed")))?;
    reader
        .by_ref()
        .take(overread)
        .read_to_end(&mut bytes)
        .map_err(|source| CliError::Io {
            context: format!("reading {label}"),
            source,
        })?;
    let actual = u64::try_from(bytes.len())
        .map_err(|_| CliError::Limit(format!("{label} byte length does not fit u64")))?;
    if actual > max_bytes {
        return Err(CliError::InputTooLarge {
            actual,
            limit: max_bytes,
        });
    }
    Ok(bytes)
}

fn load_cpu_executor(
    options: &CliOptions,
    config_bytes: &[u8],
    weight_bytes: Vec<u8>,
    tokenizer_hash: [u8; 32],
) -> Result<(Lfm2Executor<CpuBackend>, Model), CliError> {
    let config_hash = digest(config_bytes);
    let weight_hash = digest(&weight_bytes);
    let weight_length = u64::try_from(weight_bytes.len())
        .map_err(|_| CliError::Limit("model.safetensors length does not fit u64".into()))?;
    let request = Lfm2LoadRequest::new(
        config_bytes.to_owned(),
        config_hash,
        weight_length,
        weight_hash,
        LoaderLimits {
            max_asset_bytes: options.max_asset_bytes,
            max_header_bytes: MAX_CONFIG_BYTES,
            max_source_tensor_bytes: options.max_asset_bytes,
            max_retained_host_bytes: loader_retained_host_bound(options.max_asset_bytes)?,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
    )?;
    let model = Model::from_plan(
        request.plan(),
        &std::collections::HashMap::new(),
        [config_hash, weight_hash, tokenizer_hash],
    );
    let mut backend = CpuBackend::new(
        CPU_BACKEND_OWNER,
        ResourceLimits {
            max_allocation_bytes: options.max_runtime_bytes,
            max_total_bytes: options.max_runtime_bytes,
            max_pending_operations: 256,
        },
    );
    let mut provider = MemoryAssetProvider::new(weight_bytes, options.max_asset_bytes);
    let mut task = Lfm2WeightLoadTask::begin(request)?;
    let weights = drive_loader(&mut task, &mut provider, &mut backend)?;
    let executor = Lfm2Executor::new(
        backend,
        weights,
        Lfm2ExecutionLimits {
            max_logical_tokens: u64::try_from(options.max_context_tokens).map_err(|_| {
                CliError::Limit("--max-context-tokens does not fit executor capacity".into())
            })?,
        },
    )
    .map_err(CliError::from)?;
    Ok((executor, model))
}

fn loader_retained_host_bound(max_asset_bytes: u64) -> Result<u64, CliError> {
    max_asset_bytes
        .checked_mul(3)
        .and_then(|value| value.checked_add(MAX_CONFIG_BYTES))
        .ok_or_else(|| CliError::Limit("loader retained byte limit overflows u64".into()))
}

fn drive_loader(
    task: &mut Lfm2WeightLoadTask<
        minifield_engine_api::MemoryAssetRead,
        minifield_backend_cpu::CpuCompletion<()>,
        minifield_backend_cpu::CpuBuffer,
    >,
    provider: &mut MemoryAssetProvider,
    backend: &mut CpuBackend,
) -> Result<minifield_executor_core::Lfm2TypedWeights<minifield_backend_cpu::CpuBuffer>, CliError> {
    for _ in 0..MAX_LOADER_STEPS {
        match task.poll_step(provider, backend) {
            LoaderPoll::Pending => {}
            LoaderPoll::Ready(result) => return result.map_err(CliError::from),
        }
    }
    Err(CliError::Limit(
        "model loader exceeded bounded completion steps".into(),
    ))
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn arguments(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn parser_requires_explicit_model_bos_and_limits() {
        let options = parse_args(arguments(&[
            "--model-dir",
            "bundle",
            "--bos",
            "true",
            "--max-output-tokens",
            "0",
            "--max-context-tokens",
            "16",
        ]))
        .unwrap_or_else(|error| panic!("complete explicit options should parse: {error}"));
        assert_eq!(options.model_dir, PathBuf::from("bundle"));
        assert!(options.add_bos);
        assert_eq!(options.max_output_tokens, 0);
        assert_eq!(options.max_context_tokens, 16);

        assert!(matches!(
            parse_args(arguments(&[
                "--model-dir",
                "bundle",
                "--max-output-tokens",
                "1",
                "--max-context-tokens",
                "16",
            ])),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn parser_rejects_ambiguous_or_invalid_numeric_options() {
        assert!(matches!(
            parse_args(arguments(&[
                "--model-dir",
                "bundle",
                "--bos",
                "false",
                "--max-output-tokens",
                "+1",
                "--max-context-tokens",
                "16",
            ])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            parse_args(arguments(&[
                "--model-dir",
                "bundle",
                "--bos",
                "false",
                "--max-output-tokens",
                "1",
                "--max-output-tokens",
                "2",
                "--max-context-tokens",
                "16",
            ])),
            Err(CliError::Usage(message)) if message.contains("--max-output-tokens")
        ));
        assert!(matches!(
            parse_args(arguments(&[
                "--model-dir",
                "bundle",
                "--bos",
                "false",
                "--max-output-tokens",
                "1",
                "--max-context-tokens",
                "0",
            ])),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn loader_host_limit_covers_the_documented_bf16_decode_peak() {
        assert_eq!(
            loader_retained_host_bound(4)
                .unwrap_or_else(|error| panic!("small loader bound: {error}")),
            3 * 4 + MAX_CONFIG_BYTES
        );
        assert!(matches!(
            loader_retained_host_bound(u64::MAX),
            Err(CliError::Limit(_))
        ));
    }
    #[test]
    fn prompt_reader_preserves_whitespace_and_rejects_overflow_or_invalid_utf8() {
        assert_eq!(
            read_prompt(Cursor::new(b"  hello\n".to_vec()), 8)
                .unwrap_or_else(|error| panic!("valid bounded UTF-8 prompt should read: {error}")),
            "  hello\n"
        );
        assert!(matches!(
            read_prompt(Cursor::new(b"123".to_vec()), 2),
            Err(CliError::InputTooLarge {
                actual: 3,
                limit: 2
            })
        ));
        assert!(matches!(
            read_prompt(Cursor::new(vec![0xFF]), 1),
            Err(CliError::InvalidUtf8Prompt)
        ));
    }
}
