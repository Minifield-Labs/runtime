//! Native classifier qualification with explicit host policy and JSON evidence.
//! Usage: `classify BUNDLE_DIR PROMPTS_JSON [--classes 8] [--context 512]`
//! [--tokenizer tokenizer/tokenizer.json] [--lut2 raw|down|auto]
//! [--prefix true|false] [--max-lut2-bytes 67108864] [--deadline-seconds 120].

use minifield_backend_wgpu::{Nf4Staging, WgpuBackend, WgpuBuffer, WgpuOptions};
use minifield_engine_api::{
    CompletionPoll, InferenceCompletion, MemoryAssetProvider, ResourceLimits, TokenChunk,
};
use minifield_executor_core::{
    Lfm2Classifier, Lfm2ExecutionLimits, Lfm2ExecutionOptions, Lfm2LayerWeightRole,
    Lfm2LoadRequest, Lfm2Lut2Mode, Lfm2TypedWeights, Lfm2WeightFormat, Lfm2WeightLoadTask,
    Lfm2WeightRole, LoaderLimits, LoaderPoll, detect_lfm2_weight_format,
    parse_lfm2_tensor_quantization,
};
use minifield_text_tokenizer::{EncodeOptions, Tokenizer, TokenizerLimits};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    path::PathBuf,
    str::FromStr,
    time::{Duration, Instant},
};

type HostResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Options {
    bundle: PathBuf,
    prompts: PathBuf,
    tokenizer: PathBuf,
    classes: u32,
    context: u64,
    lut2: Lfm2Lut2Mode,
    cached: bool,
    max_lut2_bytes: u64,
    deadline: Duration,
    diagnostics: bool,
    warmups: usize,
}

fn boolean(value: &str) -> HostResult<bool> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(format!("expected true, false, 1, or 0; got {value:?}").into()),
    }
}

fn lut2_mode(value: &str) -> HostResult<Lfm2Lut2Mode> {
    match value {
        "raw" | "off" => Ok(Lfm2Lut2Mode::Off),
        "down" => Ok(Lfm2Lut2Mode::DownOnly),
        "auto" => Ok(Lfm2Lut2Mode::Auto),
        _ => Err(format!("expected raw, off, down, or auto; got {value:?}").into()),
    }
}

fn mode_name(mode: Lfm2Lut2Mode) -> &'static str {
    match mode {
        Lfm2Lut2Mode::Off => "raw",
        Lfm2Lut2Mode::DownOnly => "down",
        Lfm2Lut2Mode::Auto => "auto",
    }
}

fn numeric<T: FromStr>(value: &str, field: &str) -> HostResult<T> {
    value
        .parse()
        .map_err(|_| format!("invalid {field}: {value:?}").into())
}

fn argument_pairs(args: &[String]) -> HostResult<BTreeMap<&str, &str>> {
    if args.len() < 2
        || args[..2]
            .iter()
            .any(|arg| arg.is_empty() || arg.starts_with("--"))
    {
        return Err("usage: classify BUNDLE_DIR PROMPTS_JSON [OPTIONS]".into());
    }
    let mut pairs = BTreeMap::new();
    let mut options = args[2..].chunks_exact(2);
    for pair in &mut options {
        match pair[0].as_str() {
            "--classes" | "--context" | "--tokenizer" | "--lut2" | "--prefix"
            | "--max-lut2-bytes" | "--deadline-seconds" | "--diagnostics" | "--warmups" => {}
            other => return Err(format!("unknown classify option {other:?}").into()),
        }
        if pairs.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err(format!("duplicate classify option {:?}", pair[0]).into());
        }
    }
    if !options.remainder().is_empty() {
        return Err("every classify option needs a value".into());
    }
    Ok(pairs)
}

fn options(args: &[String], environment: &BTreeMap<String, String>) -> HostResult<Options> {
    let pairs = argument_pairs(args)?;
    let value = |flag: &str, variable: &str, default: &'static str| {
        pairs
            .get(flag)
            .copied()
            .or_else(|| environment.get(variable).map(String::as_str))
            .unwrap_or(default)
    };
    let bundle = PathBuf::from(&args[0]);
    let tokenizer = PathBuf::from(
        pairs
            .get("--tokenizer")
            .copied()
            .unwrap_or("tokenizer/tokenizer.json"),
    );
    let classes = numeric::<u32>(value("--classes", "", "8"), "classes")?;
    let context = numeric::<u64>(value("--context", "", "512"), "context")?;
    if classes == 0 || context == 0 {
        return Err("classes and context must be positive".into());
    }
    let seconds = numeric::<f64>(value("--deadline-seconds", "", "120"), "deadline-seconds")?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("deadline-seconds must be finite and positive".into());
    }
    Ok(Options {
        tokenizer: if tokenizer.is_absolute() {
            tokenizer
        } else {
            bundle.join(tokenizer)
        },
        bundle,
        prompts: PathBuf::from(&args[1]),
        classes,
        context,
        lut2: lut2_mode(value("--lut2", "MINI_FFN_LUT2", "auto"))?,
        cached: boolean(value("--prefix", "CLASSIFY_PREFIX", "false"))?,
        max_lut2_bytes: numeric(value("--max-lut2-bytes", "", "67108864"), "max-lut2-bytes")?,
        deadline: Duration::try_from_secs_f64(seconds)?,
        diagnostics: boolean(value("--diagnostics", "MINIFIELD_WGPU_STATS", "false"))?,
        warmups: numeric(value("--warmups", "", "0"), "warmups")?,
    })
}

fn wait<C: InferenceCompletion>(task: &mut C, deadline: Duration) -> HostResult<C::Output> {
    let limit = Instant::now()
        .checked_add(deadline)
        .ok_or("poll deadline overflows clock")?;
    loop {
        if Instant::now() >= limit {
            let _ = task.cancel();
            return Err("classifier completion exceeded its polling deadline".into());
        }
        match task.poll_step() {
            CompletionPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            CompletionPoll::Ready(result) => return Ok(result?),
        }
    }
}

fn common_prefix(encoded: &[Vec<u32>]) -> usize {
    let Some(first) = encoded.first() else {
        return 0;
    };
    let mut common = encoded
        .iter()
        .map(|ids| ids.len().saturating_sub(1))
        .min()
        .unwrap_or(0);
    for ids in &encoded[1..] {
        while common > 0 && ids[..common] != first[..common] {
            common -= 1;
        }
    }
    common
}

fn samples(
    classifier: &mut Lfm2Classifier<WgpuBackend>,
    encoded: &[Vec<u32>],
    opts: &Options,
) -> HostResult<(Vec<Value>, f64)> {
    let mut rows = Vec::with_capacity(encoded.len());
    let started = Instant::now();
    let common = if opts.cached {
        common_prefix(encoded)
    } else {
        0
    };
    if opts.cached && common == 0 {
        return Err(
            "cached qualification requires a nonempty common prefix and nonempty tails".into(),
        );
    }
    let base = if opts.cached {
        Some(wait(
            &mut classifier.prefill_base(TokenChunk::all(&encoded[0][..common]))?,
            opts.deadline,
        )?)
    } else {
        None
    };
    let base_seconds = if base.is_some() {
        started.elapsed().as_secs_f64()
    } else {
        0.0
    };
    for (index, ids) in encoded.iter().enumerate() {
        let started = Instant::now();
        let logits = if let Some(base) = &base {
            wait(
                &mut classifier.classify_tail(base, TokenChunk::all(&ids[common..]))?,
                opts.deadline,
            )?
        } else {
            wait(
                &mut classifier.classify(TokenChunk::all(ids))?,
                opts.deadline,
            )?
        };
        if logits.len() != usize::try_from(opts.classes)?
            || logits.iter().any(|value| !value.is_finite())
        {
            return Err("classifier emitted incomplete or nonfinite logits".into());
        }
        rows.push(json!({"prompt_index":index,"ids":ids,"logits":logits,"seconds":started.elapsed().as_secs_f64()}));
    }
    Ok((rows, base_seconds))
}

#[derive(Debug, Default)]
struct Lut2Expectation {
    roles: Vec<Lfm2WeightRole>,
    dispatches: BTreeMap<&'static str, u64>,
    submitted_rows: Vec<usize>,
}

fn lut2_expectation(
    formats: &[[Lfm2WeightFormat; 3]],
    mode: Lfm2Lut2Mode,
    encoded: &[Vec<u32>],
    cached: bool,
    passes: usize,
) -> HostResult<Lut2Expectation> {
    let mut expected = Lut2Expectation::default();
    let common = if cached { common_prefix(encoded) } else { 0 };
    if cached {
        expected.submitted_rows.push(common);
    }
    expected
        .submitted_rows
        .extend(encoded.iter().map(|ids| ids.len() - common));
    // Match append_tokens: the fused FFN crossover is 96 rows. The final
    // layer always reduces its FFN to 1 row, including cached-base prefill.
    let tiles = expected
        .submitted_rows
        .iter()
        .filter(|rows| **rows >= 96)
        .count();
    let executions = u64::try_from(tiles)?
        .checked_mul(u64::try_from(passes)?)
        .ok_or("expected LUT2 dispatch count overflows")?;
    if executions == 0 || mode == Lfm2Lut2Mode::Off {
        return Ok(expected);
    }
    let packed = |format| {
        matches!(
            format,
            Lfm2WeightFormat::TernaryV1 | Lfm2WeightFormat::Nf4V1
        )
    };
    for (index, [gate, up, down]) in formats.iter().copied().enumerate() {
        if index + 1 == formats.len() || gate != up || !packed(gate) || !packed(down) {
            continue;
        }
        let mut add = |role, kernel| -> HostResult<()> {
            expected.roles.push(Lfm2WeightRole::Layer { index, role });
            let count = expected.dispatches.entry(kernel).or_default();
            *count = count
                .checked_add(executions)
                .ok_or("expected LUT2 dispatch count overflows")?;
            Ok(())
        };
        if down == Lfm2WeightFormat::TernaryV1 {
            add(Lfm2LayerWeightRole::FfnW2, "packed_gemm_ternary_lut2")?;
        }
        if mode == Lfm2Lut2Mode::Auto && gate == Lfm2WeightFormat::TernaryV1 {
            add(Lfm2LayerWeightRole::FfnW1, "packed_gemm_pair_swiglu_lut2")?;
            expected.roles.push(Lfm2WeightRole::Layer {
                index,
                role: Lfm2LayerWeightRole::FfnW3,
            });
        }
    }
    Ok(expected)
}

fn load(
    opts: &Options,
    config: &[u8],
    weights: Vec<u8>,
    backend: &mut WgpuBackend,
) -> HostResult<Lfm2TypedWeights<WgpuBuffer>> {
    let size = u64::try_from(weights.len())?;
    let request = Lfm2LoadRequest::new_classifier_with_quantization(
        config.to_vec(),
        Sha256::digest(config).into(),
        size,
        Sha256::digest(&weights).into(),
        LoaderLimits {
            max_asset_bytes: size,
            max_header_bytes: 1 << 20,
            max_source_tensor_bytes: size,
            max_retained_host_bytes: size.checked_mul(6).ok_or("host load budget overflows")?,
            max_tensor_name_bytes: 1024,
            max_tensors: 4096,
            max_rank: 4,
        },
        opts.classes,
        detect_lfm2_weight_format(&weights)?,
        &parse_lfm2_tensor_quantization(&weights)?,
    )?;
    let mut provider = MemoryAssetProvider::new(weights, size);
    let mut task = Lfm2WeightLoadTask::begin(request)?;
    let deadline = Instant::now()
        .checked_add(opts.deadline)
        .ok_or("loader deadline overflows")?;
    loop {
        if Instant::now() >= deadline {
            let _ = task.cancel();
            return Err("classifier weight load exceeded its polling deadline".into());
        }
        match task.poll_step(&mut provider, backend) {
            LoaderPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            LoaderPoll::Ready(result) => return Ok(result?),
        }
    }
}

fn diagnostics(
    classifier: &Lfm2Classifier<WgpuBackend>,
    expected: &Lut2Expectation,
) -> HostResult<Value> {
    let (counts, stats, adapter) = classifier.inspect_backend(|backend| {
        (
            backend.dispatch_counts(),
            backend.stats(),
            backend.adapter_info(),
        )
    })?;
    let (fallback, effective) = lut2_status(classifier.lut2_mode(), expected, &counts, |role| {
        classifier.has_lut2_codes(role)
    });
    let memory = classifier.resource_report()?;
    Ok(json!({
        "requested_lut2_mode":mode_name(classifier.lut2_mode()),"effective_lut2_mode":effective,
        "lut2_fallback":fallback,"dispatch_counts":counts,
        "expected_lut2_dispatch_counts":expected.dispatches,"submitted_token_rows":expected.submitted_rows,
        "skipped_lut2_roles":classifier.skipped_lut2_roles().iter().map(|role|format!("{role:?}")).collect::<Vec<_>>(),
        "memory_scope":"post_run_backend_accounted_bytes",
        "memory":{"resident_weight_bytes":memory.resident_weight_bytes,"cache_bytes":memory.cache_bytes,
            "scratch_bytes":memory.scratch_bytes,"staged_branch_bytes":memory.staged_branch_bytes,
            "pending_operation_bytes":memory.pending_operation_bytes,"pending_operations":memory.pending_operations},
        "device_stats":{"dispatches":stats.dispatches,"copies":stats.copies,"clears":stats.clears,
            "compute_passes":stats.compute_passes,"submits":stats.submits,"fences":stats.fences,
            "readbacks":stats.readbacks,"encode_ns":stats.encode_ns,"submit_ns":stats.submit_ns,
            "fence_wait_ns":stats.fence_wait_ns,"readback_wait_ns":stats.readback_wait_ns},
        "adapter":{"name":adapter.name,"vendor":adapter.vendor,"device":adapter.device,
            "driver":adapter.driver,"driver_info":adapter.driver_info,"backend":format!("{:?}",adapter.backend),
            "device_type":format!("{:?}",adapter.device_type)},
        "platform":{"os":env::consts::OS,"arch":env::consts::ARCH},
    }))
}

fn missing_lut2_dispatch(expected: &Lut2Expectation, counts: &BTreeMap<&str, u64>) -> bool {
    expected
        .dispatches
        .iter()
        .any(|(kernel, minimum)| counts.get(kernel).copied().unwrap_or_default() < *minimum)
}

fn lut2_status(
    mode: Lfm2Lut2Mode,
    expected: &Lut2Expectation,
    counts: &BTreeMap<&str, u64>,
    has_codes: impl Fn(Lfm2WeightRole) -> bool,
) -> (bool, &'static str) {
    let fallback = expected.roles.iter().any(|role| !has_codes(*role))
        || missing_lut2_dispatch(expected, counts);
    let dispatched = counts
        .iter()
        .any(|(name, count)| name.contains("lut2") && !name.contains("repack") && *count > 0);
    let effective = if fallback {
        if dispatched {
            "partial"
        } else {
            "raw_fallback"
        }
    } else if expected.roles.is_empty() && mode != Lfm2Lut2Mode::Off {
        "not_applicable"
    } else {
        mode_name(mode)
    };
    (fallback, effective)
}

fn qualify(opts: &Options) -> HostResult<Value> {
    let started = Instant::now();
    let config = fs::read(opts.bundle.join("config.json"))?;
    let weights = fs::read(opts.bundle.join("model.safetensors"))?;
    let tokenizer_bytes = fs::read(&opts.tokenizer)?;
    let prompt_bytes = fs::read(&opts.prompts)?;
    let artifacts = json!({
        "config_sha256":format!("{:x}",Sha256::digest(&config)),
        "weights_sha256":format!("{:x}",Sha256::digest(&weights)),
        "tokenizer_sha256":format!("{:x}",Sha256::digest(&tokenizer_bytes)),
        "prompts_sha256":format!("{:x}",Sha256::digest(&prompt_bytes)),
    });
    let tokenizer = Tokenizer::from_json_bytes(&tokenizer_bytes, TokenizerLimits::default())?;
    let prompts: Vec<String> = serde_json::from_slice(&prompt_bytes)?;
    if prompts.is_empty() || prompts.iter().any(|prompt| prompt.trim().is_empty()) {
        return Err("prompts must be a nonempty array of nonempty strings".into());
    }
    let encoded = prompts
        .iter()
        .map(|prompt| {
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if encoded.iter().any(|ids| {
        ids.is_empty() || u64::try_from(ids.len()).map_or(true, |len| len > opts.context)
    }) {
        return Err("encoded prompts must be nonempty and fit the explicit context".into());
    }
    let mut backend = WgpuBackend::new_with_options(
        19,
        ResourceLimits {
            max_allocation_bytes: 1 << 30,
            max_total_bytes: 2 << 30,
            max_pending_operations: 512,
        },
        WgpuOptions {
            diagnostics: opts.diagnostics,
            nf4_staging: Nf4Staging::default(),
        },
    )?;
    if format!("{:?}", backend.adapter_info().device_type) == "Cpu" {
        return Err("native GPU qualification rejects a software CPU adapter".into());
    }
    let typed = load(opts, &config, weights, &mut backend)?;
    let formats = (0..typed.config().layers.len())
        .map(|index| {
            [
                Lfm2LayerWeightRole::FfnW1,
                Lfm2LayerWeightRole::FfnW3,
                Lfm2LayerWeightRole::FfnW2,
            ]
            .map(|role| typed.role_quant(Lfm2WeightRole::Layer { index, role }))
        })
        .collect::<Vec<_>>();
    let passes = opts
        .warmups
        .checked_add(1)
        .ok_or("warmup count overflows")?;
    let expected = lut2_expectation(&formats, opts.lut2, &encoded, opts.cached, passes)?;
    let mut classifier = Lfm2Classifier::new_with_options(
        backend,
        typed,
        Lfm2ExecutionLimits {
            max_logical_tokens: opts.context,
        },
        Lfm2ExecutionOptions {
            lut2_mode: opts.lut2,
            max_lut2_bytes: opts.max_lut2_bytes,
        },
    )?;
    let initialization_seconds = started.elapsed().as_secs_f64();
    let warmup_started = Instant::now();
    for _ in 0..opts.warmups {
        let _ = samples(&mut classifier, &encoded, opts)?;
    }
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();
    let (rows, base_prefill_seconds) = samples(&mut classifier, &encoded, opts)?;
    let mut result = diagnostics(&classifier, &expected)?;
    result["schema_version"] = json!(1);
    result["classes"] = json!(opts.classes);
    result["context"] = json!(opts.context);
    result["artifacts"] = artifacts;
    result["cached"] = json!(opts.cached);
    result["initialization_seconds"] = json!(initialization_seconds);
    result["warmup_seconds"] = json!(warmup_seconds);
    result["base_prefill_seconds"] = json!(base_prefill_seconds);
    result["samples"] = json!(rows);
    Ok(result)
}

fn main() -> HostResult<()> {
    let mut environment = BTreeMap::new();
    for name in ["MINI_FFN_LUT2", "CLASSIFY_PREFIX", "MINIFIELD_WGPU_STATS"] {
        match env::var(name) {
            Ok(value) => {
                environment.insert(name.to_owned(), value);
            }
            Err(env::VarError::NotPresent) => {}
            Err(error) => return Err(error.into()),
        }
    }
    let opts = options(&env::args().skip(1).collect::<Vec<_>>(), &environment)?;
    println!("{}", qualify(&opts)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{boolean, common_prefix, lut2_expectation, lut2_mode, lut2_status, options};
    use minifield_executor_core::{Lfm2Lut2Mode, Lfm2WeightFormat};
    use std::collections::BTreeMap;

    #[test]
    fn dispatch_evidence_checks_down_and_pair_separately() {
        let formats = [[Lfm2WeightFormat::TernaryV1; 3]; 2];
        let expected = lut2_expectation(&formats, Lfm2Lut2Mode::Auto, &[vec![1; 96]], false, 1)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut counts = BTreeMap::from([("packed_gemm_ternary_lut2", 1)]);
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &counts, |_| true),
            (true, "partial")
        );
        counts.insert("packed_gemm_pair_swiglu_lut2", 1);
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &counts, |_| true),
            (false, "auto")
        );
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &counts, |_| false),
            (true, "partial")
        );
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &BTreeMap::new(), |_| false),
            (true, "raw_fallback")
        );
    }

    #[test]
    fn short_prompts_have_no_expected_lut2_tiles_even_when_codes_are_unavailable() {
        let formats = [[Lfm2WeightFormat::TernaryV1; 3]; 3];
        let expected = lut2_expectation(&formats, Lfm2Lut2Mode::Auto, &[vec![1; 95]], false, 2)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(expected.submitted_rows, [95]);
        assert!(expected.dispatches.is_empty());
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &BTreeMap::new(), |_| false),
            (false, "not_applicable")
        );
    }

    #[test]
    fn cached_long_base_and_short_tails_expect_only_base_tiles() {
        let formats = [[Lfm2WeightFormat::TernaryV1; 3]; 3];
        let mut first = vec![1; 96];
        first.push(2);
        let mut second = vec![1; 96];
        second.extend([3, 4]);
        let expected = lut2_expectation(&formats, Lfm2Lut2Mode::Auto, &[first, second], true, 2)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(expected.submitted_rows, [96, 1, 2]);
        assert_eq!(expected.dispatches["packed_gemm_ternary_lut2"], 4);
        assert_eq!(expected.dispatches["packed_gemm_pair_swiglu_lut2"], 4);
        assert_eq!(expected.roles.len(), 6);
        let counts = expected.dispatches.clone();
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::Auto, &expected, &counts, |_| true),
            (false, "auto")
        );
    }

    #[test]
    fn full_long_prompts_count_each_nonfinal_layer_and_repeated_pass() {
        let formats = [[Lfm2WeightFormat::TernaryV1; 3]; 3];
        let expected = lut2_expectation(
            &formats,
            Lfm2Lut2Mode::DownOnly,
            &[vec![1; 96], vec![2; 128]],
            false,
            2,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            expected.dispatches,
            BTreeMap::from([("packed_gemm_ternary_lut2", 8)])
        );
        let counts = BTreeMap::from([("packed_gemm_ternary_lut2", 7)]);
        assert_eq!(
            lut2_status(Lfm2Lut2Mode::DownOnly, &expected, &counts, |_| true),
            (true, "partial")
        );
    }

    #[test]
    fn single_layer_classifier_ffn_always_has_one_row() {
        let formats = [[Lfm2WeightFormat::TernaryV1; 3]; 1];
        for cached in [false, true] {
            let expected =
                lut2_expectation(&formats, Lfm2Lut2Mode::Auto, &[vec![1; 128]], cached, 1)
                    .unwrap_or_else(|error| panic!("{error}"));
            assert!(expected.dispatches.is_empty());
            assert_eq!(
                lut2_status(Lfm2Lut2Mode::Auto, &expected, &BTreeMap::new(), |_| true),
                (false, "not_applicable")
            );
        }
    }

    #[test]
    fn unfused_mixed_ffn_formats_do_not_expect_lut2() {
        let formats = [[
            Lfm2WeightFormat::Dense,
            Lfm2WeightFormat::Dense,
            Lfm2WeightFormat::TernaryV1,
        ]; 2];
        let expected = lut2_expectation(&formats, Lfm2Lut2Mode::Auto, &[vec![1; 128]], false, 1)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(expected.dispatches.is_empty());
    }

    #[test]
    fn strict_host_environment_preserves_false_and_rejects_typos() {
        assert_eq!(boolean("0").ok(), Some(false));
        assert_eq!(boolean("false").ok(), Some(false));
        assert!(boolean("nope").is_err());
        assert!(lut2_mode("unknown").is_err());
        let args = vec!["bundle".to_owned(), "prompts.json".to_owned()];
        let environment = BTreeMap::from([("CLASSIFY_PREFIX".to_owned(), "0".to_owned())]);
        assert!(options(&args, &environment).is_ok_and(|options| !options.cached));
        let environment = BTreeMap::from([("MINI_FFN_LUT2".to_owned(), "unknown".to_owned())]);
        assert!(options(&args, &environment).is_err());
    }

    #[test]
    fn explicit_arguments_override_defaults_and_accept_legacy_tokenizer() {
        let args = [
            "bundle",
            "prompts.json",
            "--classes",
            "3",
            "--context",
            "64",
            "--tokenizer",
            "tokenizer.json",
            "--lut2",
            "raw",
            "--prefix",
            "false",
        ]
        .map(str::to_owned);
        let environment = BTreeMap::from([("MINI_FFN_LUT2".to_owned(), "unknown".to_owned())]);
        let opts = options(&args, &environment).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(opts.classes, 3);
        assert_eq!(opts.context, 64);
        assert_eq!(
            opts.tokenizer,
            std::path::PathBuf::from("bundle/tokenizer.json")
        );
        assert!(!opts.cached);
    }

    #[test]
    fn common_prefix_retains_one_token_for_every_tail() {
        assert_eq!(common_prefix(&[]), 0);
        assert_eq!(common_prefix(&[vec![1, 2, 3], vec![1, 2]]), 1);
        assert_eq!(common_prefix(&[vec![1], vec![1]]), 0);
        assert_eq!(common_prefix(&[vec![1, 2], vec![3, 4]]), 0);
    }
}
