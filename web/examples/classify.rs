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

fn lut2_roles(weights: &Lfm2TypedWeights<WgpuBuffer>, mode: Lfm2Lut2Mode) -> Vec<Lfm2WeightRole> {
    let mut roles = Vec::new();
    for index in 0..weights.config().layers.len() {
        let role = |role| Lfm2WeightRole::Layer { index, role };
        let down = role(Lfm2LayerWeightRole::FfnW2);
        if mode != Lfm2Lut2Mode::Off && weights.role_quant(down) == Lfm2WeightFormat::TernaryV1 {
            roles.push(down);
        }
        let gate = role(Lfm2LayerWeightRole::FfnW1);
        let up = role(Lfm2LayerWeightRole::FfnW3);
        if mode == Lfm2Lut2Mode::Auto
            && weights.role_quant(gate) == Lfm2WeightFormat::TernaryV1
            && weights.role_quant(up) == Lfm2WeightFormat::TernaryV1
        {
            roles.extend([gate, up]);
        }
    }
    roles
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
    roles: &[Lfm2WeightRole],
) -> HostResult<Value> {
    let (counts, stats, adapter) = classifier.inspect_backend(|backend| {
        (
            backend.dispatch_counts(),
            backend.stats(),
            backend.adapter_info(),
        )
    })?;
    let lut2_dispatches = counts
        .iter()
        .filter(|(name, _)| name.contains("lut2") && !name.contains("repack"))
        .map(|(_, count)| *count)
        .sum::<u64>();
    let fallback = roles.iter().any(|role| !classifier.has_lut2_codes(*role))
        || missing_lut2_dispatch(roles, &counts);
    let effective = if fallback {
        if lut2_dispatches > 0 {
            "partial"
        } else {
            "raw_fallback"
        }
    } else if roles.is_empty() && classifier.lut2_mode() != Lfm2Lut2Mode::Off {
        "not_applicable"
    } else {
        mode_name(classifier.lut2_mode())
    };
    let memory = classifier.resource_report()?;
    Ok(json!({
        "requested_lut2_mode":mode_name(classifier.lut2_mode()),"effective_lut2_mode":effective,
        "lut2_fallback":fallback,"dispatch_counts":counts,
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

fn missing_lut2_dispatch(roles: &[Lfm2WeightRole], counts: &BTreeMap<&str, u64>) -> bool {
    roles.iter().any(|role| {
        let kernel = match role {
            Lfm2WeightRole::Layer {
                role: Lfm2LayerWeightRole::FfnW2,
                ..
            } => "packed_gemm_ternary_lut2",
            Lfm2WeightRole::Layer {
                role: Lfm2LayerWeightRole::FfnW1 | Lfm2LayerWeightRole::FfnW3,
                ..
            } => "packed_gemm_pair_swiglu_lut2",
            _ => return false,
        };
        counts.get(kernel).copied().unwrap_or_default() == 0
    })
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
    let roles = lut2_roles(&typed, opts.lut2);
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
    let mut result = diagnostics(&classifier, &roles)?;
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
    use super::{boolean, common_prefix, lut2_mode, missing_lut2_dispatch, options};
    use minifield_executor_core::{Lfm2LayerWeightRole, Lfm2WeightRole};
    use std::collections::BTreeMap;

    #[test]
    fn dispatch_evidence_checks_down_and_pair_separately() {
        let roles = [
            Lfm2WeightRole::Layer {
                index: 0,
                role: Lfm2LayerWeightRole::FfnW2,
            },
            Lfm2WeightRole::Layer {
                index: 0,
                role: Lfm2LayerWeightRole::FfnW1,
            },
        ];
        let mut counts = BTreeMap::from([("packed_gemm_ternary_lut2", 1)]);
        assert!(missing_lut2_dispatch(&roles, &counts));
        counts.insert("packed_gemm_pair_swiglu_lut2", 1);
        assert!(!missing_lut2_dispatch(&roles, &counts));
        assert!(!missing_lut2_dispatch(&[], &BTreeMap::new()));
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
