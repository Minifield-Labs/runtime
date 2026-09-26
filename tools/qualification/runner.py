#!/usr/bin/env python3
"""Run actual bundle qualification and matched implementation benchmarks."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from profile import QualificationError, integer, load_profile, number, read_json


def digest(path):
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError as error:
        raise QualificationError(
            f"missing or unreadable asset {path}: {error}"
        ) from error


def artifact_identity(profile):
    return {
        "weights_sha256": digest(profile.bundle / "model.safetensors"),
        "config_sha256": digest(profile.bundle / "config.json"),
        "tokenizer_sha256": digest(profile.tokenizer),
        "prompts_sha256": digest(profile.prompts),
    }


def source_identity():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    try:
        files = sorted(
            set(
                git(
                    "ls-files", "--cached", "--others", "--exclude-standard", "-z"
                ).split(b"\0")
            )
            - {b""}
        )
        hasher = hashlib.sha256()
        for relative in files:
            hasher.update(relative + b"\0")
            path = root / os.fsdecode(relative)
            hasher.update(digest(path).encode() if path.is_file() else b"deleted")
        return {
            "revision": git("rev-parse", "HEAD").decode().strip(),
            "working_tree_dirty": bool(
                git("status", "--porcelain", "--untracked-files=all")
            ),
            "source_sha256": hasher.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualificationError(f"cannot fingerprint source tree: {error}") from error


def command_identity(command):
    executable = shutil.which(command[0])
    if executable is None:
        raise QualificationError(f"missing classifier executable {command[0]}")
    files = [Path(executable).resolve()]
    files.extend(Path(arg).resolve() for arg in command[1:] if Path(arg).is_file())
    return [{"path": str(path), "sha256": digest(path)} for path in files]


def execute(command, timeout):
    started = time.monotonic()
    environment = os.environ.copy()
    for name in (
        "MINI_FFN_LUT2",
        "CLASSIFY_PREFIX",
        "MINIFIELD_WGPU_STATS",
        "MINI_NF4_STAGE_F16",
        "MINI_LOWBITS_EXPERIMENT",
    ):
        environment.pop(name, None)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise QualificationError(f"cannot start classifier: {error}") from error
    try:
        output, errors = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise QualificationError(f"classifier timed out after {timeout}s") from error
    if process.returncode:
        raise QualificationError(
            f"classifier exited {process.returncode}: {errors[-4096:]}"
        )
    try:
        result = json.loads(
            output,
            parse_constant=lambda text: (_ for _ in ()).throw(
                QualificationError(f"classifier emitted nonfinite JSON {text}")
            ),
        )
    except json.JSONDecodeError as error:
        raise QualificationError(
            f"classifier output is not one JSON result: {error}"
        ) from error
    return result, time.monotonic() - started


def logits_matrix(rows, prompts, classes):
    if not isinstance(rows, list) or len(rows) != prompts:
        raise QualificationError("logit row count must match the nonempty prompt array")
    for row in rows:
        if not isinstance(row, list) or len(row) != classes:
            raise QualificationError("every logit row must have exactly classes values")
        for value in row:
            # Logits may be negative, but never NaN or infinity.
            number(value, "logit", minimum=-sys.float_info.max)
    return rows


def validate_result(result, profile, identities, prompt_count, mode, cached):
    if not isinstance(result, dict):
        raise QualificationError("classifier result must be a JSON object")
    if integer(result.get("schema_version"), "classifier schema_version") != 1:
        raise QualificationError("classifier must emit schema_version 1")
    integer(result.get("classes"), "classifier classes", minimum=1)
    integer(result.get("context"), "classifier context", minimum=1)
    if not isinstance(result.get("cached"), bool):
        raise QualificationError("classifier cached must be a boolean")
    for field, expected in (
        ("classes", profile.classes),
        ("context", profile.context),
        ("requested_lut2_mode", mode),
        ("cached", cached),
    ):
        if result.get(field) != expected:
            raise QualificationError(
                f"classifier {field} disagrees with explicit request"
            )
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QualificationError("classifier must report artifact hashes")
    for field in (
        "weights_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "prompts_sha256",
    ):
        if artifacts.get(field) != identities[field]:
            raise QualificationError(
                f"classifier {field} does not match the input asset"
            )
    effective = result.get("effective_lut2_mode")
    if effective not in (
        "raw",
        "down",
        "auto",
        "not_applicable",
        "partial",
        "raw_fallback",
    ):
        raise QualificationError("classifier must report its effective LUT2 mode")
    fallback = result.get("lut2_fallback")
    if not isinstance(fallback, bool):
        raise QualificationError("classifier must report boolean lut2_fallback")
    if (
        fallback or effective in ("partial", "raw_fallback")
    ) and not profile.allow_lut2_fallback:
        raise QualificationError("unexpected LUT2 fallback")
    if not fallback and effective not in (mode, "not_applicable"):
        raise QualificationError("effective LUT2 mode disagrees with requested mode")
    number(result.get("initialization_seconds"), "initialization_seconds")
    number(result.get("warmup_seconds"), "warmup_seconds")
    number(result.get("base_prefill_seconds"), "base_prefill_seconds")
    samples = result.get("samples")
    if not isinstance(samples, list) or len(samples) != prompt_count:
        raise QualificationError("classifier must return one sample for every prompt")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("prompt_index") != index:
            raise QualificationError(
                "classifier prompt indices must preserve input order"
            )
        number(sample.get("seconds"), "sample inference seconds")
        ids = sample.get("ids")
        if not isinstance(ids, list) or not ids or len(ids) > profile.context:
            raise QualificationError(
                "classifier must report nonempty token IDs within context"
            )
        for token in ids:
            integer(token, "token ID")
    logits_matrix(
        [sample.get("logits") for sample in samples], prompt_count, profile.classes
    )
    counts = result.get("dispatch_counts")
    if not isinstance(counts, dict) or not counts:
        raise QualificationError("classifier must report actual kernel dispatch counts")
    for value in counts.values():
        integer(value, "dispatch count")
    if sum(counts.values()) == 0:
        raise QualificationError("classifier reported no kernel dispatches")
    adapter = result.get("adapter")
    if not isinstance(adapter, dict) or any(
        not isinstance(adapter.get(field), str) or not adapter[field]
        for field in ("name", "backend", "device_type")
    ):
        raise QualificationError("classifier must report the actual adapter")
    if adapter["device_type"] == "Cpu":
        raise QualificationError("unexpected software CPU adapter fallback")
    memory = result.get("memory")
    if not isinstance(memory, dict):
        raise QualificationError("classifier must report accounted backend memory")
    for field in (
        "resident_weight_bytes",
        "cache_bytes",
        "scratch_bytes",
        "staged_branch_bytes",
        "pending_operation_bytes",
        "pending_operations",
    ):
        integer(memory.get(field), f"memory.{field}")
    return result


def compare(actual, expected, profile, kind, label):
    maximum_absolute = 0.0
    maximum_relative = 0.0
    agrees = 0
    passed = True
    for row, reference in zip(actual, expected, strict=True):
        agrees += max(range(len(row)), key=row.__getitem__) == max(
            range(len(reference)), key=reference.__getitem__
        )
        for value, target in zip(row, reference, strict=True):
            delta = abs(value - target)
            if not math.isfinite(delta):
                raise QualificationError("logit comparison overflowed")
            maximum_absolute = max(maximum_absolute, delta)
            relative_delta = delta / max(abs(target), 1e-12)
            if not math.isfinite(relative_delta):
                raise QualificationError("relative logit comparison overflowed")
            maximum_relative = max(maximum_relative, relative_delta)
            passed &= (
                delta
                <= profile.absolute_tolerance + profile.relative_tolerance * abs(target)
            )
    agreement = agrees / len(actual)
    thresholds = None
    if kind == "implementation_parity":
        passed &= agrees == len(actual)
    else:
        thresholds = profile.quality_thresholds
        passed = (
            maximum_absolute <= thresholds["max_absolute_delta"]
            and agreement >= thresholds["minimum_argmax_agreement"]
        )
    return {
        "kind": kind,
        "reference": label,
        "passed": passed,
        "max_absolute_delta": maximum_absolute,
        "max_relative_delta": maximum_relative,
        "argmax_agreement": agreement,
        "quality_thresholds": thresholds,
    }


def validate_reference(reference, identities, profile):
    if not isinstance(reference, dict):
        raise QualificationError("reference must be a schema_version 1 JSON object")
    if integer(reference.get("schema_version"), "reference schema_version") != 1:
        raise QualificationError("reference must be a schema_version 1 JSON object")
    integer(reference.get("classes"), "reference classes", minimum=1)
    integer(reference.get("context"), "reference context", minimum=1)
    if reference.get("status", "passed") != "passed":
        raise QualificationError("a failed reference run cannot qualify a new run")
    if (
        reference.get("classes") != profile.classes
        or reference.get("context") != profile.context
    ):
        raise QualificationError("reference classes and context must match")
    fields = ["config_sha256", "tokenizer_sha256", "prompts_sha256"]
    if profile.comparison_kind == "implementation_parity":
        fields.append("weights_sha256")
    artifacts = reference.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QualificationError("reference must report artifact hashes")
    for field in fields:
        if artifacts.get(field) != identities[field]:
            raise QualificationError(
                f"reference {field} differs for {profile.comparison_kind}"
            )


def expected_reference(profile, identities, prompt_count):
    if profile.expected_logits:
        reference = read_json(profile.expected_logits)
        validate_reference(reference, identities, profile)
        return logits_matrix(
            reference.get("logits"), prompt_count, profile.classes
        ), str(profile.expected_logits)
    if profile.reference_run:
        reference = read_json(profile.reference_run)
        validate_reference(reference, identities, profile)
        runs = reference.get("runs", [])
        if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
            raise QualificationError("reference runs must be an array of run objects")
        measured = [run for run in runs if not run.get("warmup")]
        if not measured:
            raise QualificationError("reference run has no measured results")
        result = measured[0].get("result")
        if not isinstance(result, dict) or not isinstance(result.get("samples"), list):
            raise QualificationError("reference must contain measured sample objects")
        if any(not isinstance(sample, dict) for sample in result["samples"]):
            raise QualificationError("reference sample must be a JSON object")
        rows = [sample["logits"] for sample in result["samples"]]
        return logits_matrix(rows, prompt_count, profile.classes), str(
            profile.reference_run
        )
    return None


def latency_summary(runs):
    variants = {}
    for run in runs:
        if run["warmup"]:
            continue
        key = (run["lut2_mode"], run["cached"])
        variants.setdefault(key, []).append(run["result"])
    return [
        {
            "lut2_mode": mode,
            "cached": cached,
            "repetitions": len(results),
            "initialization_seconds": [
                result["initialization_seconds"] for result in results
            ],
            "base_prefill_seconds": [
                result["base_prefill_seconds"] for result in results
            ],
            "warmup_seconds": [result["warmup_seconds"] for result in results],
            "sample_seconds": [
                [sample["seconds"] for sample in result["samples"]]
                for result in results
            ],
            "median_inference_seconds": statistics.median(
                statistics.mean(sample["seconds"] for sample in result["samples"])
                for result in results
            ),
        }
        for (mode, cached), results in variants.items()
    ]


def check_slowdown(summary, profile, identities):
    if profile.max_relative_slowdown is None:
        return []
    reference = read_json(profile.reference_run)
    validate_reference(reference, identities, profile)
    # A performance regression check always compares identical weights.
    if reference.get("artifacts") != identities:
        raise QualificationError(
            "latency regression reference must use identical artifacts"
        )
    previous = reference.get("latency", [])
    checks = []
    for variant in summary:
        matches = [
            entry
            for entry in previous
            if entry.get("lut2_mode") == variant["lut2_mode"]
            and entry.get("cached") == variant["cached"]
        ]
        if len(matches) != 1:
            raise QualificationError(
                "reference latency must cover each matching benchmark variant"
            )
        baseline = number(
            matches[0].get("median_inference_seconds"),
            "reference latency",
            minimum=1e-12,
        )
        ratio = variant["median_inference_seconds"] / baseline
        if not math.isfinite(ratio):
            raise QualificationError("latency comparison overflowed")
        checks.append(
            {
                "kind": "latency_regression",
                "lut2_mode": variant["lut2_mode"],
                "cached": variant["cached"],
                "ratio": ratio,
                "max_ratio": profile.max_relative_slowdown,
                "passed": ratio <= profile.max_relative_slowdown,
            }
        )
    return checks


def run_profile(profile, kind):
    prompts = read_json(profile.prompts)
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise QualificationError("prompts must be a nonempty array of nonempty strings")
    if kind == "bench" and profile.repetitions < 2:
        raise QualificationError("bench requires at least 2 matched repetitions")
    if kind == "qualify" and not (profile.expected_logits or profile.reference_run):
        raise QualificationError(
            "qualify requires expected_logits or reference_run evidence"
        )
    identities = artifact_identity(profile)
    source = source_identity()
    executable = command_identity(profile.command)
    expected = expected_reference(profile, identities, len(prompts))
    variants = (
        [(mode, cached) for mode in ("raw", "down", "auto") for cached in (False, True)]
        if kind == "bench"
        else [(profile.lut2_mode, profile.cached)]
    )
    runs = []
    # Interleave and reverse variants on alternating repetitions. Each host
    # warms its loaded model before measurement and reports warmup separately.
    for repetition in range(profile.repetitions):
        ordered_variants = variants if repetition % 2 == 0 else list(reversed(variants))
        for mode, cached in ordered_variants:
            if artifact_identity(profile) != identities:
                raise QualificationError(
                    "an input artifact changed during the matched run"
                )
            if command_identity(profile.command) != executable:
                raise QualificationError("classifier executable changed during the run")
            command = [
                *profile.command,
                str(profile.bundle),
                str(profile.prompts),
                "--classes",
                str(profile.classes),
                "--context",
                str(profile.context),
                "--tokenizer",
                str(profile.tokenizer),
                "--lut2",
                mode,
                "--prefix",
                str(cached).lower(),
                "--max-lut2-bytes",
                str(profile.max_lut2_bytes),
                "--deadline-seconds",
                str(profile.timeout_seconds),
                "--warmups",
                str(profile.warmups),
            ]
            result, process_seconds = execute(command, profile.timeout_seconds)
            validate_result(result, profile, identities, len(prompts), mode, cached)
            runs.append(
                {
                    "lut2_mode": mode,
                    "cached": cached,
                    "repetition": repetition,
                    "warmup": False,
                    "process_seconds": process_seconds,
                    "result": result,
                }
            )
    measured = [run for run in runs if not run["warmup"]]
    baseline = [sample["logits"] for sample in measured[0]["result"]["samples"]]
    comparisons = []
    for run in measured:
        rows = [sample["logits"] for sample in run["result"]["samples"]]
        path = "cached" if run["cached"] else "full"
        label = f"{run['lut2_mode']}/{path}/{run['repetition']}"
        if kind == "bench":
            comparisons.append(
                compare(
                    rows,
                    baseline,
                    profile,
                    "implementation_parity",
                    "matched raw/full: " + label,
                )
            )
        if expected:
            comparisons.append(
                compare(
                    rows,
                    expected[0],
                    profile,
                    profile.comparison_kind,
                    expected[1] + ": " + label,
                )
            )
    summary = latency_summary(runs)
    comparisons.extend(check_slowdown(summary, profile, identities))
    return {
        "schema_version": 1,
        "status": "passed"
        if all(check["passed"] for check in comparisons)
        else "failed",
        "kind": kind,
        "profile": profile.name,
        "profile_sha256": digest(profile.path),
        "revision": source["revision"],
        "source": source,
        "source_after": source_identity(),
        "command_files": executable,
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "artifacts": identities,
        "classes": profile.classes,
        "context": profile.context,
        "comparison_kind": profile.comparison_kind,
        "runs": runs,
        "latency": summary,
        "comparisons": comparisons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("qualify", "bench"))
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="save the same machine-readable result printed to stdout",
    )
    args = parser.parse_args(argv)
    try:
        result = run_profile(load_profile(args.profile), args.kind)
    except (QualificationError, KeyError, TypeError, ValueError, OSError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "kind": args.kind,
            "profile": str(args.profile),
            "error": str(error),
        }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
