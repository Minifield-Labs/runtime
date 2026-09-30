"""Validate complete prediction responses and compute paired cell scores."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .schema import (
    CampaignError,
    canonical,
    fields,
    integer,
    number,
    positive,
    string,
    vector,
)


def build_argv(build: dict[str, Any], backend: str, target: Path) -> list[str]:
    argv = [
        build["cargo"],
        f"+{build['rust_toolchain']}",
        "build",
        "--release",
        "--locked",
        "--package",
        build["package"],
        "--bin",
        build["binary"],
        "--no-default-features",
        "--target-dir",
        str(target),
    ]
    features = build["features_by_backend"][backend]
    if features:
        argv.extend(["--features", ",".join(features)])
    return argv


def validate_response(
    response: Any,
    cell: dict[str, Any],
    reference: dict[str, Any],
    criteria: dict[str, Any],
    cycles: int,
) -> dict[str, Any]:
    response = fields(
        response,
        {
            "schema_version",
            "backend",
            "artifacts",
            "initialization_seconds",
            "warmup_seconds",
            "cases",
            "dispatch_counts",
            "resources",
        },
    )
    if response["schema_version"] != 1:
        raise CampaignError("unsupported host response schema")
    if response["artifacts"] != reference["artifacts"]:
        raise CampaignError("host-loaded artifact hashes differ from frozen oracle")
    backend = fields(response["backend"], {"implementation", "api", "device", "driver"})
    api = "cpu" if cell["backend"] == "cpu_reference" else "metal"
    if backend["implementation"] != cell["backend"] or backend["api"] != api:
        raise CampaignError("actual backend differs from requested backend")
    for key in ("device", "driver"):
        string(backend[key], key)
    number(response["initialization_seconds"], "initialization_seconds")
    number(response["warmup_seconds"], "warmup_seconds")
    counts = response["dispatch_counts"]
    if not isinstance(counts, dict):
        raise CampaignError("dispatch_counts must be an object")
    for name, count in counts.items():
        string(name, "dispatch name")
        integer(count, "dispatch count")
    for name, minimum in cell["dispatch_requirements"].items():
        if counts.get(name, 0) < minimum:
            raise CampaignError(f"missing required dispatch evidence: {name}")
    resources = fields(
        response["resources"], {"accounted_bytes", "peak_accounted_bytes"}
    )
    accounted = integer(resources["accounted_bytes"], "accounted_bytes")
    peak = integer(resources["peak_accounted_bytes"], "peak_accounted_bytes")
    if cell["backend"] != "cpu_reference" and peak == 0:
        raise CampaignError("GPU peak memory instrumentation is missing or zero")
    if peak < accounted:
        raise CampaignError("peak memory is below ending accounted memory")
    if peak > criteria["resource_limit_bytes"]:
        raise CampaignError("peak backend-accounted memory exceeds frozen budget")
    cases = response["cases"]
    if not isinstance(cases, list) or len(cases) != len(reference["cases"]):
        raise CampaignError("host response case count mismatch")
    total_predictions, total_seconds = 0, 0.0
    for case, expected in zip(cases, reference["cases"], strict=True):
        fields(
            case,
            {
                "id",
                "completed_predictions",
                "elapsed_seconds",
                "latencies_seconds",
                "outputs",
                "predictions",
            },
        )
        if case["id"] != expected["id"]:
            raise CampaignError("host response case IDs/order mismatch")
        if integer(case["completed_predictions"], "completed_predictions") != cycles:
            raise CampaignError("host did not complete every fixed-work prediction")
        elapsed = positive(case["elapsed_seconds"], "elapsed_seconds")
        for key in ("latencies_seconds", "outputs", "predictions"):
            if not isinstance(case[key], list) or len(case[key]) != cycles:
                raise CampaignError(f"incomplete {key}")
        latencies = [positive(x, "latency") for x in case["latencies_seconds"]]
        if sum(latencies) > elapsed * (1 + 1e-6):
            raise CampaignError("prediction latencies exceed measured case time")
        target = expected["output"]
        for output, prediction in zip(
            case["outputs"], case["predictions"], strict=True
        ):
            values = vector(output, "output")
            if len(values) != len(target):
                raise CampaignError("output width differs from immutable oracle")
            for actual, original in zip(values, target, strict=True):
                tolerance = criteria["absolute_tolerance"] + criteria[
                    "relative_tolerance"
                ] * abs(original)
                if abs(actual - original) > tolerance:
                    raise CampaignError("output differs from same-precision oracle")
            if canonical(prediction) != canonical(expected["prediction"]):
                raise CampaignError("prediction differs from immutable oracle")
            if cell["task"] == "classifier" and prediction != max(
                range(len(values)), key=values.__getitem__
            ):
                raise CampaignError(
                    "reported classification differs from actual argmax"
                )
        total_predictions += cycles
        total_seconds += elapsed
    positive(total_seconds, "total measured seconds")
    response["sustained_predictions_per_second"] = positive(
        total_predictions / total_seconds, "derived sustained prediction rate"
    )
    return response


def paired_statistics(
    samples: list[dict[str, Any]], criteria: dict[str, Any]
) -> dict[str, Any]:
    means = {}
    for role in ("champion", "candidate"):
        values = [
            s["response"]["sustained_predictions_per_second"]
            for s in samples
            if s["role"] == role
        ]
        if len(values) != criteria["independent_runs"]:
            raise CampaignError("incomplete independent run count")
        for value in values:
            positive(value, "independent prediction rate")
        means[role] = positive(sum(values) / len(values), "mean prediction rate")
    ratios = []
    paired_totals = []
    for block in range(criteria["independent_runs"] // 2):
        block_samples = [s for s in samples if s["block"] == block]
        if [s["role"] for s in block_samples] != [
            "champion",
            "candidate",
            "candidate",
            "champion",
        ]:
            raise CampaignError("invalid ABBA block")
        a, b, c, d = [
            s["response"]["sustained_predictions_per_second"] for s in block_samples
        ]
        numerator = positive(b + c, "paired candidate rate total")
        denominator = positive(a + d, "paired champion rate total")
        ratios.append(positive(numerator / denominator, "paired rate ratio"))
        paired_totals.append((numerator, denominator))
    result: dict[str, Any] = {
        "mean_predictions_per_second": means,
        "candidate_over_champion": positive(
            means["candidate"] / means["champion"], "mean rate ratio"
        ),
        "paired_block_ratios": ratios,
        "status": "inconclusive",
    }
    if len(ratios) < criteria["minimum_paired_blocks"]:
        result["reason"] = "fewer independent paired blocks than frozen minimum"
        return result
    rng = random.Random(criteria["seed"])
    # Entire ABBA blocks are the sampling units; dispatches are never independent.
    boot = []
    for _ in range(criteria["bootstrap_resamples"]):
        chosen = rng.choices(paired_totals, k=len(paired_totals))
        numerator = positive(
            sum(pair[0] for pair in chosen), "bootstrap candidate total"
        )
        denominator = positive(
            sum(pair[1] for pair in chosen), "bootstrap champion total"
        )
        boot.append(positive(numerator / denominator, "bootstrap rate ratio"))
    boot.sort()
    tail = (1 - criteria["confidence"]) / 2
    low = boot[math.floor(tail * (len(boot) - 1))]
    high = boot[math.ceil((1 - tail) * (len(boot) - 1))]
    result["paired_ratio_interval"] = [low, high]
    result["confidence"] = criteria["confidence"]
    guard = 1 / (1 + criteria["maximum_slowdown"])
    if high < guard:
        result["status"] = "regressed"
    elif low < guard:
        result["reason"] = "uncertainty crosses the protected slowdown bound"
    elif low >= 1 + criteria["minimum_improvement"]:
        result["status"] = "improved"
    else:
        result["status"] = "within_budget"
    return result
