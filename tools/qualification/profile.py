"""Explicit, portable admission rules for local bundle qualification."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


class QualificationError(ValueError):
    """A profile or measured result cannot support the requested check."""


def number(value, label, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise QualificationError(f"{label} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum:
        raise QualificationError(f"{label} must be finite and >= {minimum}")
    return value


def integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(f"{label} must be an integer >= {minimum}")
    return value


def read_json(path):
    def object_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise QualificationError(f"{path}: duplicate JSON field {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_fields,
            parse_constant=lambda text: (_ for _ in ()).throw(
                QualificationError(f"{path}: nonfinite JSON constant {text}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read JSON {path}: {error}") from error


@dataclass(frozen=True)
class Profile:
    path: Path
    name: str
    bundle: Path
    prompts: Path
    tokenizer: Path
    command: tuple[str, ...]
    classes: int
    context: int
    repetitions: int
    warmups: int
    timeout_seconds: float
    max_lut2_bytes: int
    lut2_mode: str
    cached: bool
    allow_lut2_fallback: bool
    expected_logits: Path | None
    reference_run: Path | None
    comparison_kind: str
    absolute_tolerance: float
    relative_tolerance: float
    max_relative_slowdown: float | None
    quality_thresholds: dict | None


def load_profile(path):
    path = Path(path).resolve()
    data = read_json(path)
    if not isinstance(data, dict):
        raise QualificationError("profile must be a JSON object")
    allowed = {
        "schema_version",
        "name",
        "bundle",
        "prompts",
        "tokenizer",
        "command",
        "classes",
        "context",
        "repetitions",
        "warmups",
        "timeout_seconds",
        "max_lut2_bytes",
        "lut2_mode",
        "cached",
        "allow_lut2_fallback",
        "expected_logits",
        "reference_run",
        "comparison_kind",
        "tolerances",
        "max_relative_slowdown",
        "quality_thresholds",
    }
    if unknown := data.keys() - allowed:
        raise QualificationError(f"unknown profile fields: {sorted(unknown)}")
    if data.get("schema_version") != 1 or isinstance(data["schema_version"], bool):
        raise QualificationError("profile schema_version must be 1")

    def local(field, default=None, *, required=False):
        value = data.get(field, default)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            raise QualificationError(f"{field} must be a nonempty path")
        result = Path(value)
        return (path.parent / result).resolve() if not result.is_absolute() else result

    name = data.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise QualificationError("name must be a nonempty string")
    bundle = local("bundle", required=True)
    tokenizer_value = data.get("tokenizer", "tokenizer/tokenizer.json")
    if not isinstance(tokenizer_value, str) or not tokenizer_value.strip():
        raise QualificationError(
            "tokenizer must be a nonempty path relative to the bundle"
        )
    tokenizer = Path(tokenizer_value)
    tokenizer = tokenizer if tokenizer.is_absolute() else bundle / tokenizer
    command = data.get(
        "command",
        [str(Path(__file__).resolve().parents[2] / "target/release/examples/classify")],
    )
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or not arg for arg in command)
    ):
        raise QualificationError("command must be a nonempty array of argument strings")
    if "/" in command[0] and not Path(command[0]).is_absolute():
        command[0] = str((path.parent / command[0]).resolve())
    tolerances = data.get("tolerances", {"absolute": 1e-4, "relative": 1e-4})
    if not isinstance(tolerances, dict) or tolerances.keys() - {"absolute", "relative"}:
        raise QualificationError("tolerances accepts only absolute and relative")
    mode = data.get("lut2_mode", "auto")
    if mode not in ("raw", "down", "auto"):
        raise QualificationError("lut2_mode must be raw, down, or auto")
    comparison_kind = data.get("comparison_kind", "implementation_parity")
    if comparison_kind not in ("implementation_parity", "quantization_quality"):
        raise QualificationError(
            "comparison_kind must be implementation_parity or quantization_quality"
        )
    for field in ("cached", "allow_lut2_fallback"):
        if field in data and not isinstance(data[field], bool):
            raise QualificationError(f"{field} must be a boolean")
    quality = data.get("quality_thresholds")
    if quality is not None:
        if not isinstance(quality, dict) or quality.keys() != {
            "max_absolute_delta",
            "minimum_argmax_agreement",
        }:
            raise QualificationError(
                "quality_thresholds requires max_absolute_delta "
                "and minimum_argmax_agreement"
            )
        number(quality["max_absolute_delta"], "quality maximum absolute delta")
        agreement = number(
            quality["minimum_argmax_agreement"], "quality minimum argmax agreement"
        )
        if agreement > 1:
            raise QualificationError("quality minimum argmax agreement must be <= 1")
    if comparison_kind == "quantization_quality" and quality is None:
        raise QualificationError(
            "quantization_quality requires explicit quality_thresholds"
        )
    slowdown = data.get("max_relative_slowdown")
    if slowdown is not None:
        number(slowdown, "max_relative_slowdown", minimum=1.0)
        if not data.get("reference_run"):
            raise QualificationError("max_relative_slowdown requires reference_run")
    return Profile(
        path=path,
        name=name,
        bundle=bundle,
        prompts=local("prompts", required=True),
        tokenizer=tokenizer.resolve(),
        command=tuple(command),
        classes=integer(data.get("classes", 8), "classes", minimum=1),
        context=integer(data.get("context", 512), "context", minimum=1),
        repetitions=integer(data.get("repetitions", 3), "repetitions", minimum=1),
        warmups=integer(data.get("warmups", 1), "warmups"),
        timeout_seconds=number(
            data.get("timeout_seconds", 120), "timeout_seconds", minimum=0.001
        ),
        max_lut2_bytes=integer(
            data.get("max_lut2_bytes", 64 * 1024 * 1024), "max_lut2_bytes"
        ),
        lut2_mode=mode,
        cached=data.get("cached", False),
        allow_lut2_fallback=data.get("allow_lut2_fallback", False),
        expected_logits=local("expected_logits"),
        reference_run=local("reference_run"),
        comparison_kind=comparison_kind,
        absolute_tolerance=number(
            tolerances.get("absolute", 1e-4), "absolute tolerance"
        ),
        relative_tolerance=number(
            tolerances.get("relative", 1e-4), "relative tolerance"
        ),
        max_relative_slowdown=slowdown,
        quality_thresholds=quality,
    )
