"""Validate campaign inputs before any build or GPU work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

BACKENDS = {"cpu_reference", "wgpu_metal", "native_metal"}
PRECISIONS = {"fp16", "int8", "nf4", "ternary", "mixed_qat"}


class CampaignError(ValueError):
    """A campaign cannot establish admissible evidence."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def loads(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise CampaignError(f"nonfinite JSON constant: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise CampaignError(f"nonfinite JSON number: {value}")
        return parsed

    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except json.JSONDecodeError as error:
        raise CampaignError(f"invalid JSON: {error}") from error


def read_json(path: Path) -> Any:
    try:
        return loads(path.read_text())
    except OSError as error:
        raise CampaignError(f"cannot read {path}: {error}") from error


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
    except OSError as error:
        raise CampaignError(f"cannot hash {path}: {error}") from error
    return sha.hexdigest()


def fields(
    value: Any, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError("expected JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise CampaignError(
            f"missing fields {sorted(missing)}; unknown fields {sorted(unknown)}"
        )
    return value


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{name} must be an integer >= {minimum}")
    return value


def number(value: Any, name: str, minimum: float = 0) -> float:
    try:
        finite = math.isfinite(value) if isinstance(value, float | int) else False
    except OverflowError:
        finite = False
    if (
        isinstance(value, bool)
        or not isinstance(value, float | int)
        or not finite
        or value < minimum
    ):
        raise CampaignError(f"{name} must be finite and >= {minimum}")
    return float(value)


def positive(value: Any, name: str) -> float:
    result = number(value, name)
    if result == 0:
        raise CampaignError(f"{name} must be positive")
    return result


def string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{name} must be a nonempty string")
    return value


def vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{name} must be a nonempty vector")
    return [number(x, name, -float("inf")) for x in value]


def validate_inputs(value: Any, task: str) -> list[dict[str, Any]]:
    root = fields(value, {"schema_version", "task", "cases"})
    if root["schema_version"] != 1 or root["task"] != task:
        raise CampaignError("input schema/task mismatch")
    if not isinstance(root["cases"], list) or not root["cases"]:
        raise CampaignError("inputs need cases")
    seen: set[str] = set()
    for case in root["cases"]:
        if task == "classifier":
            fields(case, {"id"}, {"text", "token_ids"})
            if ("text" in case) == ("token_ids" in case):
                raise CampaignError("classifier case needs text XOR token_ids")
            if "text" in case:
                string(case["text"], "text")
        else:
            fields(case, {"id", "token_ids", "questions"}, {"segments"})
        if "token_ids" in case:
            if not isinstance(case["token_ids"], list) or not case["token_ids"]:
                raise CampaignError("token_ids must be nonempty")
            for token in case["token_ids"]:
                integer(token, "token ID")
        if task == "pointer":
            validate_pointer_case(case)
        case_id = string(case["id"], "case ID")
        if case_id in seen:
            raise CampaignError(f"duplicate case ID: {case_id}")
        seen.add(case_id)
    return root["cases"]


def validate_pointer_case(case: dict[str, Any]) -> None:
    tokens = len(case["token_ids"])
    segments = case.get("segments", [1] * tokens)
    if not isinstance(segments, list) or len(segments) != tokens:
        raise CampaignError("pointer segments must match token count")
    for segment in segments:
        integer(segment, "segment ID")
    if not isinstance(case["questions"], list) or not case["questions"]:
        raise CampaignError("pointer case needs questions")
    for question in case["questions"]:
        fields(question, {"query_index", "option_indices", "kind"})
        query = integer(question["query_index"], "query index")
        if query >= tokens or segments[query] == 0:
            raise CampaignError("pointer query must name an active token")
        options = question["option_indices"]
        if not isinstance(options, list):
            raise CampaignError("pointer options must be an array")
        for option in options:
            integer(option, "option index")
            if option >= tokens or segments[option] != segments[query]:
                raise CampaignError("pointer option must share query segment")
        if len(options) != len(set(options)):
            raise CampaignError("pointer options must be distinct")
        kind = question["kind"]
        if not isinstance(kind, dict) or kind.get("type") not in {
            "choice",
            "ordinal",
            "binary",
            "extract",
        }:
            raise CampaignError("unknown pointer question kind")
        if kind["type"] == "extract":
            fields(
                kind,
                {
                    "type",
                    "absent_index",
                    "source_start",
                    "selectable",
                    "presence_threshold",
                },
            )
            absent = integer(kind["absent_index"], "absent index")
            start = integer(kind["source_start"], "source start")
            selectable = kind["selectable"]
            if not isinstance(selectable, list) or any(
                type(x) is not bool for x in selectable
            ):
                raise CampaignError("selectable must be a boolean array")
            if start + len(selectable) > tokens:
                raise CampaignError("pointer selectable source exceeds token count")
            if absent >= tokens or segments[absent] != segments[query]:
                raise CampaignError("pointer absent marker must share query segment")
            for offset, selected in enumerate(selectable):
                if selected and (
                    start + offset == absent
                    or segments[start + offset] != segments[query]
                ):
                    raise CampaignError("selectable token must share query segment")
            if number(kind["presence_threshold"], "presence threshold") > 1:
                raise CampaignError("presence threshold must be <= 1")
        else:
            if not options:
                raise CampaignError("choice/ordinal/binary need option indices")
            if kind["type"] == "binary":
                fields(kind, {"type", "positive_option"})
                if (
                    len(options) != 2
                    or integer(kind["positive_option"], "positive option") > 1
                ):
                    raise CampaignError(
                        "binary pointer needs two options and positive option"
                    )
            else:
                fields(kind, {"type"})


def validate_pointer_reference(case: dict[str, Any], supplied: dict[str, Any]) -> None:
    expected_width = 2 * len(case["token_ids"]) * len(case["questions"])
    predictions = supplied["prediction"]
    if not isinstance(predictions, list) or len(predictions) != len(case["questions"]):
        raise CampaignError("pointer reference needs one decision per question")
    for question, prediction in zip(case["questions"], predictions, strict=True):
        kind = question["kind"]["type"]
        if kind == "choice":
            expected_width += len(question["option_indices"])
            fields(prediction, {"type", "index"})
            decision = integer(prediction["index"], "choice index")
            if prediction["type"] != "choice" or decision >= len(
                question["option_indices"]
            ):
                raise CampaignError("invalid pointer choice decision")
        elif kind == "ordinal":
            expected_width += 1 + len(question["option_indices"])
            fields(prediction, {"type", "level"})
            decision = integer(prediction["level"], "ordinal level")
            if prediction["type"] != "ordinal" or decision >= len(
                question["option_indices"]
            ):
                raise CampaignError("invalid pointer ordinal decision")
        elif kind == "binary":
            expected_width += 1 + len(question["option_indices"])
            fields(prediction, {"type", "value"})
            if prediction["type"] != "binary" or type(prediction["value"]) is not bool:
                raise CampaignError("invalid pointer binary decision")
        else:
            expected_width += 1
            if not isinstance(prediction, dict):
                raise CampaignError("pointer decision must be an object")
            if prediction.get("type") == "absent":
                fields(prediction, {"type"})
            else:
                fields(prediction, {"type", "start", "end"})
                start = integer(prediction["start"], "span start")
                end = integer(prediction["end"], "span end", 1)
                if (
                    prediction["type"] != "span"
                    or not start < end
                    or end > len(question["kind"]["selectable"])
                ):
                    raise CampaignError("invalid pointer span decision")
    if len(supplied["output"]) != expected_width:
        raise CampaignError("pointer reference width differs from task output contract")


CRITERIA_FIELDS = {
    "independent_runs",
    "cooldown_seconds",
    "minimum_improvement",
    "maximum_slowdown",
    "absolute_tolerance",
    "relative_tolerance",
    "bootstrap_resamples",
    "confidence",
    "seed",
    "minimum_paired_blocks",
    "resource_limit_bytes",
    "build_deadline_seconds",
}


def validate_criteria(value: Any) -> dict[str, Any]:
    criteria = fields(value, CRITERIA_FIELDS)
    runs = integer(criteria["independent_runs"], "independent_runs", 4)
    if runs % 2:
        raise CampaignError("independent_runs must be even for ABBA blocks")
    number(criteria["cooldown_seconds"], "cooldown_seconds", 15)
    for key in ("minimum_improvement", "maximum_slowdown"):
        if number(criteria[key], key) >= 1:
            raise CampaignError(f"{key} must be < 1")
    for key in ("absolute_tolerance", "relative_tolerance"):
        number(criteria[key], key)
    integer(criteria["bootstrap_resamples"], "bootstrap_resamples", 1000)
    confidence = number(criteria["confidence"], "confidence")
    if not 0.8 <= confidence < 1:
        raise CampaignError("confidence must be in [0.8, 1)")
    integer(criteria["seed"], "seed")
    integer(criteria["minimum_paired_blocks"], "minimum_paired_blocks", 4)
    integer(criteria["resource_limit_bytes"], "resource_limit_bytes", 1)
    positive(criteria["build_deadline_seconds"], "build_deadline_seconds")
    return criteria


CELL_FIELDS = {
    "id",
    "model",
    "precision",
    "arithmetic",
    "backend",
    "bundle",
    "inputs",
    "reference",
    "task",
    "context",
    "mode",
    "warmups",
    "measured_cycles",
    "deadline_seconds",
    "lut2_mode",
    "max_lut2_bytes",
    "dispatch_requirements",
}


def resolve_cell(value: Any, base: Path) -> dict[str, Any]:
    cell = dict(fields(value, CELL_FIELDS, {"classes", "tokenizer"}))
    for key in ("id", "model"):
        string(cell[key], key)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cell["id"]):
        raise CampaignError("cell ID must be a safe single directory name")
    if cell["precision"] not in PRECISIONS:
        raise CampaignError("unknown storage precision")
    if cell["arithmetic"] != "f32":
        raise CampaignError("this schema freezes f32 arithmetic")
    if cell["backend"] not in BACKENDS:
        raise CampaignError("unknown backend")
    if cell["task"] not in {"classifier", "pointer"}:
        raise CampaignError("unknown task")
    if cell["mode"] not in {"full", "cached"}:
        raise CampaignError("unknown inference mode")
    if cell["lut2_mode"] not in {"off", "down", "auto"}:
        raise CampaignError("unknown LUT2 mode")
    integer(cell["context"], "context", 1)
    integer(cell["warmups"], "warmups", 1)
    integer(cell["measured_cycles"], "measured_cycles", 1)
    integer(cell["max_lut2_bytes"], "max_lut2_bytes")
    positive(cell["deadline_seconds"], "deadline_seconds")
    if cell["task"] == "classifier":
        integer(cell.get("classes"), "classes", 1)
    elif "classes" in cell:
        raise CampaignError("pointer cells don't declare classes")
    requirements = cell["dispatch_requirements"]
    if not isinstance(requirements, dict):
        raise CampaignError("dispatch_requirements must be an object")
    for name, count in requirements.items():
        string(name, "dispatch name")
        integer(count, "dispatch minimum", 1)
    if cell["backend"] != "cpu_reference" and not requirements:
        raise CampaignError("GPU cells need explicit dispatch requirements")
    for key in ("bundle", "inputs", "reference"):
        cell[key] = str((base / string(cell[key], key)).resolve())
    tokenizer = string(cell.get("tokenizer", "tokenizer/tokenizer.json"), "tokenizer")
    cell["tokenizer"] = str((Path(cell["bundle"]) / tokenizer).resolve())
    return cell


def artifact_hashes(cell: dict[str, Any]) -> dict[str, str]:
    bundle = Path(cell["bundle"])
    return {
        "weights_sha256": file_digest(bundle / "model.safetensors"),
        "config_sha256": file_digest(bundle / "config.json"),
        "tokenizer_sha256": file_digest(Path(cell["tokenizer"])),
        "inputs_sha256": file_digest(Path(cell["inputs"])),
    }


def validate_reference(cell: dict[str, Any]) -> dict[str, Any]:
    inputs = validate_inputs(read_json(Path(cell["inputs"])), cell["task"])
    reference = fields(
        read_json(Path(cell["reference"])),
        {"schema_version", "task", "artifacts", "cases"},
        {"provenance"},
    )
    if reference["schema_version"] != 1 or reference["task"] != cell["task"]:
        raise CampaignError("reference schema/task mismatch")
    if reference["artifacts"] != artifact_hashes(cell):
        raise CampaignError("reference does not bind the exact cell artifacts")
    if not isinstance(reference["cases"], list):
        raise CampaignError("reference cases must be an array")
    if len(reference["cases"]) != len(inputs):
        raise CampaignError("reference case count differs from inputs")
    for expected, supplied in zip(inputs, reference["cases"], strict=True):
        fields(supplied, {"id", "output", "prediction"})
        if expected["id"] != supplied["id"]:
            raise CampaignError("reference cases must match input IDs/order")
        values = vector(supplied["output"], "reference output")
        if cell["task"] == "classifier":
            if len(values) != cell["classes"]:
                raise CampaignError("reference class count mismatch")
            if type(supplied["prediction"]) is not int or supplied["prediction"] != max(
                range(len(values)), key=values.__getitem__
            ):
                raise CampaignError("reference prediction differs from stable argmax")
        else:
            validate_pointer_reference(expected, supplied)
        canonical(supplied["prediction"])
    return reference
