"""Bounded, inference-only views of the versioned full training state."""

import struct
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError, safe_open

from .container import MAX_HEADER_BYTES, MAX_WEIGHTS_BYTES, TensorFile
from .errors import ConversionError
from .jsonio import parse_json, read_json

ModelMode = Literal["classifier", "pointer-encoder"]


def _normalize_parameter(name: str, mode: ModelMode) -> str:
    if mode == "pointer-encoder":
        if name.startswith("lfm2."):
            return "model." + name.removeprefix("lfm2.")
        if name.startswith("magicbox.pointer."):
            return "pointer." + name.removeprefix("magicbox.pointer.") + ".weight"
    return name


class CheckpointParameters(TensorFile):
    """A bounded header view exposing only FP32 masters from ``params/``.

    Inherited tensor reads seek to individual byte ranges. The broader full-state
    header may contain scalar optimizer counters and is checked by safetensors.
    """

    def __init__(self, path: Path, mode: ModelMode):
        self.path = path
        size = path.stat().st_size
        if not 8 <= size <= MAX_WEIGHTS_BYTES:
            raise ConversionError("checkpoint exceeds byte limit or is truncated")
        with path.open("rb") as stream:
            prefix = stream.read(8)
            (length,) = struct.unpack("<Q", prefix)
            if length > MAX_HEADER_BYTES or length > size - 8:
                raise ConversionError("checkpoint header exceeds byte limit")
            header = parse_json(stream.read(length))
        if not isinstance(header, dict) or len(header) > 10001:
            raise ConversionError("checkpoint header must be a bounded object")
        self.data_start = 8 + length
        self.metadata = header.pop("__metadata__", {})
        try:
            with safe_open(path, framework="numpy") as artifact:
                if set(artifact.keys()) != set(header):
                    raise ConversionError("checkpoint parser inventory disagreement")
        except SafetensorError as error:
            raise ConversionError(f"invalid checkpoint container: {error}") from error
        self.entries = {}
        self.source_names = {}
        for name, entry in header.items():
            if not name.startswith("params/"):
                continue
            canonical = _normalize_parameter(name.removeprefix("params/"), mode)
            if canonical in self.entries:
                raise ConversionError(f"parameter normalization collision: {canonical}")
            if entry["dtype"] != "F32" or not 1 <= len(entry["shape"]) <= 3:
                raise ConversionError(
                    f"checkpoint parameters must be FP32 masters: {name}"
                )
            self.entries[canonical] = entry
            self.source_names[canonical] = name
        if not self.entries:
            raise ConversionError("full checkpoint contains no params/ masters")
        leaves = [name.removeprefix("params/") for name in self.source_names.values()]
        expected = {
            f"{group}/{name}" for group in ("params", "m", "v") for name in leaves
        }
        expected.add("step")
        if set(header) != expected:
            raise ConversionError(
                "full-state inventory must contain params, m, v, and step"
            )
        for name in leaves:
            parameter = header["params/" + name]
            for group in ("m", "v"):
                moment = header[group + "/" + name]
                if moment["dtype"] != "F32" or moment["shape"] != parameter["shape"]:
                    raise ConversionError(
                        "optimizer header differs from parameter master"
                    )
        if header["step"]["dtype"] != "I32" or header["step"]["shape"] != []:
            raise ConversionError("full-state step must be a scalar I32")


def read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    value = read_json(path)
    keys = {"format", "tensor_sha256", "inventory_sha256", "optimizer_id", "cursor"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["format"] != "minifield.full-training-state/1"
    ):
        raise ConversionError("expected minifield.full-training-state/1 manifest")
    for name in ("tensor_sha256", "inventory_sha256", "optimizer_id"):
        field = value[name]
        if (
            type(field) is not str
            or len(field) != 64
            or any(c not in "0123456789abcdef" for c in field)
        ):
            raise ConversionError(f"invalid checkpoint identity hash: {name}")
    cursor = value["cursor"]
    if not isinstance(cursor, dict) or set(cursor) != {
        "run_id",
        "data_sha256",
        "source_id",
        "next_batch",
    }:
        raise ConversionError("invalid checkpoint cursor")
    if (
        any(
            type(cursor[name]) is not str or not cursor[name]
            for name in ("run_id", "data_sha256", "source_id")
        )
        or type(cursor["next_batch"]) is not int
        or cursor["next_batch"] < 0
    ):
        raise ConversionError("invalid checkpoint cursor values")
    return value
