"""Explicit inference-only extraction with protected-role precision.

This accepts one named full-training-state contract, not arbitrary checkpoints.
Optimizer arrays are validated structurally by safetensors but never loaded.
"""

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import __version__
from .checkpoint import CheckpointParameters, ModelMode, read_checkpoint_manifest
from .config import Lfm2Config, validate_config
from .container import (
    Tensor,
    TensorFile,
    write_tensor_stream,
)
from .errors import ConversionError
from .jsonio import file_record, json_bytes, parse_json, read_json
from .paths import (
    ASSETS,
    MANIFEST,
    reject_symlinks,
    source_directory,
    staged_destination,
)
from .quantization import ALGORITHMS, dequantize, quantize
from .tokenizer import validate_tokenizer

SCHEMA = "minifield.prepared-evaluation-bundle/1"
POLICY = "minifield/protected-embedding-and-output-heads/1"
Precision = Literal["fp16", "int8", "nf4", "ternary"]
PACKED = {"ternary-v1": "ternary", "nf4-v1": "nf4", "int8-v1": "int8"}
DENSE = {"f16": "F16", "bf16": "BF16", "f32": "F32"}
POINTER_NAMES = tuple(
    f"pointer.{end}_{role}.weight"
    for end in ("start", "end")
    for role in ("query", "key")
)


def protected_role(name: str, shape: tuple[int, ...]) -> bool:
    """Only backbone rank-2 projections are eligible for low-bit storage."""
    return (
        len(shape) != 2
        or name == "model.embed_tokens.weight"
        or name == "lm_head.weight"
        or name == "classification_head.weight"
        or name in POINTER_NAMES
    )


def evaluation_inventory(
    config: Lfm2Config,
    mode: ModelMode,
    *,
    classes: int | None,
    pointer_width: int | None,
) -> dict[str, tuple[int, ...]]:
    if mode == "classifier":
        if classes is None or pointer_width is not None:
            raise ConversionError("classifier requires classes and no pointer_width")
        return config.inventory(classes)
    if mode != "pointer-encoder" or classes is not None:
        raise ConversionError("expected classifier or pointer-encoder mode")
    if type(pointer_width) is not int or not 0 < pointer_width <= 65536:
        raise ConversionError("pointer_width must be a positive integer <= 65536")
    return {
        **config.inventory(),
        **{name: (pointer_width, config.hidden) for name in POINTER_NAMES},
    }


def _model_config(
    raw: Any, mode: ModelMode, pointer_width: int | None
) -> tuple[dict[str, Any], Lfm2Config]:
    config = validate_config(raw)
    output = dict(raw)
    if mode == "pointer-encoder":
        if raw.get("architectures") != ["Lfm2BidirectionalForMaskedLM"]:
            raise ConversionError(
                "pointer encoder requires the bidirectional architecture"
            )
        if raw.get("use_cache") is not False:
            raise ConversionError("bidirectional encoder must declare use_cache=false")
        output["minifield_runtime_model"] = "lfm2-pointer-encoder/1"
        output["minifield_pointer"] = {
            "format": "minifield.magicbox-joint-pointer/1",
            "projection_dim": pointer_width,
        }
    output["dtype"] = "float16"
    return output, config


def _physical_inventory(
    inventory: dict[str, tuple[int, ...]], formats: dict[str, str]
) -> dict[str, tuple[str, tuple[int, ...]]]:
    output = {}
    for name, shape in inventory.items():
        representation = formats[name]
        if type(representation) is not str:
            raise ConversionError("role representations must be strings")
        if representation in DENSE:
            output[name] = (DENSE[representation], shape)
        elif representation in PACKED:
            if protected_role(name, shape):
                raise ConversionError(
                    f"protected role must remain high precision: {name}"
                )
            if shape[1] % 128:
                raise ConversionError(f"packed columns must divide group128: {name}")
            divisor = {"ternary-v1": 4, "nf4-v1": 2, "int8-v1": 1}[representation]
            output[name + ".codes"] = ("U8", (shape[0], shape[1] // divisor))
            output[name + ".scales"] = ("F16", (shape[0], shape[1] // 128))
        else:
            raise ConversionError(f"unknown role representation: {representation}")
    return output


def _validate_artifact(
    artifact: TensorFile,
    inventory: dict[str, tuple[int, ...]],
    formats: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(formats, dict):
        raise ConversionError("role precision must be an object")
    if set(formats) != set(inventory):
        raise ConversionError("role precision map differs from model inventory")
    expected = _physical_inventory(inventory, formats)
    if set(artifact.entries) != set(expected):
        raise ConversionError("prepared weight inventory mismatch")
    records = []
    for name, (dtype, shape) in sorted(expected.items()):
        entry = artifact.entries[name]
        if entry["dtype"] != dtype or tuple(entry["shape"]) != shape:
            raise ConversionError(f"wrong prepared tensor shape/dtype: {name}")
        artifact.validate_finite(name)
        records.append({"name": name, "dtype": dtype, "shape": list(shape)})
    for name, representation in formats.items():
        if protected_role(name, inventory[name]) and representation not in (
            "f16",
            "bf16",
        ):
            raise ConversionError(f"protected role requires F16 or BF16: {name}")
        if representation in PACKED:
            codes, scales = (
                artifact.tensor(name + ".codes"),
                artifact.tensor(name + ".scales"),
            )
            dequantize(
                np.frombuffer(codes.data, dtype=np.uint8).reshape(codes.shape),
                np.frombuffer(scales.data, dtype="<f2").reshape(scales.shape),
                PACKED[representation],
            )
    return records


def _asset(path: Path) -> Path:
    path = path.absolute()
    reject_symlinks(path)
    if not path.is_file():
        raise ConversionError(f"required regular source file is missing: {path}")
    return path


def prepare_checkpoint(
    checkpoint: Path,
    config_path: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    precision: Precision,
    mode: ModelMode,
    source_model: str,
    source_revision: str,
    classes: int | None = None,
    pointer_width: int | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract one supported full-state checkpoint without reading optimizer arrays."""
    if precision not in ("fp16", "int8", "nf4", "ternary"):
        raise ConversionError("precision must be fp16, int8, nf4, or ternary")
    if not source_model or not source_revision:
        raise ConversionError("source model and revision are required")
    checkpoint, config_path, tokenizer_path = map(
        _asset, (checkpoint, config_path, tokenizer_path)
    )
    checkpoint_manifest = _asset(checkpoint.parent / "manifest.json")
    state = read_checkpoint_manifest(checkpoint_manifest)
    source_files = [
        file_record(checkpoint, "checkpoint"),
        file_record(checkpoint_manifest, "checkpoint-manifest"),
        file_record(config_path, "config"),
        file_record(tokenizer_path, "tokenizer"),
    ]
    if source_files[0]["sha256"] != state.get("tensor_sha256"):
        raise ConversionError(
            "checkpoint tensor SHA-256 differs from training manifest"
        )
    if (
        expected_tokenizer_sha256 is not None
        and source_files[3]["sha256"] != expected_tokenizer_sha256
    ):
        raise ConversionError("tokenizer SHA-256 differs from pinned source")
    raw_config = read_json(config_path, 1 << 20)
    output_config, config = _model_config(raw_config, mode, pointer_width)
    validate_tokenizer(read_json(tokenizer_path), config.vocab)
    inventory = evaluation_inventory(
        config, mode, classes=classes, pointer_width=pointer_width
    )
    parameters = CheckpointParameters(checkpoint, mode)
    if set(parameters.entries) != set(inventory):
        missing = sorted(set(inventory) - set(parameters.entries))
        extra = sorted(set(parameters.entries) - set(inventory))
        raise ConversionError(
            f"checkpoint parameter inventory mismatch; missing={missing}, extra={extra}"
        )
    for name, shape in inventory.items():
        if tuple(parameters.entries[name]["shape"]) != shape:
            raise ConversionError(f"checkpoint parameter shape mismatch: {name}")
    formats = {
        name: "f16"
        if precision == "fp16" or protected_role(name, shape)
        else precision + "-v1"
        for name, shape in inventory.items()
    }
    metadata = (
        {"format": "pt"}
        if precision == "fp16"
        else {
            "format": "minifield.mixed.v1",
            "tensor_quantization": json_bytes(formats).decode().strip(),
            "group_size": "128",
            "code_order": "little-endian-byte-sequential",
            "quantizer": ALGORITHMS[precision],
            "source_model": source_model,
            "source_revision": source_revision,
            "producer": f"minifield-converters {__version__}",
        }
    )
    source_tensors = []

    def tensors() -> Iterator[tuple[str, Tensor]]:
        for name, shape in sorted(inventory.items()):
            tensor = parameters.tensor(name)
            values = np.frombuffer(tensor.data, dtype="<f4").reshape(tensor.shape)
            if not np.isfinite(values).all():
                raise ConversionError(f"non-finite checkpoint parameter: {name}")
            source_tensors.append(
                {
                    "source_name": parameters.source_names[name],
                    "name": name,
                    "dtype": tensor.dtype,
                    "shape": list(shape),
                    "sha256": hashlib.sha256(tensor.data).hexdigest(),
                }
            )
            if formats[name] == "f16":
                with np.errstate(over="ignore", invalid="ignore"):
                    stored = values.astype(np.float16)
                if not np.isfinite(stored).all():
                    raise ConversionError(f"FP16 storage overflow: {name}")
                yield name, Tensor.from_array(stored)
            else:
                codes, scales = quantize(values, precision)
                yield name + ".codes", Tensor.from_array(codes)
                yield name + ".scales", Tensor.from_array(scales)

    with staged_destination(output, checkpoint.parent, False) as pending:
        (pending / "tokenizer").mkdir()
        (pending / "config.json").write_bytes(json_bytes(output_config))
        shutil.copyfile(tokenizer_path, pending / "tokenizer/tokenizer.json")
        write_tensor_stream(
            pending / "model.safetensors",
            _physical_inventory(inventory, formats),
            tensors(),
            metadata,
        )
        # Detect mutation between provenance capture and extraction.
        if source_files != [
            file_record(path, record["path"])
            for path, record in zip(
                (checkpoint, checkpoint_manifest, config_path, tokenizer_path),
                source_files,
                strict=True,
            )
        ]:
            raise ConversionError("source files changed during extraction")
        manifest = {
            "schema": SCHEMA,
            "producer": f"minifield-converters {__version__}",
            "operation": {
                "kind": "prepare-checkpoint",
                "precision": precision,
                "role_policy": POLICY,
                "algorithm": "fp16-storage/1"
                if precision == "fp16"
                else ALGORITHMS[precision],
            },
            "model": {"mode": mode, "classes": classes, "pointer_width": pointer_width},
            "source_identity": {"model": source_model, "revision": source_revision},
            "checkpoint_manifest": state,
            "source_files": source_files,
            "source_tensors": source_tensors,
            "role_precision": formats,
            "files": [file_record(pending / relative, relative) for relative in ASSETS],
            "tensors": _validate_artifact(
                TensorFile(pending / "model.safetensors"), inventory, formats
            ),
            "weight_format": metadata["format"],
        }
        manifest["bundle_digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        (pending / MANIFEST).write_bytes(json_bytes(manifest))
        validate_prepared_bundle(pending)
    return manifest


def validate_prepared_bundle(directory: Path) -> dict[str, Any]:
    directory = source_directory(directory)
    manifest = read_json(_asset(directory / MANIFEST))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ConversionError("unsupported prepared-evaluation manifest")
    unsigned = dict(manifest)
    expected_digest = unsigned.pop("bundle_digest", None)
    if expected_digest != hashlib.sha256(json_bytes(unsigned)).hexdigest():
        raise ConversionError("prepared bundle provenance digest mismatch")
    if manifest.get("files") != [
        file_record(_asset(directory / relative), relative) for relative in ASSETS
    ]:
        raise ConversionError("prepared asset hashes/lengths differ from manifest")
    raw = read_json(directory / "config.json", 1 << 20)
    config = validate_config(raw, allow_f16=True)
    validate_tokenizer(read_json(directory / "tokenizer/tokenizer.json"), config.vocab)
    model = manifest.get("model")
    if not isinstance(model, dict) or set(model) != {
        "mode",
        "classes",
        "pointer_width",
    }:
        raise ConversionError(
            "prepared manifest must declare model mode and head dimensions"
        )
    if model["mode"] == "pointer-encoder":
        expected = {
            "format": "minifield.magicbox-joint-pointer/1",
            "projection_dim": model["pointer_width"],
        }
        if (
            raw.get("minifield_pointer") != expected
            or raw.get("architectures") != ["Lfm2BidirectionalForMaskedLM"]
            or raw.get("use_cache") is not False
        ):
            raise ConversionError(
                "pointer encoder config does not match prepared head contract"
            )
    inventory = evaluation_inventory(
        config,
        model["mode"],
        classes=model["classes"],
        pointer_width=model["pointer_width"],
    )
    artifact = TensorFile(directory / "model.safetensors")
    formats = manifest.get("role_precision")
    if not isinstance(formats, dict):
        raise ConversionError("prepared manifest requires role_precision")
    marker = artifact.metadata.get("format")
    if marker == "minifield.mixed.v1":
        if (
            artifact.metadata.get("group_size") != "128"
            or artifact.metadata.get("code_order") != "little-endian-byte-sequential"
        ):
            raise ConversionError("invalid prepared packed metadata")
        if (
            parse_json(artifact.metadata.get("tensor_quantization", "").encode())
            != formats
        ):
            raise ConversionError("container role precision differs from manifest")
    elif marker != "pt" or any(value != "f16" for value in formats.values()):
        raise ConversionError("unsupported prepared weight format")
    if manifest.get("weight_format") != marker or manifest.get(
        "tensors"
    ) != _validate_artifact(artifact, inventory, formats):
        raise ConversionError("prepared tensor inventory differs from manifest")
    return manifest


def package_mixed_bundle(
    weights: Path,
    config_path: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    classes: int,
    source_model: str,
    source_revision: str,
) -> dict[str, Any]:
    """Package an existing QAT artifact without rewriting any tensor bytes."""
    weights, config_path, tokenizer_path = map(
        _asset, (weights, config_path, tokenizer_path)
    )
    config = validate_config(read_json(config_path, 1 << 20), allow_f16=True)
    validate_tokenizer(read_json(tokenizer_path), config.vocab)
    artifact = TensorFile(weights)
    if artifact.metadata.get("format") != "minifield.mixed.v1":
        raise ConversionError("QAT packaging requires minifield.mixed.v1")
    formats = parse_json(artifact.metadata.get("tensor_quantization", "").encode())
    inventory = evaluation_inventory(
        config, "classifier", classes=classes, pointer_width=None
    )
    tensor_inventory = _validate_artifact(artifact, inventory, formats)
    inputs = (weights, config_path, tokenizer_path)
    source_files = [
        file_record(path, label)
        for path, label in zip(inputs, ("weights", "config", "tokenizer"), strict=True)
    ]
    with staged_destination(output, weights.parent, False) as pending:
        (pending / "tokenizer").mkdir()
        for path, relative in zip(
            inputs,
            ("model.safetensors", "config.json", "tokenizer/tokenizer.json"),
            strict=True,
        ):
            shutil.copyfile(path, pending / relative)
        files = [file_record(pending / relative, relative) for relative in ASSETS]
        if files[1]["sha256"] != source_files[0]["sha256"]:
            raise ConversionError("QAT artifact bytes changed while packaging")
        if source_files != [
            file_record(path, record["path"])
            for path, record in zip(inputs, source_files, strict=True)
        ]:
            raise ConversionError("QAT source assets changed while packaging")
        manifest = {
            "schema": SCHEMA,
            "producer": f"minifield-converters {__version__}",
            "operation": {
                "kind": "package-mixed-qat",
                "lossless": True,
                "role_policy": POLICY,
            },
            "model": {"mode": "classifier", "classes": classes, "pointer_width": None},
            "source_identity": {"model": source_model, "revision": source_revision},
            "source_files": source_files,
            "role_precision": formats,
            "files": files,
            "tensors": tensor_inventory,
            "weight_format": "minifield.mixed.v1",
        }
        manifest["bundle_digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        (pending / MANIFEST).write_bytes(json_bytes(manifest))
        validate_prepared_bundle(pending)
    return manifest
