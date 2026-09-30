"""Validated lossless asset packaging and explicitly requested quantization."""

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import Lfm2Config, is_quantized_role, validate_config
from .container import Tensor, TensorFile, float_values, write_tensors
from .errors import ConversionError
from .jsonio import file_record, json_bytes, read_json
from .paths import (
    ASSETS,
    MANIFEST,
    SCHEMA,
    reject_symlinks,
    source_asset,
    source_directory,
    staged_destination,
)
from .quantization import ALGORITHMS, FORMATS, Scheme, dequantize, quantize
from .tokenizer import validate_tokenizer

PRODUCER = f"minifield-converters {__version__}"


def _format(artifact: TensorFile) -> str:
    marker = artifact.metadata.get("format")
    if marker is None or marker == "pt":
        return "dense"
    if marker in FORMATS.values():
        for key, value in (
            ("group_size", "128"),
            ("code_order", "little-endian-byte-sequential"),
        ):
            if artifact.metadata.get(key) != value:
                raise ConversionError(f"invalid packed metadata: {key}")
        for key in ("quantizer", "source_model", "producer"):
            if not artifact.metadata.get(key):
                raise ConversionError(f"packed metadata is missing: {key}")
        if marker == FORMATS["ternary"] and not artifact.metadata.get(
            "source_revision"
        ):
            raise ConversionError("ternary metadata is missing source_revision")
        return marker
    raise ConversionError(f"unsupported weights format marker: {marker}")


def validate_weights(
    artifact: TensorFile, config: Lfm2Config, classes: int | None = None
) -> list[dict[str, Any]]:
    weight_format = _format(artifact)
    scheme = next(
        (key for key, value in FORMATS.items() if value == weight_format), None
    )
    inventory = config.inventory(classes)
    expected = {}
    for name, shape in inventory.items():
        if scheme is not None and is_quantized_role(name, shape):
            if shape[1] % 128:
                raise ConversionError(f"packed columns must divide group128: {name}")
            width = {"ternary": 4, "nf4": 2, "int8": 1}[scheme]
            expected[name + ".codes"] = ("U8", (shape[0], shape[1] // width))
            expected[name + ".scales"] = ("F16", (shape[0], shape[1] // 128))
        else:
            expected[name] = (config.dtype, shape)
    if set(artifact.entries) != set(expected):
        missing = sorted(set(expected) - set(artifact.entries))
        extra = sorted(set(artifact.entries) - set(expected))
        raise ConversionError(
            f"weight inventory mismatch; missing={missing}, extra={extra}"
        )
    records = []
    for name, (dtype, shape) in sorted(expected.items()):
        entry = artifact.entries[name]
        if entry["dtype"] != dtype or tuple(entry["shape"]) != shape:
            raise ConversionError(f"wrong weight shape/dtype: {name}")
        artifact.validate_finite(name)
        records.append({"name": name, "dtype": dtype, "shape": list(shape)})
    if scheme is not None:
        for name, shape in inventory.items():
            if is_quantized_role(name, shape):
                codes, scales = (
                    artifact.tensor(name + ".codes"),
                    artifact.tensor(name + ".scales"),
                )
                # This validates reserved codes, widths, finite and nonnegative scales.
                dequantize(
                    np.frombuffer(codes.data, dtype=np.uint8).reshape(codes.shape),
                    np.frombuffer(scales.data, dtype="<f2").reshape(scales.shape),
                    scheme,
                )
    return records


def _tokenizer_source(source: Path) -> tuple[Path, str]:
    candidates = [
        name
        for name in ("tokenizer.json", "tokenizer/tokenizer.json")
        if (source / name).exists() or (source / name).is_symlink()
    ]
    if len(candidates) != 1:
        raise ConversionError(
            "provide exactly one tokenizer.json at root or tokenizer/"
        )
    return source_asset(source, candidates[0]), candidates[0]


def convert_bundle(
    source: Path,
    output: Path,
    *,
    scheme: Scheme | None = None,
    source_model: str | None = None,
    source_revision: str | None = None,
    overwrite: bool = False,
    classes: int | None = None,
) -> dict[str, Any]:
    """Package dense canonical LFM2 inputs, optionally using an explicit quantizer."""
    source = source_directory(source)
    config_path = source_asset(source, "config.json")
    weights_path = source_asset(source, "model.safetensors")
    tokenizer_path, tokenizer_relative = _tokenizer_source(source)
    config = validate_config(read_json(config_path, 1 << 20))
    validate_tokenizer(read_json(tokenizer_path), config.vocab)
    artifact = TensorFile(weights_path)
    if _format(artifact) != "dense":
        raise ConversionError(
            "conversion requires canonical dense inputs; requantization is unsupported"
        )
    source_inventory = validate_weights(artifact, config, classes)
    if scheme is not None:
        if scheme not in ("ternary", "nf4"):
            raise ConversionError("quantization scheme must be ternary or nf4")
        if not source_model or not source_revision:
            raise ConversionError(
                "quantization requires source_model and source_revision"
            )
    inputs = [
        (config_path, "config.json"),
        (weights_path, "model.safetensors"),
        (tokenizer_path, tokenizer_relative),
    ]
    source_files = [file_record(path, relative) for path, relative in inputs]
    with staged_destination(output, source, overwrite) as pending:
        (pending / "tokenizer").mkdir()
        shutil.copyfile(config_path, pending / "config.json")
        shutil.copyfile(tokenizer_path, pending / "tokenizer/tokenizer.json")
        if scheme is None:
            shutil.copyfile(weights_path, pending / "model.safetensors")
        else:
            tensors = {}
            for name, shape in sorted(config.inventory(classes).items()):
                tensor = artifact.tensor(name)
                if not is_quantized_role(name, shape):
                    tensors[name] = tensor
                else:
                    codes, scales = quantize(float_values(tensor), scheme)
                    tensors[name + ".codes"] = Tensor.from_array(codes)
                    tensors[name + ".scales"] = Tensor.from_array(scales)
            metadata = dict(artifact.metadata)
            metadata.update(
                {
                    "format": FORMATS[scheme],
                    "group_size": "128",
                    "code_order": "little-endian-byte-sequential",
                    "quantizer": ALGORITHMS[scheme],
                    "source_model": source_model,
                    "source_revision": source_revision,
                    "producer": PRODUCER,
                }
            )
            write_tensors(pending / "model.safetensors", tensors, metadata)
        if source_files != [file_record(path, relative) for path, relative in inputs]:
            raise ConversionError("source assets changed during conversion")
        output_files = [
            file_record(pending / relative, relative) for relative in ASSETS
        ]
        # Copied assets retain exact source bytes, including JSON formatting.
        if (
            output_files[0]["sha256"] != source_files[0]["sha256"]
            or output_files[2]["sha256"] != source_files[2]["sha256"]
        ):
            raise ConversionError("copied source assets changed during conversion")
        if scheme is None and output_files[1]["sha256"] != source_files[1]["sha256"]:
            raise ConversionError("lossless weights copy changed bytes")
        output_artifact = TensorFile(pending / "model.safetensors")
        output_inventory = validate_weights(output_artifact, config, classes)
        manifest = {
            "schema": SCHEMA,
            "producer": PRODUCER,
            "operation": {
                "kind": "quantize" if scheme is not None else "convert",
                "algorithm": ALGORITHMS[scheme] if scheme else "canonical-lfm2-copy/1",
                "lossless": scheme is None,
            },
            "weight_format": _format(output_artifact),
            "model": {
                "mode": "classifier" if classes is not None else "lm",
                "classes": classes,
            },
            "source_identity": {"model": source_model, "revision": source_revision},
            "source_files": source_files,
            "files": output_files,
            "source_tensors": source_inventory,
            "tensors": output_inventory,
        }
        manifest["bundle_digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        (pending / MANIFEST).write_bytes(json_bytes(manifest))
        validate_bundle(pending)
    return manifest


def validate_bundle(directory: Path) -> dict[str, Any]:
    directory = source_directory(directory)
    manifest_path = directory / MANIFEST
    reject_symlinks(manifest_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ConversionError("unsupported converter manifest")
    actual = [
        file_record(source_asset(directory, relative), relative) for relative in ASSETS
    ]
    if manifest.get("files") != actual:
        raise ConversionError("bundle asset hashes/lengths do not match manifest")
    unsigned = dict(manifest)
    expected_digest = unsigned.pop("bundle_digest", None)
    if expected_digest != hashlib.sha256(json_bytes(unsigned)).hexdigest():
        raise ConversionError("bundle provenance digest does not match manifest")
    config = validate_config(read_json(directory / "config.json", 1 << 20))
    validate_tokenizer(read_json(directory / "tokenizer/tokenizer.json"), config.vocab)
    artifact = TensorFile(directory / "model.safetensors")
    model = manifest.get("model")
    if not isinstance(model, dict) or set(model) != {"mode", "classes"}:
        raise ConversionError("manifest must declare model mode and classes")
    classes = model["classes"]
    expected_mode = "classifier" if classes is not None else "lm"
    if model["mode"] != expected_mode:
        raise ConversionError("manifest model mode and classes disagree")
    if manifest.get("weight_format") != _format(artifact):
        raise ConversionError("manifest weight format differs from container")
    if manifest.get("tensors") != validate_weights(artifact, config, classes):
        raise ConversionError("manifest tensor inventory differs from container")
    return manifest
