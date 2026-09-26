"""Strict, deterministic JSON and streaming hashes."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .errors import ConversionError


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ConversionError(f"non-finite JSON constant: {value}")


def parse_json(data: bytes) -> Any:
    try:
        result = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
        pending = [(result, 0)]
        while pending:
            value, depth = pending.pop()
            if depth > 128:
                raise ConversionError("JSON nesting exceeds depth 128")
            if isinstance(value, str):
                value.encode("utf-8")
            elif isinstance(value, float) and not math.isfinite(value):
                raise ConversionError("JSON number is non-finite")
            elif isinstance(value, dict):
                for key, child in value.items():
                    key.encode("utf-8")
                    pending.append((child, depth + 1))
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ConversionError(f"invalid JSON: {error}") from error


def read_json(path: Path, max_bytes: int = 8 << 20) -> Any:
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ConversionError(f"JSON asset exceeds {max_bytes} bytes: {path.name}")
    return parse_json(data)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def file_record(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
