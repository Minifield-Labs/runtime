"""Bounded safetensors admission and deterministic container writing."""

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open

from .errors import ConversionError
from .jsonio import json_bytes, parse_json

WIDTHS = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1}
MAX_HEADER_BYTES = 8 << 20
MAX_WEIGHTS_BYTES = 8 << 30


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def from_array(cls, value: np.ndarray) -> "Tensor":
        dtype = {
            np.dtype("float32"): "F32",
            np.dtype("float16"): "F16",
            np.dtype("uint8"): "U8",
        }.get(value.dtype)
        if dtype is None:
            raise ConversionError(f"unsupported array dtype: {value.dtype}")
        return cls(
            dtype,
            value.shape,
            value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(),
        )


class TensorFile:
    def __init__(self, path: Path):
        self.path = path
        size = path.stat().st_size
        if size > MAX_WEIGHTS_BYTES or size < 8:
            raise ConversionError(
                "weights exceed byte limit or have a truncated prefix"
            )
        with path.open("rb") as stream:
            (header_bytes,) = struct.unpack("<Q", stream.read(8))
            if header_bytes > MAX_HEADER_BYTES or header_bytes > size - 8:
                raise ConversionError(
                    "safetensors header exceeds limit or available bytes"
                )
            header = parse_json(stream.read(header_bytes))
        if not isinstance(header, dict) or len(header) > 10001:
            raise ConversionError("safetensors header must be a bounded object")
        self.data_start = 8 + header_bytes
        self.metadata = header.pop("__metadata__", {})
        if not isinstance(self.metadata, dict) or any(
            type(value) is not str for value in self.metadata.values()
        ):
            raise ConversionError("safetensors metadata values must be strings")
        self.entries = header
        intervals = []
        for name, entry in header.items():
            if not name or len(name.encode()) > 1024 or not isinstance(entry, dict):
                raise ConversionError("invalid tensor name/header")
            if set(entry) != {"dtype", "shape", "data_offsets"}:
                raise ConversionError(f"unexpected tensor header fields: {name}")
            dtype, shape, offsets = (
                entry["dtype"],
                entry["shape"],
                entry["data_offsets"],
            )
            if type(dtype) is not str or dtype not in WIDTHS:
                raise ConversionError(f"unsupported tensor dtype: {name}")
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 3
                or any(type(x) is not int or x <= 0 for x in shape)
            ):
                raise ConversionError(f"invalid tensor dimensions: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)
            ):
                raise ConversionError(f"invalid tensor offsets: {name}")
            start, end = offsets
            if (
                end - start != math.prod(shape) * WIDTHS[dtype]
                or end > size - self.data_start
            ):
                raise ConversionError(f"tensor bytes differ from dimensions: {name}")
            intervals.append((start, end))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ConversionError("safetensors payload has a hole or overlap")
            cursor = end
        if cursor != size - self.data_start:
            raise ConversionError("safetensors payload has trailing bytes")
        try:
            with safe_open(path, framework="numpy") as artifact:
                if set(artifact.keys()) != set(header):
                    raise ConversionError("safetensors parser inventory disagreement")
        except SafetensorError as error:
            raise ConversionError(f"invalid safetensors container: {error}") from error

    def tensor(self, name: str) -> Tensor:
        entry = self.entries[name]
        start, end = entry["data_offsets"]
        with self.path.open("rb") as stream:
            stream.seek(self.data_start + start)
            data = stream.read(end - start)
        if len(data) != end - start:
            raise ConversionError(f"tensor became truncated: {name}")
        return Tensor(entry["dtype"], tuple(entry["shape"]), data)

    def validate_finite(self, name: str) -> None:
        entry = self.entries[name]
        start, end = entry["data_offsets"]
        if entry["dtype"] == "U8":
            return
        with self.path.open("rb") as stream:
            stream.seek(self.data_start + start)
            remaining = end - start
            while remaining:
                data = stream.read(min(remaining, 1 << 20))
                if (
                    not data
                    or not np.isfinite(
                        float_values(Tensor(entry["dtype"], (), data))
                    ).all()
                ):
                    raise ConversionError(f"non-finite or truncated tensor: {name}")
                remaining -= len(data)


def float_values(tensor: Tensor) -> np.ndarray:
    if tensor.dtype == "BF16":
        result = (np.frombuffer(tensor.data, dtype="<u2").astype(np.uint32) << 16).view(
            np.float32
        )
    elif tensor.dtype in ("F32", "F16"):
        result = np.frombuffer(
            tensor.data, dtype="<f4" if tensor.dtype == "F32" else "<f2"
        )
        result = result.astype(np.float32)
    else:
        raise ConversionError("tensor is not floating-point")
    return result.reshape(tensor.shape) if tensor.shape else result


def write_tensors(
    path: Path, tensors: dict[str, Tensor], metadata: dict[str, str]
) -> None:
    """Write sorted names and metadata with exact little-endian payload bytes."""
    header = {"__metadata__": metadata}
    cursor = 0
    for name, tensor in sorted(tensors.items()):
        if (
            tensor.dtype not in WIDTHS
            or len(tensor.data) != math.prod(tensor.shape) * WIDTHS[tensor.dtype]
        ):
            raise ConversionError(f"invalid tensor serialization: {name}")
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + len(tensor.data)],
        }
        cursor += len(tensor.data)
    data = json_bytes(header).rstrip(b"\n")
    data += b" " * (-len(data) % 8)
    if len(data) > MAX_HEADER_BYTES or cursor + len(data) + 8 > MAX_WEIGHTS_BYTES:
        raise ConversionError("output safetensors exceeds container limits")
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        for _, tensor in sorted(tensors.items()):
            stream.write(tensor.data)


def write_tensor_stream(
    path: Path,
    inventory: dict[str, tuple[str, tuple[int, ...]]],
    tensors: Iterable[tuple[str, Tensor]],
    metadata: dict[str, str],
) -> None:
    """Write a declared inventory while retaining only the current tensor pair.

    The caller supplies sorted physical names. Checking them against the header
    catches accidental omissions and keeps offsets independent of generator state.
    """
    header = {"__metadata__": metadata}
    cursor = 0
    names = sorted(inventory)
    for name in names:
        dtype, shape = inventory[name]
        if (
            dtype not in WIDTHS
            or not 1 <= len(shape) <= 3
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ConversionError(f"invalid streamed tensor descriptor: {name}")
        size = math.prod(shape) * WIDTHS[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    data = json_bytes(header).rstrip(b"\n")
    data += b" " * (-len(data) % 8)
    if len(data) > MAX_HEADER_BYTES or cursor + len(data) + 8 > MAX_WEIGHTS_BYTES:
        raise ConversionError("output safetensors exceeds container limits")
    iterator = iter(tensors)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        for expected in names:
            item = next(iterator, None)
            if item is None or item[0] != expected:
                raise ConversionError("streamed tensor inventory differs from header")
            name, tensor = item
            dtype, shape = inventory[name]
            if (
                tensor.dtype != dtype
                or tensor.shape != shape
                or len(tensor.data) != math.prod(shape) * WIDTHS[dtype]
            ):
                raise ConversionError(f"streamed tensor shape/dtype differs: {name}")
            stream.write(tensor.data)
        if next(iterator, None) is not None:
            raise ConversionError("streamed tensor inventory has extra entries")
