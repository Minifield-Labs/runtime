"""Versioned offline quantizers and lossless split-stream encode/decode."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from .errors import ConversionError

Scheme = Literal["ternary", "nf4", "int8"]
GROUP_SIZE = 128
FORMATS = {
    "ternary": "minifield.ternary.v1",
    "nf4": "minifield.nf4.v1",
    "int8": "minifield.int8.v1",
}
ALGORITHMS = {
    "ternary": "minifield-converters/ternary-absmax-stored-fp16/1",
    "nf4": "minifield-converters/nf4-absmax-stored-fp16/1",
    "int8": "minifield-converters/int8-absmax-stored-fp16/1",
}
NF4 = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)
NF4.flags.writeable = False


def _scheme(scheme: str) -> int:
    if scheme not in FORMATS:
        raise ConversionError("quantization scheme must be ternary, nf4, or int8")
    return {"ternary": 2, "nf4": 4, "int8": 8}[scheme]


def pack_codes(codes: npt.NDArray[np.uint8], scheme: Scheme) -> npt.NDArray[np.uint8]:
    """Pack row-major codes, first weight in the least-significant bits."""
    bits = _scheme(scheme)
    width = 8 // bits
    maximum = {"ternary": 2, "nf4": 15, "int8": 255}[scheme]
    if (
        codes.dtype != np.uint8
        or codes.ndim != 2
        or 0 in codes.shape
        or codes.shape[1] % GROUP_SIZE
        or np.any(codes > maximum)
    ):
        raise ConversionError("codes must be non-empty valid U8 rows with K%128=0")
    if scheme == "int8" and np.any(codes == 128):
        raise ConversionError("INT8 code -128 is reserved")
    grouped = codes.reshape(codes.shape[0], -1, width)
    packed = np.zeros(grouped.shape[:2], dtype=np.uint8)
    for index in range(width):
        packed |= grouped[:, :, index] << (index * bits)
    return packed


def unpack_codes(
    packed: npt.NDArray[np.uint8], scheme: Scheme
) -> npt.NDArray[np.uint8]:
    bits = _scheme(scheme)
    width = 8 // bits
    if (
        packed.dtype != np.uint8
        or packed.ndim != 2
        or 0 in packed.shape
        or (packed.shape[1] * width) % GROUP_SIZE
    ):
        raise ConversionError("packed codes must be non-empty U8 rows with K%128=0")
    codes = np.empty((packed.shape[0], packed.shape[1] * width), dtype=np.uint8)
    for index in range(width):
        codes[:, index::width] = (packed >> (index * bits)) & ((1 << bits) - 1)
    if scheme == "ternary" and np.any(codes == 3):
        raise ConversionError("ternary code 3 is reserved")
    if scheme == "int8" and np.any(codes == 128):
        raise ConversionError("INT8 code -128 is reserved")
    return codes


def dequantize(
    packed: npt.NDArray[np.uint8], scales: npt.NDArray[np.float16], scheme: Scheme
) -> npt.NDArray[np.float32]:
    codes = unpack_codes(packed, scheme)
    if (
        scales.dtype != np.dtype("float16")
        or scales.shape != (codes.shape[0], codes.shape[1] // GROUP_SIZE)
        or not np.all(np.isfinite(scales) & (scales >= 0))
    ):
        raise ConversionError("scales must be finite nonnegative F16 row/group values")
    if scheme == "ternary":
        levels = codes.astype(np.float32) - 1
    elif scheme == "int8":
        levels = codes.view(np.int8).astype(np.float32)
    else:
        levels = NF4[codes]
    return levels * np.repeat(scales.astype(np.float32), GROUP_SIZE, axis=1)


def quantize(
    weights: npt.ArrayLike, scheme: Scheme
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float16]]:
    """Use stored-FP16 absmax scales and explicitly versioned code selection.

    Ternary quotients use FP64 divide then FP32 rounding, matching the training
    exporter. NF4 uses FP32 distances and ties to the lower codebook index.
    Groups whose scale rounds to zero encode only the zero level.
    """
    _scheme(scheme)
    source = np.asarray(weights)
    if source.dtype.kind not in "fiu" or source.ndim != 2 or 0 in source.shape:
        raise ConversionError("weights must be a non-empty real numeric matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        values = source.astype(np.float32)
    if not np.isfinite(values).all() or values.shape[1] % GROUP_SIZE:
        raise ConversionError("weights must be finite FP32 rows with K%128=0")
    grouped = values.reshape(values.shape[0], -1, GROUP_SIZE)
    with np.errstate(over="ignore", under="ignore"):
        maximum = np.max(np.abs(grouped), axis=-1)
        scales = (maximum / np.float32(127) if scheme == "int8" else maximum).astype(
            np.float16
        )
    if not np.isfinite(scales).all():
        raise ConversionError("group absmax scale overflows FP16")
    safe = np.where(scales > 0, scales, np.float16(1)).astype(np.float64)
    normalized = (grouped.astype(np.float64) / safe[:, :, None]).astype(np.float32)
    if scheme in ("ternary", "int8"):
        rounded = np.sign(normalized) * np.floor(np.abs(normalized) + np.float32(0.5))
        if scheme == "ternary":
            codes = np.clip(rounded + 1, 0, 2).astype(np.uint8)
            codes[scales == 0] = 1
        else:
            codes = np.clip(rounded, -127, 127).astype(np.int8).view(np.uint8)
            codes[scales == 0] = 0
    else:
        # Keep scratch proportional to one codebook level, rather than 16x model size.
        distances = np.full(grouped.shape, np.inf, dtype=np.float32)
        codes = np.zeros(grouped.shape, dtype=np.uint8)
        for index, level in enumerate(NF4):
            candidate = np.abs(normalized - level)
            better = candidate < distances
            codes[better] = index
            distances[better] = candidate[better]
        codes[scales == 0] = 7
    return pack_codes(codes.reshape(values.shape), scheme), scales
