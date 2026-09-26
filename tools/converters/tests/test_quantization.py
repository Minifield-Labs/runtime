import numpy as np
import pytest

from minifield_converters.errors import ConversionError
from minifield_converters.quantization import (
    NF4,
    dequantize,
    pack_codes,
    quantize,
    unpack_codes,
)


def test_ternary_hand_calculated_bytes_scales_and_half_ties():
    weights = np.zeros((2, 128), dtype=np.float32)
    weights[0] = np.tile([2, -2, 0, 1], 32)
    codes, scales = quantize(weights, "ternary")
    assert codes.tobytes() == bytes([0x92]) * 32 + bytes([0x55]) * 32
    assert scales.astype("<f2").tobytes() == bytes.fromhex("0040 0000")
    np.testing.assert_array_equal(
        dequantize(codes, scales, "ternary")[0, :4], [2, -2, 0, 2]
    )
    np.testing.assert_array_equal(dequantize(codes, scales, "ternary")[1], 0)


def test_nf4_hand_calculated_nibble_order_and_codebook_roundtrip():
    raw = np.tile(np.arange(16, dtype=np.uint8), (2, 8))
    packed = pack_codes(raw, "nf4")
    assert packed.tobytes() == bytes.fromhex("10 32 54 76 98 ba dc fe") * 16
    np.testing.assert_array_equal(unpack_codes(packed, "nf4"), raw)
    scales = np.array([[1], [2]], dtype=np.float16)
    decoded = dequantize(packed, scales, "nf4")
    np.testing.assert_array_equal(decoded[0], np.tile(NF4, 8))
    np.testing.assert_array_equal(decoded[1], np.tile(NF4 * 2, 8))
    codes, stored = quantize(decoded[:1], "nf4")
    np.testing.assert_array_equal(codes, packed[:1])
    np.testing.assert_array_equal(stored, scales[:1])


@pytest.mark.parametrize("scheme,zero", [("ternary", 0x55), ("nf4", 0x77)])
def test_zero_and_fp16_underflow_groups(scheme, zero):
    values = np.zeros((2, 128), dtype=np.float32)
    values[1] = np.float32(1e-10)
    codes, scales = quantize(values, scheme)
    assert codes.tobytes() == bytes([zero]) * codes.size
    np.testing.assert_array_equal(scales, 0)
    np.testing.assert_array_equal(dequantize(codes, scales, scheme), 0)


def test_nf4_ties_choose_lower_index():
    values = np.zeros((1, 128), dtype=np.float32)
    values[0, 0] = (NF4[7] + NF4[8]) / np.float32(2)
    values[0, 1] = 1
    codes, _ = quantize(values, "nf4")
    assert codes[0, 0] == 0xF7


@pytest.mark.parametrize("scheme", ["ternary", "nf4"])
def test_stored_fp16_scale_is_semantic_value(scheme):
    values = np.zeros((1, 128), dtype=np.float32)
    values[0, 0] = 1.0004
    codes, scales = quantize(values, scheme)
    assert scales[0, 0] == np.float16(1)
    assert dequantize(codes, scales, scheme)[0, 0] == np.float32(1)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((1, 127)),
        np.zeros((128,)),
        np.zeros((0, 128)),
        np.full((1, 128), np.nan),
        np.full((1, 128), 70000),
        np.full((1, 128), np.inf),
    ],
)
def test_invalid_quantization_inputs(values):
    with pytest.raises(ConversionError):
        quantize(values, "ternary")


def test_invalid_code_and_scale_parts():
    with pytest.raises(ConversionError, match="codes"):
        pack_codes(np.full((1, 128), 3, dtype=np.uint8), "ternary")
    with pytest.raises(ConversionError, match="reserved"):
        unpack_codes(np.full((1, 32), 0xFF, dtype=np.uint8), "ternary")
    packed = np.full((1, 32), 0x55, dtype=np.uint8)
    for scales in (
        np.ones((1, 1), dtype=np.float32),
        np.full((1, 1), -1, dtype=np.float16),
        np.full((1, 1), np.nan, dtype=np.float16),
    ):
        with pytest.raises(ConversionError, match="scales"):
            dequantize(packed, scales, "ternary")
    with pytest.raises(ConversionError, match="scheme"):
        quantize(np.zeros((1, 128)), "unknown")
