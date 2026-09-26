import struct

import numpy as np
import pytest
from safetensors import safe_open

from minifield_converters.cli import main
from minifield_converters.container import Tensor, TensorFile, write_tensors
from minifield_converters.errors import ConversionError
from minifield_converters.jsonio import json_bytes


def test_deterministic_writer_roundtrips_through_independent_safetensors_reader(
    tmp_path,
):
    first, second = tmp_path / "first", tmp_path / "second"
    arrays = {
        "b": np.arange(8, dtype=np.float32).reshape(2, 4),
        "a": np.array([0.5, 1.0], dtype=np.float16),
    }
    tensors = {name: Tensor.from_array(value) for name, value in arrays.items()}
    write_tensors(first, tensors, {"producer": "test", "format": "pt"})
    write_tensors(
        second,
        dict(reversed(list(tensors.items()))),
        {"format": "pt", "producer": "test"},
    )
    assert first.read_bytes() == second.read_bytes()
    with safe_open(first, framework="numpy") as artifact:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(artifact.get_tensor(name), expected)


@pytest.mark.parametrize(
    "header,payload",
    [
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, bytes(8)),
        ({"a": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}, bytes(4)),
        ({"a": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]}}, bytes(8)),
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, bytes(8)),
        ({"__metadata__": {"format": 3}}, b""),
    ],
)
def test_invalid_container_headers_are_rejected(tmp_path, header, payload):
    path = tmp_path / "bad.safetensors"
    data = json_bytes(header)
    path.write_bytes(struct.pack("<Q", len(data)) + data + payload)
    with pytest.raises(ConversionError):
        TensorFile(path)


def test_cli_help_fixture_convert_quantize_and_validate(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_code:
        main(["--help"])
    assert exit_code.value.code == 0
    assert "quantize" in capsys.readouterr().out
    source, dense, packed = tmp_path / "source", tmp_path / "dense", tmp_path / "packed"
    assert main(["fixture", str(source)]) == 0
    assert main(["convert", str(source), str(dense)]) == 0
    assert (
        main(
            [
                "quantize",
                str(source),
                str(packed),
                "--scheme",
                "nf4",
                "--source-model",
                "synthetic/lfm2",
                "--source-revision",
                "v1",
            ]
        )
        == 0
    )
    assert main(["validate", str(packed)]) == 0
    assert main(["convert", str(source), str(dense)]) == 2
    assert "overwrite" in capsys.readouterr().err
