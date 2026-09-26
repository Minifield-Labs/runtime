import numpy as np
import pytest

from minifield_converters.bundle import convert_bundle, validate_bundle
from minifield_converters.cli import main
from minifield_converters.container import TensorFile
from minifield_converters.errors import ConversionError
from minifield_converters.fixture import write_fixture


@pytest.mark.parametrize("scheme", [None, "ternary", "nf4"])
def test_explicit_classifier_mode_preserves_dense_head(tmp_path, scheme):
    source, output = tmp_path / "source", tmp_path / "output"
    write_fixture(source, classes=8)
    options = {"scheme": scheme, "source_model": "synthetic", "source_revision": "v1"}
    manifest = convert_bundle(source, output, classes=8, **options)
    assert manifest["model"] == {"mode": "classifier", "classes": 8}
    assert validate_bundle(output) == manifest
    name = "classification_head.weight"
    original = TensorFile(source / "model.safetensors").tensor(name)
    actual = TensorFile(output / "model.safetensors").tensor(name)
    assert actual == original
    assert actual.shape == (8, 128)
    values = np.frombuffer(actual.data, dtype="<f4")
    assert set(values) == {-0.125, 0.0, 0.125}
    with pytest.raises(ConversionError, match="inventory"):
        convert_bundle(source, tmp_path / "implicit")
    with pytest.raises(ConversionError, match="shape/dtype"):
        convert_bundle(source, tmp_path / "wrong", classes=7)


@pytest.mark.parametrize("classes", [0, -1, 65537, True])
def test_invalid_class_counts_fail_without_output(tmp_path, classes):
    output = tmp_path / "source"
    with pytest.raises(ConversionError, match="classes"):
        write_fixture(output, classes=classes)
    assert not output.exists()


def test_classifier_cli_fixture_conversion_and_validation(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    assert main(["fixture", str(source), "--classes", "8"]) == 0
    assert main(["convert", str(source), str(output), "--classes", "8"]) == 0
    assert main(["validate", str(output)]) == 0
