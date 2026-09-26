import json

import numpy as np
import pytest

from minifield_converters.bundle import convert_bundle, validate_bundle
from minifield_converters.container import (
    Tensor,
    TensorFile,
    float_values,
    write_tensors,
)
from minifield_converters.errors import ConversionError
from minifield_converters.jsonio import digest, json_bytes, read_json
from minifield_converters.paths import ASSETS, MANIFEST
from minifield_converters.quantization import dequantize


def test_structural_conversion_is_byte_lossless_and_deterministic(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = convert_bundle(source, first)
    assert manifest == convert_bundle(source, second)
    assert manifest["operation"]["lossless"] is True
    assert manifest["weight_format"] == "dense"
    assert (first / MANIFEST).read_bytes() == (second / MANIFEST).read_bytes()
    for relative in ASSETS:
        assert (first / relative).read_bytes() == (source / relative).read_bytes()
    assert validate_bundle(first) == manifest
    assert TensorFile(first / "model.safetensors").metadata == {"format": "pt"}


@pytest.mark.parametrize("scheme", ["ternary", "nf4"])
def test_explicit_quantization_packs_only_matmul_roles(source, tmp_path, scheme):
    first, second = tmp_path / "first", tmp_path / "second"
    options = {
        "scheme": scheme,
        "source_model": "synthetic/lfm2",
        "source_revision": "fixture-v1",
    }
    manifest = convert_bundle(source, first, **options)
    assert manifest == convert_bundle(source, second, **options)
    assert manifest["operation"]["lossless"] is False
    assert manifest["source_files"][1]["sha256"] != manifest["files"][1]["sha256"]
    original, packed = (
        TensorFile(source / "model.safetensors"),
        TensorFile(first / "model.safetensors"),
    )
    for name, entry in original.entries.items():
        source_tensor = original.tensor(name)
        if len(entry["shape"]) != 2:
            assert packed.tensor(name) == source_tensor
        else:
            codes, scales = (
                packed.tensor(name + ".codes"),
                packed.tensor(name + ".scales"),
            )
            decoded = dequantize(
                np.frombuffer(codes.data, dtype=np.uint8).reshape(codes.shape),
                np.frombuffer(scales.data, dtype="<f2").reshape(scales.shape),
                scheme,
            )
            np.testing.assert_array_equal(decoded, float_values(source_tensor))
    validate_bundle(first)


def test_bf16_dense_bytes_survive_structural_and_quantized_outputs(source, tmp_path):
    artifact = TensorFile(source / "model.safetensors")
    tensors = {}
    for name in artifact.entries:
        original = artifact.tensor(name)
        bits = float_values(original).view(np.uint32)
        tensors[name] = Tensor(
            "BF16", original.shape, (bits >> 16).astype("<u2").tobytes()
        )
    (source / "model.safetensors").unlink()
    write_tensors(source / "model.safetensors", tensors, {"format": "pt"})
    config = read_json(source / "config.json")
    config["dtype"] = "bfloat16"
    (source / "config.json").write_bytes(json_bytes(config))
    dense, packed = tmp_path / "dense", tmp_path / "packed"
    convert_bundle(source, dense)
    assert digest(source / "model.safetensors") == digest(dense / "model.safetensors")
    convert_bundle(
        source, packed, scheme="ternary", source_model="synthetic", source_revision="v1"
    )
    result = TensorFile(packed / "model.safetensors")
    for name, tensor in tensors.items():
        if len(tensor.shape) != 2:
            assert result.tensor(name) == tensor


def test_existing_bundle_requires_explicit_managed_overwrite(source, tmp_path):
    output = tmp_path / "output"
    original = convert_bundle(source, output)
    with pytest.raises(ConversionError, match="overwrite"):
        convert_bundle(source, output)
    assert convert_bundle(source, output, overwrite=True) == original
    (output / "personal-note.txt").write_text("keep this")
    with pytest.raises(ConversionError, match="unrelated"):
        convert_bundle(source, output, overwrite=True)
    assert (output / "personal-note.txt").read_text() == "keep this"


def test_failed_quantization_leaves_existing_destination_unchanged(source, tmp_path):
    output = tmp_path / "output"
    convert_bundle(source, output)
    before = (output / MANIFEST).read_bytes()
    artifact = TensorFile(source / "model.safetensors")
    tensors = {name: artifact.tensor(name) for name in artifact.entries}
    name = "model.embed_tokens.weight"
    values = float_values(tensors[name])
    values[0, 0] = 70000
    tensors[name] = Tensor.from_array(values)
    (source / "model.safetensors").unlink()
    write_tensors(source / "model.safetensors", tensors, {"format": "pt"})
    with pytest.raises(ConversionError, match="overflows FP16"):
        convert_bundle(
            source,
            output,
            scheme="ternary",
            source_model="synthetic",
            source_revision="v1",
            overwrite=True,
        )
    assert (output / MANIFEST).read_bytes() == before
    assert not list(tmp_path.glob(".output.pending-*"))


def test_paths_symlinks_and_unmanaged_overwrite_are_refused(source, tmp_path):
    with pytest.raises(ConversionError, match="overlap"):
        convert_bundle(source, source / "output")
    output = tmp_path / "unmanaged"
    output.mkdir()
    (output / "keep").write_text("user file")
    with pytest.raises(ConversionError, match="managed"):
        convert_bundle(source, output, overwrite=True)
    assert (output / "keep").read_text() == "user file"
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(ConversionError, match="symlink"):
        convert_bundle(linked, tmp_path / "output")
    token = source / "tokenizer/tokenizer.json"
    copy = tmp_path / "external-tokenizer.json"
    token.rename(copy)
    token.symlink_to(copy)
    with pytest.raises(ConversionError, match="symlink"):
        convert_bundle(source, tmp_path / "output")


def test_manifest_hashes_and_paths_are_checked_before_asset_use(source, tmp_path):
    output = tmp_path / "output"
    convert_bundle(source, output)
    manifest = read_json(output / MANIFEST)
    manifest["files"][0]["path"] = "../../external.json"
    (output / MANIFEST).write_bytes(json_bytes(manifest))
    with pytest.raises(ConversionError, match="hashes"):
        validate_bundle(output)
    convert_bundle(source, output, overwrite=True)
    (output / "config.json").write_text("{}")
    with pytest.raises(ConversionError, match="hashes"):
        validate_bundle(output)


@pytest.mark.parametrize(
    "marker", ["minifield.ternary.v2", "minifield.polyomino-weights/1", "mystery"]
)
def test_unknown_container_formats_are_refused(source, tmp_path, marker):
    artifact = TensorFile(source / "model.safetensors")
    tensors = {name: artifact.tensor(name) for name in artifact.entries}
    (source / "model.safetensors").unlink()
    write_tensors(source / "model.safetensors", tensors, {"format": marker})
    with pytest.raises(ConversionError, match="format marker"):
        convert_bundle(source, tmp_path / "output")


def test_ambiguous_tokenizer_sources_and_requantization_are_refused(source, tmp_path):
    (source / "tokenizer.json").write_bytes(
        (source / "tokenizer/tokenizer.json").read_bytes()
    )
    with pytest.raises(ConversionError, match="exactly one"):
        convert_bundle(source, tmp_path / "output")
    (source / "tokenizer.json").unlink()
    packed = tmp_path / "packed"
    convert_bundle(
        source, packed, scheme="nf4", source_model="synthetic", source_revision="v1"
    )
    with pytest.raises(ConversionError, match="requantization"):
        convert_bundle(packed, tmp_path / "again")


def test_source_profile_inventory_and_nonfinite_weights_are_refused(source, tmp_path):
    artifact = TensorFile(source / "model.safetensors")
    tensors = {name: artifact.tensor(name) for name in artifact.entries}
    name = "model.embedding_norm.weight"
    values = float_values(tensors[name])
    values[0] = np.nan
    tensors[name] = Tensor.from_array(values)
    (source / "model.safetensors").unlink()
    write_tensors(source / "model.safetensors", tensors, {})
    with pytest.raises(ConversionError, match="non-finite"):
        convert_bundle(source, tmp_path / "output")
    tensors.pop(name)
    (source / "model.safetensors").unlink()
    write_tensors(source / "model.safetensors", tensors, {})
    with pytest.raises(ConversionError, match="inventory"):
        convert_bundle(source, tmp_path / "output")


def test_root_tokenizer_input_is_normalized_to_canonical_directory(source, tmp_path):
    (source / "tokenizer/tokenizer.json").rename(source / "tokenizer.json")
    output = tmp_path / "output"
    manifest = convert_bundle(source, output)
    assert manifest["source_files"][2]["path"] == "tokenizer.json"
    assert manifest["files"][2]["path"] == "tokenizer/tokenizer.json"
    assert (
        json.loads((output / "tokenizer/tokenizer.json").read_bytes())["version"]
        == "1.0"
    )
