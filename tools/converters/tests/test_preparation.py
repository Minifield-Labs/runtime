"""Golden representation bytes and explicit full-state extraction boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from minifield_converters.container import TensorFile, float_values
from minifield_converters.errors import ConversionError
from minifield_converters.fixture import write_fixture
from minifield_converters.jsonio import digest, json_bytes
from minifield_converters.preparation import (
    CheckpointParameters,
    package_mixed_bundle,
    prepare_checkpoint,
    validate_prepared_bundle,
)
from minifield_converters.quantization import (
    dequantize,
    pack_codes,
    quantize,
    unpack_codes,
)


def checkpoint(tmp_path: Path, *, pointer: bool = False):
    source = tmp_path / "source"
    write_fixture(source, classes=None if pointer else 8)
    artifact = TensorFile(source / "model.safetensors")
    arrays = {}
    for name in artifact.entries:
        raw = name.replace("model.", "lfm2.", 1) if pointer else name
        arrays["params/" + raw] = float_values(artifact.tensor(name))
    config = json.loads((source / "config.json").read_text())
    if pointer:
        config["architectures"] = ["Lfm2BidirectionalForMaskedLM"]
        config["use_cache"] = False
        for end in ("start", "end"):
            for role in ("query", "key"):
                arrays[f"params/magicbox.pointer.{end}_{role}"] = np.full(
                    (16, 128), 0.1234, dtype=np.float32
                )
    # Distinct sentinel moments must never become inference weights.
    for name, value in list(arrays.items()):
        arrays["m/" + name.removeprefix("params/")] = np.full_like(value, 42)
        arrays["v/" + name.removeprefix("params/")] = np.full_like(value, 84)
    arrays["step"] = np.array(5, dtype=np.int32)
    state = tmp_path / "checkpoint"
    state.mkdir()
    path = state / "state.safetensors"
    save_file(arrays, str(path))
    (state / "manifest.json").write_bytes(
        json_bytes(
            {
                "format": "minifield.full-training-state/1",
                "tensor_sha256": digest(path),
                "inventory_sha256": "a" * 64,
                "optimizer_id": "b" * 64,
                "cursor": {
                    "next_batch": 5,
                    "run_id": "synthetic-v1",
                    "data_sha256": "c" * 64,
                    "source_id": "d" * 64,
                },
            }
        )
    )
    (source / "config.json").write_bytes(json_bytes(config))
    return path, source


def prepare(tmp_path, *, precision="ternary", pointer=False):
    state, source = checkpoint(tmp_path, pointer=pointer)
    output = tmp_path / "prepared"
    manifest = prepare_checkpoint(
        state,
        source / "config.json",
        source / "tokenizer/tokenizer.json",
        output,
        precision=precision,
        mode="pointer-encoder" if pointer else "classifier",
        source_model="synthetic/lfm2",
        source_revision="fixture-v1",
        classes=None if pointer else 8,
        pointer_width=16 if pointer else None,
        expected_tokenizer_sha256=digest(source / "tokenizer/tokenizer.json"),
    )
    return state, source, output, manifest


def test_int8_hand_calculated_codes_scales_and_half_away_ties():
    values = np.zeros((2, 128), dtype=np.float32)
    values[0, :7] = [127, -127, 0, 0.5, -0.5, 1.5, -1.5]
    codes, scales = quantize(values, "int8")
    assert codes[0, :7].tobytes() == bytes([127, 129, 0, 1, 255, 2, 254])
    assert scales.astype("<f2").tobytes() == bytes.fromhex("003c 0000")
    np.testing.assert_array_equal(
        dequantize(codes, scales, "int8")[0, :7], [127, -127, 0, 1, -1, 2, -2]
    )
    np.testing.assert_array_equal(
        unpack_codes(pack_codes(codes, "int8"), "int8"), codes
    )
    np.testing.assert_array_equal(dequantize(codes, scales, "int8")[1], 0)


def test_int8_stored_scale_and_underflow_are_semantic():
    values = np.zeros((2, 128), dtype=np.float32)
    values[0, 0] = np.float32(127.05)
    values[1] = np.float32(1e-10)
    codes, scales = quantize(values, "int8")
    assert scales[0, 0] == np.float16(1)
    assert dequantize(codes, scales, "int8")[0, 0] == np.float32(127)
    assert scales[1, 0] == 0
    np.testing.assert_array_equal(codes[1], 0)
    with pytest.raises(ConversionError, match="reserved"):
        unpack_codes(np.full((1, 128), 128, dtype=np.uint8), "int8")


@pytest.mark.parametrize("precision", ["fp16", "int8", "nf4", "ternary"])
def test_checkpoint_export_preserves_protected_roles_and_excludes_optimizer(
    tmp_path, precision, monkeypatch
):
    original = CheckpointParameters.tensor
    reads = []

    def guarded(self, name):
        assert not name.startswith(("m/", "v/", "params/")) and name != "step"
        reads.append(name)
        return original(self, name)

    monkeypatch.setattr(CheckpointParameters, "tensor", guarded)
    state, source, output, manifest = prepare(tmp_path, precision=precision)
    assert reads and len(reads) == len(manifest["source_tensors"])
    assert manifest == validate_prepared_bundle(output)
    artifact = TensorFile(output / "model.safetensors")
    source_artifact = TensorFile(source / "model.safetensors")
    for name in (
        "model.embed_tokens.weight",
        "classification_head.weight",
        "model.embedding_norm.weight",
        "model.layers.0.conv.conv.weight",
    ):
        assert artifact.entries[name]["dtype"] == "F16"
        np.testing.assert_array_equal(
            float_values(artifact.tensor(name)),
            float_values(source_artifact.tensor(name))
            .astype(np.float16)
            .astype(np.float32),
        )
    assert all(
        not name.startswith(("params/", "m/", "v/")) for name in artifact.entries
    )
    assert (state.parent / "manifest.json").is_file()
    assert json.loads((output / "config.json").read_text())["dtype"] == "float16"
    role = "model.layers.0.feed_forward.w1.weight"
    assert manifest["role_precision"][role] == (
        "f16" if precision == "fp16" else precision + "-v1"
    )
    if precision == "int8":
        assert artifact.entries[role + ".codes"]["shape"] == [128, 128]


def test_pointer_normalization_and_dense_output_heads(tmp_path):
    _, _, output, manifest = prepare(tmp_path, pointer=True, precision="nf4")
    artifact = TensorFile(output / "model.safetensors")
    for end in ("start", "end"):
        for role in ("query", "key"):
            name = f"pointer.{end}_{role}.weight"
            assert artifact.entries[name]["dtype"] == "F16"
            assert artifact.entries[name]["shape"] == [16, 128]
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Lfm2BidirectionalForMaskedLM"]
    assert config["minifield_pointer"]["projection_dim"] == 16
    assert config["minifield_runtime_model"] == "lfm2-pointer-encoder/1"
    assert any(
        record["source_name"].startswith("params/lfm2.")
        for record in manifest["source_tensors"]
    )


def test_package_qat_preserves_every_weight_byte_and_rejects_wrong_assets(tmp_path):
    _, _, prepared, _ = prepare(tmp_path, precision="ternary")
    output = tmp_path / "qat-copy"
    manifest = package_mixed_bundle(
        prepared / "model.safetensors",
        prepared / "config.json",
        prepared / "tokenizer/tokenizer.json",
        output,
        classes=8,
        source_model="synthetic/qat",
        source_revision="qat-fixture-v1",
    )
    assert manifest["operation"]["lossless"]
    assert (output / "model.safetensors").read_bytes() == (
        prepared / "model.safetensors"
    ).read_bytes()
    assert manifest == validate_prepared_bundle(output)
    with pytest.raises(ConversionError, match="shape/dtype"):
        package_mixed_bundle(
            prepared / "model.safetensors",
            prepared / "config.json",
            prepared / "tokenizer/tokenizer.json",
            tmp_path / "wrong",
            classes=7,
            source_model="synthetic/qat",
            source_revision="qat-fixture-v1",
        )
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize(
    "failure", ["nonfinite", "overflow", "extra", "hash", "tokenizer"]
)
def test_checkpoint_rejects_invalid_source_and_leaves_no_output(tmp_path, failure):
    state, source = checkpoint(tmp_path)
    if failure in ("nonfinite", "overflow", "extra"):
        from safetensors.numpy import load_file

        arrays = load_file(str(state))  # Small synthetic fixture only.
        if failure == "extra":
            arrays["params/extra.weight"] = np.zeros((1, 128), dtype=np.float32)
        else:
            arrays["params/classification_head.weight"][0, 0] = (
                np.nan if failure == "nonfinite" else 70000
            )
        save_file(arrays, str(state))
        metadata = json.loads((state.parent / "manifest.json").read_text())
        metadata["tensor_sha256"] = digest(state)
        (state.parent / "manifest.json").write_bytes(json_bytes(metadata))
    if failure == "hash":
        metadata = json.loads((state.parent / "manifest.json").read_text())
        metadata["tensor_sha256"] = "0" * 64
        (state.parent / "manifest.json").write_bytes(json_bytes(metadata))
    output = tmp_path / "invalid"
    with pytest.raises(ConversionError):
        prepare_checkpoint(
            state,
            source / "config.json",
            source / "tokenizer/tokenizer.json",
            output,
            precision="ternary",
            mode="classifier",
            classes=8,
            source_model="synthetic/lfm2",
            source_revision="fixture-v1",
            expected_tokenizer_sha256="0" * 64 if failure == "tokenizer" else None,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid.pending-*"))


def test_cli_preparation_and_validation_use_explicit_new_contract(tmp_path, capsys):
    from minifield_converters.cli import main

    state, source = checkpoint(tmp_path)
    output = tmp_path / "cli-prepared"
    assert (
        main(
            [
                "prepare-checkpoint",
                str(state),
                str(output),
                "--config",
                str(source / "config.json"),
                "--tokenizer",
                str(source / "tokenizer/tokenizer.json"),
                "--precision",
                "fp16",
                "--mode",
                "classifier",
                "--classes",
                "8",
                "--source-model",
                "synthetic/classifier",
                "--source-revision",
                "fixture-v1",
            ]
        )
        == 0
    )
    assert main(["validate", str(output)]) == 0
    assert '"weight_format": "pt"' in capsys.readouterr().out


def test_mixed_packaging_rejects_low_bit_protected_embedding(tmp_path):
    from minifield_converters.container import write_tensors

    _, _, prepared, _ = prepare(tmp_path)
    artifact = TensorFile(prepared / "model.safetensors")
    metadata = dict(artifact.metadata)
    formats = json.loads(metadata["tensor_quantization"])
    formats["model.embed_tokens.weight"] = "ternary-v1"
    metadata["tensor_quantization"] = json.dumps(formats)
    modified = tmp_path / "bad-protected.safetensors"
    write_tensors(
        modified, {name: artifact.tensor(name) for name in artifact.entries}, metadata
    )
    with pytest.raises(ConversionError, match="protected role"):
        package_mixed_bundle(
            modified,
            prepared / "config.json",
            prepared / "tokenizer/tokenizer.json",
            tmp_path / "refused",
            classes=8,
            source_model="synthetic/qat",
            source_revision="fixture-v1",
        )
    assert not (tmp_path / "refused").exists()
