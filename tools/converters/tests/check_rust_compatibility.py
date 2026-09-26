"""Explicit cross-language check. Unit pytest never imports or invokes Rust."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from minifield_converters.bundle import convert_bundle, validate_bundle
from minifield_converters.config import is_quantized_role
from minifield_converters.container import TensorFile, float_values
from minifield_converters.fixture import write_fixture
from minifield_converters.jsonio import digest
from minifield_converters.paths import ASSETS
from minifield_converters.quantization import dequantize


def probe(repo: Path, bundle: Path, classes: int | None) -> dict:
    command = [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "-p",
        "minifield-executor-core",
        "--example",
        "converter_probe",
        "--",
        str(bundle),
    ]
    if classes is not None:
        command.extend(["--classes", str(classes)])
    result = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, timeout=180, check=False
    )
    if result.returncode:
        raise RuntimeError(f"Rust probe failed ({result.returncode}):\n{result.stderr}")
    return json.loads(result.stdout)


def check_output(output: dict, classes: int | None) -> None:
    assert output["mode"] == ("lm" if classes is None else "classifier")
    assert output["tokens"] == [1, 3]
    assert output["append_tokens"] == [4]
    assert output["base_state"] == {"logical_length": 2, "token_history": [1, 3]}
    if classes is None:
        assert output["append_state"] == {
            "logical_length": 3,
            "token_history": [1, 3, 4],
        }
    else:
        assert output["append_sequence"] == [1, 3, 4]
    for key in ("logits", "append_logits", "fresh_logits"):
        values = np.asarray(output[key])
        assert values.shape == (32 if classes is None else classes,)
        assert np.isfinite(values).all()
        assert np.max(np.abs(values)) > 0.001
        assert np.ptp(values) > 0.001
    np.testing.assert_allclose(
        output["append_logits"], output["fresh_logits"], rtol=2e-5, atol=2e-5
    )


def check_case(repo: Path, directory: Path, classes: int | None) -> None:
    source, dense = directory / "source", directory / "dense"
    write_fixture(source, classes=classes)
    convert_bundle(source, dense, classes=classes)
    # The dense baseline is the exact original source, including producer metadata.
    for relative in ASSETS:
        assert digest(source / relative) == digest(dense / relative)
    baseline = probe(repo, dense, classes)
    check_output(baseline, classes)
    original = TensorFile(source / "model.safetensors")
    for scheme in ("ternary", "nf4"):
        output = directory / scheme
        convert_bundle(
            source,
            output,
            scheme=scheme,
            source_model="synthetic/lfm2",
            source_revision="converter-fixture-v1",
            classes=classes,
        )
        validate_bundle(output)
        packed = TensorFile(output / "model.safetensors")
        for name in original.entries:
            tensor = original.tensor(name)
            if is_quantized_role(name, tensor.shape):
                codes = packed.tensor(name + ".codes")
                scales = packed.tensor(name + ".scales")
                decoded = dequantize(
                    np.frombuffer(codes.data, dtype=np.uint8).reshape(codes.shape),
                    np.frombuffer(scales.data, dtype="<f2").reshape(scales.shape),
                    scheme,
                )
                np.testing.assert_array_equal(decoded, float_values(tensor))
            else:
                assert packed.tensor(name) == tensor
        actual = probe(repo, output, classes)
        check_output(actual, classes)
        for key in ("logits", "append_logits", "fresh_logits"):
            np.testing.assert_allclose(actual[key], baseline[key], rtol=2e-5, atol=2e-5)
    mode = "LM" if classes is None else f"classifier/{classes}"
    print(f"{mode}: dense/ternary/NF4 loader, logits, append, hashes passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    if (
        not (repo / "Cargo.toml").is_file()
        or not (repo / "crates/executor-core/examples/converter_probe.rs").is_file()
    ):
        parser.error(
            "--repo-root must contain this runtime workspace and converter_probe"
        )
    # resolve() handles platform temporary-directory aliases before safe path admission.
    with tempfile.TemporaryDirectory(prefix="minifield-converter-rust-") as temporary:
        root = Path(temporary).resolve(strict=True)
        for name, classes in (("lm", None), ("classifier", 8)):
            case = root / name
            case.mkdir()
            check_case(repo, case, classes)


if __name__ == "__main__":
    main()
