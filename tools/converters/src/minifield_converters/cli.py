"""Offline commands with explicit structural conversion and quantization."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .bundle import convert_bundle, validate_bundle
from .errors import ConversionError
from .fixture import write_fixture
from .jsonio import read_json
from .paths import MANIFEST
from .preparation import SCHEMA as PREPARED_SCHEMA
from .preparation import (
    package_mixed_bundle,
    prepare_checkpoint,
    validate_prepared_bundle,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Offline canonical LFM2 asset converter"
    )
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("convert", "losslessly package dense canonical LFM2 assets"),
        ("quantize", "explicitly quantize LFM2 backbone matmul roles"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("source", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument(
            "--classes", type=int, help="explicit classifier head width; omitted for LM"
        )
        if name == "quantize":
            command.add_argument("--scheme", choices=("ternary", "nf4"), required=True)
            command.add_argument("--source-model", required=True)
            command.add_argument("--source-revision", required=True)
    command = commands.add_parser(
        "validate", help="check bundle hashes and supported assets"
    )
    command.add_argument("bundle", type=Path)
    command = commands.add_parser(
        "fixture", help="generate a tiny synthetic LFM2 source"
    )
    command.add_argument("output", type=Path)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument(
        "--classes", type=int, help="generate an independent dense classifier head"
    )
    command = commands.add_parser(
        "prepare-checkpoint", help="extract FP32 masters with protected-role precision"
    )
    command.add_argument("checkpoint", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument(
        "--precision", choices=("fp16", "int8", "nf4", "ternary"), required=True
    )
    command.add_argument(
        "--mode", choices=("classifier", "pointer-encoder"), required=True
    )
    command.add_argument("--classes", type=int)
    command.add_argument("--pointer-width", type=int)
    command.add_argument("--source-model", required=True)
    command.add_argument("--source-revision", required=True)
    command.add_argument("--expected-tokenizer-sha256")
    command = commands.add_parser(
        "package-mixed-qat", help="copy a mixed QAT artifact byte-for-byte"
    )
    command.add_argument("weights", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--classes", type=int, required=True)
    command.add_argument("--source-model", required=True)
    command.add_argument("--source-revision", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "fixture":
            write_fixture(
                arguments.output,
                overwrite=arguments.overwrite,
                classes=arguments.classes,
            )
            print("synthetic source fixture written")
            return 0
        if arguments.command == "validate":
            raw = read_json(arguments.bundle / MANIFEST)
            validate = (
                validate_prepared_bundle
                if isinstance(raw, dict) and raw.get("schema") == PREPARED_SCHEMA
                else validate_bundle
            )
            manifest = validate(arguments.bundle)
        elif arguments.command == "prepare-checkpoint":
            manifest = prepare_checkpoint(
                arguments.checkpoint,
                arguments.config,
                arguments.tokenizer,
                arguments.output,
                precision=arguments.precision,
                mode=arguments.mode,
                classes=arguments.classes,
                pointer_width=arguments.pointer_width,
                source_model=arguments.source_model,
                source_revision=arguments.source_revision,
                expected_tokenizer_sha256=arguments.expected_tokenizer_sha256,
            )
        elif arguments.command == "package-mixed-qat":
            manifest = package_mixed_bundle(
                arguments.weights,
                arguments.config,
                arguments.tokenizer,
                arguments.output,
                classes=arguments.classes,
                source_model=arguments.source_model,
                source_revision=arguments.source_revision,
            )
        else:
            manifest = convert_bundle(
                arguments.source,
                arguments.output,
                overwrite=arguments.overwrite,
                scheme=getattr(arguments, "scheme", None),
                source_model=getattr(arguments, "source_model", None),
                source_revision=getattr(arguments, "source_revision", None),
                classes=arguments.classes,
            )
        print(
            json.dumps(
                {
                    "bundle_digest": manifest["bundle_digest"],
                    "weight_format": manifest["weight_format"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (ConversionError, OSError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
