"""Contained source assets and staged, explicit bundle replacement."""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .errors import ConversionError
from .jsonio import read_json

MANIFEST = "conversion-manifest.json"
SCHEMA = "minifield.converter-bundle/1"
ASSETS = ("config.json", "model.safetensors", "tokenizer/tokenizer.json")


def reject_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ConversionError(f"symlink paths are unsupported: {component}")


def source_directory(path: Path) -> Path:
    path = path.absolute()
    reject_symlinks(path)
    if not path.is_dir():
        raise ConversionError(f"source must be a directory: {path}")
    return path.resolve()


def source_asset(directory: Path, relative: str) -> Path:
    if relative not in ASSETS and relative != "tokenizer.json":
        raise ConversionError("unsupported asset path")
    path = directory / relative
    reject_symlinks(path)
    if not path.is_file():
        raise ConversionError(f"required regular asset is missing: {relative}")
    return path


def _managed_destination(path: Path) -> None:
    if not path.is_dir() or not (path / MANIFEST).is_file():
        raise ConversionError("overwrite requires an existing converter-managed bundle")
    reject_symlinks(path / MANIFEST)
    manifest = read_json(path / MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ConversionError("overwrite requires a recognized converter manifest")
    allowed = {*ASSETS, MANIFEST, "tokenizer"}
    for item in path.rglob("*"):
        reject_symlinks(item)
        if item.relative_to(path).as_posix() not in allowed:
            raise ConversionError("overwrite refuses unrelated files in a bundle")


@contextmanager
def staged_destination(output: Path, source: Path | None, overwrite: bool):
    output = output.absolute()
    reject_symlinks(output)
    output = output.resolve()
    if source is not None and (
        output == source or output in source.parents or source in output.parents
    ):
        raise ConversionError("source and output directories must not overlap")
    if output.exists():
        if not overwrite:
            raise ConversionError("destination exists; pass --overwrite explicitly")
        _managed_destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent)
    )
    backup = None
    try:
        yield staging
        if output.exists():
            if not overwrite:
                raise ConversionError("destination appeared during conversion")
            _managed_destination(output)
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent)
            )
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except OSError:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
