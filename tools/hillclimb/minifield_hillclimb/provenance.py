"""Freeze the environment and identify the compiler/config that built a run."""

from __future__ import annotations

import os
import shutil
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .schema import CampaignError, fields, file_digest, string

BUILD_ENV_KEYS = {
    "PATH",
    "HOME",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "SOURCE_DATE_EPOCH",
}
HOST_ENV_KEYS = {"PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
DENIED_EXACT = {
    "PROFILE",
    "OPT_LEVEL",
    "DEBUG",
    "NUM_JOBS",
    "OUT_DIR",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "RUSTFLAGS",
    "RUSTDOCFLAGS",
}
DENIED_PREFIXES = (
    "CARGO_",
    "RUST",
    "DYLD_",
    "WGPU_",
    "MINIFIELD_",
    "CLASSIFY_",
    "GIT_",
)


def ambient_environment() -> dict[str, str]:
    return dict(os.environ)


def validate_environment(value: Any) -> dict[str, dict[str, str]]:
    root = fields(value, {"build", "host"})
    result = {}
    for phase, allowed in (("build", BUILD_ENV_KEYS), ("host", HOST_ENV_KEYS)):
        values = root[phase]
        if not isinstance(values, dict) or set(values) - allowed:
            raise CampaignError(f"{phase} environment contains unreviewable overrides")
        if not {"PATH", "HOME"} <= values.keys():
            raise CampaignError(
                f"{phase} environment must explicitly pin PATH and HOME"
            )
        for name, value in values.items():
            string(value, f"environment {name}")
            if "\0" in value:
                raise CampaignError("environment values cannot contain NUL")
            if (
                name in {"HOME", "CARGO_HOME", "RUSTUP_HOME"}
                and not Path(value).is_absolute()
            ):
                raise CampaignError(f"environment {name} must be an absolute path")
        result[phase] = {**values, "MINIFIELD_TELEMETRY": "0"}
    return result


def verify_environment(
    frozen: dict[str, dict[str, str]], ambient: Mapping[str, str]
) -> None:
    permitted = BUILD_ENV_KEYS | HOST_ENV_KEYS | {"MINIFIELD_TELEMETRY"}
    for name in ambient:
        if name not in permitted and (
            name in DENIED_EXACT or name.startswith(DENIED_PREFIXES)
        ):
            raise CampaignError(f"unreviewed ambient build/host override: {name}")
    for phase, values in frozen.items():
        for name, value in values.items():
            if name == "MINIFIELD_TELEMETRY":
                if value != "0":
                    raise CampaignError(
                        "telemetry must stay disabled during measurement"
                    )
            elif ambient.get(name) != value:
                raise CampaignError(f"{phase} environment changed: {name}")
        for name in BUILD_ENV_KEYS if phase == "build" else HOST_ENV_KEYS:
            if name in ambient and name not in values:
                raise CampaignError(
                    f"{phase} environment omitted ambient setting: {name}"
                )


def cargo_configs(source: Path, environment: dict[str, str]) -> dict[str, str | None]:
    """Include missing paths, so creating a higher-priority config changes identity."""
    home = Path(environment["HOME"])
    cargo_home = Path(environment.get("CARGO_HOME", str(home / ".cargo")))
    directories = {
        cargo_home,
        *(parent / ".cargo" for parent in (source, *source.parents)),
    }
    result = {}
    for directory in sorted(directories):
        legacy = directory / "config"
        selected = legacy if legacy.exists() else directory / "config.toml"
        for name in ("config", "config.toml"):
            path = directory / name
            if path == selected and path.exists():
                admit_cargo_config(path)
            result[str(path)] = file_digest(path) if path.exists() else None
    return result


def admit_cargo_config(path: Path) -> None:
    """The selected compiler must remain the independently probed toolchain."""
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CampaignError(f"cannot admit Cargo config {path}: {error}") from error
    if "include" in config:
        raise CampaignError(f"unsupported Cargo config include: {path}")
    build = config.get("build", {})
    if not isinstance(build, dict):
        raise CampaignError(f"Cargo config [build] must be a table: {path}")
    for selector in ("rustc", "rustc-wrapper", "rustc-workspace-wrapper"):
        if selector in build:
            raise CampaignError(
                f"unsupported Cargo compiler selector build.{selector}: {path}"
            )
    env = config.get("env", {})
    if not isinstance(env, dict):
        raise CampaignError(f"Cargo config [env] must be a table: {path}")
    for name in env:
        if name in DENIED_EXACT | {"HOME", "PATH"} or name.startswith(
            ("CARGO_", "RUST")
        ):
            raise CampaignError(
                f"unsupported Cargo [env] compiler override {name}: {path}"
            )


def resolve_tool(command: str, environment: dict[str, str]) -> str:
    found = shutil.which(command, path=environment["PATH"])
    if found is None:
        raise CampaignError(f"required tool missing from frozen PATH: {command}")
    # Preserve the rustup shim's filename. Resolving the cargo symlink to rustup
    # would change argv[0] and therefore the program's behavior.
    return str(Path(found).absolute())


def tool_identity(
    build: dict[str, Any],
    source: Path,
    environment: dict[str, str],
    command: Callable[..., Any],
) -> dict[str, Any]:
    cargo = resolve_tool(build["cargo"], environment)
    rustup = resolve_tool("rustup", environment)

    def checked(argv: list[str]) -> str:
        result = command(argv, source, 30, environment)
        if result.returncode:
            raise CampaignError(f"toolchain probe failed: {argv[0]}")
        return result.stdout.strip()

    paths = {
        "cargo_entry": cargo,
        "rustup_entry": rustup,
        "cargo_selected": checked(
            [rustup, "which", "--toolchain", build["rust_toolchain"], "cargo"]
        ),
        "rustc_selected": checked(
            [rustup, "which", "--toolchain", build["rust_toolchain"], "rustc"]
        ),
    }
    result = {
        name: {"path": path, "sha256": file_digest(Path(path))}
        for name, path in paths.items()
    }
    result["cargo_version"] = checked(
        [cargo, f"+{build['rust_toolchain']}", "--version"]
    )
    result["rustc_vv"] = checked([paths["rustc_selected"], "-vV"])
    release = next(
        (
            line.split(": ", 1)[1]
            for line in result["rustc_vv"].splitlines()
            if line.startswith("release: ")
        ),
        None,
    )
    if release != build["rust_toolchain"]:
        raise CampaignError(
            "actual compiler release differs from frozen Rust toolchain"
        )
    return result
