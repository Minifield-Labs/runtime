"""Build, validate, and measure independently identified runtime candidates."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import campaign_status, repeated_improvement, validate_screen
from .ledger import reserve_attempt, validate_budget
from .provenance import (
    ambient_environment,
    cargo_configs,
    tool_identity,
    validate_environment,
    verify_environment,
)
from .schema import (
    BACKENDS,
    CampaignError,
    artifact_hashes,
    canonical,
    digest,
    fields,
    file_digest,
    loads,
    read_json,
    resolve_cell,
    string,
    validate_criteria,
    validate_reference,
)
from .scoring import build_argv, paired_statistics, validate_response


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value) + b"\n")


def harness_digest() -> str:
    root = Path(__file__).parent
    return digest({p.name: file_digest(p) for p in sorted(root.glob("*.py"))})


def host_source_digest(root: Path) -> str:
    if not root.is_dir():
        raise CampaignError(f"native evaluation host source is missing: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CampaignError("native evaluation host source cannot use symlinks")
        if path.is_file() and not set(path.relative_to(root).parts) & {
            "target",
            ".git",
            "__pycache__",
        }:
            files[str(path.relative_to(root))] = file_digest(path)
    if not files:
        raise CampaignError("native evaluation host source is empty")
    return digest(files)


def freeze(
    manifest_path: Path,
    output: Path,
    environment_provider: Callable[[], dict[str, str]] = ambient_environment,
) -> dict[str, Any]:
    manifest = fields(
        read_json(manifest_path),
        {
            "schema_version",
            "name",
            "build",
            "criteria",
            "cells",
            "environment",
            "attempt_budget",
        },
    )
    if manifest["schema_version"] != 1:
        raise CampaignError("unsupported campaign schema")
    string(manifest["name"], "name")
    build = fields(
        manifest["build"],
        {
            "cargo",
            "rust_toolchain",
            "package",
            "binary",
            "features_by_backend",
            "host_source",
            "host_relative_path",
        },
    )
    for key in ("cargo", "rust_toolchain", "package", "binary"):
        string(build[key], key)
    if build["rust_toolchain"] != "1.89.0":
        raise CampaignError("campaigns pin the repository's Rust 1.89.0 toolchain")
    build = dict(build)
    host_relative = Path(string(build["host_relative_path"], "host_relative_path"))
    if host_relative.is_absolute() or ".." in host_relative.parts:
        raise CampaignError("host_relative_path must stay within each source checkout")
    build["host_source"] = str(
        (manifest_path.parent / string(build["host_source"], "host_source")).resolve()
    )
    features = build["features_by_backend"]
    if not isinstance(features, dict) or set(features) != BACKENDS:
        raise CampaignError("build needs explicit features for all backend identities")
    for values in features.values():
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise CampaignError("features must be distinct arrays")
        for value in values:
            string(value, "feature")
    criteria = validate_criteria(manifest["criteria"])
    environment = validate_environment(manifest["environment"])
    verify_environment(environment, environment_provider())
    if not isinstance(manifest["cells"], list) or not manifest["cells"]:
        raise CampaignError("campaign needs evaluation cells")
    cells = [resolve_cell(c, manifest_path.parent) for c in manifest["cells"]]
    if len({c["id"] for c in cells}) != len(cells):
        raise CampaignError("duplicate cell ID")
    for cell in cells:
        validate_reference(cell)
        cell["artifacts"] = artifact_hashes(cell)
        cell["reference_sha256"] = file_digest(Path(cell["reference"]))
    lock = {
        "schema_version": 1,
        "name": manifest["name"],
        "build": build,
        "criteria": criteria,
        "criteria_sha256": digest(criteria),
        "environment": environment,
        "attempt_budget": validate_budget(
            manifest["attempt_budget"], manifest_path.parent
        ),
        "harness_sha256": harness_digest(),
        "native_host_sha256": host_source_digest(Path(build["host_source"])),
        "cells": cells,
    }
    lock["campaign_sha256"] = digest(lock)
    # A freeze is deliberate and never overwrites existing campaign evidence.
    with output.open("xb") as stream:
        stream.write(canonical(lock) + b"\n")
    return lock


def validate_lock(path: Path, expected_digest: str) -> dict[str, Any]:
    lock = fields(
        read_json(path),
        {
            "schema_version",
            "name",
            "build",
            "criteria",
            "criteria_sha256",
            "environment",
            "attempt_budget",
            "harness_sha256",
            "native_host_sha256",
            "cells",
            "campaign_sha256",
        },
    )
    unhashed = {k: v for k, v in lock.items() if k != "campaign_sha256"}
    if digest(unhashed) != lock["campaign_sha256"]:
        raise CampaignError("campaign digest mismatch")
    if lock["campaign_sha256"] != expected_digest:
        raise CampaignError("campaign differs from externally pinned digest")
    if lock["schema_version"] != 1:
        raise CampaignError("unsupported lock schema")
    validate_criteria(lock["criteria"])
    if digest(lock["criteria"]) != lock["criteria_sha256"]:
        raise CampaignError("criteria digest mismatch")
    verify_frozen_inputs(lock)
    return lock


def verify_frozen_inputs(lock: dict[str, Any]) -> None:
    if harness_digest() != lock["harness_sha256"]:
        raise CampaignError("evaluation harness changed after campaign freeze")
    if (
        host_source_digest(Path(lock["build"]["host_source"]))
        != lock["native_host_sha256"]
    ):
        raise CampaignError("native evaluation host changed after campaign freeze")
    for cell in lock["cells"]:
        if artifact_hashes(cell) != cell["artifacts"]:
            raise CampaignError(f"frozen artifacts changed: {cell['id']}")
        if file_digest(Path(cell["reference"])) != cell["reference_sha256"]:
            raise CampaignError(f"frozen oracle changed: {cell['id']}")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def execute(
    argv: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> CommandResult:
    """Use exact arguments and kill the entire build/host process group on timeout."""
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise ProcessDeadline(argv, stdout, stderr) from error
    return CommandResult(process.returncode, stdout, stderr)


class ProcessDeadline(CampaignError):
    def __init__(self, argv: list[str], stdout: str, stderr: str):
        super().__init__(f"process deadline exceeded: {argv[0]}")
        self.stdout = stdout
        self.stderr = stderr


def source_identity(source: Path) -> dict[str, Any]:
    listing = execute(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        source,
        30,
    )
    if listing.returncode:
        raise CampaignError(f"source is not a readable Git checkout: {source}")
    names = set(listing.stdout.split("\0")) - {""}
    excluded = {".git", "target", ".venv", "__pycache__", "node_modules"}
    source_suffixes = {
        ".rs",
        ".wgsl",
        ".metal",
        ".py",
        ".toml",
        ".lock",
        ".json",
        ".js",
        ".mjs",
        ".ts",
        ".h",
        ".c",
        ".cpp",
    }
    # Git-ignored source can still enter a build through include!/module/config
    # paths. Include it while excluding only generated dependency/build trees.
    for directory, child_dirs, child_files in os.walk(source):
        for name in child_dirs:
            if name not in excluded and (Path(directory) / name).is_symlink():
                raise CampaignError(
                    "symlinked source directories need explicit vendoring"
                )
        child_dirs[:] = [name for name in child_dirs if name not in excluded]
        for filename in child_files:
            path = Path(directory) / filename
            if path.suffix in source_suffixes:
                names.add(str(path.relative_to(source)))
    files = {}
    for name in sorted(names):
        path = source / name
        if path.is_file():
            files[name] = file_digest(path)
            if path.is_symlink():
                files[f"symlink:{name}"] = os.readlink(path)
        elif not path.exists():
            files[name] = "deleted"
        else:
            raise CampaignError(f"unsupported source entry: {name}")
    revision = execute(["git", "rev-parse", "HEAD"], source, 30)
    dirty = execute(["git", "status", "--porcelain"], source, 30)
    if revision.returncode or dirty.returncode:
        raise CampaignError("cannot identify source checkout")
    return {
        "path": str(source),
        "revision": revision.stdout.strip(),
        "dirty": bool(dirty.stdout),
        "source_sha256": digest(files),
        "source_files": len(files),
    }


class Controller:
    """Dependencies are injectable for decision tests, never via campaign files."""

    def __init__(
        self,
        command: Callable[..., CommandResult] = execute,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        identify: Callable[[Path], dict[str, Any]] = source_identity,
        environment_provider: Callable[[], dict[str, str]] = ambient_environment,
        tool_probe: Callable[..., dict[str, Any]] | None = None,
    ):
        self.command = command
        self.clock = clock
        self.sleep = sleep
        self.identify = identify
        self.environment_provider = environment_provider
        self.tool_probe = tool_probe or (
            lambda build, source, env: tool_identity(build, source, env, execute)
        )
        self.execution_context: dict[str, Any] = {}
        self.screen_pin: tuple[Path, str] | None = None
        self.last_measurement_end: float | None = None

    def _command(
        self, argv: list[str], cwd: Path, deadline: float, log: Path, phase: str
    ) -> CommandResult:
        try:
            result = self.command(
                argv, cwd, deadline, self.execution_context["environment"][phase]
            )
        except ProcessDeadline as error:
            log.with_suffix(".stdout").write_text(error.stdout)
            log.with_suffix(".stderr").write_text(error.stderr)
            raise
        log.with_suffix(".stdout").write_text(result.stdout)
        log.with_suffix(".stderr").write_text(result.stderr)
        if result.returncode:
            raise CampaignError(f"process failed ({result.returncode}): {argv[0]}")
        return result

    def _cooldown(self, seconds: float) -> None:
        if self.last_measurement_end is None:
            return
        deadline = self.last_measurement_end + seconds
        while (remaining := deadline - self.clock()) > 0:
            self.sleep(remaining)

    def run(
        self,
        lock_path: Path,
        expected_digest: str,
        champion: Path,
        candidate: Path,
        output: Path,
        confirmation_of: Path | None = None,
        expected_screen_digest: str | None = None,
    ) -> dict[str, Any]:
        lock = validate_lock(lock_path, expected_digest)
        confirmation_of = confirmation_of.resolve() if confirmation_of else None
        champion, candidate, output = (
            champion.resolve(),
            candidate.resolve(),
            output.resolve(),
        )
        if output.is_relative_to(champion) or output.is_relative_to(candidate):
            raise CampaignError("run evidence must be outside both source checkouts")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "campaign.lock.json", lock)
        result: dict[str, Any] = {
            "schema_version": 1,
            "campaign_sha256": expected_digest,
            "criteria_sha256": lock["criteria_sha256"],
            "status": "failed",
            "builds": {},
            "cells": [],
            "errors": [],
        }
        sources = {"champion": champion, "candidate": candidate}
        self.screen_pin = None
        try:
            identities = {role: self.identify(path) for role, path in sources.items()}
            result["sources"] = identities
            self._verify_hosts(lock, sources)
            self.execution_context = self._context(lock, sources)
            result["execution_context"] = self.execution_context
            ledger_path = Path(lock["attempt_budget"]["ledger"])
            if any(ledger_path.is_relative_to(source) for source in sources.values()):
                raise CampaignError(
                    "attempt ledger must stay outside measured checkouts"
                )
            if confirmation_of is not None:
                if not expected_screen_digest:
                    raise CampaignError(
                        "confirmation requires externally pinned screen SHA-256"
                    )
                if file_digest(confirmation_of) != expected_screen_digest:
                    raise CampaignError("screen differs from externally pinned digest")
                self.screen_pin = confirmation_of, expected_screen_digest
                screen = validate_screen(
                    read_json(confirmation_of),
                    lock,
                    identities,
                    self.execution_context,
                    confirmation_of.parent,
                )
                result["confirmation_of_sha256"] = expected_screen_digest
            elif expected_screen_digest is not None:
                raise CampaignError("screen digest requires --confirmation-of")
            result["attempt_sha256"] = reserve_attempt(
                lock,
                "finalist" if confirmation_of else "candidate",
                output,
                identities,
                expected_screen_digest,
            )
            builds = self._build(lock, sources, output, result)
            if confirmation_of is not None:
                for key, record in result["builds"].items():
                    if (
                        record["executable_sha256"]
                        != screen["builds"][key]["executable_sha256"]
                    ):
                        raise CampaignError(
                            "confirmation executable differs from finalist"
                        )
            for cell in lock["cells"]:
                evidence: dict[str, Any] = {
                    "id": cell["id"],
                    "model": cell["model"],
                    "precision": cell["precision"],
                    "arithmetic": cell["arithmetic"],
                    "backend": cell["backend"],
                    "mode": cell["mode"],
                    "correctness": [],
                    "samples": [],
                    "status": "failed",
                }
                result["cells"].append(evidence)
                screen_device = (
                    None
                    if confirmation_of is None
                    else next(
                        c["device"] for c in screen["cells"] if c["id"] == cell["id"]
                    )
                )
                self._cell(
                    lock,
                    cell,
                    builds,
                    sources,
                    identities,
                    output,
                    evidence,
                    screen_device,
                )
                evidence["statistics"] = paired_statistics(
                    evidence["samples"], lock["criteria"]
                )
                evidence["status"] = evidence["statistics"]["status"]
            verify_frozen_inputs(lock)
            self._verify_hosts(lock, sources)
            for role, path in sources.items():
                if self.identify(path) != identities[role]:
                    raise CampaignError(f"{role} source changed during evaluation")
            self._verify_builds(builds)
            self._verify_context(lock, sources, reprobe_tools=True)
            result["status"] = campaign_status(
                [cell["status"] for cell in result["cells"]]
            )
            if (
                confirmation_of is not None
                and result["status"] == "eligible_for_confirmation"
            ):
                result["confirmed_cell_ids"] = repeated_improvement(screen, result)
                result["status"] = (
                    "promoted"
                    if result["confirmed_cell_ids"]
                    else "confirmation_failed"
                )
        except (CampaignError, OSError, KeyError, TypeError) as error:
            result["errors"].append(str(error))
        finally:
            write_json(output / "result.json", result)
        return result

    def _context(
        self, lock: dict[str, Any], sources: dict[str, Path]
    ) -> dict[str, Any]:
        verify_environment(lock["environment"], self.environment_provider())
        tools = {
            role: self.tool_probe(lock["build"], source, lock["environment"]["build"])
            for role, source in sources.items()
        }
        if tools["champion"] != tools["candidate"]:
            raise CampaignError("matched builds selected different toolchains")
        configs = {
            role: cargo_configs(source, lock["environment"]["build"])
            for role, source in sources.items()
        }
        active = {}
        for role, values in configs.items():
            identities = {}
            for filename, sha in values.items():
                if sha is not None:
                    path = Path(filename)
                    key = (
                        "checkout:" + str(path.relative_to(sources[role]))
                        if path.is_relative_to(sources[role])
                        else "external:" + str(path)
                    )
                    identities[key] = sha
            active[role] = identities
        if active["champion"] != active["candidate"]:
            raise CampaignError("matched builds have different active Cargo configs")
        return {
            "environment": lock["environment"],
            "tools": tools,
            "cargo_configs": configs,
        }

    def _verify_context(
        self,
        lock: dict[str, Any],
        sources: dict[str, Path],
        reprobe_tools: bool = False,
    ) -> None:
        verify_environment(lock["environment"], self.environment_provider())
        for role, source in sources.items():
            actual_configs = cargo_configs(source, lock["environment"]["build"])
            if actual_configs != self.execution_context["cargo_configs"][role]:
                raise CampaignError("Cargo configuration changed during evaluation")
            if (
                reprobe_tools
                and self.tool_probe(lock["build"], source, lock["environment"]["build"])
                != self.execution_context["tools"][role]
            ):
                raise CampaignError("compiler/tool identity changed during evaluation")
        if self.screen_pin and file_digest(self.screen_pin[0]) != self.screen_pin[1]:
            raise CampaignError("pinned screening evidence changed during confirmation")

    @staticmethod
    def _verify_builds(builds: dict[tuple[str, str], dict[str, Any]]) -> None:
        for build in builds.values():
            if file_digest(Path(build["executable"])) != build["executable_sha256"]:
                raise CampaignError("built executable changed during evaluation")

    @staticmethod
    def _verify_hosts(lock: dict[str, Any], sources: dict[str, Path]) -> None:
        for role, source in sources.items():
            actual = host_source_digest(source / lock["build"]["host_relative_path"])
            if actual != lock["native_host_sha256"]:
                raise CampaignError(f"{role} native host differs from frozen harness")

    def _build(
        self,
        lock: dict[str, Any],
        sources: dict[str, Path],
        output: Path,
        result: dict[str, Any],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        builds = {}
        for role, source in sources.items():
            for backend in sorted({cell["backend"] for cell in lock["cells"]}):
                target = output / "targets" / role / backend
                argv = build_argv(lock["build"], backend, target)
                argv[0] = self.execution_context["tools"][role]["cargo_entry"]["path"]
                record: dict[str, Any] = {"argv": argv, "target": str(target)}
                result["builds"][f"{role}/{backend}"] = record
                self._verify_context(lock, sources, reprobe_tools=True)
                self._command(
                    argv,
                    source,
                    lock["criteria"]["build_deadline_seconds"],
                    output / f"build-{role}-{backend}",
                    "build",
                )
                self._verify_context(lock, sources, reprobe_tools=True)
                binary = target / "release" / lock["build"]["binary"]
                record["executable"] = str(binary)
                record["executable_sha256"] = file_digest(binary)
                builds[role, backend] = record
        return builds

    def _cell(
        self,
        lock: dict[str, Any],
        cell: dict[str, Any],
        builds: dict[tuple[str, str], dict[str, Any]],
        sources: dict[str, Path],
        identities: dict[str, Any],
        output: Path,
        evidence: dict[str, Any],
        screen_device: dict[str, Any] | None = None,
    ) -> None:
        reference = validate_reference(cell)
        device = None
        sequence = [("correctness", -1, role) for role in sources]
        for block in range(lock["criteria"]["independent_runs"] // 2):
            sequence.extend(
                ("measure", block, role)
                for role in ("champion", "candidate", "candidate", "champion")
            )
        cell_dir = output / "cells" / cell["id"]
        cell_dir.mkdir(parents=True)
        for index, (phase, block, role) in enumerate(sequence):
            verify_frozen_inputs(lock)
            self._verify_hosts(lock, sources)
            self._verify_context(lock, sources)
            if self.identify(sources[role]) != identities[role]:
                raise CampaignError(f"{role} source changed during evaluation")
            build = builds[role, cell["backend"]]
            self._verify_builds(builds)
            cycles = 1 if phase == "correctness" else cell["measured_cycles"]
            request = {
                "schema_version": 1,
                **{
                    key: cell[key]
                    for key in (
                        "backend",
                        "task",
                        "bundle",
                        "inputs",
                        "tokenizer",
                        "arithmetic",
                        "context",
                        "mode",
                        "deadline_seconds",
                        "lut2_mode",
                        "max_lut2_bytes",
                    )
                },
                "phase": phase,
                "warmups": 0 if phase == "correctness" else cell["warmups"],
                "measured_cycles": cycles,
            }
            if "classes" in cell:
                request["classes"] = cell["classes"]
            request_path = cell_dir / f"{index:03}-{phase}-{role}.request.json"
            write_json(request_path, request)
            if phase == "measure":
                self._cooldown(lock["criteria"]["cooldown_seconds"])
            self._verify_context(lock, sources)
            self._verify_builds(builds)
            verify_frozen_inputs(lock)
            started = self.clock()
            try:
                host = self._command(
                    [build["executable"], "--request", str(request_path)],
                    sources[role],
                    cell["deadline_seconds"],
                    request_path,
                    "host",
                )
            finally:
                ended = self.clock()
                if phase == "measure":
                    self.last_measurement_end = ended
            response = validate_response(
                loads(host.stdout), cell, reference, lock["criteria"], cycles
            )
            case_seconds = sum(case["elapsed_seconds"] for case in response["cases"])
            if case_seconds > (ended - started) * (1 + 1e-6):
                raise CampaignError("reported measurement exceeds process wall time")
            if device is None:
                device = response["backend"]
                evidence["device"] = device
                if screen_device is not None and device != screen_device:
                    raise CampaignError(
                        "confirmation device differs from pinned screen"
                    )
            elif response["backend"] != device:
                raise CampaignError(
                    "backend/device identity changed between matched runs"
                )
            verify_frozen_inputs(lock)
            self._verify_hosts(lock, sources)
            self._verify_builds(builds)
            self._verify_context(lock, sources)
            for checked_role, source in sources.items():
                if self.identify(source) != identities[checked_role]:
                    raise CampaignError(
                        f"{checked_role} source changed during evaluation"
                    )
            entry = {
                "role": role,
                "block": block,
                "response": response,
                "process_wall_seconds": ended - started,
                "started_monotonic": started,
                "ended_monotonic": ended,
            }
            evidence["correctness" if phase == "correctness" else "samples"].append(
                entry
            )
