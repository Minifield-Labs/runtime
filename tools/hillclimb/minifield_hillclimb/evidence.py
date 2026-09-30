"""Reconstruct screening decisions from complete immutable raw evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import (
    CampaignError,
    canonical,
    fields,
    number,
    positive,
    validate_reference,
)
from .scoring import build_argv, paired_statistics, validate_response


def campaign_status(statuses: list[str]) -> str:
    if "regressed" in statuses:
        return "rejected"
    if "inconclusive" in statuses:
        return "inconclusive"
    if "improved" in statuses:
        return "eligible_for_confirmation"
    return "no_confirmed_improvement"


def repeated_improvement(
    screen: dict[str, Any], confirmation: dict[str, Any]
) -> list[str]:
    screen_winners = {c["id"] for c in screen["cells"] if c["status"] == "improved"}
    return sorted(
        c["id"]
        for c in confirmation["cells"]
        if c["status"] == "improved" and c["id"] in screen_winners
    )


def validate_screen(
    screen: Any,
    lock: dict[str, Any],
    identities: dict[str, Any],
    context: dict[str, Any],
    directory: Path,
) -> dict[str, Any]:
    fields(
        screen,
        {
            "schema_version",
            "campaign_sha256",
            "criteria_sha256",
            "status",
            "builds",
            "cells",
            "errors",
            "sources",
            "execution_context",
            "attempt_sha256",
        },
    )
    if (
        screen["schema_version"] != 1
        or screen["campaign_sha256"] != lock["campaign_sha256"]
        or screen["criteria_sha256"] != lock["criteria_sha256"]
        or screen["sources"] != identities
        or screen["execution_context"] != context
        or screen["errors"] != []
    ):
        raise CampaignError("confirmation does not bind the frozen finalist context")
    backends = {cell["backend"] for cell in lock["cells"]}
    expected_builds = {
        f"{role}/{backend}" for role in identities for backend in backends
    }
    if (
        not isinstance(screen["builds"], dict)
        or set(screen["builds"]) != expected_builds
    ):
        raise CampaignError("screen build evidence is incomplete")
    for key, build in screen["builds"].items():
        fields(build, {"argv", "target", "executable", "executable_sha256"})
        role, backend = key.split("/")
        target = directory / "targets" / role / backend
        expected_argv = build_argv(lock["build"], backend, target)
        expected_argv[0] = context["tools"][role]["cargo_entry"]["path"]
        if (
            build["target"] != str(target)
            or build["executable"] != str(target / "release" / lock["build"]["binary"])
            or build["argv"] != expected_argv
        ):
            raise CampaignError("screen build arguments differ from frozen build")
    if (
        not isinstance(screen["cells"], list)
        or not all(isinstance(c, dict) for c in screen["cells"])
        or [c.get("id") for c in screen["cells"]]
        != [cell["id"] for cell in lock["cells"]]
    ):
        raise CampaignError("screen cell IDs/count/order differ from frozen campaign")
    previous_end = None
    for recorded, cell in zip(screen["cells"], lock["cells"], strict=True):
        fields(
            recorded,
            {
                "id",
                "model",
                "precision",
                "arithmetic",
                "backend",
                "mode",
                "correctness",
                "samples",
                "status",
                "statistics",
                "device",
            },
        )
        for key in ("id", "model", "precision", "arithmetic", "backend", "mode"):
            if recorded[key] != cell[key]:
                raise CampaignError("screen cell metadata differs from frozen campaign")
        if (
            not isinstance(recorded["correctness"], list)
            or len(recorded["correctness"]) != 2
        ):
            raise CampaignError("screen correctness evidence is incomplete")
        expected_sequence = [("champion", -1, 1), ("candidate", -1, 1)]
        for block in range(lock["criteria"]["independent_runs"] // 2):
            expected_sequence.extend(
                (role, block, cell["measured_cycles"])
                for role in ("champion", "candidate", "candidate", "champion")
            )
        if (
            not isinstance(recorded["samples"], list)
            or len(recorded["samples"]) != len(expected_sequence) - 2
        ):
            raise CampaignError("screen measured samples are incomplete")
        reference = validate_reference(cell)
        entries = recorded["correctness"] + recorded["samples"]
        for entry, (role, block, cycles) in zip(
            entries, expected_sequence, strict=True
        ):
            fields(
                entry,
                {
                    "role",
                    "block",
                    "response",
                    "process_wall_seconds",
                    "started_monotonic",
                    "ended_monotonic",
                },
            )
            if entry["role"] != role or entry["block"] != block:
                raise CampaignError(
                    "screen process order differs from fixed ABBA sequence"
                )
            response = dict(entry["response"])
            saved_rate = response.pop("sustained_predictions_per_second", None)
            checked = validate_response(
                response, cell, reference, lock["criteria"], cycles
            )
            if saved_rate != checked["sustained_predictions_per_second"]:
                raise CampaignError("screen saved rate differs from raw elapsed work")
            if checked["backend"] != recorded["device"]:
                raise CampaignError(
                    "screen changed backend/device across matched processes"
                )
            start = number(entry["started_monotonic"], "screen monotonic start")
            end = number(entry["ended_monotonic"], "screen monotonic end")
            wall = positive(entry["process_wall_seconds"], "screen process wall time")
            if end - start != wall:
                raise CampaignError("screen wall-time evidence is inconsistent")
            measured = sum(c["elapsed_seconds"] for c in checked["cases"])
            if measured > wall * (1 + 1e-6):
                raise CampaignError(
                    "screen reported measurement exceeds process wall time"
                )
            if block >= 0:
                if (
                    previous_end is not None
                    and start - previous_end < lock["criteria"]["cooldown_seconds"]
                ):
                    raise CampaignError("screen measured runs violate frozen cooldown")
                previous_end = end
        recomputed = paired_statistics(recorded["samples"], lock["criteria"])
        if (
            canonical(recomputed) != canonical(recorded["statistics"])
            or recorded["status"] != recomputed["status"]
        ):
            raise CampaignError(
                "screen decision differs from recomputed raw-sample statistics"
            )
    expected_status = campaign_status([cell["status"] for cell in screen["cells"]])
    if (
        screen["status"] != expected_status
        or expected_status != "eligible_for_confirmation"
    ):
        raise CampaignError("screen is not an eligible frozen finalist")
    return screen
