"""Serialize and retain the predeclared campaign attempt budget."""

from __future__ import annotations

import fcntl
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import CampaignError, canonical, digest, fields, integer, loads, string


def validate_budget(value: Any, base: Path) -> dict[str, Any]:
    budget = dict(fields(value, {"candidate_limit", "finalist_limit", "ledger"}))
    integer(budget["candidate_limit"], "candidate_limit", 1)
    integer(budget["finalist_limit"], "finalist_limit", 1)
    budget["ledger"] = str((base / string(budget["ledger"], "ledger")).resolve())
    return budget


def reserve_attempt(
    lock: dict[str, Any],
    kind: str,
    output: Path,
    sources: dict[str, Any],
    screen_sha256: str | None,
) -> str:
    """Reserve before work starts; failures consume attempts and remain recorded."""
    budget = lock["attempt_budget"]
    path = Path(budget["ledger"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        raw = stream.read()
        ledger = (
            loads(raw)
            if raw
            else {
                "schema_version": 1,
                "campaign_sha256": lock["campaign_sha256"],
                "attempts": [],
            }
        )
        fields(ledger, {"schema_version", "campaign_sha256", "attempts"})
        if (
            ledger["schema_version"] != 1
            or ledger["campaign_sha256"] != lock["campaign_sha256"]
        ):
            raise CampaignError("attempt ledger belongs to another campaign")
        if not isinstance(ledger["attempts"], list):
            raise CampaignError("invalid attempt ledger")
        previous = lock["campaign_sha256"]
        counts = {"candidate": 0, "finalist": 0}
        for attempt in ledger["attempts"]:
            fields(
                attempt,
                {
                    "kind",
                    "output",
                    "sources",
                    "screen_sha256",
                    "recorded_at",
                    "previous_sha256",
                    "sha256",
                },
            )
            if attempt["kind"] not in counts or attempt["previous_sha256"] != previous:
                raise CampaignError("invalid attempt ledger chain")
            unhashed = {k: v for k, v in attempt.items() if k != "sha256"}
            if digest(unhashed) != attempt["sha256"]:
                raise CampaignError("attempt ledger digest mismatch")
            previous = attempt["sha256"]
            counts[attempt["kind"]] += 1
        if kind not in counts or counts[kind] >= budget[f"{kind}_limit"]:
            raise CampaignError(f"predeclared {kind} attempt budget exhausted")
        attempt = {
            "kind": kind,
            "output": str(output),
            "sources": sources,
            "screen_sha256": screen_sha256,
            "recorded_at": datetime.now(UTC).isoformat(),
            "previous_sha256": previous,
        }
        attempt["sha256"] = digest(attempt)
        ledger["attempts"].append(attempt)
        stream.seek(0)
        stream.truncate()
        stream.write(canonical(ledger).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return attempt["sha256"]
