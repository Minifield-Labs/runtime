"""Command-line entry point; arguments never execute through a shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import Controller, freeze
from .schema import CampaignError, file_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser(
        "freeze", help="bind criteria and exact oracles"
    )
    freeze_parser.add_argument("manifest", type=Path)
    freeze_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser(
        "run", help="build and evaluate against frozen lock"
    )
    run_parser.add_argument("lock", type=Path)
    run_parser.add_argument("--expected-campaign-sha256", required=True)
    run_parser.add_argument("--champion", type=Path, required=True)
    run_parser.add_argument("--candidate", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--confirmation-of", type=Path)
    run_parser.add_argument("--expected-screen-sha256")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            lock = freeze(args.manifest.resolve(), args.output.resolve())
            print(json.dumps({"campaign_sha256": lock["campaign_sha256"]}))
            return 0
        result = Controller().run(
            args.lock.resolve(),
            args.expected_campaign_sha256,
            args.champion.resolve(),
            args.candidate.resolve(),
            args.output.resolve(),
            args.confirmation_of.resolve() if args.confirmation_of else None,
            args.expected_screen_sha256,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output.resolve()),
                    "campaign_sha256": result["campaign_sha256"],
                    "result_sha256": file_digest(args.output.resolve() / "result.json"),
                }
            )
        )
        return (
            1
            if result["status"]
            in {"failed", "rejected", "inconclusive", "confirmation_failed"}
            else 0
        )
    except (CampaignError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
