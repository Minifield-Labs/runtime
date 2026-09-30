from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from minifield_hillclimb.controller import (
    CommandResult,
    Controller,
    ProcessDeadline,
    build_argv,
    execute,
    freeze,
    paired_statistics,
    source_identity,
    validate_lock,
    validate_response,
    write_json,
)
from minifield_hillclimb.evidence import repeated_improvement
from minifield_hillclimb.provenance import cargo_configs, tool_identity
from minifield_hillclimb.schema import (
    CampaignError,
    artifact_hashes,
    digest,
    file_digest,
    loads,
    read_json,
    validate_inputs,
    validate_pointer_reference,
)

CRITERIA = {
    "independent_runs": 8,
    "cooldown_seconds": 15,
    "minimum_improvement": 0.01,
    "maximum_slowdown": 0.02,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.0001,
    "bootstrap_resamples": 1000,
    "confidence": 0.95,
    "seed": 47,
    "minimum_paired_blocks": 4,
    "resource_limit_bytes": 1000000,
    "build_deadline_seconds": 5,
}


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, amount):
        self.sleeps.append(amount)
        self.value += amount


class Fixture:
    def __init__(self, root):
        self.root = root
        self.ambient = {"PATH": os.environ["PATH"], "HOME": str(root / "home")}
        (root / "home").mkdir()
        self.host_source = root / "frozen-host"
        self.host_source.mkdir()
        (self.host_source / "main.rs").write_text("frozen synthetic native host")
        bundle = root / "bundle"
        (bundle / "tokenizer").mkdir(parents=True)
        (bundle / "model.safetensors").write_bytes(b"synthetic weights")
        (bundle / "config.json").write_text("{}")
        (bundle / "tokenizer/tokenizer.json").write_text("{}")
        write_json(
            root / "inputs.json",
            {
                "schema_version": 1,
                "task": "classifier",
                "cases": [{"id": "a", "token_ids": [1, 2]}],
            },
        )
        self.cell = {
            "id": "model-nf4-cpu",
            "model": "synthetic",
            "precision": "nf4",
            "arithmetic": "f32",
            "backend": "cpu_reference",
            "bundle": str(bundle),
            "inputs": str(root / "inputs.json"),
            "reference": str(root / "oracle.json"),
            "tokenizer": str(bundle / "tokenizer/tokenizer.json"),
            "task": "classifier",
            "classes": 2,
            "context": 32,
            "mode": "full",
            "warmups": 1,
            "measured_cycles": 2,
            "deadline_seconds": 5,
            "lut2_mode": "off",
            "max_lut2_bytes": 0,
            "dispatch_requirements": {},
        }
        self.reference = {
            "schema_version": 1,
            "task": "classifier",
            "artifacts": artifact_hashes(self.cell),
            "cases": [{"id": "a", "output": [0.1, 0.9], "prediction": 1}],
        }
        write_json(root / "oracle.json", self.reference)
        self.manifest = {
            "schema_version": 1,
            "name": "synthetic-decision-tests",
            "build": {
                "cargo": "cargo",
                "rust_toolchain": "1.89.0",
                "package": "minifield-evaluation",
                "binary": "minifield-eval",
                "host_source": str(self.host_source),
                "host_relative_path": "evaluation-host",
                "features_by_backend": {
                    "cpu_reference": [],
                    "wgpu_metal": ["wgpu"],
                    "native_metal": ["metal"],
                },
            },
            "criteria": copy.deepcopy(CRITERIA),
            "environment": {"build": dict(self.ambient), "host": dict(self.ambient)},
            "attempt_budget": {
                "candidate_limit": 20,
                "finalist_limit": 5,
                "ledger": str(root / "attempts.json"),
            },
            "cells": [self.cell],
        }
        self.manifest_path = root / "manifest.json"
        write_json(self.manifest_path, self.manifest)
        self.lock_path = root / "campaign.lock.json"
        self.lock = self.freeze(self.lock_path)

    def freeze(self, output):
        return freeze(self.manifest_path, output, lambda: self.ambient)

    def host(self, role, speed=10, addition=""):
        source = self.root / role
        source.mkdir()
        shutil.copytree(self.host_source, source / "evaluation-host")
        script = source / "host.py"
        script.write_text(f"""#!{sys.executable}
import hashlib, json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[2]).read_text())
inputs = json.loads(Path(request["inputs"]).read_text())
n = request["measured_cycles"]
bundle = Path(request["bundle"])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
artifacts = {{"weights_sha256": sha(bundle / "model.safetensors"),
 "config_sha256": sha(bundle / "config.json"),
 "tokenizer_sha256": sha(request["tokenizer"]),
 "inputs_sha256": sha(request["inputs"])}}
cases = [{{"id": c["id"], "completed_predictions": n,
          "elapsed_seconds": n / {speed}, "latencies_seconds": [1 / {speed}] * n,
          "outputs": [[0.1, 0.9]] * n, "predictions": [1] * n}}
         for c in inputs["cases"]]
response = {{"schema_version": 1, "artifacts": artifacts,
 "backend": {{"implementation": request["backend"], "api": "cpu",
             "device": "synthetic", "driver": "test"}},
 "initialization_seconds": 0.01, "warmup_seconds": 0.01,
 "cases": cases, "dispatch_counts": {{}},
 "resources": {{"accounted_bytes": 100, "peak_accounted_bytes": 200}}}}
{addition}
print(json.dumps(response))
""")
        script.chmod(0o755)
        return source


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def response(self):
        return {
            "schema_version": 1,
            "artifacts": self.fixture.reference["artifacts"],
            "backend": {
                "implementation": "cpu_reference",
                "api": "cpu",
                "device": "synthetic",
                "driver": "test",
            },
            "initialization_seconds": 0,
            "warmup_seconds": 0,
            "cases": [
                {
                    "id": "a",
                    "completed_predictions": 2,
                    "elapsed_seconds": 0.2,
                    "latencies_seconds": [0.1, 0.1],
                    "outputs": [[0.1, 0.9], [0.1, 0.9]],
                    "predictions": [1, 1],
                }
            ],
            "dispatch_counts": {},
            "resources": {"accounted_bytes": 100, "peak_accounted_bytes": 200},
        }

    def validate(self, response, cell=None):
        return validate_response(
            response, cell or self.fixture.cell, self.fixture.reference, CRITERIA, 2
        )

    def controller(self, after_host=None, after_build=None, rate_overrides=None):
        clock = Clock()
        build_calls = []

        def identify(source):
            return {
                "path": str(source),
                "revision": "synthetic",
                "source_sha256": file_digest(source / "host.py"),
                "dirty": False,
            }

        def command(argv, cwd, deadline, environment):
            clock.value += 1
            if "--target-dir" in argv:
                build_calls.append(argv)
                target = Path(argv[argv.index("--target-dir") + 1])
                binary = target / "release/minifield-eval"
                binary.parent.mkdir(parents=True)
                shutil.copy2(cwd / "host.py", binary)
                if after_build:
                    after_build(argv)
                return CommandResult(0, "synthetic build", "")
            result = execute(argv, cwd, deadline, environment)
            if rate_overrides and cwd.name == "candidate":
                cell_id = Path(argv[-1]).parent.name
                if cell_id in rate_overrides:
                    response = loads(result.stdout)
                    rate = rate_overrides[cell_id]
                    for case in response["cases"]:
                        case["elapsed_seconds"] = case["completed_predictions"] / rate
                        case["latencies_seconds"] = [1 / rate] * case[
                            "completed_predictions"
                        ]
                    result = CommandResult(0, json.dumps(response), result.stderr)
            if after_host:
                after_host(argv)
            return result

        controller = Controller(
            command,
            clock.now,
            clock.sleep,
            identify,
            environment_provider=lambda: self.fixture.ambient,
            tool_probe=lambda _build, _source, _env: {
                "cargo_entry": {"path": "synthetic-cargo", "sha256": "synthetic"},
                "rustc_vv": "release: 1.89.0",
            },
        )
        return controller, clock, build_calls

    def run_campaign(
        self, champion, candidate, controller, name="run", confirmation=None
    ):
        return controller.run(
            self.fixture.lock_path,
            self.fixture.lock["campaign_sha256"],
            champion,
            candidate,
            self.root / name,
            confirmation,
            file_digest(confirmation) if confirmation else None,
        )

    def test_bootstrap_matches_declared_mean_rate_estimator(self):
        # Fast conditions carry more rate in the declared arithmetic mean.
        # Averaging ratios would hide the 10% regression in the first block.
        samples = []
        for block, (champion, candidate) in enumerate(
            [(1000, 900), (1, 1.5), (1, 1.5), (1, 1.5)]
        ):
            for role in ("champion", "candidate", "candidate", "champion"):
                samples.append(
                    {
                        "role": role,
                        "block": block,
                        "response": {
                            "sustained_predictions_per_second": champion
                            if role == "champion"
                            else candidate
                        },
                    }
                )
        result = paired_statistics(samples, CRITERIA)
        self.assertLess(result["candidate_over_champion"], 0.91)
        self.assertNotEqual(result["status"], "improved")
        self.assertLess(result["paired_ratio_interval"][0], 0.91)

    def test_lock_rejects_changed_weights_or_oracle(self):
        (Path(self.fixture.cell["bundle"]) / "model.safetensors").write_bytes(
            b"changed"
        )
        with self.assertRaisesRegex(CampaignError, "artifacts changed"):
            validate_lock(self.fixture.lock_path, self.fixture.lock["campaign_sha256"])

    def test_native_host_is_part_of_the_frozen_harness(self):
        (self.fixture.host_source / "main.rs").write_text("changed timing behavior")
        with self.assertRaisesRegex(CampaignError, "native evaluation host changed"):
            validate_lock(self.fixture.lock_path, self.fixture.lock["campaign_sha256"])

    def test_candidate_cannot_change_the_native_host(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        (candidate / "evaluation-host/main.rs").write_text("changed counters")
        controller, _, builds = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("native host differs", result["errors"][0])
        self.assertFalse(builds)

    def test_lock_digest_is_externally_pinned(self):
        lock = copy.deepcopy(self.fixture.lock)
        lock["criteria"]["minimum_improvement"] = 0
        lock["criteria_sha256"] = digest(lock["criteria"])
        lock["campaign_sha256"] = digest(
            {k: v for k, v in lock.items() if k != "campaign_sha256"}
        )
        write_json(self.fixture.lock_path, lock)
        with self.assertRaisesRegex(CampaignError, "externally pinned"):
            validate_lock(self.fixture.lock_path, self.fixture.lock["campaign_sha256"])

    def test_freeze_rejects_missing_reference_and_short_cooldown(self):
        Path(self.fixture.cell["reference"]).unlink()
        with self.assertRaises(CampaignError):
            self.fixture.freeze(self.root / "another.lock.json")
        self.fixture.manifest["criteria"]["cooldown_seconds"] = 14.9
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        with self.assertRaisesRegex(CampaignError, "cooldown_seconds"):
            self.fixture.freeze(self.root / "another.lock.json")

    def test_json_rejects_duplicates_and_nonfinite_constants(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises(CampaignError):
                loads(raw)

    def test_response_rejects_incomplete_changed_and_nonfinite_outputs(self):
        for edit in (
            lambda r: r["cases"][0]["outputs"].pop(),
            lambda r: r["cases"][0]["predictions"].__setitem__(1, 0),
            lambda r: r["cases"][0]["outputs"][1].__setitem__(0, float("nan")),
            lambda r: r["cases"][0]["outputs"][1].__setitem__(0, 0.5),
            lambda r: r["cases"][0].__setitem__("completed_predictions", 1),
            lambda r: r["cases"][0].__setitem__("id", "wrong"),
        ):
            response = self.response()
            edit(response)
            with self.subTest(edit=edit), self.assertRaises(CampaignError):
                self.validate(response)

    def test_response_rejects_fallback_and_missing_dispatch(self):
        cell = {
            **self.fixture.cell,
            "backend": "wgpu_metal",
            "dispatch_requirements": {"gemm": 1},
        }
        with self.assertRaisesRegex(CampaignError, "actual backend"):
            self.validate(self.response(), cell)
        response = self.response()
        response["backend"].update(implementation="wgpu_metal", api="metal")
        with self.assertRaisesRegex(CampaignError, "dispatch evidence"):
            self.validate(response, cell)
        response["dispatch_counts"] = {"gemm": 1}
        self.validate(response, cell)

    def test_gpu_peak_counter_cannot_be_missing_or_zero(self):
        response = self.response()
        cell = {**self.fixture.cell, "backend": "wgpu_metal"}
        response["backend"].update(implementation="wgpu_metal", api="metal")
        response["resources"] = {"accounted_bytes": 0, "peak_accounted_bytes": 0}
        with self.assertRaisesRegex(CampaignError, "instrumentation"):
            self.validate(response, cell)

    def test_finite_input_times_cannot_create_infinite_rates(self):
        response = self.response()
        response["cases"][0]["elapsed_seconds"] = 1e-320
        response["cases"][0]["latencies_seconds"] = [5e-321, 5e-321]
        with self.assertRaisesRegex(CampaignError, "finite"):
            self.validate(response)

    def test_response_rejects_memory_and_impossible_timing(self):
        response = self.response()
        response["resources"]["peak_accounted_bytes"] = 1000001
        with self.assertRaisesRegex(CampaignError, "budget"):
            self.validate(response)

    def test_host_must_report_exact_loaded_artifact_hashes(self):
        response = self.response()
        response["artifacts"] = {**response["artifacts"], "weights_sha256": "wrong"}
        with self.assertRaisesRegex(CampaignError, "host-loaded artifact"):
            self.validate(response)
        response = self.response()
        response["cases"][0]["elapsed_seconds"] = 0.1
        with self.assertRaisesRegex(CampaignError, "latencies exceed"):
            self.validate(response)

    def test_real_fake_hosts_use_abba_cooldown_and_confirmation(self):
        champion = self.fixture.host("champion", speed=10)
        candidate = self.fixture.host("candidate", speed=12)
        controller, clock, builds = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "eligible_for_confirmation")
        samples = result["cells"][0]["samples"]
        self.assertEqual(
            [s["role"] for s in samples],
            ["champion", "candidate", "candidate", "champion"] * 4,
        )
        self.assertEqual(clock.sleeps, [15] * 15)
        for a, b in zip(samples, samples[1:], strict=False):
            self.assertGreaterEqual(b["started_monotonic"] - a["ended_monotonic"], 15)
        self.assertEqual(
            result["cells"][0]["statistics"]["mean_predictions_per_second"],
            {"champion": 10, "candidate": 12},
        )
        self.assertEqual(len(builds), 2)
        self.assertNotEqual(
            builds[0][builds[0].index("--target-dir") + 1],
            builds[1][builds[1].index("--target-dir") + 1],
        )
        confirmation, _, _ = self.controller()
        promoted = self.run_campaign(
            champion,
            candidate,
            confirmation,
            "confirmation",
            self.root / "run/result.json",
        )
        self.assertEqual(promoted["status"], "promoted")

    def test_mutation_during_host_run_is_retained_failure(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")

        def mutate(_argv):
            Path(self.fixture.cell["reference"]).write_text("{}")

        controller, _, _ = self.controller(mutate)
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("oracle changed", result["errors"][0])
        self.assertTrue((self.root / "run/result.json").is_file())
        self.assertTrue(list((self.root / "run/cells").rglob("*.stdout")))

    def test_regression_rejects_and_too_few_blocks_are_inconclusive(self):
        samples = [
            {
                "role": role,
                "block": block,
                "response": {"sustained_predictions_per_second": speed},
            }
            for block in range(4)
            for role, speed in [
                ("champion", 10),
                ("candidate", 8),
                ("candidate", 8),
                ("champion", 10),
            ]
        ]
        self.assertEqual(paired_statistics(samples, CRITERIA)["status"], "regressed")
        criteria = {**CRITERIA, "minimum_paired_blocks": 5}
        self.assertEqual(paired_statistics(samples, criteria)["status"], "inconclusive")

    def test_build_arguments_pin_release_lock_toolchain_and_backend(self):
        argv = build_argv(self.fixture.manifest["build"], "wgpu_metal", Path("/target"))
        self.assertEqual(
            argv[:5], ["cargo", "+1.89.0", "build", "--release", "--locked"]
        )
        self.assertIn("--no-default-features", argv)
        self.assertEqual(argv[-2:], ["--features", "wgpu"])

    def test_process_deadline_keeps_output_and_kills_process(self):
        script = self.root / "slow.py"
        script.write_text("import time\nprint('started', flush=True)\ntime.sleep(20)\n")
        with self.assertRaises(ProcessDeadline) as raised:
            execute([sys.executable, str(script)], self.root, 1.0)
        self.assertEqual(raised.exception.stdout.strip(), "started")

    def test_full_controller_deadline_retains_partial_logs(self):
        champion = self.fixture.host(
            "champion",
            addition=("import time\nprint('partial', flush=True)\ntime.sleep(20)"),
        )
        candidate = self.fixture.host("candidate")
        # Allow interpreter startup under concurrent compilation; the 20-second
        # fixture still exceeds the deadline. Production criteria are untouched.
        self.fixture.manifest["cells"][0]["deadline_seconds"] = 3.0
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        self.fixture.lock_path.unlink()
        self.fixture.lock = self.fixture.freeze(self.fixture.lock_path)
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("deadline exceeded", result["errors"][0])
        logs = list((self.root / "run/cells").rglob("*.stdout"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text().strip(), "partial")

    def test_oracle_width_and_predictions_are_frozen(self):
        reference = copy.deepcopy(self.fixture.reference)
        reference["cases"][0]["prediction"] = 0
        write_json(Path(self.fixture.cell["reference"]), reference)
        with self.assertRaisesRegex(CampaignError, "argmax"):
            self.fixture.freeze(self.root / "bad.lock.json")

    def test_four_run_exploration_cannot_qualify_for_confirmation(self):
        champion = self.fixture.host("champion", speed=10)
        candidate = self.fixture.host("candidate", speed=15)
        self.fixture.manifest["criteria"]["independent_runs"] = 4
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        self.fixture.lock_path.unlink()
        self.fixture.lock = self.fixture.freeze(self.fixture.lock_path)
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "inconclusive")
        confirmation, _, _ = self.controller()
        result = self.run_campaign(
            champion,
            candidate,
            confirmation,
            "confirmation",
            self.root / "run/result.json",
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("frozen finalist", result["errors"][0])

    def test_executable_mutation_after_launch_fails_even_last_process(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        count = 0

        def mutate(argv):
            nonlocal count
            count += 1
            if count == 18:  # 2 correctness plus 16 measured processes.
                Path(argv[0]).write_text("replaced binary")

        controller, _, _ = self.controller(mutate)
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("executable changed", result["errors"][0])

    def test_source_mutation_after_last_host_does_not_pass(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        count = 0

        def mutate(_argv):
            nonlocal count
            count += 1
            if count == 18:
                (candidate / "host.py").write_text("source changed")

        controller, _, _ = self.controller(mutate)
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate source changed", result["errors"][0])

    def test_reported_duration_cannot_exceed_actual_process_wall(self):
        champion = self.fixture.host(
            "champion", addition=('response["cases"][0]["elapsed_seconds"] = 100')
        )
        candidate = self.fixture.host("candidate")
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("process wall time", result["errors"][0])

    def test_backend_driver_cannot_change_between_builds(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host(
            "candidate", addition=('response["backend"]["driver"] = "different"')
        )
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("backend/device identity", result["errors"][0])

    def test_source_identity_includes_dirty_untracked_and_ignored_code(self):
        repository = self.root / "source"
        repository.mkdir()
        for argv in (
            ["git", "init", "-b", "main"],
            ["git", "config", "user.name", "Synthetic Test"],
            ["git", "config", "user.email", "test@example.invalid"],
        ):
            self.assertEqual(execute(argv, repository, 10).returncode, 0)
        tracked = repository / "tracked.rs"
        tracked.write_text("original")
        (repository / ".gitignore").write_text("ignored.rs\ntarget/\n")
        execute(["git", "add", "."], repository, 10)
        committed = execute(
            ["git", "commit", "-m", "test: seed source identity"], repository, 10
        )
        self.assertEqual(committed.returncode, 0, committed.stderr)
        first = source_identity(repository)
        tracked.write_text("dirty")
        dirty = source_identity(repository)
        self.assertNotEqual(first["source_sha256"], dirty["source_sha256"])
        (repository / "untracked.rs").write_text("new module")
        untracked = source_identity(repository)
        self.assertNotEqual(dirty["source_sha256"], untracked["source_sha256"])
        (repository / "ignored.rs").write_text("ignored included module")
        ignored = source_identity(repository)
        self.assertNotEqual(untracked["source_sha256"], ignored["source_sha256"])
        (repository / "target").mkdir()
        (repository / "target/generated.rs").write_text("build output")
        self.assertEqual(
            ignored["source_sha256"], source_identity(repository)["source_sha256"]
        )

    def test_pointer_inputs_and_reference_shape_match_complete_contract(self):
        case = {
            "id": "pointer",
            "token_ids": [1, 2, 3, 4],
            "questions": [
                {
                    "query_index": 0,
                    "option_indices": [1, 2],
                    "kind": {"type": "choice"},
                },
                {
                    "query_index": 0,
                    "option_indices": [],
                    "kind": {
                        "type": "extract",
                        "absent_index": 1,
                        "source_start": 2,
                        "selectable": [True, True],
                        "presence_threshold": 0.5,
                    },
                },
            ],
        }
        validate_inputs(
            {"schema_version": 1, "task": "pointer", "cases": [case]}, "pointer"
        )
        supplied = {
            "id": "pointer",
            "output": [0.1] * 19,
            "prediction": [
                {"type": "choice", "index": 0},
                {"type": "span", "start": 0, "end": 2},
            ],
        }
        validate_pointer_reference(case, supplied)
        supplied["output"].pop()
        with self.assertRaisesRegex(CampaignError, "width"):
            validate_pointer_reference(case, supplied)
        invalid = copy.deepcopy(case)
        invalid["segments"] = [1, 1, 2, 2]
        with self.assertRaisesRegex(CampaignError, "query segment"):
            validate_inputs(
                {"schema_version": 1, "task": "pointer", "cases": [invalid]}, "pointer"
            )

    def screen(self):
        champion = self.fixture.host("champion", speed=10)
        candidate = self.fixture.host("candidate", speed=12)
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "eligible_for_confirmation")
        return champion, candidate, self.root / "run/result.json", result

    def test_confirmation_requires_external_screen_digest(self):
        champion, candidate, path, _ = self.screen()
        controller, _, _ = self.controller()
        result = controller.run(
            self.fixture.lock_path,
            self.fixture.lock["campaign_sha256"],
            champion,
            candidate,
            self.root / "confirmation",
            path,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("externally pinned screen", result["errors"][0])

    def test_changed_screen_cannot_use_its_original_external_digest(self):
        champion, candidate, path, screen = self.screen()
        original = file_digest(path)
        screen["cells"][0]["statistics"]["candidate_over_champion"] = 999
        write_json(path, screen)
        controller, _, _ = self.controller()
        result = controller.run(
            self.fixture.lock_path,
            self.fixture.lock["campaign_sha256"],
            champion,
            candidate,
            self.root / "confirmation",
            path,
            original,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("externally pinned digest", result["errors"][0])

    def test_screen_decisions_are_reconstructed_even_if_rehashed(self):
        champion, candidate, path, original = self.screen()
        mutations = [
            lambda s: s.__setitem__("status", "no_confirmed_improvement"),
            lambda s: s["cells"][0].__setitem__("status", "within_budget"),
            lambda s: s["cells"][0]["statistics"].__setitem__(
                "candidate_over_champion", 999
            ),
            lambda s: s["cells"].clear(),
            lambda s: s["cells"].__setitem__(0, None),
            lambda s: s["cells"].append(copy.deepcopy(s["cells"][0])),
            lambda s: s["cells"][0]["samples"].pop(),
            lambda s: s["cells"][0]["correctness"].pop(),
            lambda s: s["cells"][0]["samples"][0]["response"]["cases"][0][
                "outputs"
            ].pop(),
            lambda s: s["cells"][0]["samples"][0]["response"].__setitem__(
                "sustained_predictions_per_second", 999
            ),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                screen = copy.deepcopy(original)
                mutate(screen)
                write_json(path, screen)
                controller, _, builds = self.controller()
                result = self.run_campaign(
                    champion, candidate, controller, f"confirmation-{index}", path
                )
                self.assertEqual(result["status"], "failed")
                self.assertFalse(builds)

    def test_confirmation_must_repeat_the_same_improved_cell(self):
        second = {**self.fixture.cell, "id": "second-model", "model": "second"}
        self.fixture.manifest["cells"].append(second)
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        self.fixture.lock_path.unlink()
        self.fixture.lock = self.fixture.freeze(self.fixture.lock_path)
        champion = self.fixture.host("champion", speed=10)
        candidate = self.fixture.host("candidate", speed=12)
        controller, _, _ = self.controller(
            rate_overrides={self.fixture.cell["id"]: 12, "second-model": 10}
        )
        screen = self.run_campaign(champion, candidate, controller)
        self.assertEqual(screen["status"], "eligible_for_confirmation")
        self.assertEqual(
            [c["status"] for c in screen["cells"]], ["improved", "within_budget"]
        )
        confirmation, _, _ = self.controller(
            rate_overrides={self.fixture.cell["id"]: 10, "second-model": 12}
        )
        result = self.run_campaign(
            champion,
            candidate,
            confirmation,
            "confirmation",
            self.root / "run/result.json",
        )
        self.assertEqual(result["status"], "confirmation_failed")
        self.assertEqual(result["confirmed_cell_ids"], [])
        self.assertEqual(repeated_improvement(screen, result), [])

    def test_unreviewed_profile_and_rustflags_reject_before_build(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        for index, name in enumerate(
            ("PROFILE", "RUSTFLAGS", "CARGO_PROFILE_RELEASE_OPT_LEVEL", "RUSTC_WRAPPER")
        ):
            with self.subTest(name=name):
                self.fixture.ambient[name] = "changed"
                controller, _, builds = self.controller()
                result = self.run_campaign(
                    champion, candidate, controller, f"run-{index}"
                )
                self.assertEqual(result["status"], "failed")
                self.assertIn(name, result["errors"][0])
                self.assertFalse(builds)
                del self.fixture.ambient[name]

    def test_manifest_cannot_allowlist_rustflags_or_host_profile(self):
        for phase, key in (("build", "RUSTFLAGS"), ("host", "PROFILE")):
            with self.subTest(phase=phase):
                manifest = copy.deepcopy(self.fixture.manifest)
                manifest["environment"][phase][key] = "changed"
                write_json(self.fixture.manifest_path, manifest)
                with self.assertRaisesRegex(CampaignError, "unreviewable overrides"):
                    self.fixture.freeze(self.root / f"{phase}.lock.json")

    def test_environment_mutation_during_build_is_rejected(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        controller, _, _ = self.controller(
            after_build=lambda _argv: self.fixture.ambient.update(PROFILE="debug")
        )
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("PROFILE", result["errors"][0])

    def test_environment_mutation_after_host_is_rejected(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        controller, _, _ = self.controller(
            after_host=lambda _argv: self.fixture.ambient.update(PATH="changed")
        )
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("environment changed", result["errors"][0])

    def test_inherited_telemetry_is_forced_off_in_real_hosts(self):
        self.fixture.ambient["MINIFIELD_TELEMETRY"] = "1"
        check = (
            'import os\nassert os.environ["MINIFIELD_TELEMETRY"] == "0"\n'
            'assert "PROFILE" not in os.environ'
        )
        champion = self.fixture.host("champion", addition=check)
        candidate = self.fixture.host("candidate", addition=check)
        controller, _, _ = self.controller()
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "no_confirmed_improvement")
        self.assertEqual(
            result["execution_context"]["environment"]["host"]["MINIFIELD_TELEMETRY"],
            "0",
        )

    def test_global_and_ancestor_cargo_config_mutations_reject(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        global_config = self.root / "home/.cargo/config.toml"
        ancestor_config = self.root / ".cargo/config.toml"
        for index, config in enumerate((global_config, ancestor_config)):
            with self.subTest(config=config):

                def mutate(_argv, config=config):
                    config.parent.mkdir(exist_ok=True)
                    config.write_text("[profile.release]\nopt-level=0\n")

                controller, _, _ = self.controller(after_build=mutate)
                result = self.run_campaign(
                    champion, candidate, controller, f"run-{index}"
                )
                self.assertEqual(result["status"], "failed")
                self.assertIn("Cargo configuration changed", result["errors"][0])
                config.unlink()

    def test_confirmation_rejects_new_global_cargo_config(self):
        champion, candidate, path, _ = self.screen()
        config = self.root / "home/.cargo/config.toml"
        config.parent.mkdir()
        config.write_text("[build]\nincremental=false\n")
        controller, _, builds = self.controller()
        result = self.run_campaign(
            champion, candidate, controller, "confirmation", path
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("finalist context", result["errors"][0])
        self.assertFalse(builds)

    def test_equal_config_bytes_at_distinct_ancestor_paths_are_not_equated(self):
        paths = {role: self.root / role / "checkout" for role in ("left", "right")}
        for source in paths.values():
            source.mkdir(parents=True)
            config = source.parent / ".cargo/config.toml"
            config.parent.mkdir()
            config.write_text("[build]\nincremental=false\n")
        controller, _, _ = self.controller()
        with self.assertRaisesRegex(CampaignError, "different active Cargo configs"):
            controller._context(
                self.fixture.lock,
                {
                    "champion": paths["left"],
                    "candidate": paths["right"],
                },
            )

    def test_selected_compiler_identity_cannot_change_between_builds(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        changed = False

        def mutate(_argv):
            nonlocal changed
            changed = True

        controller, _, _ = self.controller(after_build=mutate)
        controller.tool_probe = lambda _build, _source, _env: {
            "cargo_entry": {
                "path": "synthetic-cargo",
                "sha256": "new" if changed else "old",
            },
        }
        result = self.run_campaign(champion, candidate, controller)
        self.assertEqual(result["status"], "failed")
        self.assertIn("compiler/tool identity changed", result["errors"][0])

    def test_tool_probe_records_actual_selected_paths_hashes_and_rustc_vv(self):
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("cargo", "rustup", "selected-cargo", "selected-rustc"):
            path = tools / name
            path.write_text(f"synthetic {name}")
            path.chmod(0o755)
        calls = []

        def probe(argv, _cwd, _timeout, environment):
            calls.append(argv)
            self.assertEqual(environment["MINIFIELD_TELEMETRY"], "0")
            if "which" in argv:
                stdout = str(tools / f"selected-{argv[-1]}")
            elif argv[-1] == "-vV":
                stdout = "rustc 1.89.0\nrelease: 1.89.0\ncommit-hash: synthetic\n"
            else:
                stdout = "cargo 1.89.0 (synthetic)"
            return CommandResult(0, stdout, "")

        environment = {
            "PATH": str(tools),
            "HOME": str(self.root / "home"),
            "MINIFIELD_TELEMETRY": "0",
        }
        identity = tool_identity(
            self.fixture.manifest["build"], self.root, environment, probe
        )
        self.assertEqual(
            identity["rustc_selected"]["sha256"], file_digest(tools / "selected-rustc")
        )
        self.assertEqual(calls[-1][-1], "-vV")

    def test_persistent_budget_counts_failures_and_blocks_extra_attempts(self):
        self.fixture.manifest["attempt_budget"]["candidate_limit"] = 1
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        self.fixture.lock_path.unlink()
        self.fixture.lock = self.fixture.freeze(self.fixture.lock_path)
        champion = self.fixture.host(
            "champion", addition='response["cases"][0]["predictions"] = [0]'
        )
        candidate = self.fixture.host("candidate")
        controller, _, _ = self.controller()
        first = self.run_campaign(champion, candidate, controller)
        self.assertEqual(first["status"], "failed")
        controller, _, builds = self.controller()
        second = self.run_campaign(champion, candidate, controller, "another")
        self.assertEqual(second["status"], "failed")
        self.assertIn("budget exhausted", second["errors"][0])
        self.assertFalse(builds)
        ledger = read_json(self.root / "attempts.json")
        self.assertEqual(len(ledger["attempts"]), 1)

    def test_manifest_home_paths_must_be_absolute(self):
        for phase, name in (
            ("build", "HOME"),
            ("host", "HOME"),
            ("build", "CARGO_HOME"),
            ("build", "RUSTUP_HOME"),
        ):
            with self.subTest(phase=phase, name=name):
                manifest = copy.deepcopy(self.fixture.manifest)
                manifest["environment"][phase][name] = "relative/home"
                write_json(self.fixture.manifest_path, manifest)
                with self.assertRaisesRegex(
                    CampaignError, f"{name} must be an absolute path"
                ):
                    self.fixture.freeze(self.root / f"{phase}-{name}.lock.json")

    def test_unchanged_cargo_config_cannot_select_a_mutable_external_compiler(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        directory = self.root / "home/.cargo"
        directory.mkdir()
        config = directory / "config.toml"
        wrapper = self.root / "external-wrapper"
        wrapper.write_text("mutable external compiler wrapper")
        for selector in ("rustc", "rustc-wrapper", "rustc-workspace-wrapper"):
            with self.subTest(selector=selector):
                config.write_text(f"[build]\n{selector} = {json.dumps(str(wrapper))}\n")
                before = file_digest(config)
                for revision in range(2):
                    wrapper.write_text(f"externally changed implementation {revision}")
                    controller, _, builds = self.controller()
                    result = self.run_campaign(
                        champion, candidate, controller, f"{selector}-{revision}"
                    )
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(f"build.{selector}", result["errors"][0])
                    self.assertFalse(builds)
                    self.assertEqual(file_digest(config), before)

    def test_cargo_env_compiler_and_wrapper_overrides_reject_before_build(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        directory = self.root / "home/.cargo"
        directory.mkdir()
        config = directory / "config.toml"
        for name in (
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_RUSTC",
            "RUSTUP_TOOLCHAIN",
            "PATH",
            "HOME",
        ):
            with self.subTest(name=name):
                config.write_text(
                    f'[env]\n{name} = {{value="external", force=true, relative=true}}\n'
                )
                controller, _, builds = self.controller()
                result = self.run_campaign(
                    champion, candidate, controller, f"env-{name}"
                )
                self.assertEqual(result["status"], "failed")
                self.assertIn(f"compiler override {name}", result["errors"][0])
                self.assertFalse(builds)

    def test_active_cargo_legacy_filename_priority_preserves_both_hashes(self):
        directory = self.root / "home/.cargo"
        directory.mkdir()
        legacy, modern = directory / "config", directory / "config.toml"
        legacy.write_text("[build]\nincremental=false\n")
        modern.write_text('[build]\nrustc-wrapper="inactive-wrapper"\n')
        environment = self.fixture.lock["environment"]["build"]
        records = cargo_configs(self.root, environment)
        self.assertEqual(records[str(legacy)], file_digest(legacy))
        self.assertEqual(records[str(modern)], file_digest(modern))
        legacy.unlink()
        with self.assertRaisesRegex(CampaignError, "build.rustc-wrapper"):
            cargo_configs(self.root, environment)

    def test_unchanged_cargo_config_cannot_include_mutable_external_settings(self):
        champion = self.fixture.host("champion")
        candidate = self.fixture.host("candidate")
        directory = self.root / "home/.cargo"
        directory.mkdir()
        config = directory / "config.toml"
        external = self.root / "external-config.toml"
        for form, include in enumerate((str(external), [str(external)])):
            with self.subTest(include=include):
                config.write_text(f"include = {json.dumps(include)}\n")
                before = file_digest(config)
                for revision in range(2):
                    external.write_text(f"[build]\njobs = {revision + 1}\n")
                    controller, _, builds = self.controller()
                    result = self.run_campaign(
                        champion, candidate, controller, f"include-{form}-{revision}"
                    )
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("Cargo config include", result["errors"][0])
                    self.assertFalse(builds)
                    self.assertEqual(file_digest(config), before)

    def test_invalid_active_cargo_toml_rejects_before_build(self):
        directory = self.root / "home/.cargo"
        directory.mkdir()
        (directory / "config.toml").write_text("[build]\ninvalid = [\n")
        with self.assertRaisesRegex(CampaignError, "cannot admit Cargo config"):
            cargo_configs(self.root, self.fixture.lock["environment"]["build"])


if __name__ == "__main__":
    unittest.main()
