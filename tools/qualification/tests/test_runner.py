"""Synthetic transport and admission tests; these make no model claim."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "runner.py"
FAKE = r"""
import argparse, hashlib, json, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("bundle")
p.add_argument("prompts")
p.add_argument("--classes", type=int)
p.add_argument("--context", type=int)
p.add_argument("--tokenizer")
p.add_argument("--lut2")
p.add_argument("--prefix")
p.add_argument("--max-lut2-bytes")
p.add_argument("--deadline-seconds")
p.add_argument("--warmups")
a = p.parse_args()
b = pathlib.Path(a.bundle)
case = (b / "case.txt").read_text()
if case == "timeout":
    time.sleep(5)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
prompts = json.loads(pathlib.Path(a.prompts).read_text())
rows = []
for index, prompt in enumerate(prompts):
    logits = [0.5 + index, 0.25, -0.5]
    if case == "nonfinite":
        logits[0] = float("nan")
    if case == "mismatch":
        logits[0] += 1
    if case == "argmax":
        logits[1] = logits[0] + 0.01
    seconds = 0.01 if case != "slow" else 1
    rows.append({"prompt_index": index, "ids": [1, 2, index + 3],
                 "logits": logits, "seconds": seconds})
fallback = case == "fallback"
result = {
    "schema_version": 1, "classes": a.classes, "context": a.context,
    "requested_lut2_mode": a.lut2,
    "effective_lut2_mode": "partial" if fallback else a.lut2,
    "lut2_fallback": fallback, "cached": a.prefix == "true",
    "initialization_seconds": 0.2,
    "warmup_seconds": 0.01 * int(a.warmups),
    "base_prefill_seconds": 0.003 if a.prefix == "true" else 0,
    "samples": rows, "dispatch_counts": {"synthetic_transport": 1},
    "memory": {"resident_weight_bytes": 12, "cache_bytes": 0, "scratch_bytes": 0,
               "staged_branch_bytes": 0, "pending_operation_bytes": 0,
               "pending_operations": 0},
    "adapter": {"name": "synthetic transport fixture", "backend": "synthetic",
                "device_type": "Synthetic"},
    "artifacts": {"weights_sha256": sha(b / "model.safetensors"),
                  "config_sha256": sha(b / "config.json"),
                  "tokenizer_sha256": sha(pathlib.Path(a.tokenizer)),
                  "prompts_sha256": sha(pathlib.Path(a.prompts))},
}
if case == "empty_logits":
    result["samples"][0]["logits"] = []
if case == "missing_dispatch":
    result["dispatch_counts"] = {}
if case == "wrong_hash":
    result["artifacts"]["weights_sha256"] = "0" * 64
if case == "wrong_artifacts_type":
    result["artifacts"] = []
if case == "boolean_schema":
    result["schema_version"] = True
print(json.dumps(result))
"""


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        (self.bundle / "tokenizer").mkdir(parents=True)
        (self.bundle / "config.json").write_text("{}")
        (self.bundle / "model.safetensors").write_bytes(b"synthetic transport fixture")
        (self.bundle / "tokenizer/tokenizer.json").write_text("{}")
        (self.bundle / "case.txt").write_text("ok")
        self.prompts = self.root / "prompts.json"
        self.prompts.write_text(json.dumps(["common first", "common second"]))
        self.fake = self.root / "fake.py"
        self.fake.write_text(FAKE)
        self.profile = {
            "schema_version": 1,
            "name": "synthetic-transport",
            "bundle": "bundle",
            "prompts": "prompts.json",
            "command": [sys.executable, str(self.fake)],
            "classes": 3,
            "context": 64,
            "repetitions": 2,
            "warmups": 1,
            "timeout_seconds": 2,
            "expected_logits": "expected.json",
        }
        self.expected = {
            "schema_version": 1,
            "classes": 3,
            "context": 64,
            "artifacts": self.artifacts(),
            "logits": [[0.5, 0.25, -0.5], [1.5, 0.25, -0.5]],
        }
        self.write_expected()

    def artifacts(self):
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("weights_sha256", self.bundle / "model.safetensors"),
                ("config_sha256", self.bundle / "config.json"),
                ("tokenizer_sha256", self.bundle / "tokenizer/tokenizer.json"),
                ("prompts_sha256", self.prompts),
            )
        }

    def write_expected(self):
        (self.root / "expected.json").write_text(json.dumps(self.expected))

    def run_tool(self, kind="qualify"):
        profile = self.root / "profile.json"
        profile.write_text(json.dumps(self.profile))
        process = subprocess.run(
            [sys.executable, str(RUNNER), kind, str(profile)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(process.stderr, "", process.stderr)
        return process.returncode, json.loads(process.stdout)

    def test_qualification_binds_hashes_and_separates_initialization(self):
        code, result = self.run_tool()
        self.assertEqual(code, 0, result)
        self.assertEqual(result["artifacts"], self.artifacts())
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(len(result["latency"][0]["initialization_seconds"]), 2)
        self.assertEqual(result["latency"][0]["median_inference_seconds"], 0.01)
        self.assertEqual(result["comparisons"][0]["kind"], "implementation_parity")
        self.assertTrue(result["comparisons"][0]["passed"])
        self.assertIsInstance(result["source"]["working_tree_dirty"], bool)
        self.assertEqual(len(result["source"]["source_sha256"]), 64)
        self.assertEqual(len(result["command_files"]), 2)

    def test_benchmark_matches_all_modes_and_paths_per_repetition(self):
        code, result = self.run_tool("bench")
        self.assertEqual(code, 0, result)
        self.assertEqual(len(result["runs"]), 12)
        self.assertEqual(len(result["latency"]), 6)
        self.assertEqual(
            {(entry["lut2_mode"], entry["cached"]) for entry in result["latency"]},
            {
                (mode, cached)
                for mode in ("raw", "down", "auto")
                for cached in (False, True)
            },
        )
        self.assertTrue(all(entry["repetitions"] == 2 for entry in result["latency"]))

    def test_empty_prompts_missing_assets_and_bad_profile_fail(self):
        for prompts in ([], [""], ["  "], [1]):
            with self.subTest(prompts=prompts):
                self.prompts.write_text(json.dumps(prompts))
                code, result = self.run_tool()
                self.assertEqual(code, 1)
                self.assertIn("nonempty", result["error"])
        self.prompts.write_text(json.dumps(["common first", "common second"]))
        (self.bundle / "model.safetensors").unlink()
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("missing", result["error"])
        self.profile["classes"] = True
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("classes", result["error"])
        self.profile["classes"] = 3
        self.profile["typo"] = 1
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("unknown profile fields", result["error"])

    def test_nonfinite_mismatch_fallback_and_missing_evidence_fail(self):
        for case in (
            "nonfinite",
            "mismatch",
            "fallback",
            "empty_logits",
            "missing_dispatch",
            "wrong_hash",
            "wrong_artifacts_type",
            "boolean_schema",
        ):
            with self.subTest(case=case):
                (self.bundle / "case.txt").write_text(case)
                code, result = self.run_tool()
                self.assertEqual(code, 1, result)
                self.assertEqual(result["status"], "failed")
        (self.bundle / "case.txt").write_text("ok")
        self.profile.pop("expected_logits")
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("requires", result["error"])

    def test_timeout_is_a_failed_machine_readable_result(self):
        self.profile["timeout_seconds"] = 0.05
        (self.bundle / "case.txt").write_text("timeout")
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("timed out", result["error"])

    def test_different_weights_require_quantization_quality_label(self):
        self.expected["artifacts"]["weights_sha256"] = "f" * 64
        self.write_expected()
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("weights_sha256", result["error"])
        self.profile["comparison_kind"] = "quantization_quality"
        self.profile["quality_thresholds"] = {
            "max_absolute_delta": 0.1,
            "minimum_argmax_agreement": 0.9,
        }
        code, result = self.run_tool()
        self.assertEqual(code, 0, result)
        self.assertTrue(
            all(
                entry["kind"] == "quantization_quality"
                for entry in result["comparisons"]
            )
        )

    def test_loose_tolerance_still_requires_argmax_for_implementation_parity(self):
        self.profile["tolerances"] = {"absolute": 10, "relative": 10}
        (self.bundle / "case.txt").write_text("argmax")
        code, result = self.run_tool()
        self.assertEqual(code, 1, result)
        self.assertEqual(result["comparisons"][0]["argmax_agreement"], 0)
        self.profile["comparison_kind"] = "quantization_quality"
        self.profile["quality_thresholds"] = {
            "max_absolute_delta": 2,
            "minimum_argmax_agreement": 0,
        }
        code, result = self.run_tool()
        self.assertEqual(code, 0, result)
        self.profile["quality_thresholds"]["minimum_argmax_agreement"] = 0.5
        code, result = self.run_tool()
        self.assertEqual(code, 1, result)

    def test_duplicate_profile_fields_and_implicit_quality_thresholds_reject(self):
        profile = self.root / "duplicate.json"
        profile.write_text('{"schema_version":1,"schema_version":1}')
        process = subprocess.run(
            [sys.executable, str(RUNNER), "qualify", str(profile)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("duplicate JSON field", json.loads(process.stdout)["error"])
        self.profile["comparison_kind"] = "quantization_quality"
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("explicit quality_thresholds", result["error"])

    def test_relative_slowdown_checks_same_weight_matching_variant(self):
        code, previous = self.run_tool()
        self.assertEqual(code, 0, previous)
        (self.root / "reference.json").write_text(json.dumps(previous))
        self.profile["reference_run"] = "reference.json"
        self.profile["max_relative_slowdown"] = 1.1
        (self.bundle / "case.txt").write_text("slow")
        code, result = self.run_tool()
        self.assertEqual(code, 1)
        checks = [
            entry
            for entry in result["comparisons"]
            if entry["kind"] == "latency_regression"
        ]
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0]["passed"])
        self.assertEqual(checks[0]["ratio"], 100)


if __name__ == "__main__":
    unittest.main()
