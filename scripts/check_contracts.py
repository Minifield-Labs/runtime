"""Check this repository's pinned contracts and synthetic handoff fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(root, relative):
    path = (root / relative).resolve()
    need(root.resolve() in path.parents and path.is_file(), f"Invalid local file: {relative}")
    return path


def artifact(root, ref):
    path = local_file(root, ref["path"])
    need(digest(path) == ref["sha256"], f"Hash mismatch: {path.name}")
    if "bytes" in ref:
        need(path.stat().st_size == ref["bytes"], f"Size mismatch: {path.name}")
    return path


def validators():
    lock = read(ROOT / "contracts/lock.json")
    pinned = {}
    for contract in lock["contracts"]:
        for path, expected in contract.get("artifacts", {}).items():
            need(digest(local_file(ROOT, path)) == expected, f"Contract artifact pin changed: {path}")
        for path, expected in contract["files"].items():
            need(path not in pinned, f"Duplicate schema pin: {path}")
            pinned[path] = expected
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / "contracts").rglob("*.schema.json")}
    need(set(pinned) == actual, "Schema files and contract lock disagree")
    schemas = {}
    for relative, expected in pinned.items():
        path = local_file(ROOT, relative)
        need(digest(path) == expected, f"Schema pin changed: {relative}")
        value = read(path)
        Draft202012Validator.check_schema(value)
        need(value["$id"] not in schemas, f"Duplicate schema ID: {value['$id']}")
        schemas[value["$id"]] = value
    registry = Registry().with_resources(
        (key, Resource.from_contents(value)) for key, value in schemas.items()
    )
    return {
        key: Draft202012Validator(value, registry=registry, format_checker=FormatChecker())
        for key, value in schemas.items()
    }


VALIDATORS = validators()
NEGATIVE_COUNT = 0


def validate(kind, value):
    VALIDATORS[f"urn:minifield:{kind}:0.1.0"].validate(value)


def rejects(kind, value):
    global NEGATIVE_COUNT
    try:
        validate(kind, value)
    except ValidationError:
        NEGATIVE_COUNT += 1
        return
    raise ValueError(f"Malformed {kind} fixture was accepted")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset(path):
    manifest = read(path)
    validate("dataset", manifest)
    root = path.parent
    product = read(artifact(root, manifest["product"]))
    validate("product", product)
    tool_names = [tool["name"] for tool in product["tools"]]
    need(len(tool_names) == len(set(tool_names)), "Duplicate product tool names")
    for tool in product["tools"]:
        Draft202012Validator.check_schema(tool["input_schema"])
        if "output_schema" in tool:
            Draft202012Validator.check_schema(tool["output_schema"])
    seen, families = set(), {}
    for shard in manifest["shards"]:
        trajectories = rows(artifact(root, shard["trajectories"]))
        judgments = rows(artifact(root, shard["judgments"]))
        need(len(trajectories) == len(judgments) == shard["count"], "Shard counts disagree")
        judged = {}
        for row in judgments:
            validate("judgment", row)
            need(row["trajectory_id"] not in judged, "Duplicate judgment")
            need(row["verdict"] == "accepted" and row["semantic_review"] == "approved",
                 "Export contains an unapproved example")
            need(all(c["passed"] for c in row["checks"]), "Accepted judgment has a failed check")
            need(row["forbidden_effects"] == 0, "Accepted example has forbidden effects")
            judged[row["trajectory_id"]] = row
        shard_ids = set()
        for row in trajectories:
            validate("trajectory", row)
            tid = row["trajectory_id"]
            need(tid not in seen, "Duplicate trajectory")
            need(row["split"] == shard["split"], "Split mismatch")
            need((row["product_id"], row["product_version"]) ==
                 (product["product_id"], product["product_version"]), "Product mismatch")
            for family in [row["task_id"], row["lineage"]["root_task_id"],
                           *row["lineage"]["parent_task_ids"]]:
                need(families.setdefault(family, row["split"]) == row["split"],
                     "Task family crosses splits")
            calls, pending = set(), set()
            for message in row["messages"]:
                if message["role"] == "assistant":
                    need(not pending, "Assistant resumed before tool results")
                    for call in message["tool_calls"]:
                        need(call["id"] not in calls, "Duplicate tool-call ID")
                        calls.add(call["id"])
                        pending.add(call["id"])
                elif message["role"] == "tool":
                    need(message["tool_call_id"] in pending, "Unmatched tool result")
                    pending.remove(message["tool_call_id"])
                else:
                    need(not pending, "Conversation advanced before tool results")
            need(not pending, "Unresolved tool call")
            seen.add(tid)
            shard_ids.add(tid)
        need(shard_ids == set(judged), "Trajectory/judgment IDs disagree")
    malformed = copy.deepcopy(trajectories[0])
    malformed["private_judge"] = {"expected_state": "must stay out"}
    rejects("trajectory", malformed)
    malformed = copy.deepcopy(trajectories[0])
    malformed["schema_version"] = "999.0.0"
    rejects("trajectory", malformed)
    malformed = copy.deepcopy(trajectories[0])
    assistant = next(m for m in malformed["messages"] if m["role"] == "assistant")
    del assistant["trainable"]
    rejects("trajectory", malformed)
    return len(seen)


def bundle(path):
    value = read(path)
    validate("model-bundle", value)
    roles, paths = set(), set()
    for ref in value["files"]:
        need(ref["path"] not in paths, "Duplicate bundle asset path")
        paths.add(ref["path"])
        asset = artifact(path.parent, ref)
        roles.add(ref["role"])
        if ref["role"] == "product_contract":
            product = read(asset)
            need((product["product_id"], product["product_version"]) ==
                 (value["product_id"], value["product_version"]), "Bundle product mismatch")
    need({"weights", "tokenizer", "chat_template", "product_contract"} <= roles,
         "Missing required model-bundle asset")
    if value["purpose"] == "contract_fixture":
        need(value["engine"]["name"] == "fixture", "Fixture claims a real inference engine")
    malformed = copy.deepcopy(value)
    malformed["files"][0]["path"] = "../outside.bin"
    rejects("model-bundle", malformed)
    malformed = copy.deepcopy(value)
    malformed["files"][0]["sha256"] = "unknown"
    rejects("model-bundle", malformed)


def main():
    records = sum(dataset(p) for p in sorted((ROOT / "examples").glob("dataset-*/manifest.json")))
    bundles = list((ROOT / "examples").glob("model-bundle-*/manifest.json"))
    for path in bundles:
        bundle(path)
    protocol = list((ROOT / "examples").glob("environment-*/*.json"))
    for path in protocol:
        value = read(path)
        validate("environment", value)
        malformed = copy.deepcopy(value)
        malformed["protocol_version"] = "999.0.0"
        rejects("environment", malformed)
    print(f"{ROOT.name}: {len(VALIDATORS)} schemas, {records} trajectory fixtures, "
          f"{len(bundles)} bundle fixtures, {len(protocol)} protocol fixtures, "
          f"{NEGATIVE_COUNT} rejection checks passed.")


if __name__ == "__main__":
    main()
