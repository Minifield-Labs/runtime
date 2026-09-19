#!/usr/bin/env python3
"""Build manifest.json for the T2 llama.cpp oracle fixtures.

Reads the artifacts produced by run-oracle.sh under tmp/oracle/out/ and
records: model identity + sha256, llama.cpp tag/commit/build config,
tool versions, and per-prompt prompt token ids, generated ids, output
hashes, BOS behavior, and any emitted id >= 64402.

Stdlib only. Run after run-oracle.sh steps 1-4.
"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "oracle"
ORACLE = ROOT / "tmp" / "oracle"
LLAMA = ROOT / "tmp" / "llama.cpp"
GGUF = ORACLE / "LFM2.5-230M-BF16.gguf"
OUT = ORACLE / "out"
PROMPTS = TOOLS / "prompts"

EXPECTED_GGUF_SHA256 = "9a47cffa8c86d071e4cdb2adf6861251fecd6191c226c88607a193a0d9cc5a38"
FIRST_UNMAPPED_MODEL_TOKEN_ID = 64402  # crates/text-tokenizer: tokenizer rejects ids >= this


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(LLAMA), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def parse_kv_file(path: Path) -> dict:
    """Parse 'key=value' lines written by dump_logits."""
    kv = {}
    if not path.exists():
        return kv
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def prompt_ids_from_stderr(path: Path) -> list[int]:
    """llama-completion --verbose-prompt prints `%6d -> 'piece'` lines."""
    ids = []
    if not path.exists():
        return ids
    for m in re.finditer(r"^\s*(\d+) -> '", path.read_text(errors="replace"), re.M):
        ids.append(int(m.group(1)))
    return ids


def main() -> int:
    prompts = ["p01", "p02", "p03", "p04", "p05", "p06"]

    head = git("rev-parse", "HEAD")
    tags = git("tag", "--points-at", "HEAD").splitlines()

    gguf_sha = sha256_file(GGUF) if GGUF.exists() else None
    manifest = {
        "status": "complete" if gguf_sha else "pending_execution",
        "purpose": "llama.cpp oracle fixtures for Rust LFM2.5-230M executor validation (task T2)",
        "model": {
            "file": "LFM2.5-230M-BF16.gguf",
            "source": "https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-BF16.gguf",
            "hf_repo_commit": "cdf97bd8205908758f44aec508d68ac1aef98f5c",
            "expected_sha256_lfs": EXPECTED_GGUF_SHA256,
            "expected_size_bytes": 461884256,
            "sha256": gguf_sha,
            "sha256_matches_lfs": (gguf_sha == EXPECTED_GGUF_SHA256) if gguf_sha else None,
            "size_bytes": GGUF.stat().st_size if GGUF.exists() else None,
        },
        "model_facts": {
            "architecture": "lfm2",
            "vocab_size": 65536,
            "n_layers": 14,
            "dtype": "bf16",
            "bos_token_id": 1,
            "eos_token_id": 7,
            "pad_token_id": 0,
            "first_unmapped_model_token_id": FIRST_UNMAPPED_MODEL_TOKEN_ID,
        },
        "llama_cpp": {
            "repo": "https://github.com/ggml-org/llama.cpp",
            "tag": tags[0] if tags else "b11046",
            "commit": head,
            "cmake_flags": "-DGGML_METAL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release",
            "binary_used": "llama-completion",
            "binary_note": (
                "in b11046 `llama-cli` is a chat REPL that applies the chat "
                "template; `llama-completion` is the classic -f/-n/--verbose-prompt "
                "batch interface used for the oracle"
            ),
            "run_flags": (
                "-t 1 -ngl 0 --temp 0 --top-k 1 --seed 42 -ctk f32 -ctv f32 "
                "-f <prompt> -n 128 --no-warmup --no-display-prompt --verbose-prompt"
            ),
        },
        "tool_versions": {
            "dump_logits": "scripts/oracle/dump_logits.c (links built libllama; f32 KV; "
                           "add_special=true parse_special=true)",
            "logits_format": "u32 n_tokens, u32 n_vocab, then f32[n_tokens][n_vocab] "
                             "little-endian rows via llama_get_logits_ith",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "prompts": {},
        "anomalies": [],
    }

    for p in prompts:
        raw = (PROMPTS / f"{p}.txt").read_bytes()
        cli_kv = OUT / f"{p}.stderr.txt"
        dl_kv = parse_kv_file(OUT / f"{p}.logits.stderr.txt")
        gen_kv = parse_kv_file(OUT / f"{p}.logits.f32.gen.txt")
        logits = OUT / f"{p}.logits.f32"
        tokens_txt = OUT / f"{p}.tokens.txt"

        cli_ids = prompt_ids_from_stderr(cli_kv)
        dl_ids = [int(x) for x in dl_kv.get("prompt_token_ids", "").split(",") if x]
        emitted = [int(x) for x in gen_kv.get("emitted_ids", "").split(",") if x]

        entry = {
            "prompt_file": f"scripts/oracle/prompts/{p}.txt",
            "prompt_text": raw.decode("utf-8", errors="replace"),
            "prompt_sha256": hashlib.sha256(raw).hexdigest(),
            "prompt_token_ids_llama_cli": cli_ids,
            "prompt_token_ids_dump_logits": dl_ids,
            "prompt_ids_agree": (cli_ids == dl_ids) if (cli_ids and dl_ids) else None,
            "bos_prepended": dl_kv.get("bos_prepended"),
            "vocab_add_bos_flag": dl_kv.get("vocab_add_bos"),
            "n_prompt_tokens": int((dl_kv.get("n_prompt_tokens", "0") or "0").split()[0]),
            "n_generated": int(gen_kv.get("n_generated", "0") or 0),
            "emitted_ids": emitted,
            "stopped_on_eog": gen_kv.get("stopped_on_eog"),
            "max_emitted_id": int(gen_kv.get("max_emitted_id", "0") or 0),
            "any_emitted_id_ge_64402": gen_kv.get("any_emitted_id_ge_64402"),
            "emitted_text_dump_logits": gen_kv.get("emitted_text"),
            "generated_text_llama_cli": tokens_txt.read_text(errors="replace")
            if tokens_txt.exists() else None,
            "logits_file": f"out/{p}.logits.f32",
            "logits_sha256": sha256_file(logits) if logits.exists() else None,
            "logits_bytes": logits.stat().st_size if logits.exists() else None,
            "tokens_file": f"out/{p}.tokens.txt",
            "stderr_file": f"out/{p}.stderr.txt",
        }

        if entry["max_emitted_id"] >= FIRST_UNMAPPED_MODEL_TOKEN_ID:
            manifest["anomalies"].append(
                f"{p}: emitted token id {entry['max_emitted_id']} >= "
                f"{FIRST_UNMAPPED_MODEL_TOKEN_ID} (Rust tokenizer rejects)"
            )
        if entry["prompt_ids_agree"] is False:
            manifest["anomalies"].append(f"{p}: llama-cli and dump_logits prompt ids differ")

        manifest["prompts"][p] = entry

    (ORACLE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {ORACLE / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
