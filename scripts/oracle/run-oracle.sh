#!/usr/bin/env bash
# run-oracle.sh — T2 oracle capture for the Rust LFM2.5-230M executor.
#
# Downloads the published BF16 GGUF, builds llama.cpp CPU-only plus the
# dump_logits tool, captures per-prompt raw f32 logits and greedy token
# streams, then writes manifest.json. All artifacts stay under tmp/
# (gitignored); this script's sources live in scripts/oracle/.
#
# Steps: download + verify GGUF -> clone + build llama.cpp -> build logits
# tool -> per-prompt llama-completion + logits capture -> manifest.json.

set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TOOLS/../.." && pwd)"
ORACLE="$ROOT/tmp/oracle"
LLAMA="$ROOT/tmp/llama.cpp"
GGUF="$ORACLE/LFM2.5-230M-BF16.gguf"
EXPECTED_SHA256="9a47cffa8c86d071e4cdb2adf6861251fecd6191c226c88607a193a0d9cc5a38"
EXPECTED_SIZE=461884256
HF_URL="https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-BF16.gguf"
LLAMA_TAG="b11046"

mkdir -p "$ORACLE/out"

echo "=== step 1: download GGUF (~441 MB) ==="
if [ ! -f "$GGUF" ]; then
    curl -L --fail --progress-bar -o "$GGUF" "$HF_URL"
fi
SIZE=$(stat -f%z "$GGUF" 2>/dev/null || stat -c%s "$GGUF")
SHA=$(shasum -a 256 "$GGUF" | awk '{print $1}')
echo "size=$SIZE sha256=$SHA"
[ "$SIZE" = "$EXPECTED_SIZE" ] || { echo "SIZE MISMATCH: got $SIZE want $EXPECTED_SIZE"; exit 1; }
[ "$SHA" = "$EXPECTED_SHA256" ] || { echo "SHA MISMATCH: got $SHA want $EXPECTED_SHA256"; exit 1; }

echo "=== step 2: clone + build llama.cpp (CPU-only, $LLAMA_TAG) ==="
if [ ! -d "$LLAMA/.git" ]; then
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
# NOTE: in b11046, `llama-cli` is a chat REPL that applies the chat template.
# The classic batch interface (-f, -n, --verbose-prompt) is `llama-completion`.
cmake -B "$LLAMA/build" -S "$LLAMA" \
    -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA/build" -j --target llama-completion
BIN="$LLAMA/build/bin/llama-completion"
"$BIN" --version || true

echo "=== step 3: build dump_logits tool ==="
cc -O2 -std=c11 -c "$TOOLS/dump_logits.c" -o "$ORACLE/dump_logits.o" \
    -I"$LLAMA/include" -I"$LLAMA/ggml/include"
# link with clang++ (libllama is C++; shared libs carry their own deps via rpath)
clang++ -o "$ORACLE/dump_logits" "$ORACLE/dump_logits.o" \
    -L"$LLAMA/build/bin" -lllama \
    -L"$LLAMA/build/ggml/src" -lggml-cpu -lggml-base -lggml \
    -Wl,-rpath,"$LLAMA/build/bin" -Wl,-rpath,"$LLAMA/build/ggml/src"

echo "=== step 4: per-prompt captures ==="
for p in p01 p02 p03 p04 p05 p06; do
    echo "--- $p ---"
    "$BIN" -m "$GGUF" -t 1 -ngl 0 --temp 0 --top-k 1 --seed 42 \
        -ctk f32 -ctv f32 -f "$TOOLS/prompts/$p.txt" -n 128 \
        --no-warmup --no-display-prompt --verbose-prompt \
        2> "$ORACLE/out/$p.stderr.txt" > "$ORACLE/out/$p.tokens.txt"
    "$ORACLE/dump_logits" "$GGUF" "$TOOLS/prompts/$p.txt" "$ORACLE/out/$p.logits.f32" 128 \
        2> "$ORACLE/out/$p.logits.stderr.txt"
done

echo "=== step 5: hashes + manifest ==="
shasum -a 256 "$ORACLE"/out/*.logits.f32 "$ORACLE"/out/*.tokens.txt
python3 "$TOOLS/build_manifest.py"

echo "=== done. manifest at $ORACLE/manifest.json ==="
