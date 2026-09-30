#!/usr/bin/env bash
# Portable checks and explicit hardware release gates.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    printf '%s\n' 'usage: scripts/check.sh quick|ci|gpu|wasm|browser [ARGS...]|qualify PROFILE|bench PROFILE' >&2
    exit 2
}

require() {
    command -v "$1" >/dev/null 2>&1 || { printf 'Required tool is missing: %s\n' "$1" >&2; exit 1; }
}

qualification_tests() {
    require uv
    uv run --locked --project tools/qualification python -m unittest discover -s tools/qualification/tests -v
}

hillclimb_checks() {
    require uv
    (
        cd tools/hillclimb
        uv run --locked --project ../qualification python -m unittest discover -s tests -v
        uv run --locked --project ../qualification ruff check .
        uv run --locked --project ../qualification ruff format --check .
    )
}

converter_checks() {
    require uv
    (
        cd tools/converters
        uv run --locked pytest
        uv run --locked ruff check .
        uv run --locked ruff format --check .
        uv run --locked python tests/check_rust_compatibility.py --repo-root ../..
    )
}

[[ $# -ge 1 ]] || usage
mode="$1"
shift
require cargo
case "$mode" in
    quick)
        [[ $# -eq 0 ]] || usage
        cargo +1.89.0 fmt --all --check
        cargo +1.89.0 test --locked -p minifield-backend-cpu -p minifield-kernels-simd \
            -p minifield-executor-core -p minifield-json-grammar -p minifield-text-tokenizer
        qualification_tests
        hillclimb_checks
        ;;
    ci)
        [[ $# -eq 0 ]] || usage
        require uv
        require node
        require npm
        node --input-type=module -e 'if (Number(process.versions.node.split(".")[0]) < 22) throw Error("Node 22 or newer is required")'
        cargo +1.89.0 fmt --all --check
        cargo +1.89.0 test --workspace --all-targets --locked
        cargo +1.89.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
        RUSTDOCFLAGS="-D warnings" cargo +1.89.0 doc --workspace --all-features --no-deps --locked
        uv run --locked --project tools/qualification python scripts/check_contracts.py
        uv run --locked --project tools/qualification python scripts/check_repository.py
        cargo +1.89.0 test --manifest-path tools/quant-reference/Cargo.toml --locked
        qualification_tests
        hillclimb_checks
        uv run --locked --project tools/qualification ruff check tools/qualification
        uv run --locked --project tools/qualification ruff format --check tools/qualification
        npm test
        converter_checks
        ;;
    gpu)
        [[ $# -eq 0 ]] || usage
        MINIFIELD_REQUIRE_GPU=1 cargo +1.89.0 test -p minifield-backend-wgpu \
            --features experimental-kernels --locked --lib --test lowbits --test parity \
            -- --test-threads=1
        ;;
    wasm)
        [[ $# -eq 0 ]] || usage
        cargo +1.89.0 check --workspace --target wasm32-unknown-unknown --locked
        printf '%s\n' 'WASM compilation passed. Browser execution requires scripts/check.sh browser.' >&2
        ;;
    browser)
        [[ $# -ge 1 ]] || usage
        require node
        require wasm-bindgen
        cargo +1.89.0 build --release --locked -p minifield-web-demo --target wasm32-unknown-unknown
        wasm-bindgen target/wasm32-unknown-unknown/release/minifield_web_demo.wasm --target web --out-dir web/pkg
        node scripts/check_browser.mjs "$@"
        ;;
    qualify|bench)
        [[ $# -eq 1 ]] || usage
        require uv
        cargo +1.89.0 build --release --locked -p minifield-web-demo --example classify
        uv run --locked --project tools/qualification python tools/qualification/runner.py "$mode" "$1"
        ;;
    *) usage ;;
esac
