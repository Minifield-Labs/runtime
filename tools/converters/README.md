# Minifield converters

Independent Python tooling for offline LFM2 asset packaging and explicit weight
quantization. Rust builds don't invoke this package. It imports no training,
runtime, or sibling source trees.

`convert` validates a canonical dense LFM2 export and copies its bytes unchanged.
`quantize` applies one declared lossy quantizer to every backbone rank-2 matmul
role, including the tied embedding/head. An explicitly declared classifier head,
norms, and convolution kernels retain their original BF16 or F32 bytes.
Each operation writes deterministic hashes, source
inventory, output inventory, algorithm/version, and provenance digest.

## Commands

Run commands inside this directory:

```sh
uv sync --locked
uv run --locked minifield-convert convert /absolute/source /absolute/bundle
uv run --locked minifield-convert quantize /absolute/source /absolute/ternary-bundle \
  --scheme ternary --source-model LiquidAI/LFM2.5-230M --source-revision EXACT_REVISION
uv run --locked minifield-convert quantize /absolute/source /absolute/nf4-bundle \
  --scheme nf4 --source-model LiquidAI/LFM2.5-230M --source-revision EXACT_REVISION
uv run --locked minifield-convert validate /absolute/bundle
uv run --locked minifield-convert convert /absolute/classifier-source \
  /absolute/classifier-bundle --classes 8
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The source must contain `config.json`, `model.safetensors`, and exactly one
`tokenizer.json` at the root or under `tokenizer/`. Output uses the Rust CLI's
layout: `config.json`, `model.safetensors`, `tokenizer/tokenizer.json`, and
`conversion-manifest.json`. Config and tokenizer formatting stays unchanged.
Missing or known dense `format=pt` safetensors metadata is accepted and preserved
by structural conversion. Unknown format markers are rejected.

An existing destination is refused unless `--overwrite` is explicit. Replacement
only accepts a converter-managed directory containing its known files. Source
and output directories mustn't overlap. Symlink assets and path components are
rejected; resolve a platform alias such as macOS `/tmp` to `/private/tmp` first.
Failures leave the existing bundle intact and remove pending output.

## Supported scope

The admitted architecture is `model_type=lfm2`, tied embedding/LM head, SwiGLU,
default unscaled RoPE, bias-free convolution, and `conv`/`full_attention` layers.
Configuration aliases and effective FF dimension adjustment follow the runtime.
The source tensor inventory must match the config exactly; storage is uniformly
F32 or BF16 as declared by `dtype`. `--classes N` explicitly admits an independent
dense `classification_head.weight` `[N,hidden_size]`; the manifest records this
mode and validation reads it from the manifest. Omit the flag for an LM bundle.
The classifier backbone still requires the supported tied-embedding config.
We reject unknown architectures, F16 dense sources, shards, adapters, separate
untied heads, packed source inputs, mixed quantization, and general checkpoint
recovery.

Tokenizer admission supports the pinned Split/ByteLevel/BPE/BOS-only profile.
These checks establish supported structure and hashes. Model execution and
quality require validation through the Rust loader and the delivered model.

This package's `minifield.converter-bundle/1` manifest describes inference
assets and conversion provenance. The training-owned `model-bundle/0.1.0`
release contract also requires product and training identities, chat template,
and product contract. Those aren't supplied by the 3 canonical assets; this
package doesn't create or claim a complete product release.

## Packed contracts and quantizers

`minifield.ternary.v1` stores U8 `[N,K/4]` codes and F16 `[N,K/128]` scales.
Weight 0 occupies the lowest 2 bits; codes 0, 1, 2 decode to -1, 0, +1 times
the stored scale. Code 3 is rejected. `minifield.nf4.v1` stores U8 `[N,K/2]`
codes, low nibble first, using the exact runtime NF4 codebook and the same
scales. Both require non-empty rank-2 matrices and `K % 128 == 0`.

The offline algorithms are versioned separately from those storage contracts:

- `ternary-absmax-stored-fp16/1`: FP32 source values; group absmax rounded to
  FP16; FP64 divide by the stored scale rounded to FP32; half-away-from-zero
  rounding and clipping. This matches the inspected training exporter.
- `nf4-absmax-stored-fp16/1`: the same stored scale; FP32 normalized distances
  to the pinned NF4 levels; ties choose the lower index.

An all-zero or FP16-underflowed scale group gets scale 0 and zero-level codes.
FP16 scale overflow and non-finite weights are rejected. `pack_codes`,
`unpack_codes`, and `dequantize` provide lossless code-stream operations;
`quantize` explicitly changes weight values. MFQ8 is a separate single-matrix
INT8 parity format and isn't emitted as an LFM2 bundle. No QAT, optimizer state,
training, KV-cache quantization, or external-format adapters are included.

Quantization processes one source matrix at a time and retains output streams
until writing. The converter caps weights at 8 GiB and safetensors headers at
8 MiB; memory still depends on the largest source matrix and total output.

## Synthetic fixture and Rust integration

```sh
uv run --locked minifield-convert fixture /private/tmp/lfm2-converter-source
uv run --locked minifield-convert convert /private/tmp/lfm2-converter-source \
  /private/tmp/lfm2-converter-dense
uv run --locked minifield-convert quantize /private/tmp/lfm2-converter-source \
  /private/tmp/lfm2-converter-ternary --scheme ternary \
  --source-model synthetic/lfm2 --source-revision converter-fixture-v1
uv run --locked minifield-convert quantize /private/tmp/lfm2-converter-source \
  /private/tmp/lfm2-converter-nf4 --scheme nf4 \
  --source-model synthetic/lfm2 --source-revision converter-fixture-v1
uv run --locked minifield-convert fixture /private/tmp/lfm2-classifier-source --classes 8
uv run --locked minifield-convert convert /private/tmp/lfm2-classifier-source \
  /private/tmp/lfm2-classifier-dense --classes 8
uv run --locked python tests/check_rust_compatibility.py --repo-root ../..
```

The generator in `minifield_converters.fixture` emits a deterministic synthetic
2-layer conv/attention model with hidden and FF widths 128 and vocabulary 32.
Its weights are hand-defined ternary values with scale 0.125, so both ternary
and NF4 packing/decode are exact. `--classes 8` adds a dense 8-class head for
browser qualification without private weights. The tokenizer encodes `ab` as
`[3]` and optional BOS as `[1,3]`. It contains no trained weights
or customer data. Generate files outside Git; the package tests use temporary
directories and source-defined golden bytes.

The explicit compatibility command generates LM and classifier inputs and their
dense, ternary, and NF4 bundles in a temporary directory. It checks original vs
dense asset hashes, exact Python decode, actual Rust loader/inference logits
against the dense baseline (`rtol=2e-5`, `atol=2e-5`), and cached append vs fresh
prefill. It requires Cargo and fails if Rust cannot build or run; pytest doesn't
invoke it. Run it as a separate CI step.

The CPU example is a development probe with config capped at 1 MiB, weights at
16 MiB, and backend storage at 64 MiB:

```sh
cargo run --locked -p minifield-executor-core --example converter_probe -- \
  /private/tmp/lfm2-classifier-dense --classes 8
```

Run the Cargo command from the repository root. Omit `--classes` for an LM. JSON
contains prompt `[1,3]` and appended `[4]` logits. LM output includes actual base
and appended prefix length/history. Classifier output includes its reusable base
state and effective append sequence; the classifier's branch prefix is private.
The probe uses the standard checked loader and CPU backend and needs no Python.
