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

## Protected evaluation preparation

`prepare-checkpoint` extracts the FP32 `params/` masters from a
`minifield.full-training-state/1` checkpoint. It validates the full-state header
and manifest hash, seeks to each parameter's byte range, and streams inference
output. Optimizer moments and the scalar step aren't loaded or exported.
The input checkpoint and its manifest stay unchanged.

```sh
uv run --locked minifield-convert prepare-checkpoint /absolute/state.safetensors \
  /absolute/classifier-int8 --config /absolute/config.json \
  --tokenizer /absolute/tokenizer.json --precision int8 --mode classifier \
  --classes 8 --source-model evaluation/classifier --source-revision EXACT_REVISION
uv run --locked minifield-convert prepare-checkpoint /absolute/state.safetensors \
  /absolute/pointer-nf4 --config /absolute/encoder-config.json \
  --tokenizer /absolute/encoder-tokenizer.json --precision nf4 \
  --mode pointer-encoder --pointer-width 256 \
  --source-model evaluation/encoder --source-revision EXACT_REVISION \
  --expected-tokenizer-sha256 EXACT_SHA256
uv run --locked minifield-convert package-mixed-qat /absolute/model.safetensors \
  /absolute/qat-bundle --config /absolute/config.json \
  --tokenizer /absolute/tokenizer.json --classes 8 \
  --source-model evaluation/mixed-qat --source-revision EXACT_REVISION
uv run --locked minifield-convert validate /absolute/pointer-nf4
```

Preparation has its own `minifield.prepared-evaluation-bundle/1` provenance
manifest. It records checkpoint, config, tokenizer and output hashes; the
training cursor; normalized parameter names and hashes; and the exact role
precision policy. It refuses existing destinations. Legacy `convert` and
`quantize` keep their original admission and quantization policy.

The supported preparation modes are a causal classifier with an independent
head, and the bidirectional LFM2 MagicBox joint-pointer encoder with 4 output
projections. Encoder `lfm2.*` names become `model.*`; the 4
`magicbox.pointer.{start,end}_{query,key}` masters become
`pointer.{start,end}_{query,key}.weight`. The encoder config retains its
bidirectional architecture and declares the versioned pointer dimensions.

`fp16` stores every master as F16. `int8`, `nf4`, and `ternary` store only
backbone rank-2 projections at the requested precision. Embeddings (including
the tied LM head), classifier and pointer heads, normalization gains, and
convolution taps remain F16. These are stored-weight precisions; runtime
activations and dense arithmetic remain F32. Each precision gets its own
implementation reference over the exact decoded stored weights.

Low-bit preparation emits `minifield.mixed.v1` with an exhaustive
`tensor_quantization` map using `f16`, `ternary-v1`, `nf4-v1`, or `int8-v1`.
The NF4 and ternary algorithms are the same versioned quantizers described
above. The new `minifield.int8.v1` representation uses U8 `[N,K]` bytes with
signed two's-complement codes `-127..127`; byte `128` is reserved. Scales are
F16 `[N,K/128]`. INT8 scale is `absmax/127` rounded to stored F16, and code
selection divides by that stored scale, rounds half away from 0, and clips to
`-127..127`. Decode is F32 signed code times the stored scale. All-zero and
underflowed groups encode 0. This is separate from MFQ8's single-matrix format.

`package-mixed-qat` verifies the same protected-role policy and copies the
supplied weights byte-for-byte. Its config and tokenizer must match the packed
inventory, including any previously pruned vocabulary. It performs no new
quantization, pruning, or training.
