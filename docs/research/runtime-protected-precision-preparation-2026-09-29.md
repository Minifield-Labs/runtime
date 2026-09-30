# Protected precision for the runtime hill-climb

## Hypothesis

Runtime parity needs the actual deployed model families, with a fixed reference
for each stored precision. A runtime change should preserve the predictions and
floating outputs of the same artifact. Comparing separately quantized artifacts
would mix implementation errors with model quality changes.

The supplied classifier and MagicBox encoder are full training states. Their
files contain FP32 masters, 2 optimizer moment arrays for each master, and a
scalar update counter. Loading those files as inference assets would retain
irrelevant state and obscure the identity of the weights being measured.

The supplied production QAT artifact already has a deliberate precision split:
ternary backbone projections, BF16 embeddings and output head, and a previously
pruned vocabulary. Its bytes and matching tokenizer are a distinct evaluation
case.

## Fix

Added an explicit offline preparation path in `tools/converters`. It validates
the full-state safetensors header and training manifest, then reads only
`params/` byte ranges. The writer declares the physical inventory up front and
streams one parameter's transformed output at a time. It never allocates the
optimizer arrays and never overwrites an original checkpoint.

Preparation emits FP16, INT8, NF4, and ternary variants. Low-bit variants protect
embeddings, the tied vocabulary head, classifier/pointer heads, norms, and
convolution taps at F16. QAT packaging keeps the supplied BF16 and packed bytes
unchanged. The legacy structural converter and its quantization defaults stay
compatible.

The new INT8 contract is explicit: U8 row-major signed two's-complement codes,
`-127..127`, with reserved byte `128`; F16 group-128 scales; F32 decode. The
quantizer stores `absmax/127` in F16 before choosing codes. It rounds half away
from 0 and clips to the admitted range. NF4 and ternary reuse their existing
versioned algorithms.

MagicBox preparation normalizes `lfm2.*` to `model.*` and names the 4 high
precision heads `pointer.{start,end}_{query,key}.weight`. It preserves the
bidirectional architecture and declares the pointer contract separately from
the causal classifier.

Each generated bundle records its source file hashes, normalized source tensor
hashes, output inventory, role precision, training cursor, and provenance digest
under `minifield.prepared-evaluation-bundle/1`. Those inference assets aren't
presented as a complete training-owned product release.

## Test

The converter suite passes 77 tests. New hand-defined cases establish INT8
signed byte order, positive and negative half ties, stored scale semantics,
underflow, and the reserved `-128` code. Full-state fixtures verify that only
parameter masters are read, protected roles remain dense F16, output head names
are normalized, and optimizers never enter inference output.

The suite also rejects nonfinite masters, F16 overflow in protected roles,
extra parameters, stale checkpoint hashes, and wrong tokenizer pins. Failed
preparation removes pending output. Mixed QAT packaging verifies exact weight
byte equality and rejects mismatched head dimensions. Ruff lint and formatting
checks pass.

Prepared all 8 precision variants and the supplied QAT artifact. Every bundle
passed inventory, finite-value, packed-code, asset-hash, and provenance
validation. The encoder tokenizer was fetched at the published pinned revision
and matched SHA-256
`1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711`.

| Model | Stored precision | Weight file bytes |
| --- | --- | ---: |
| Model100k classifier | FP16 | 459,417,456 |
| Model100k classifier | INT8 + protected F16 | 299,444,936 |
| Model100k classifier | NF4 + protected F16 | 218,180,144 |
| Model100k classifier | Ternary + protected F16 | 177,548,144 |
| MagicBox encoder | FP16 | 711,081,864 |
| MagicBox encoder | INT8 + protected F16 | 428,280,648 |
| MagicBox encoder | NF4 + protected F16 | 284,625,568 |
| MagicBox encoder | Ternary + protected F16 | 212,798,480 |
| Supplied production QAT classifier | Original mixed precision | 60,343,032 |

These sizes describe model files. The QAT model has a different, previously
pruned vocabulary and is evaluated independently. Each newly prepared model's
65,536 by 1,024 protected embedding occupies 134,217,728 bytes in F16.

The original classifier checkpoint hash is
`93893515c830f5248c88159f1293e8b9609d3a0d7ff09e287849ba17803e59ed`.
The original MagicBox checkpoint hash is
`6a39c95e18fa5e8abb0926f7cb4c177ca0b8c1c0bdb682fe3615402494ee167a`.
The copied QAT weight hash is
`9f72504f3dc674352b6eaab0198f6966ff81e6c0b513999ce18142911b7f8e0b`,
identical to the supplied file.

Generated assets, preparation logs, per-tensor provenance, and the complete
size/hash index live outside Git in the dated hill-climb experiment. Its
`prepare_models.py` consumes the installed converter package. No runtime or
training sibling source import is part of the conversion tool.

The checkpoint manifests don’t include a training Git revision. A provenance
review caught an initial use of the inspected math commit as the checkpoint
revision. The final source identity uses the checkpoint SHA-256 and recorded
run/source IDs; inspected code revisions are separate. Low-bit headers were
rewritten atomically after reference computation, and every stored tensor’s
SHA-256 stayed identical. The correction adds 32 header bytes to each new
low-bit file. Its full before/after proof is archived with the experiment.

## Resolution

The evaluation inputs now have reproducible precision boundaries. Embeddings
and output heads retain the requested information-bearing precision, optimizer
state stays outside inference, and the supplied QAT artifact is preserved.

This establishes representation and artifact identity. Runtime admission,
mathematical output parity, and latency qualification remain separate gates.
Every candidate is compared with a reference for its exact model, stored
precision, and backend. The FP16 label describes weight storage; the current
runtime expands dense values and computes activations in F32.

## Source pins

- Training checkpoint contract and MagicBox math:
  [`96fd486d1601bb910b390652bb21d53463e8237a`](https://github.com/Minifield-Labs/minifield-training/commit/96fd486d1601bb910b390652bb21d53463e8237a).
- Published encoder assets:
  [`LiquidAI/LFM2.5-Encoder-350M`, revision `b886781f7c6f10ca9b7096e21b83e30a073c2f39`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/tree/b886781f7c6f10ca9b7096e21b83e30a073c2f39).
- Precision policy: `minifield/protected-embedding-and-output-heads/1`.
- Preparation provenance: `minifield.prepared-evaluation-bundle/1`.
