# Native packed tile8 candidate

The native candidate introduces one fixed 8×32×32 cooperative family with 256 threads. Formatting, strict Clippy, portable checks, quick/CI/WASM and all 17 native hardware tests passed. Actual-model qualification and frozen performance screening are pending.

This branch follows WGPU PR25's report child, `90242f27af533c288b5a32ca22de2e22a7270b8d`. Its first commit, `9a3780a5cb15f094681c71ee088ae58799eaab79`, restores all 28 closed WGPU implementation/test/check paths to exact `415007ea32509ff72cd130d8c4de87860f75e74e` presence, blobs and modes. The full tracked tree equals that scalar baseline except the preserved WGPU experiment report; the native implementation follows separately.

Root's independently reviewed qualification-only amendment released the prior scheduling hold after WGPU's rejected screen, audit, report bridge and publication completed. Both native timing controls remain inconclusive, and the original MagicBox ternary independent numerical failure remains visible. Frozen assets, criteria, host, runner, references, dependencies, Cargo files and environment policy stay unchanged; screening requires a separate checked-evidence admission.

The precommitted mathematical contract is the repaired `prove_native_tile8_layout.py`, SHA256 `72a5a861c0b00a0c797ff07977aeed869ef8e16afbc94749874456f0fc0c2a2f`. Its active evidence SHA256 is `3ff4dc7d950aad0f80df8c667ac47373f2899113d8e890ea04b21a28a693cda8`, with canonical result SHA256 `2753516750795b0c2485bf0c90851b7114ee2d7c337fe6bf4b33dc3bcc7045d5`. Those retained outside-Git artifacts establish a bounded address/provenance model. They don't qualify this runtime source or compiled shader. The original T4 proof's broad input-oracle claims were withdrawn separately; it doesn't govern admission.

## Hypothesis

The 8-row choice doubles source-level decoded-weight reuse versus the earlier 4-row proposal while adding 512 declared shared bytes. The design keeps one output per thread and one ascending-K accumulator per output, with two accumulators for a pair. No speed or occupancy conclusion follows from that source structure.

| Item | Implementation contract |
|---|---|
| Tile | T8, N32, K32 |
| Complete group | `(256,1,1)` |
| Group grid | `(ceil(N/32),ceil(T/8),1)` |
| Input storage | F32 `[8][32]` |
| Weight storage | F32 `[32][33]`, one array per matrix |
| Declared shared bytes | 5,248 single; 9,472 pair |
| Barriers per group | `2*(K/32)` |
| Selection | `8<=T<=512`, `32<=N<=8192`, `128<=K<=8192`, `K%128==0` |

## Fix

`kernels.rs` owns a finite typed registry of the 15 existing kernels and the two new names, `packed_linear_tile8` and `packed_pair_tile8`. Existing entry-point and raw-counter names remain stable. The bridge compiles the baseline and tiled MSL sources into one independent native library, preserving the canonical NF4 constants and SiLU expression. It retains the existing `fastMathEnabled=false` policy.

`packed.rs` owns canonical format inference, four semantic modes, checked complete-group geometry and scalar/tile selection. Modes remain plain single, input-SwiGLU single, plain pair and paired-SwiGLU epilogue. Format IDs remain 0=ternary/P4, 1=NF4/P2, 2=INT8/P1. Both pair branches retain independent packing and scales, including all nine format combinations with distinct same-format streams.

The stored representation remains row-major U8 codes and group-128 source F16 scales converted to GPU F32. Ternary is LSB-first `(symbol-1)`, NF4 is low nibble first, and INT8 uses signed two's-complement bytes. Canonical artifact admission excludes ternary symbol3, INT8 byte128, nonfinite scales and negative nonzero scales. The backend retains its existing tensor/ownership validation boundary; it doesn't rescan or repack artifact contents.

The bridge queries execution width, maximum total threads, static threadgroup memory and device threadgroup capacity from the actual compiled objects at construction. Selection uses the cached observed values for the specific single/pair pipeline. A tile requires execution width32, at least256 threads, and reported static memory within device capacity. There is no lower bound tying actual allocation to declared source-array totals, and no dynamic threadgroup allocation.

Those limits are pipeline-specific and per-group. They don't establish per-core occupancy or shared-memory bank behavior. Apple documents the compiled [thread limit](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup), [execution width](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth), [static allocation](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/staticthreadgroupmemorylength) and [device capacity](https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength). The Objective-C calls compiled against `objc2-metal 0.3.2`, and hardware construction retained both actual pipeline records.

For local thread `q`, token slot is `q/32` and output slot is `q%32`. Each thread stages one activation. A packed-byte owner iterates `j=q; j<32*(32/P); j+=256`, loading consecutive canonical K bytes within an output row and decoding them into `weights[k][output]` with stride33. Component `c` writes local K `P*(j%(32/P))+c`; global scale is `row*(K/128)+k_base/128`. K32 tiles never cross a scale block.

Invalid token slots stage zero without reading input/gate/up. Invalid output columns stage zero without code/scale reads. The padding column32 is untouched. All threads reach both barriers, including masked rows and columns, before a final guarded write. The second barrier protects shared arrays against the next tile's overwrite. The bridge uses explicit `dispatchThreadgroups` with full256 groups for tiled entries; it retains the baseline clamped flat-grid path for scalar kernels.

Each decoded coefficient is multiplied by its F32 scale before staging and activation multiplication. Each output accumulates ascending K32 chunks and ascending local K. Input fusion stages `SiLU(gate)*up`; pair epilogue applies `SiLU(sum_a)*sum_b` only after both reductions. The fused pair aliases the unused B output to A and issues no separate B store.

The existing 16-word parameter block stays in slot8. Single begins `[T,N,K,format,input_fusion]`, with bindings output/input-or-gate/codes/scales/up in slots0..4. Pair begins `[T,N,K,format_a,format_b,epilogue_fusion]`, with output_a/output_b/input/codes_a/scales_a/codes_b/scales_b in slots0..6. Plain single retains input as its unused-up binding.

All tensor, alias and parameter validation completes before packed selection and recording. The common encoding preflight rejects incompatible kernel/grid kinds and counter overflow before creating a command or extending retention. Buffers still flow through the existing allocation references, batch retention, submissions, fences and retirement. No extra device buffer, repack, pool or resource-accounting class is introduced.

Unsupported tile shapes/capabilities route through the original validated scalar kernels. T7 falls back, T8 fills one token group and T9 masks the second group. Numerical core code reads no environment settings. Host dispatch aggregation already admits new raw names without kernel-name pins; one packed operation still records one dispatch.

## Test

The 15 passing portable tests cover registry identity/source definitions, all four routing modes, T7/T8/T9, shape and u32/address limits, compiled255/256 boundaries, actual-memory admission without a declared lower bound, canonical format inference, shared-array ownership and independent flat-input/weight consumer coordinates. The capability unit stays explicitly ignored in portable checks and ran separately on hardware.

All four ignored `tests/packed_tile8.rs` hardware cases passed with real native construction and MSL compilation. Their loops cover 308 packed operation calls: 236 tiled calls and 72 scalar threshold fallbacks, across all formats/pairs/fusions, N33/K128/256/384 boundaries, deployment token lengths and actual matrix dimensions. Each call asserts an exact raw-counter delta, so an unexpected reference-device fallback fails qualification.

Qualification planning found that those dispatch assertions prove eligibility without retaining the four observed pipeline-limit numbers. The separate ignored private unit passed and retained two JSON records from the compiled capability cache, one for each tiled pipeline. The hardware gate selects `--lib` and `--nocapture`, preserving these records alongside all 12 existing parity tests and the four new integration tests.

| Actual pipeline | Registry ID | Execution width | Maximum threads | Static bytes | Device threadgroup bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `packed_linear_tile8` | 4294970055 | 32 | 1024 | 5248 | 32768 |
| `packed_pair_tile8` | 4294970055 | 32 | 1024 | 9472 | 32768 |

Numerical references decode canonical bytes and scales directly, with F32 scale-before-product, ascending-K accumulation and the same SiLU expression. Positive non-power F16 scales, zero/extreme values, exact cancellation, group boundaries and sentinel outputs are included. Primitive comparisons use absolute `1e-4` plus relative `1e-4`; no frozen numerical guard changes.

Static test review found that the original six-value scale palette advanced by three positions per K group, repeating every two groups. A shader using `(column/128)%2` could therefore pass its scale checks. The corrected source assigns positive finite F16 bits `0x2e01 + bounded_row_seed_offset + 32*group`, with distinct scales across every group at every fixture width. The offset is 0..127 and the group is 0..63, keeping the generated bits within `0x2e01..=0x3660`. Source assertions check full sequence uniqueness and explicitly distinguish groups 0/1/2 at K384. The constant zero, maximum and minimum-subnormal special rows remain separate. The corrected fixture passed actual MSL compilation and numerical comparison against the ordered scalar reference.

The bounded row offset repeats after 128 ordinary rows, but the zero/maximum/subnormal exceptions expose a scale-only `row%128` fault in the existing actual-width fixture. Static tracing of the N512/K1024/T9 ternary case gives row133/token1 a correct value of `-0.0139007568359375`; aliasing its scales to maximum-scale row5 gives `-16376`. That's a manually traced fixture distinction; the containing hardware case passed against the ordered scalar reference.

Actual-shaped synthetic T9 fixtures cover the seven unique packed `[N,K]` pairs: `[512,1024]`, `[1024,1024]`, `[1024,2560]`, `[2560,1024]`, `[3072,1024]`, `[1024,4608]`, `[4608,1024]`. One T345/N2560/K1024 FFN fixture uses impulses at K127/128. Shapes came from small classifier/pointer conversion-manifest metadata; no real tensor contents are included. Sparse references omit finite zero products while retaining ascending order of nonzero terms, keeping fixture reference work bounded. Full actual-model qualification remains separate and required.

## Resolution

The local qualification gates passed: formatting, 15 portable tests, hardware-source compilation, strict Clippy, native evaluation-host compilation, repository quick/CI/WASM and all 17 native hardware tests. The restored scalar WGPU hardware gate passed 55 tests, with its 2 existing development benchmarks still ignored. Independent review of the final implementation found no substantive correctness, ownership or resource issue; packed artifact bytes and the numerical contracts remain unchanged.

The initial formatting failure and two Clippy `collapsible_if` findings remain in the evidence. Root applied rustfmt and flattened the two guards while preserving capability-before-shape and geometry-before-group evaluation. The first quick check hit sandbox DNS restrictions while fetching locked Python dependencies; the same gate passed with network access and fresh logs, using an isolated CPython 3.13.5 environment for portable checks only.

The next gates bind a clean reviewed implementation, its complete source closure and a fresh release executable to all 14 original native model anchors. Preserve every independent NumPy diagnostic, then separately admit the unchanged full native screen and eligible fresh confirmation. Model parity, performance and final adoption remain unproven at this implementation checkpoint; actual resolution belongs in the report-only child after measurement.
