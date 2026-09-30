# WGPU signed-INT8 coherent word loader

Status: rejected by the frozen 14-cell WGPU performance screen on September 30, 2026. Local CI/GPU/WASM, all 14 same-precision real-model anchor checks and 5 browser lanes passed. The independent audit reproduced the rejection; confirmation and promotion weren't admitted. Measured implementation: `3511a572f49bbd1e34f741e408cc457c10f0787e`.

## Hypothesis

The existing signed-INT8 projection loader extracts 1 byte per coefficient, repeating a U32 code-word and group-128 scale access for each of its 4 bytes. Loading 1 word and 1 scale, then staging 4 decoded F32 coefficients, could reduce redundant loader work across the existing 4 projection compositions.

The proposed source-level access count falls from 512 to 128 word/scale reads per valid 32-column K16 weight tile, for each weight stream. Compiler elimination, GPU transactions, cache behavior, shared-memory scheduling and latency require measurement. That count doesn't establish a speedup.

The frozen measured/controller baseline is `bf32d7a`. This feature worktree starts at `415007ea32509ff72cd130d8c4de87860f75e74e`, which adds browser startup qualification plumbing and retains the same backend source. The implementation changes `crates/backend-wgpu/`, this research note and root's addition of the new hardware test target to `scripts/check.sh gpu`. Native Metal, executor, host, controller, artifact bytes, public APIs and allocation contracts stay outside its scope.

## Fix

The shared F32 arithmetic template is named `packed_prefill.wgsl`, reflecting its existing use by the entire packed projection family. It has 1 finite weight-loader insertion point. NF4/ternary receive the original scalar load text. INT8 receives the coherent word loader plus explicit vec4 accessors in its 4 binding headers. A shared decoder extracts low byte first, sign-extends 2's-complement bytes and multiplies each coefficient by its F32 scale before any activation product.

For a local invocation `(x,y)` in the existing 16×16 workgroup:

```text
L = x + 16*y
active weight owners: L < 128
c = L / 4
q = 4*(L % 4)
row = col0 + c
word index = row*(K/4) + (base+q)/4
scale index = row*(K/128) + (base+q)/128
shared index for component j = (q+j)*32 + c, j = 0..3
```

Those owners stage every 512-slot weight tile once. Invalid N columns stage zeros without reading either buffer. The explicit whole-word K guard is safe because operand admission derives K from the scale width in groups of 128; every admitted positive K is divisible by 128, 16 and 4. Quartets can't cross a scale group.

All 256 invocations stage activations and reach both barriers. The output tile remains 64×32, reduction traversal remains K16 with ascending `base` and `d`, and each invocation retains its existing 4×2 output fragment. Shared stride32, F32 scaling, MAC expressions, SwiGLU expressions, output stores, bindings, access masks and optional vector aliases remain fixed. No geometry variants, alternate loader controls, GEMV path, dense path or repack are introduced.

| Composition | Weight streams | Storage bindings | Read-only mask |
| --- | --- | --- | --- |
| Single projection | A | 5 | `0b11_1100` |
| Paired projection | A, B | 8 | `0b1_1111_1000` |
| SwiGLU input then projection | A; separate gate/up | 7 | `0b1111_1100` |
| Paired projection then SwiGLU | A, B | 7 | `0b1111_1100` |

INT8 keeps its F32 path under all 4 staging selections. Non-INT8 F16 template assembly remains unchanged. Output-owner correspondence doesn't constrain cooperative weight staging; the existing consumers read the staged tile after its barrier.

## Test

The source identity oracle lives in `crates/backend-wgpu/tests/fixtures/packed-scalar-415007e/`. Its 12 small text fragments were copied before editing from clean `415007ea32509ff72cd130d8c4de87860f75e74e`. The test independently assembles all 8 NF4/ternary projection compositions for F32, F16 weights, F16 activations and F16 both, then compares complete source bytes. It also restores the shared scalar body and compares it with the frozen body. The oracle doesn't reconstruct its expected source through the new registry.

The portable INT8 tests check all 4 compositions across all 4 staging selections, resolved hooks, workgroup size, binding names/access, storage-count/read-mask registry contracts and barriers. Existing device-free Naga validation covers the complete registry. A separate host-only ownership test compares independent former scalar indexing, candidate flat shared writes and direct tensor provenance. Its consumers read actual stored slots using the unchanged fragment coordinates; expected values don't bypass the proposed stores.

The ownership grid contains 120 shapes and 4 forms: M=1/2/4/8/16/63/64/65, N=1/17/31/32/33, K=128/256/384. It checks separate A/B provenance, all 256 input owners, 1 write per shared slot, no masked reads, and 4-to-1 scalar/word read multiplicity. Negative controls include row aliasing, shifted scales, duplicate/missing stores, B-to-A aliasing and masked reads hidden by later zero fill. Decoder checks cover every byte, including raw `0x80`, plus non-power-of-2 scales. Bundle admission's separate prohibition on -128 remains unchanged.

The hardware test source uses independent ascending-K F32 references that dequantize before multiplying input. All 120 boundary shapes exercise all 4 compositions with distinct A/B codes/scales and separate gate/up inputs. Scales are exact widenings of positive normal F16 values with odd mantissas. Comparison uses `atol=rtol=1e-4`; word/group impulses use exact coefficient equality.

The general fixture's mantissa is `1 + 2*((37*row + 83*group + 19*stream) % 511)`. Since 83 and 511 are coprime, fixed-row group scales have period 511, exceeding the maximum 36 groups in these K shapes.

The same formula repeats every 511 rows. A scale-only `row % 511` bug could therefore survive the wide synthetic cases, while the original N<=33 boundary grid doesn't reach those rows.

A separate N=513/K=384 fixture patches rows 0/511/512 with 18 explicitly distinct positive finite exact-F16 scales across their 3 groups and 2 weight streams. Nonzero signed codes at columns 0/128/256 prevent the default row 512 zero codes from hiding a scale-address error. Host assertions distinguish every selected row/group/stream and reject hypothetical `row % 511` and `group % 2` scale results.

Plain projection uses exact impulse readback comparisons against direct canonical scalar coefficients and a separate signed-byte conversion, without the candidate word/quartet or generic coefficient helper. The same patched matrices exercise all 4 forms with separate gate/up inputs and the established numerical tolerance. These checks passed on the real adapter, including the original 120-shape count assertions.

The older primitive oracle repair changes `(activation * signed_code) * scale` to `activation * (signed_code * scale)` in `tests/int8.rs`. At frozen `415007ea32509ff72cd130d8c4de87860f75e74e`, `src/shaders/int8_linear_header.wgsl` line 11 returns `signed_code * scale`; the frozen `nf4_prefill.wgsl` stores that F32 coefficient at lines 45–48 and multiplies it by activation at lines 53–57. The new decoder and shared body preserve that written order.

The old fixture's small dyadic activations and power-of-2 scales make its products identical under both associations, so it couldn't expose the general F32 rounding difference. Its tolerance and F32 summation stay unchanged. Host/controller source and frozen real-model references stay unchanged.

Deployment-width fixtures cover the union `[N,K]` = `[512,1024]`, `[1024,1024]`, `[1024,2560]`, `[2560,1024]`, `[3072,1024]`, `[1024,4608]`, `[4608,1024]`, with M65. A sparse M346/N2560/K1024 case checks a production-sized classifier FFN output tile. Eight ordered nonzero inputs make their expected outputs independently auditable without a full dense CPU matrix reference. Protected dense pointer heads `[256,1024]` stay outside this candidate. These synthetic widths came from small conversion-manifest metadata; no model files are read by the tests.

Invalid-admission tests reject wrong code width, scale row count, input rank/width, output shape, paired shapes, gate/up layout and nonfinite upload before any INT8 dispatch. Cancellation source tests cover pending readback accounting, dropped logical output ownership, fence settlement, cancelled admission and a correct subsequent submission. Existing allocator and completion tests remain in place.

| Gate | Current status |
| --- | --- |
| Independent analytical address proof and read-only review | Completed before this draft; scoped to ownership/provenance |
| Implementation source and diff review | Independent static review completed; no remaining material findings |
| Formatter, Clippy, compilation and device-free Naga validation | Passed; 5 new portable tests and the exact registry validation test executed |
| Required `quick` and full `ci` | Passed, including all 48 hill-climbing controller tests |
| Boundary, deployment-width and cancellation hardware tests | All 6 new tests passed in CI and the explicit GPU gate |
| WASM compilation | Passed; browser execution is a separate gate |
| Original-precision real-model qualification | Passed all 14 starting-runtime anchors and exact typed decisions at clean `3511a572` |
| Browser qualification | Passed FP16, INT8, NF4, ternary and mixed-QAT classifier lanes in Chrome |
| Frozen spaced ABBA screen and fresh confirmation | Screen completed all 252 captures and rejected; no confirmation admitted |

The separate workspace proof isn't imported or executed by runtime tests. Its reviewed source SHA256 is `8b54fb4aeccd1f61719c1d982b17543af5c770ab655b9fb0ce96786f90c941f2`; report SHA256 is `37ebceb20a37dfa9ebcb00903abf9705b83b6ef1ca27b19efb5bbf0856ac115b`. It established address/ownership and exact coefficient representation checks, with explicit mutation sensitivity. It didn't establish compiled control flow, dot products, GPU execution or performance.

The explicit GPU gate passed 66 tests with 2 existing development benchmarks ignored. Its 6 new hardware tests all executed with 0 ignored. The portable source/provenance tests and the complete Naga registry check also passed; these counts include different kinds of checks and don't turn portable tests into hardware evidence.

The first full CI run passed its workspace tests and caught 3 strict Clippy findings in the new test code. Descriptive names replaced 3 crowded single-letter locals, and digit separators clarified 2 exact dyadic literals. The focused all-feature backend Clippy check and full CI retry passed. No shader, fixture value, arithmetic, tolerance or harness source changed in that correction.

The existing synthetic controller hosts couldn't launch from a Python virtual-environment path containing spaces. A 2-case OS diagnostic confirmed that an existing interpreter entry with spaces failed with `ENOENT`, while its resolved entry launched. Validation used a temporary virtual-environment path without spaces and the same locked projects; all 48 controller tests then passed. This changed validation process setup only, leaving the frozen campaign build/host environment untouched.

Clean `3511a572` passed the 14-cell real-model preflight with the original absolute/relative tolerances and exact typed decisions. The complete raw dispatch dictionary matched each starting WGPU capture. The five classifier browser lanes passed full, cached-prefix and recovery comparisons. The initial browser wrapper retained a failed cleanup observation after misidentifying an unrelated user Chrome renderer; exact ancestry review preserved its two successful lanes, and the three remaining lanes completed in the retained resumption run. No user Chrome process was killed. Measured controller/host source, timing policy, tolerances and campaign artifacts remained frozen.

## Resolution

The frozen screen rejected this loader. Cached INT8 classification fell from 6.451093 to 6.180167 sustained predictions/second, a 4.20% throughput reduction. All four matched-block ratios were between 0.956877 and 0.959598; their 95% interval was [0.957179, 0.959038]. The complete interval is below the protected floor `1 / 1.02 = 0.9803921569`.

Full INT8 classification was effectively flat (ratio 1.000114). MagicBox INT8 and ternary crossed the protected bound and remain inconclusive. Final cell counts are 1 regressed, 2 inconclusive, 11 within budget and 0 improved. The source-level 4-to-1 loader access count didn't produce an end-to-end gain.

The independent outside-source auditor checked all 28 correctness processes and 224 measured processes, all 756 raw request/stdout/stderr files, exact dispatch maps in both roles, device/source/executable/context identities, resources, wall deadlines, sequential F64 PPS arithmetic, separate latency averages, ABBA order and the saved intervals. It reported complete consistent evidence, no audit errors, no missing records and the same rejected outcome. Its 44 synthetic tests and final Ruff check passed before that audit.

The 223 adjacent measured-process gaps had a minimum of 17.045795 seconds, above the fixed 15-second gap. Each cell has 8 independent processes per role grouped into 4 ABBA blocks. The 95% interval resamples those 4 complete blocks 10,000 times with seed 47; the resample count doesn't add independent observations.

### Actual end-to-end results

PPS is the arithmetic mean of 8 process PPS values per role. Prediction latency is averaged within each process and then across those 8 processes; it isn't calculated as the reciprocal of mean PPS. These columns exclude initialization and warmup; process wall time and deadlines were audited separately. Each comparison uses the identical model artifact and precision.

| Cell | Champion PPS | Candidate PPS | Champion latency ms | Candidate latency ms | PPS ratio | 95% paired interval | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| model100k-fp16-full | 1.625592 | 1.632354 | 615.193 | 612.629 | 1.004160 | [1.000877, 1.009483] | within_budget |
| model100k-fp16-cached | 1.867002 | 1.868327 | 535.654 | 535.273 | 1.000709 | [0.994364, 1.006415] | within_budget |
| model100k-int8-full | 6.020460 | 6.021143 | 166.110 | 166.082 | 1.000114 | [0.994673, 1.005611] | within_budget |
| model100k-int8-cached | 6.451093 | 6.180167 | 155.132 | 161.930 | 0.958003 | [0.957179, 0.959038] | regressed |
| model100k-nf4-full | 5.899949 | 5.871541 | 169.520 | 170.361 | 0.995185 | [0.985884, 1.002204] | within_budget |
| model100k-nf4-cached | 6.646143 | 6.626994 | 150.470 | 150.900 | 0.997119 | [0.993854, 1.001842] | within_budget |
| model100k-ternary-full | 6.497922 | 6.521405 | 153.899 | 153.342 | 1.003614 | [1.000661, 1.006169] | within_budget |
| model100k-ternary-cached | 7.097469 | 7.072590 | 140.899 | 141.400 | 0.996495 | [0.992287, 1.000718] | within_budget |
| polyomino-qat-mixed-full | 6.523626 | 6.520067 | 153.290 | 153.373 | 0.999454 | [0.997679, 1.001670] | within_budget |
| polyomino-qat-mixed-cached | 7.105576 | 7.125379 | 140.739 | 140.346 | 1.002787 | [0.996015, 1.009648] | within_budget |
| magicbox-fp16-full | 1.408077 | 1.411977 | 710.313 | 708.241 | 1.002770 | [0.997331, 1.010601] | within_budget |
| magicbox-int8-full | 4.553486 | 4.500108 | 219.647 | 222.266 | 0.988278 | [0.976557, 1.000357] | inconclusive |
| magicbox-nf4-full | 4.496099 | 4.485736 | 222.462 | 222.951 | 0.997695 | [0.987296, 1.005405] | within_budget |
| magicbox-ternary-full | 4.509362 | 4.444681 | 221.799 | 225.000 | 0.985656 | [0.976025, 0.994033] | inconclusive |

### Dispatches, memory and model size

Every candidate raw dispatch dictionary equaled its champion/start-anchor dictionary at the same work count. The table shows aggregate dispatches per fixed matrix cycle, not per prediction. Peak bytes are runtime-accounted allocations, checked against the original 4 GiB limit; packed model storage is measured separately.

| Cell | Champion peak bytes | Candidate peak bytes | Dispatches per matrix cycle (both roles) |
| --- | ---: | ---: | ---: |
| model100k-fp16-full | 1,719,700,000 | 1,719,700,000 | 895 |
| model100k-fp16-cached | 1,756,776,992 | 1,756,776,992 | 855 |
| model100k-int8-full | 1,048,644,128 | 1,048,644,128 | 725 |
| model100k-int8-cached | 1,106,184,736 | 1,106,184,736 | 685 |
| model100k-nf4-full | 967,379,488 | 967,379,488 | 725 |
| model100k-nf4-cached | 1,024,920,096 | 1,024,920,096 | 685 |
| model100k-ternary-full | 989,399,584 | 989,399,584 | 725 |
| model100k-ternary-cached | 1,046,940,192 | 1,046,940,192 | 685 |
| polyomino-qat-mixed-full | 758,712,864 | 758,712,864 | 725 |
| polyomino-qat-mixed-cached | 816,253,472 | 816,253,472 | 685 |
| magicbox-fp16-full | 3,592,036,736 | 3,592,036,736 | 672 |
| magicbox-int8-full | 2,744,918,400 | 2,744,918,400 | 672 |
| magicbox-nf4-full | 2,601,263,488 | 2,601,263,488 | 672 |
| magicbox-ternary-full | 2,532,057,472 | 2,532,057,472 | 672 |

Packed model bytes changed by 0 and every artifact hash remained identical. Classifier FP16/INT8/NF4/ternary weights remain 459,417,456 / 299,444,936 / 218,180,144 / 177,548,144 bytes. Encoder FP16/INT8/NF4/ternary remain 711,081,864 / 428,280,648 / 284,625,568 / 212,798,480 bytes. The deployed mixed-QAT weights remain 60,343,032 bytes. Embedding and head precision stayed protected.

All same-precision runtime-anchor and exact typed-output comparisons passed. MagicBox ternary retains its original strict independent NumPy numerical failure, which is also present in the starting runtime and CPU path. The other 13 independent numerical cells pass. The unchanged starting anchor accepts runtime-change parity; it doesn't erase that separate reference discrepancy.

### Decision and retained identities

The loader remains an unpromoted experiment. No fresh confirmation ran, no champion transition occurred, and the attempt ledger retains 1 of 4 candidate reservations and 0 of 2 finalist reservations consumed. Native Metal remains a separate candidate and measurement lane.

This screen establishes the cached INT8 regression and the absence of an eligible win. It doesn't isolate the physical cause. The cooperative ownership changed which invocations load and stage coefficients; compiler lowering, memory traffic, shared-memory access and live registers would need their own controlled experiment to assign the cost. The existing scalar loader remains the performance champion.

The shared packed projection template and explicit loader hook provide a small, explainable place for a future isolated variant. Operation-specific bindings and arithmetic remain visible, while backend-local selection can evolve independently. A broad registry rewrite or generic compiler isn't justified by this result.

| Evidence | Retained identity |
| --- | --- |
| Measured implementation | `3511a572f49bbd1e34f741e408cc457c10f0787e` |
| Measured source closure | `6b1c9f19d2048ef1e8f301803c8f0c45dec637f660742c8f6e99c4d0637c8ce7`, 447 entries |
| Frozen champion/controller | `bf32d7a14732b1c696e5e44768c5666cf608ba0c` |
| Frozen campaign | `330947ef687ffe6b0bed4cf3b4590bcc0d7ef4cbc62a353cced58fcca910c164` |
| Screen result | `29c9b2cd840bb51293fb52f9914bb8c3b830794a3c806a389f50d03f2b249e5f` |
| External terminal pin | `1a1849e793ec9125b5debba7037e025363d1fa27991436aa17c98556013683d9` |
| Independent screen audit | `af5f5907fa935cae1e303a94364139bca9a74f43fc5f5fb3958084b433b64158` |
| Candidate executable | `cecbe0376956b70b91ade5d1cc9680ab5f9bff510155071ad8a979746af45d76` |
| Champion executable | `fe0f1f02e7782c04decccfdab75247516e6c48f5a8ef6cba58981bcccacacaf8` |

Raw runs and audits live in the workspace experiment `experiments/2026-09-29-runtime-gpu-hill-climb`, outside source Git. The final report-only direct child is checked separately against the full measured source closure and retained evidence. It supplies this write-up and receives no separate timing or promotion claim.

Local CI/GPU/WASM and real-model/browser gates passed. GitHub's selected remote CI jobs failed before starting because of the account payment/spending limit; their annotations are retained. Remote CI is unresolved.
