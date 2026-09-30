# WGPU signed-INT8 coherent word loader

Status: local source, hardware and WASM checks passed September 30, 2026 after WGPU's 14-cell unchanged-build control and fresh evidence audit passed. Actual-model, browser and frozen performance qualification remain pending. Native Metal remains held after its separate inconclusive control repeat; no candidate performance result or promotion decision exists.

## Hypothesis

The existing signed-INT8 projection loader extracts 1 byte per coefficient, repeating a U32 code-word and group-128 scale access for each of its 4 bytes. Loading 1 word and 1 scale, then staging 4 decoded F32 coefficients, could reduce redundant loader work across the existing 4 projection compositions.

The proposed source-level access count falls from 512 to 128 word/scale reads per valid 32-column K16 weight tile, for each weight stream. Compiler elimination, GPU transactions, cache behavior, shared-memory scheduling and latency require measurement. That count doesn't establish a speedup.

The frozen measured/controller baseline is `bf32d7a`. This feature worktree starts at `415007ea32509ff72cd130d8c4de87860f75e74e`, which adds browser startup qualification plumbing and retains the same backend source. The draft changes `crates/backend-wgpu/`, this research note and root's addition of the new hardware test target to `scripts/check.sh gpu`. Native Metal, executor, host, controller, artifact bytes, public APIs and allocation contracts stay outside its scope.

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
| Draft source and diff review | Independent static review completed; no remaining material findings |
| Formatter, Clippy, compilation and device-free Naga validation | Passed; 5 new portable tests and the exact registry validation test executed |
| Required `quick` and full `ci` | Passed, including all 48 hill-climbing controller tests |
| Boundary, deployment-width and cancellation hardware tests | All 6 new tests passed in CI and the explicit GPU gate |
| WASM compilation | Passed; browser execution is a separate gate |
| Original-precision real-model qualification | Pending, requires the clean implementation commit |
| Browser qualification | Pending, requires the clean implementation commit |
| Frozen spaced ABBA screen and fresh confirmation | Pending candidate qualification |

The separate workspace proof isn't imported or executed by runtime tests. Its reviewed source SHA256 is `8b54fb4aeccd1f61719c1d982b17543af5c770ab655b9fb0ce96786f90c941f2`; report SHA256 is `37ebceb20a37dfa9ebcb00903abf9705b83b6ef1ca27b19efb5bbf0856ac115b`. It established address/ownership and exact coefficient representation checks, with explicit mutation sensitivity. It didn't establish compiled control flow, dot products, GPU execution or performance.

The explicit GPU gate passed 66 tests with 2 existing development benchmarks ignored. Its 6 new hardware tests all executed with 0 ignored. The portable source/provenance tests and the complete Naga registry check also passed; these counts include different kinds of checks and don't turn portable tests into hardware evidence.

The first full CI run passed its workspace tests and caught 3 strict Clippy findings in the new test code. Descriptive names replaced 3 crowded single-letter locals, and digit separators clarified 2 exact dyadic literals. The focused all-feature backend Clippy check and full CI retry passed. No shader, fixture value, arithmetic, tolerance or harness source changed in that correction.

The existing synthetic controller hosts couldn't launch from a Python virtual-environment path containing spaces. A 2-case OS diagnostic confirmed that an existing interpreter entry with spaces failed with `ENOENT`, while its resolved entry launched. Validation used a temporary virtual-environment path without spaces and the same locked projects; all 48 controller tests then passed. This changed validation process setup only, leaving the frozen campaign build/host environment untouched.

Original-precision real-model qualification and browser execution follow the clean implementation commit. Freeze-anchor comparisons must check numerical outputs and typed decisions before any performance screen. Measured controller/host source, timing policy, tolerances and campaign artifacts remain frozen.

## Resolution

Performance resolution is pending. The implementation and its local source/hardware gates passed. Independent static review accepted the core, all 12 frozen source fragments and the corrected row-boundary fixture; the final checked-source review precedes the implementation commit.

Promotion requires admitted execution, all correctness gates, frozen spaced ABBA screening and a fresh confirmation. No performance claim follows from the analytical read count or previous qualification samples.
