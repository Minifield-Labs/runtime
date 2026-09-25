# FFN optimization experiment log

Subject: `minifield.nf4.v1` polyomino classifier, wgpu native path.
Prompt shape: 346 tokens, hidden 1024, intermediate 2560, 14 FFNs.
Machine: development Mac (same adapter across runs; absolute numbers
comparable within this log).

Method per experiment:
- `cargo test -p minifield-backend-wgpu --test kernel_bench classifier_prefill_components -- --nocapture`
  isolates GQA, FFN pair, FFN down at m=346.
- `cargo run --release -p minifield-web-demo --example classify -- bundle prompts.json`
  gives end-to-end seconds per decision and logits vs the oracle cases.
- Correctness gate: argmax + logits on the 3 oracle cases must match the
  baseline (max |d logit| recorded; small fp drift tolerated, argmax may not flip).

## E0 baseline (faedfa8, pre-experiment)

kernel_bench `classifier_prefill_components`, release, 6 reps each:

| op | per-dispatch wait |
|---|---|
| GQA m=346 q=16 kv=8 dim=64 | 4649.9 us |
| NF4 FFN pair m=346 k=1024 n=2560 | 4433.0 us |
| NF4 FFN down (swiglu-fused) m=346 k=2560 n=1024 | 3719.2 us |

FFN per decision = 14 x (4.43 + 3.72) = ~114 ms; attention = 6 x 4.65 =
~27.9 ms. Matches the audit's 115/27 ms split.

classify end-to-end (3 oracle cases): 0.320 s, 0.269 s, 0.219 s per
decision (GPU warm-up visible; later cases faster). dispatches=432.
Live kernels confirmed: `packed_gemm_pair_nf4` x60, `packed_swiglu_gemm_nf4`
x42, `packed_gemm_nf4` x84. Argmax on all 3 oracle cases matches expected
(LEFT, LEFT, CW).

## E1 control: standalone SwiGLU + plain NF4 down

Change: executor FFN replaced `packed_swiglu_linear` with `swiglu` into a
materialized `hidden` buffer + plain `packed_linear` for down. +1 dispatch
per FFN.

kernel_bench: swiglu+down = 3940.9 us vs swiglu-fused down = 3718.6 us
(release run; a second measurement gave 3766 vs 3732). Essentially even.

classify end-to-end: 0.339, 0.280, 0.220 s (baseline 0.320/0.269/0.219).
Logits bitwise identical to baseline on all 3 oracle cases.
dispatches=474 (+42, one swiglu per FFN).

Interpretation: materialize-once vs recompute-in-consumer is a wash. The
32x silu recompute costs almost nothing; the down kernel is bound by
weight/activation traffic, not the input-side epilogue math. Reproducer
for "bandwidth-bound, not arithmetic-bound".

## E2 candidate: SwiGLU epilogue in pair kernel + plain NF4 down

Change: new `packed_swiglu_pair` op (pair GEMM template + silu(a)*b
epilogue in the store block, writes a single [m,2560] hidden buffer;
ternary + NF4 header variants). Executor uses it for fully-packed FFN
with rows>1, then plain `packed_linear` down. Same 2 dispatches/FFN as
baseline, one fewer intermediate buffer (gate+up 7.09MB -> hidden 3.54MB).

kernel_bench (6 reps, release):

| op | per-dispatch wait |
|---|---|
| pair (gate+up) | 4618.4 us |
| swiglu-fused down | 3718.6 us |
| pair+swiglu fused | 4583.7 us |
| pair+swiglu + plain down (E2 total) | 8298.6 us |

Baseline total (pair + swiglu-down) = 8337 us. E2 = 8299 us. ~0.5%
faster at kernel level: the producer epilogue is free (4584 vs 4618)
and plain down costs the same as swiglu-fused down (8299-4584 = 3715
vs 3719). Confirms E1: the epilogue silu was never the cost.

Parity: packed_nf4_ops_match_cpu + packed_swiglu_pair_ternary_matches_cpu
pass (all 29 parity tests green).

classify end-to-end: 0.430, 0.272, 0.227 s (baseline 0.320/0.269/0.219;
first case carries new-pipeline compile warmup). Steady-state cases 2,3
within noise. Max |d logit| vs recorded NF4 baseline ~4e-6, argmax
preserved on all 3 oracle cases.

Verdict: neutral-to-tiny win end-to-end, but structurally better: one
fewer intermediate buffer, activations evaluated once, and it unblocks
E4/E5 by giving the down projection a plain GEMM consumer. Keep.

## E3 audit: dequant reuse across token rows

Static read of `nf4_prefill.wgsl` (shared GEMM template, also used by
the pair and fused-swiglu variants via PAIR/input_value headers).

Within a workgroup: **no redundant decode.** Each invocation unpacks 4
weights (2x2 of the 32x32 weight tile) once per k-step into workgroup
arrays `weights_a`/`weights_b`, then the `d`-loop reuses the decoded
f32 tile across all 32 token rows. Decode count per workgroup =
weight-tile size, not weight-tile x rows.

Across workgroups: real amplification. Grid is ceil(m/32) x ceil(n/32).
At m=346: 11 row tiles, so every weight is decoded 11x per dispatch
(once per row-tile workgroup that shares its column block).
Symmetrically every activation element is re-read ceil(n/32) times
(80x for pair at n=2560, 32x for down at n=1024), though input reads
are plain f32 and benefit from cache.

So decode amplification scales with m (worse for longer prompts).
Two remedies: bigger m-tiles (fewer row tiles) or unpack-once into a
scratch f32 weight buffer + dense GEMM (= E4's control, which also
measures the ceiling if decode cost were removed entirely).

## E4 unpacked-weight control

Method: existing `dense_prefill_bench` vs `packed_prefill_bench`, same
m=346 shapes, release, 15-30 reps. Dense = f32 weights through the
ordinary `linear` op (the path an unpacked-to-scratch model would take).

| shape | dense f32 | packed ternary | packed NF4 |
|---|---|---|---|
| m=346 k=1024 n=1024 | 7199.8us | 4475.3us | 1473.2us |
| m=346 k=1024 n=3072 | 21138.4us | 25072.7us | 4459.6us |
| m=346 k=2560 n=1024 | 16687.1us | 4350.7us | 3673.5us |

Verdict: packed NF4 is 4.5-4.9x faster than the existing dense f32
GEMM everywhere. Caveat: the two paths differ in more than weight
bytes (tiling, work ownership, loading), so this does NOT isolate
weight compression as the cause; execution strategy is a competing
explanation (see ternary note below). Unpacking weights into scratch
is still a dead end on these numbers: full f32 expansion of all FFN
weights would add ~367.5MB resident (a single-projection scratch is
~10MB f32), and the GEMM itself is slower anyway.

Side note: ternary packed is ~3x slower than NF4 at n=1024 and ~5.7x
slower at n=3072, interesting given ternary decode is simpler; likely a
different kernel mapping. Not the classifier's format, deprioritized.

## E5 tile variants on the shared NF4 GEMM template

m-scaling probe first (packed_linear nf4, k=2560 n=1024, 30 reps):

| m | row tiles | workgroups | wait |
|---|---|---|---|
| 32 | 1 | 32 | 864.5us |
| 64 | 2 | 64 | 823.1us |
| 128 | 4 | 128 | 1731.3us |
| 256 | 8 | 256 | 2834.4us |
| 346 | 11 | 352 | 3811.8us |

32->64 workgroups is free (parallel lanes absorb it), then ~linear.
Per-workgroup serial cost is the k-loop: k/32 iterations x 2 barriers.
Two variants tried on `nf4_prefill.wgsl` (shared by linear/pair/swiglu
headers; dispatch row-tile divisor updated to match):

K_STEP=64 (halve barriers per k-sweep): REGRESSION. Pair 4618->13354us,
down 3719->5900us. Likely occupancy loss from three 2048-f32 workgroup
arrays (24KB), though register pressure/codegen are untested
alternatives. Reverted.

TILE_M=64 (halve row tiles; inputs 8KB, weights unchanged; 4x2
fragment/thread): WIN at classifier shape.

| op | 32-row tile | 64-row tile |
|---|---|---|
| pair m=346 | 4618us | 3881us |
| down (swiglu-fused) | 3719us | 3017us |
| pair+swiglu | 4584us | 3774us |
| E2 total | 8299us | ~6512-6878us |

packed_linear m=346: k=2560 3812->3198us; k=1024 n=2560 3643->2957us.
Cost is low-m regression: m=32 864->1114us, m=64 823->1746us (single
full tile = 2x serial work per workgroup). Classifier runs m=346 so the
trade is right for this model; rows<32 never reach this template anyway
(GEMV-MT handles them), but m in 32..64 now pays ~2x vs before. Worth
keeping an eye on for short-prompt models.

Parity: all 29 tests green after the tile change.

classify e2e (E2 + 64-row tiles): 0.408, 0.258, 0.189 s vs E0 baseline
0.320/0.269/0.219. Steady-state ~14% faster end-to-end. Logits
bit-identical to the E2 run (which matched baseline to ~4e-6).

## E6 final-row / unnecessary-row pruning

Change (executor only): in the layer loop, when `index + 1 ==
config.layers.len()` and `rows > 1`, slice `ffn_input` and `residual` to
row `rows - 1` via `copy_rect_2d`, run the whole FFN block at m = 1
(GEMV path), and let `add_row_rms_norm` produce `next_x`/`next_u` at
[1, hidden]. Logits then run `weight_linear` on `u` directly instead of
copying row `rows - 1` out. Attention/conv parts of the last layer are
unchanged (their outputs feed the FFN row and the KV/conv caches, both
still needed).

classify e2e: 0.223, 0.185, 0.171 s. vs E2+64tile (0.258/0.189) and E0
baseline (0.269/0.219): steady-state ~22% faster than baseline.
Max |d logit| vs recorded NF4 baseline ~3e-6 (m=1 GEMV reduction-order
noise). Argmax preserved on all 3 oracle cases.

executor-core tests: 32 pass, 0 fail (CPU backend covers the sliced
path through the same executor code).

## Summary so far

E0 baseline: 0.320/0.269/0.219 s, FFN ~114ms, GQA ~28ms.
E1 control (swiglu + plain down): wash. Silu recompute never mattered.
E2 producer epilogue (packed_swiglu_pair): ~neutral time, one buffer
   fewer; kept as the structural cleanup.
E3 audit: in-workgroup decode already deduped; across-workgroup decode
   amplified ceil(m/32)x, activation reads amplified ceil(n/32)x.
E4 unpacked control: dense f32 is 4.5-4.9x SLOWER than packed NF4.
   Unpacking dead end confirmed.
E5 K_STEP=64: regression (occupancy). TILE_M=64: ~17-20% FFN win.
   Kept. Low-m cost: m in 32..64 pays ~2x vs 32-row tiles.
E6 last-layer FFN at m=1: ~22% e2e vs baseline, logits unchanged.

Combined: ~0.17-0.22s/decision vs 0.27-0.32 baseline on the two
steady-state cases (~1.4x). Per-layer FFN (pair_swiglu + down) went
~8.3ms -> ~6.5-6.9ms; the last layer's FFN now runs at m=1.
Note: E5->E6 case-2 drop (258->185ms) exceeds what a ~7ms FFN removal
explains; likely also dispatch/alloc reduction plus run-to-run
variance. Only 3 cases, single pass each: not a benchmark
distribution. Follow-up: A/B repeat runs + GPU timestamps.

Current bottleneck diagnosis is explicitly UNVERIFIED. Established:
activation fusion is latency-neutral; packed >> current dense; tile
geometry matters a lot. Not established: whether the limiter is
weight bandwidth, threadgroup-memory access, arithmetic, or
occupancy.

## E7 low-m dispatch crossover (round 2)

Forced every packed_linear m through the GEMV-MT kernel (rows<1024 gate)
and compared against both tile variants:

| shape | GEMV-MT | GEMM 32-tile | GEMM 64-tile |
|---|---|---|---|
| m=32 k=2560 n=1024 | 766us | 864us | 1114us |
| m=64 k=2560 n=1024 | 893us | 823us | 1746us |
| m=128 k=2560 n=1024 | 1604us | 1731us | 1157us |
| m=346 k=2560 n=1024 | 4638us | 3812us | 3198us |
| m=32 k=1024 n=2560 | 473us | 504us | 872us |
| m=128 k=1024 n=2560 | 1722us | 1399us | 1050us |
| m=346 k=1024 n=2560 | 4578us | 3643us | 2957us |

Crossover ~96: below it MT wins or ties, above it 64-tile wins.
Changes: GEMM dispatch threshold rows >= 32 -> rows >= 96 at all three
sites that have an MT fallback (linear, pair, swiglu-linear). The fused
pair-swiglu op has no MT variant, so the executor gates it at
ffn_rows >= 96; smaller m takes pair+swiglu_linear instead (which do
have short-row kernels). m=32..95 no longer pays the fat-tile penalty.
Parity + bench suite green.

## E8 tile geometry, round 2

K_STEP=16 on the 64x32 tile (halve staging: inputs 4KB + weights 2KB
each; pair total 8KB): consistent ~8-10% win on pair kernels
(3881->3464/3509, 3774->3439/3480), flat-to-positive on down
(3017->2893/2893). Half the workgroup memory and faster. Kept.

32x64 tile (new nf4_prefill_n64.wgsl, wide N: 2x4 fragment, activation
loads halved, weight decode amplification doubled) applied to
PackedGemmNf4 only:

| shape | 64x32 K16 | 32x64 K16 |
|---|---|---|
| m=346 k=2560 n=1024 | 2893 | 2784 |
| m=346 k=1024 n=2560 | 2865 | 2474 |

Consistent small win at classifier shape; m in 96..256 appears worse
(~1.5-2x) so the MT cutoff stays load-bearing. Physically consistent
with E4: weight bytes are 1/8 of activation bytes, so activation
reloads matter more than decode amplification. Parity green.

e2e: 0.339/0.222/0.165s; steady-state 0.165s vs 0.219 baseline = ~25%
faster. Machine had ambient load during some runs; repeated
measurements at m=346 agreed within ~3%.

## E9 cooperative-matrix NF4 GEMM (native only, rejected)

Prototype: full 32x64-tile GEMM where each of 8 subgroups owns an
8x32 output slab via `coop_mat8x8<f32>` fragments
(`coopLoadT`/`coopMultiplyAdd`/`coopStoreT` against workgroup-staged
tiles). Same K16 staging loop as the scalar kernel. Partial tiles fall
back to the scalar 2x4-fragment path because `coopStoreT` cannot clip.
Reference implementation preserved at
`docs/experiments/packed_gemm_coop_nf4.wgsl`.

Adapter capability (M1 Max, Metal): 8x8x8 with F32->F32, F16->F16,
F16->F32. Requires `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` plus
`Features::SUBGROUP` (for `subgroup_id`/`num_subgroups`); both native
only. Prototype requested them behind adapter probing and an explicit
allow site for wgpu's unsafe `ExperimentalFeatures::enabled()` token.

WGSL/pipeline gotchas hit during bring-up, in case this is revisited:
- `coopLoad` is column-major, `coopLoadT` row-major; staged tiles are
  row-major so T variants are required. Column-major loads produced
  wrong numerics that still happened to keep argmax on the 3 oracle
  cases; always diff logits, not just argmax.
- naga's uniform-builtin whitelist is only
  WorkGroupId|WorkGroupSize|NumWorkGroups; `num_subgroups` and helper-
  function call results read non-uniform and poison control flow for
  coop ops. Fix: inline `flat_wg` arithmetic, bounce `nsg`/`full`
  through `workgroupUniformLoad`.
- Any per-lane `if` in a block taints subsequent statements; staging
  guards had to become select+clamp.
- `coopStoreT(matrix, pointer, stride)` argument order differs from
  the load order.

Result at m=346 (same session, alternated runs):

| shape | scalar 32x64 K16 | coop 8x8 frags |
|---|---|---|
| k=1024 n=1024 | 1182us | 1511us |
| k=1024 n=3072 | 3054us | 3924us |
| k=2560 n=1024 | 2867us | 3424us |

19-28% slower at every shape. K=32 staging variant was worse still
(3.3-3.4ms on the down shape). Interpretation: this FP32 cooperative
implementation is consistently slower than the scalar implementation
at the tested shapes; its matrix execution does not compensate for
the implementation's other costs (staging, decode, barriers, masked
fallbacks). E1/E2 measured SwiGLU placement, not MAC cost, so they
cannot establish that the multiply loop is free.

Verdict: rejected, code path removed. Workspace `unsafe_code` lint
restored to forbid; no experimental features requested on device init.
If revisited, look at removing staging entirely (direct coopLoadT from
storage into A/B frags still needs decoded f32 in memory, so decode
must land in workgroup anyway) or waiting for f16 coop configs plus
fp16 staging to cut per-fragment bytes.

## E6 invariant note

`bulk_prefill_matches_serial_token_appends` already covers the E6
invariant: bulk prefill (final-layer FFN sliced to m=1) followed by a
continuation append matches serial appends for both formats. Passed
post-E6; cache positions/sequence lengths describe the full prompt.

## E10 aligned-row tile resolution + boundary parity

Aligned-row check on the wide-tile result (PackedGemmNf4, k=2560
n=1024, K16). At m=346 the wide 32x64 tile schedules 176 workgroups
vs the narrow 64x32 tile's 192 (352 vs 384 covered rows), an ~8%
padded-work gap. At aligned rows both cover exactly m:

| m | wide 32x64 | narrow 64x32 |
|---|---|---|
| 96  | 1184us | 1144us |
| 192 | 1663us | 1632us |
| 320 | 2676us | 2413us |
| 346 | 2961us | 2747us |
| 384 | 2929us | 2884us |

Narrow wins or ties everywhere at m>=96; E8's wide-tile advantage at
m=346 was mostly the padding gap plus noise. PackedGemmNf4 reverted
to the shared 64x32 template; dispatch is a two-way choice (GEMV-MT
below 96 rows, 64x32 K16 GEMM at and above), not three-way. The 96
boundary is measured for packed_linear; parity coverage added at
m=95/96/97.

## E11 workgroup-order swizzle (rejected)

Swap row-tile/col-tile precedence in the flat workgroup index,
dispatch adjusted to match: +/-2% across all shapes, within noise.
This specific ordering provides no measurable benefit. (Narrower
claim than "no kernel-level locality win exists": working set fitting
a cache does not eliminate the cost of moving data through it.)
Reverted.

## E12 decision-level prefix reuse (executor, not kernel)

`Lfm2Classifier` gained `prefill_base`/`classify_tail` wrapping the
existing `prefill_choice_base`/`append_choice_logits` executor API
(commit 98d8498 machinery). Measured via the classify example with
`CLASSIFY_PREFIX=1` on the 3 oracle prompts:

- common prefix across decisions: 47 of ~346 tokens (~14%). Board
  state diverges early in the template, so causal reuse of the rest
  is impossible without a prompt-template change.
- per-decision: 0.61-0.68s full prefill -> 0.53-0.56s tail append.
  ~15-18% faster, logits identical (append==direct is asserted
  bit-exact by append_choice_logits_branches_off_a_shared_unscored_base).

Baseline caveat, reconciled: the classify example measures wall
time around a submit->poll->sleep(1ms) loop over ~30 sequential ops
per decision; kernel_bench measures GPU wait inside a single
30-rep submission. Under load average ~6 (measured during this
run), scheduler jitter inflates the e2e number to ~0.61s while GPU
kernel times are unchanged from earlier runs (m=346 down still
~2730us). The 15-18% ratio holds within-harness; absolute e2e
times are load-sensitive and not comparable across sessions.

Why the win can exceed the token fraction: tile quantization. 299
fresh tokens schedule ceil(299/64)=5 row tiles vs ceil(346/64)=6,
16.7% less row-tile work from 13.6% fewer tokens. Not proof of the
whole explanation, but consistent with the measured 15-18%.

Worker semantics (agreed production direction): keep the existing
template, and each decision branches from the SAME cached base,
no appending successive board states into a growing history. The
next template revision should add a second cache level: instructions
cached across the model lifetime, settled board state cached while
unchanged across a decision group, and only per-decision fields
fresh. Reordering the prompt is not numerically equivalent, so
evaluate task quality separately on that revision.

## E13 FP16 workgroup staging + occupancy analysis

Occupancy accounting correction (review): an earlier draft of this
section computed resident workgroups as
max_compute_workgroup_storage_size / staging_bytes. That is wrong:
the adapter field is a per-workgroup shader limit, not the per-core
threadgroup-memory pool, so it cannot derive residency. The earlier
768-threads-per-core figure was also asserted without an adapter
source. K16's measured win stands; the mechanism does not.

Correct statement: K16 reduces staging allocation and consistently
improves latency despite additional synchronization rounds. This is
consistent with a resource-pressure or generated-code improvement,
but the specific limiter and achieved occupancy remain unmeasured.
It follows that f16 staging is NOT proven incapable of improving
occupancy either; the measured effect size is what it is on this
adapter, regardless of which resource bound it.

Metal System Trace capture (xctrace on kernel_bench,
/tmp/mst_nf4.trace): command-buffer granularity only, and each bench
iteration encodes 30 reps inside one submission, so intra-submission
dispatch gaps are invisible at that level. Per-dispatch counters
(ALU/memory/occupancy) need an Xcode GPU frame capture with counter
sampling, which is GUI-bound; achieved occupancy and the dominant
limiter remain unmeasured on this adapter.

FP16 staging experiment (MINI_NF4_STAGE_F16=w|x|wx selects weights /
activations / both; nf4_prefill_f16.wgsl keeps f32 decode, f32
accumulate, f32 output; SHADER_F16 requested only when the knob is
set; adapter reports the feature present).

Numerics (packed_nf4_ops_match_cpu under wx): full-matrix drift at
m=346 is max_abs 1.1e-2 on GEMM outputs and 2.7e-1 after the SwiGLU
product; rel spikes only on near-zero elements. Parity holds at a
dedicated 0.1 abs / 0.02 rel bound under the knob; production stays
at 1e-4.

Timings at m=346. packed_linear runs were 30-rep interleaved (tight);
the fused rows below came from classifier_prefill_components at 6
reps with ~5% ambient swing, so treat single-run deltas under ~4% as
noise.

plain GEMM (packed_linear, interleaved, trusted):

| variant | k=2560 n=1024 | k=1024 n=2560 |
|---|---|---|
| f32 | 2729-2730 | 2536-2542 |
| w-f16 | 2708 | 2485 |
| x-f16 | 2714 | 2481 |
| wx-f16 | 2684-2688 (-1.5%) | 2460-2477 (-2.7%) |

m=320 aligned check (k=2560 n=1024 / k=1024 n=2560): f32 2316/2289,
w 2298/2231, x 2256/2235, wx 2251/2220. Same ~2-3% shape: the win is
not an m=346 padding artifact.

packed_prefill_bench corroboration under wx: nf4 k=1024 n=1024
1114.8->1076.4 (-3.4%), k=1024 n=3072 3030.6->2948.0 (-2.7%),
k=2560 n=1024 2731.2->2689.6 (-1.5%); nf4=false rows unchanged.

fused ops at m=346 (6 reps, noisy):

| op | f32 | w-f16 | x-f16 | wx-f16 |
|---|---|---|---|---|
| pair k=1024 n=2560 | 3508 | 3460 | 3624 | 3326 |
| packed_swiglu_linear k=2560 | 2887 | 2886 | 3048 | 3222 |
| pair+swiglu fused | 3436 | 3380 | 3767 | 3321 |
| E2 total | 6393 | 6083 | 6643 | 5952 |

The swilin regression isolates to x-f16 (+6%) and compounds to +12%
at wx while the same-shape plain GEMM wins -1.5%. Plausible account:
its input_value computes silu(gate)*up per staged element, so
activations-f16 pays conversions on a transcendental result with no
weight-array partner to amortize them; hypothesis, not measured
cause. Caveat for all f16 numbers: the variant also adds conversions
and changes the compiled instruction stream, so deltas are the net
effect of the precision change, not a clean measurement of staging
bandwidth alone.

Verdict: these variants did not earn production complexity (small
trusted gains, inconsistent fused results, different numerical
tolerance). That is not a demonstrated hardware ceiling. The knob +
parity tolerance stay dev-only for revisiting after the ternary QAT
changes decode cost.

## Round-3 verdicts

- E10: narrow 64x32 K16 is the production GEMM tile at m>=96; E8's
  wide-tile result was a padding artifact. Dispatch stays two-way.
- E11: this workgroup-order swap provides no measurable benefit;
  reverted.
- E12: prefix reuse is real (~15-18% per decision in this harness);
  strongest immediate next step, wasm/worker wiring agreed.
- E13: f16 staging variants did not earn production complexity
  (~1.5-2% on the plain GEMM, inconsistent fused results, f16
  drift). Dev knob kept, not productionized.

Stopping rationale: the tested NF4 micro-optimizations are showing
diminishing returns: the kernel survived several targeted
alternatives and classifier-level reuse offers a clearer measured
payoff. This is not evidence the GPU has no scheduling headroom;
the dominant limiter is unmeasured.

Caution on ternary (E4 warning): several existing ternary cases were
substantially slower than NF4. Two-bit packing halves weight-code
bytes only; decoded staging traffic, activation traffic, MACs, and
barriers are unchanged under the current decode-to-f32-tile design,
so smaller weights do not automatically mean faster kernels. Before
counting QAT as a latency win, run the same-weight comparison:
encode W = scale x {-1,0,+1} (all values present in the standard
NF4 codebook) as both NF4 and packed ternary, and measure the same
64x32 K16 template at the production shapes, including the
cached-tail m~299 shape, not just m=346. That isolates the decoder
cost from model quality and unrelated tiling differences.
