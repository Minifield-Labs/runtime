# Where the milliseconds go

346 tokens, 14 feed-forward blocks, and a small model deciding which way to move a polyomino. On our development Mac, the first recorded run took 320 milliseconds. The next 2 decisions took 269 and 219 milliseconds as the GPU warmed up.

Those are modest numbers, but this is the kind of workload we care about at Minifield: a model operating a product on someone's device, with a person waiting for the result. Every decision should be cheap enough to make again. A quarter of a second is plenty of time to become curious about what the machine is doing.

That curiosity took us through packed weights, shared memory, synchronization barriers, matrix instructions, and a particularly persuasive result that mostly disappeared when we changed the prompt length. We made the runtime faster. We also collected several explanations that the measurements couldn't support.

## One request still has plenty of work inside it

The experiments used our native wgpu runtime on an M1 Max, running an NF4-quantized classifier. NF4 stores weight codes in 4 bits; the kernel decodes them into numerical values as it works. The prompt had 346 tokens, the model's hidden width was 1024, and its feed-forward blocks expanded that to 2560.

Batch size 1 means we're serving 1 request. During prefill, the stage that processes its prompt, we can still work on many token positions together. Our matrix operations had 346 rows to chew through.

That distinction matters. Processing a whole prompt and processing 1 new token give the GPU very different jobs. A kernel that spreads a large matrix across the machine can leave a small matrix paying for arrangements it barely uses.

We started by measuring the pieces. Each feed-forward block took about 4.43 milliseconds for its paired projections and 3.72 milliseconds for its down projection. Across 14 blocks, that was roughly 114 milliseconds. The measured attention work added about 28 milliseconds.

These component timings didn't explain every millisecond of the application, but they gave us somewhere specific to dig. The feed-forward blocks had the largest measured bill.

We used 2 views of the work throughout: a kernel harness to time individual operations, and the classifier to measure complete decisions. We also compared output scores against 3 reference cases. The expected decisions were LEFT, LEFT, and CW, which would eventually prove to be a dangerously forgiving answer key.

## The expensive-looking calculation

A feed-forward block takes each token's representation, expands it, applies a nonlinear operation, and projects it back down. In this model, 2 projections produce values called gate and up. SwiGLU combines them:

```text
hidden = silu(gate) * up
output = hidden * down_weights
```

SiLU applies a smooth, sigmoid-based gate to its input. It looked like an obvious place to save work because our down-projection kernel recomputed the activation as different output tiles consumed it. At this shape, that meant evaluating it repeatedly across 32 column tiles.

We tried calculating it once, writing the result to a buffer, and letting an ordinary projection consume that buffer. This added a dispatch, which is another kernel launch for the GPU to execute.

The result was essentially even. One measurement put the separate activation and projection at 3.94 milliseconds against 3.72 milliseconds for the fused version. Another put them at 3.77 and 3.73. The classifier's output scores were bitwise identical.

Apparently we could perform that suspicious-looking calculation many times without moving the latency much. Saving arithmetic had introduced a buffer write, a later read, and another dispatch. Whatever we saved was swallowed by the rest of the arrangement.

We then moved SwiGLU into the end of the paired projection, where gate and up were already available. That let us write a single activated buffer instead of 2 separate projection outputs, cutting those intermediate buffers from about 7.09 MB to 3.54 MB.

The pair of kernels went from roughly 8.34 milliseconds to 8.30 milliseconds. A 0.5% difference barely deserves a victory lap, but the smaller buffer and simpler downstream operation were worth keeping. They also made later experiments easier.

Our early interpretation was that memory traffic dominated. The measurements supported a narrower claim: changing the placement of this activation barely affected latency. They left the cost of the matrix multiply, shared-memory traffic, and GPU scheduling unresolved.

That qualification would become a recurring chore.

## Down the memory hierarchy

The shader divided its output into tiles, initially 32 token rows by 32 output columns. A workgroup, a cooperating set of GPU invocations, owned each tile.

Inside that workgroup, the kernel loaded a slice of activations and decoded a slice of packed weights into shared workgroup memory. The invocations could then reuse those values while calculating their outputs. Once they finished that slice, they loaded the next one.

The weights were already being decoded once per tile and reused across its 32 token rows. Our suspicion that every row was separately unpacking the same weights had run into the actual shader.

Across workgroups, though, the duplication was real. A 346-row prompt needed 11 row tiles. Each group covering a different set of rows decoded the weights it needed again, so each weight was decoded 11 times per dispatch.

The activations had a matching problem in the other direction. Every group covering another set of output columns needed the inputs again. With 2560 output columns and 32 columns per tile, that was 80 reads of each activation element at the shader level. Caches could soften those reads; they still described repeated work in the program.

This is where a GPU starts to feel cramped despite having an enormous amount of parallel machinery. You want groups to share more data, which means making their working areas larger. Those areas consume resources that other groups would also like to use.

The obvious escape was to unpack the weights once into ordinary 32-bit floats and use a dense matrix multiply. We tested the existing dense path as a control.

It was about 4.5 to 4.9 times slower than the packed NF4 path across the measured shapes. Expanding all the feed-forward weights would also add roughly 367.5 MB of resident float storage. That particular route could stay closed.

The comparison had limits. The kernels differed in tiling, loading, and work ownership as well as weight format. It showed that our existing dense implementation was a poor destination for unpacked weights. It couldn't tell us how much of NF4's advantage came from compression itself.

Even the simpler ternary format was substantially slower in several of these tests. Fewer bits had failed to buy a faster implementation on their own.

## A bigger mouthful

Each tile advances through the matrix's inner dimension in chunks. We called that chunk size `K_STEP`. Between chunks, barriers keep the workgroup coordinated: the staged data must be ready before anyone consumes it, and everyone must finish before that memory gets overwritten.

Our first attempt doubled the chunk from 32 to 64. That halved the number of synchronization rounds. It also enlarged the staged data.

The paired projection went from about 4.62 milliseconds to 13.35 milliseconds. The down projection went from 3.72 to 5.90. We had saved barriers and made the expensive operation almost 3 times slower.

Larger workgroup allocations could have reduced the machine's ability to keep work resident. Register pressure or different generated instructions were also plausible. We didn't have the counters to choose between them, so we reverted the change and kept the uncertainty.

Doubling the tile's token rows worked much better. A 64-row tile reused each decoded weight across more of the prompt, reducing the number of groups repeating that work. The paired projection fell to about 3.88 milliseconds and the down projection to 3.02.

There was a catch at short prompts. At 64 rows, one tested projection slowed from 0.82 milliseconds to 1.75 milliseconds. A larger tile had concentrated more serial work into each group, and a small matrix didn't provide enough other groups to make that arrangement worthwhile.

We measured the crossover against our short-row kernel and settled on 96 rows as the dispatch boundary. Below that, the runtime used the short-row path. At 96 and above, it used the larger tiled matrix multiply. We added correctness coverage at 95, 96, and 97 rows, where an innocent boundary condition could select the wrong path.

Then we shrank `K_STEP` to 16. This increased the synchronization rounds, halved staging memory relative to the 64-by-32 tile with K32, and improved the paired kernels by another 8% to 10%.

Fewer barriers had lost badly. More barriers, with a smaller working set, won. Counting one kind of instruction was giving us a very incomplete picture of the machine.

## The result that survived until we changed the rows

Next came a wider tile: 32 rows by 64 columns. It reused activations across more output columns while repeating weight decode across more row groups.

At 346 rows it looked promising. One down-projection measurement improved from 2.89 to 2.78 milliseconds; another projection improved from 2.87 to 2.47 milliseconds. We had a plausible explanation involving activation reuse and compressed weight bytes. The explanation fit comfortably around the result.

Then we checked aligned row counts.

Tiles cover whole rectangles. With a 64-row tile, 346 rows require capacity for 384. With a 32-row tile, they require capacity for 352. Boundary guards keep the extra positions from changing the answer, but tile geometry still changes the scheduled work.

At our favourite prompt length, the wider tile scheduled 176 workgroups against 192 for the narrower tile. It had an approximately 8% advantage in padded coverage before any proposed memory benefit entered the discussion.

Across the follow-up measurements, the narrow tile won or tied, including at 346 rows. The apparent wide-tile advantage was largely padding and noise. We removed the extra variant and returned to the shared 64-by-32, K16 kernel.

A convincing explanation is easy to write after a good number appears. Changing the dimensions was a much cheaper way to find out whether we'd earned it.

## The matrix instructions arrive

We also tried cooperative matrix operations, where subgroups collaborate on small matrix fragments. The adapter exposed native support for the required operations, and the prototype used 8-by-8 float fragments with the same K16 staging loop.

Getting the first correct result took some excavation. Our staged data was row-major, while one load operation expected column-major data. Using it interpreted the tile in the wrong orientation.

The output scores were wrong. All 3 winning decisions were still correct.

LEFT, LEFT, CW had concealed a broken calculation. From then on, the score differences carried considerably more emotional weight than the winning labels.

There were also shader-validation constraints around uniform control flow. Cooperative operations need participating invocations to reach them together. Some values and helper calls that looked uniform to us weren't accepted as uniform by the validator, so the staging and control flow needed careful rearrangement.

Once correct, the prototype was 19% to 28% slower across the tested shapes. Increasing its staging chunk made it worse. The matrix operations hadn't compensated for the implementation's staging, decoding, synchronization, and boundary handling.

We removed the experimental execution path and preserved the shader as a reference. This particular FP32 implementation had lost to the scalar kernel. Other precisions and other arrangements remained open questions.

## Half the storage, a small discount

Another experiment stored staged weights, activations, or both in 16-bit floats while keeping decode, accumulation, and output in 32-bit floats. That reduced the shared-memory footprint, with extra conversions and some numerical drift attached.

The carefully interleaved plain-matrix measurements improved by about 1.5% to 2.7% when both were staged in half precision. Aligned-row checks showed similar small gains. Fused operations were less consistent, and one activation-fused path regressed.

The numerical change was measurable too. Full-matrix maximum absolute differences reached about 0.011 on GEMM outputs and 0.27 after the SwiGLU product. These were intermediate tensors, not final classification scores, but they required a separate, looser development tolerance.

We kept the experiment behind a development switch. The trusted gains were small, the fused results were mixed, and adopting it would add another numerical behaviour to maintain.

During the review, we also caught an error in our occupancy accounting. We'd tried to derive resident workgroups from an adapter limit on workgroup memory. That limit describes how much a single workgroup may allocate; it doesn't tell us the size of the per-core pool or how many groups actually reside there.

Our trace captured command buffers containing many dispatches. It couldn't resolve the per-dispatch counters we needed to settle the question. Smaller staging was faster in several experiments, but achieved occupancy and the dominant hardware limiter remained unmeasured.

The distinction is irritating when you'd like a neat ending. It also determines whether the next experiment starts from evidence or folklore.

## Climbing back out of the shader

Some of the clearest savings came from asking which outputs the classifier actually needed.

The final layer's feed-forward block was processing all 346 rows. Our decision used the final row. At that point, the feed-forward calculation for earlier rows had no further layer to feed and no part in the requested logits.

We sliced the final feed-forward input and residual down to 1 row. Attention and convolution still processed the prompt as required, including the state needed for later continuation. The last feed-forward block took the single-row path.

The recorded decisions fell to 223, 185, and 171 milliseconds. Compared with the original later cases at 269 and 219 milliseconds, that was an encouraging application-level improvement. Output-score drift stayed around a few millionths, and continuation tests checked that bulk prefill followed by an append still agreed with serial token appends.

The size of one timing drop exceeded what removing a roughly 7-millisecond feed-forward operation could explain. Dispatch and allocation changes might have contributed, alongside run-to-run variation. With 3 cases and a single pass per case, those numbers were observations rather than a stable latency distribution.

Then we tried reusing the prompt prefix across decisions. The prompts shared 47 of roughly 346 tokens before the board state diverged. We cached that base and branched each new decision from it, leaving each board state in its own branch.

In that measurement session, full decisions took 0.61 to 0.68 seconds and cached-tail decisions took 0.53 to 0.56 seconds, a reduction of about 15% to 18%. The output scores matched exactly in the shared-base append test.

Those absolute times were much slower than the earlier runs. The classifier harness submitted work, polled for completion, and slept for 1 millisecond between polls over roughly 30 sequential operations. Under host load, scheduler delays inflated its wall time while kernel timings stayed near their earlier values.

The prefix comparison was useful within that harness and session. Comparing its absolute time with a quiet earlier run would mix GPU work with a different amount of host waiting.

Even the prefix gain had a tile-shaped wrinkle. Removing 47 tokens left 299 fresh rows, taking the tiled work from 6 row tiles to 5. A 13.6% reduction in tokens produced a 16.7% reduction in row tiles. That was consistent with the observed gain, though it couldn't explain every part of the execution.

Moving more stable content to the front of the prompt could extend reuse. It would also change the model's input, so that revision needs its own task-quality evaluation. A faster answer still has to be the answer we wanted.

## What we brought back

The production kernel settled on 64-by-32 tiles with K16 staging, selected from 96 rows upward. We kept the producer-side activation fusion and the final-row feed-forward pruning. The cooperative matrix path and the wider tile were removed; half-precision staging stayed experimental.

The useful result is a set of measured changes with boundaries. Larger row tiles helped the classifier's prefill shape. Smaller staging helped despite adding barriers. Final-row pruning removed unnecessary calculation, and prefix reuse saved another 15% to 18% in its own application-level comparison. Those percentages come from different experiments and shouldn't be added together.

We still couldn't name the dominant GPU limiter with confidence. Weight traffic, workgroup-memory access, arithmetic, and occupancy remained candidates. The tested micro-optimizations were yielding smaller returns, while reuse offered a clearer next step.

That leaves another tempting idea waiting outside the shader: ternary weights. Their codes take half the bits of NF4, but the current design still decodes into float tiles, moves activations, multiplies, and synchronizes. We'll need the same representable weights in both formats, through the same tile geometry, before counting those saved bits as saved time.

We went looking for the milliseconds inside a 346-token decision. Some were hiding in tile dimensions, some in repeated work, and some in a sleeping host thread. A few survived because our tools couldn't yet tell us where to look.

The model kept choosing LEFT, LEFT, CW through almost all of it. Even when we'd loaded the matrix tiles the wrong way around.

---

*Experiment details, timings, correctness checks, and rejected variants are recorded in the [FFN optimization experiment log](ffn-prefill-experiments.md). These measurements describe one classifier on one development adapter using the native wgpu path; browser performance wasn't measured in this investigation.*
