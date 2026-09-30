# Browser qualification startup correction

## Hypothesis

The browser gate at runtime revision `bf32d7a` failed on Model100k FP16 with
CDP error `-32000`, `Execution context was destroyed.` The runner created a
target already navigating to `qualification.html`, attached its debugger,
and immediately awaited `Runtime.evaluate`. That evaluation could start in
the target's initial blank document before navigation replaced its context.

The competing hypothesis was a renderer or GPU failure while loading the
438 MiB FP16 asset. The original failure log remains unchanged outside Git
in the workspace experiment's `runs/browser-baseline/model100k-fp16.log`.
Its SHA-256 is
`f5978b7e8065921a2a1c54d4a2677167f0a1dcd44f80c92569e3283fea4fde03`.

## Trace

An instrumented copy of the original runner retained CDP messages, commands,
loopback asset requests, and Chrome stderr. A natural run passed the largest
FP16 bundle after its extra debugger round trips let navigation finish first.

A controlled 250 ms delay on the HTML response reproduced the original CDP
error. `Runtime.evaluate` started in the initial `://` context at approximately
1031 ms, and its context-destroyed error arrived at approximately 1243 ms.
The trace contains no WASM, configuration, tokenizer, or weight request before
the failure, and Chrome was still alive when the runner cleaned up.

This reproduction isolates the document-navigation race before model loading.
The retained natural pass also demonstrates that the actual FP16 bundle could
execute in this Chrome configuration.

## Fix

The runner creates the debugger target at `about:blank`, connects, enables Page
lifecycle events, and installs its listener before explicitly navigating.
It waits for a `load` event whose frame and loader IDs match the `Page.navigate`
response. Events received before that response are retained, so a fast load
is observed; an initial blank-document load cannot release the wait.

Each CDP command receives a distinct request ID. Navigation errors and missing
document loaders fail the command, and the lifecycle listener and timer are
removed on both success and failure. The existing overall deadline applies.
Failure output now includes Chrome's retained stderr tail.

Model assets, WASM, browser flags, prompts, class count, LUT2 policy, numerical
comparisons, and native evaluation source remain unchanged. The runner adds
no retry or additional inference attempt.

## Test

The same controlled 250 ms response delay passes with the correction. Its trace
records an initial blank load with one loader ID, the navigation response with
a different loader ID, the corresponding qualification-page load, and only
then `Runtime.evaluate`. Actual FP16 inference passes with maximum absolute
comparison error `3.814697265625e-6`, including full/cached parity, native
comparison, empty-input rejection, and successful recovery.

Portable checks are `node --check scripts/check_browser.mjs`, `npm test`
(3 existing tests), and `git diff --check`. The controlled real-browser
reproduction supplies the regression evidence for this asynchronous startup
boundary.

Clean-source bundle qualification runs serially for `model100k-fp16`,
`model100k-int8`, `model100k-nf4`, `model100k-ternary`, and
`polyomino-qat-mixed`, using the original hash-bound expected fixtures and
`classifier-texts.json`. Each invocation keeps 8 classes, absolute and relative
tolerances of `1e-4`, a 600-second deadline, and `MINIFIELD_TELEMETRY=0`.
The actual outcomes and source identities belong in the retained reports.

## Evidence

Generated diagnostics, archived runners, logs, and reports stay outside Git
under the named workspace experiment's `runs/browser-startup-diagnostics/`.
The clean-source five-bundle reports belong in
`runs/browser-startup-qualification/`. The original failure and expected
fixtures stay in `runs/browser-baseline/`.

The reused WASM package was built from the native baseline source and copied
only into this worktree's ignored `web/pkg/`. Its WASM SHA-256 is
`d4950cca4d93fe59b8579683d023df12591b86580fbe11a3cf282c2cb7d86329`.
Qualification reports bind that digest alongside each asset and prompt digest.

This correction stabilizes browser qualification startup. Frozen native
campaign criteria, executable hashes, and timing policy retain their original
identities.
