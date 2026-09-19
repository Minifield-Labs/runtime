# Runtime reports

Open `report.html` in a browser. It's a self-contained report with embedded Manrope and IBM Plex Mono fonts, the Minifield mark, and an interactive SVG chart. It works offline, without a build step, server, package installation, or sibling repository. The included figures are **synthetic design data**, labelled in the report.

## Reuse

Copy `report.html` into a run's evidence folder and edit the JSON block with `id="report-data"`. Put iterations in chronological order. The **last iteration** supplies the headline metrics; the first supplies the baseline. A single iteration is supported.

For automated reporting, supply the same JSON shape to the optional Node 22+ helper:

```sh
node reports/render.mjs runs/metrics.json runs/report.html
```

The helper validates the data, escapes embedded JSON, and creates a standalone HTML file. It refuses to overwrite an existing report. The output directory must already exist.

Each report lives in its own directory under `reports/<model>-<platform>/` with a canonical `metrics.json` and the rendered `report.html`, committed together for every accepted performance-affecting change. Keep bulk evidence (logs, raw captures, model files) outside Git; record its location and hashes in the `evidence` field.

Set `sample: false` only after replacing every illustrative value with a measurement or `null`. Missing measurements display as `N/A`; they never become zero. Strings render as text. If editing embedded JSON by hand, write a literal `<` as `\u003c` so strings cannot close the script block. The helper handles this automatically.

## Data fields

The report uses `schemaVersion: 1`.

| Field | Meaning |
| --- | --- |
| `model.name` | Model or delivered bundle name. Required. |
| `model.parameters` | Parameter count as an integer, displayed as M or B. |
| `model.quantization` | Exact weight format, such as Q4_K_M, Q8_0, or BF16. |
| `io.inputTokens` | Input tokens per sequence. Record cached tokens in the measurement notes. |
| `io.contextTokens` | Configured context limit, including input and generated tokens. |
| `io.outputTokens` | Actual generated tokens per sequence, not the requested maximum. |
| `io.batchSize` | Number of concurrent sequences. |
| `platform` | Hardware, installed memory, runtime/backend, engine, engine version, and environment. Runtime examples: WASM SIMD, Metal, CUDA, WebGPU, CPU. |
| `measurement.statistic` | Aggregation used for the displayed results, such as Median or Mean. |
| `measurement.repetitions` | Measured runs per iteration, excluding warmups. |
| `measurement.cache` | Model and prefix cache state. |
| `measurement.memoryScope` | What the memory measurements include and how they were collected. |
| `measurement.notes` | Timing boundaries, warmups, workload identity, decoding policy, exclusions, and the intended change. |
| `date`, `evidence` | Optional date and evidence reference. Rendered as plain text. |
| `iterations` | At least 1 row, with a unique string `id`, optional `name`, and measurements below. |

Optional text and measurements can be `null` or omitted. `sample` must be an explicit boolean. The model, I/O, platform, and measurement objects must exist, even when some fields are unknown.

Each iteration supplies:

| Metric | Unit and boundary |
| --- | --- |
| `outputTokensPerSecond` | Aggregate decode tokens/s across the batch. Normally excludes the first token. Document the timing boundary and treatment of uneven sequence lengths. |
| `prefillTokensPerSecond` | Newly processed input tokens across the batch / measured prefill seconds. Exclude reused prefix tokens from the numerator. |
| `ttftMs` | Request submission to the first emitted token, in milliseconds. Record whether queueing, tokenization, and cold loading are included. |
| `modelMemoryBytes` | Measured retained model memory in bytes, including only the allocations named by `memoryScope`. Displayed in GiB, using 2³⁰ bytes per GiB. |
| `peakMemoryBytes` | Peak bytes for the **same allocation scope** over the request. Use `null` when unavailable. Process RSS and model allocation memory need separate reports or an explicitly shared scope. |

The report formats supplied measurements; it doesn't infer TTFT from prefill speed, estimate memory from file size, or calculate benchmark aggregates from raw samples. The source table uses the same display precision as the headline. The embedded JSON retains supplied precision.

## Comparisons

Use one model, quantization, workload, platform, aggregation, repetition count, and cache policy per report. Only compare iterations collected under those shared conditions. Start a separate report when these change. Put exact build revisions in iteration names or the evidence record.

The chart switches between output speed, prefill speed, TTFT, and model memory. Each uses its own units and a zero-based axis. It connects adjacent measured iterations with straight segments, preserves gaps, and marks the first iteration as a dashed baseline. Hover or keyboard-focus a point for its value; the disclosure contains the accessible data table.

Percentage changes use the first iteration. A missing or zero baseline suppresses the percentage. Higher throughput and lower TTFT/memory are labelled according to their direction. With fewer than 2 measured points, the graph explains what's missing.

## Sharing and printing

Share the HTML file itself. Fonts, styles, chart logic, and measurements travel with it; the report makes no network requests. Font licenses are embedded and also retained in `font-licenses/`.

`Print / Save PDF` opens the browser print dialog. The print stylesheet uses an ink-friendly landscape sheet and keeps the selected chart metric. Disable browser headers and footers for a clean PDF. Measurement notes remain available in the HTML disclosure.

The report handles small screens and has no automatic animation. Invalid data produces a readable error in place of results.
