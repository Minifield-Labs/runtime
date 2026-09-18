# Runtime experiment reports

`report.html` is a minimal standalone template for one experiment comparison. It has no build step and no dependencies. It uses the landing page rules: graphite `#0b0c0d`, warm white `#efeeeb`, ember `#f09169`, Manrope for prose, IBM Plex Mono for data.

Copy it per run, then edit only the JSON block with id `report-data`:

- `titleHtml`, `question`, `runId`, `pills`
- `metrics`: three headline figures, each with treatment delta and control base
- `device`, `engine`, `decision`, `evidence`
- `record`: the run record rows from `experiments/run-record-template.md`

Keep template placeholders (`to record`, `none recorded`) until measured values exist. Do not invent results. Keep run evidence outside Git; this template holds only concise conclusions.
