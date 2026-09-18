# CR02a strict f32 rounding sentinels

Copied unchanged from the independent coordinator fixture:
experiments/2026-09-16-two-stage-routing/runs/coordinator-runtime-cr02a-resource-002/rounding-sentinels.json.

The coordinator generator is make_rounding_sentinel.py in the same source directory. This
fixture tests exact f32 bit boundaries for RoPE vector combinations and short-convolution tap
accumulation; it does not replace the 18-case mathematical model-ops-001 fixture or its tolerance.
