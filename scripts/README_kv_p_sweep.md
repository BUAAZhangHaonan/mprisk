# Qwen3-VL-8B KV-cache prompt sweep

The runner must first produce a JSON measurement file with one row for each
prompt count. The plotting consumer accepts a top-level rows, points, or
results list. Each row needs P or prompt_count; latency and stability fields
may be nested:

- latency.median_seconds, latency.p95_seconds, and optional latency.n
- stability.pattern_agreement and/or stability.metric_convergence

The consumer keeps absent metrics as null and marks them Pending; it never
creates values for incomplete P runs.

Command:

    PYTHONPATH=src python scripts/export_kv_p_sweep_curve.py \
      --input outputs/kv_validation/delivery_20260716/qwen3_vl_kv_p_sweep.json \
      --output-dir paper/figures/generated/kv_p_sweep_v1

The output directory contains kv_p_sweep_curve.csv,
kv_p_sweep_curve.json, kv_p_sweep_latency_stability.pdf, and a PNG
preview. The JSON records the source SHA-256, expected P values
1/2/4/8/16/32/64, and any missing counts.

