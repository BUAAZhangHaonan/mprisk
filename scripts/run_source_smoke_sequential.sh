#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 MODEL_KEY [MODEL_KEY ...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/team/zhanghaonan/miniconda3/envs/mind-py311/bin/python"
config="$repo_root/configs/cache/complete_cache_matrix.yaml"
log_root="$repo_root/outputs/cache_smoke_matrix_20260722/source"
driver_log="$log_root/sequential_driver.log"

mkdir -p "$log_root"
cd "$repo_root"

for model_key in "$@"; do
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$model_key" | tee -a "$driver_log"
  PYTHONPATH="$repo_root/src" "$python_bin" \
    scripts/run_cache_smoke_matrix.py \
    --config "$config" \
    --domain source \
    --model "$model_key" \
    --execute 2>&1 | tee -a "$driver_log"
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$model_key" | tee -a "$driver_log"
done
