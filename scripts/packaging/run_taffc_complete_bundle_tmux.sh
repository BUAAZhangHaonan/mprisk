#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
repo_root="/home/team/zhanghaonan/TAFFC/mprisk"
delivery="${repo_root}/outputs/deliveries/taffc_complete_bundle_20260721"
candidate="${repo_root}/outputs/deliveries/taffc_complete_bundle_20260721_dataset_reorg_candidate"
log_root="${repo_root}/outputs/deliveries/logs"

case "${mode}" in
  reorg-dry-run)
    session="taffc_bundle_reorg_dryrun_20260721"
    arguments="--dry-run --workers 1 --output ${candidate}"
    ;;
  reorg-build)
    session="taffc_bundle_reorg_build_20260721"
    arguments="--skip-media-stream-probe --workers 1 --output ${candidate}"
    ;;
  reorg-verify)
    session="taffc_bundle_reorg_verify_20260721"
    arguments="--verify-only --workers 1 --output ${candidate}"
    ;;
  reorg-promote)
    session="taffc_bundle_reorg_promote_20260721"
    arguments="--workers 1 --promote-candidate ${candidate} --verified-status ${log_root}/taffc_bundle_reorg_verify_20260721.status --verified-log ${log_root}/taffc_bundle_reorg_verify_20260721.log --output ${delivery}"
    ;;
  *)
    echo "usage: $0 {reorg-dry-run|reorg-build|reorg-verify|reorg-promote}" >&2
    exit 2
    ;;
esac

log_path="${log_root}/${session}.log"
status_path="${log_root}/${session}.status"
mkdir -p "${log_root}"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 1
fi
if [[ -e "${log_path}" || -e "${status_path}" ]]; then
  echo "durable log/status already exists for ${session}" >&2
  exit 1
fi

run_command="cd ${repo_root} && set -o pipefail; nice -n 15 ionice -c 3 python3 scripts/packaging/build_taffc_complete_bundle.py ${arguments} 2>&1 | tee ${log_path}; code=\${PIPESTATUS[0]}; printf '%s\\n' \"\${code}\" > ${status_path}; exit \"\${code}\""
printf -v quoted_command '%q' "${run_command}"
tmux new-session -d -s "${session}" "bash -lc ${quoted_command}"

echo "session=${session}"
echo "log=${log_path}"
echo "status=${status_path}"
