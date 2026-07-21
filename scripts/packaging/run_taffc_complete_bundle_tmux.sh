#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
repo_root="/home/team/zhanghaonan/TAFFC/mprisk"
delivery="${repo_root}/outputs/deliveries/taffc_complete_bundle_20260721"
log_root="${repo_root}/outputs/deliveries/logs"

case "${mode}" in
  dry-run)
    session="taffc_bundle_dryrun_20260721"
    arguments="--dry-run"
    ;;
  build)
    session="taffc_bundle_build_20260721"
    arguments=""
    ;;
  build-resume1)
    session="taffc_bundle_build_resume1_20260721"
    arguments="--skip-media-stream-probe"
    ;;
  build-resume2)
    session="taffc_bundle_build_resume2_20260721"
    arguments="--skip-media-stream-probe --resume-existing-staging"
    ;;
  build-resume3)
    session="taffc_bundle_build_resume3_20260721"
    arguments="--skip-media-stream-probe --resume-existing-staging"
    ;;
  verify)
    session="taffc_bundle_verify_20260721"
    arguments="--verify-only --output ${delivery}"
    ;;
  *)
    echo "usage: $0 {dry-run|build|build-resume1|build-resume2|build-resume3|verify}" >&2
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
