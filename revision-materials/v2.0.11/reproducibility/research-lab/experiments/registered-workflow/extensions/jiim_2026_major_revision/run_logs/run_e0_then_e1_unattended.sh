#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/CervicalCancerV2/research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision
export TORCH_HOME=/root/autodl-tmp/jiim-torch-cache
PY=/root/jiim-env/bin/python
MANIFEST=run_manifests/jiim_extension_300_v0.6.csv
LOGDIR=run_logs

check_stage() {
  local stage="$1" expected="$2" snapshot
  snapshot="$("$PY" run_manifest_queue.py --manifest "$MANIFEST" --stage "$stage" --gpu-count 2 --gpu-index 0 --status)"
  printf '%s\n' "$snapshot"
  printf '%s\n' "$snapshot" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); c=d["counts"]; n=d["stage_runs"]; expected=int(sys.argv[1]); print("CHECK",n,c); assert n==expected and c["completed_reported"]==expected and all(c[k]==0 for k in ("failed","incomplete_output","not_started","other","running_reported")), "stage is not fully complete"' "$expected"
}

run_stage() {
  local stage="$1" gpu pid rc=0
  echo "$(date '+%F %T %Z') STAGE_START $stage"
  local pids=()
  for gpu in 0 1; do
    ( export TORCH_HOME=/root/autodl-tmp/jiim-torch-cache; exec "$PY" -u run_manifest_queue.py --manifest "$MANIFEST" --stage "$stage" --gpu-count 2 --gpu-index "$gpu" --execute ) >"$LOGDIR/${stage}_gpu${gpu}_unattended.log" 2>&1 &
    pids+=("$!")
    echo "$(date '+%F %T %Z') LAUNCHED stage=$stage gpu=$gpu pid=${pids[-1]}"
  done
  for pid in "${pids[@]}"; do
    if wait "$pid"; then echo "$(date '+%F %T %Z') QUEUE_EXIT stage=$stage pid=$pid code=0"; else rc=$?; echo "$(date '+%F %T %Z') QUEUE_EXIT stage=$stage pid=$pid code=$rc"; fi
  done
  if (( rc != 0 )); then echo "$(date '+%F %T %Z') ABORT_AFTER_QUEUE_ERROR stage=$stage"; exit "$rc"; fi
}

trap 'rc=$?; echo "$(date "+%F %T %Z") ORCHESTRATOR_EXIT code=$rc"' EXIT
echo "$(date '+%F %T %Z') ORCHESTRATOR_START"
# E0 must be allowed to continue if partially complete; the queue resumes only missing runs.
run_stage e0
check_stage e0 150
echo "$(date '+%F %T %Z') E0_COMPLETE; proceeding to E1"
run_stage e1
check_stage e1 270
echo "$(date '+%F %T %Z') E0_AND_E1_COMPLETE"
