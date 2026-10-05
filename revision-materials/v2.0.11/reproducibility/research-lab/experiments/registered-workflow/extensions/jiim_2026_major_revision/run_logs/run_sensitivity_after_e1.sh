#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/CervicalCancerV2/research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision
export TORCH_HOME=/root/autodl-tmp/jiim-torch-cache
PY=/root/jiim-env/bin/python
MANIFEST=run_manifests/jiim_extension_300_v0.6.csv
LOGDIR=run_logs
ORCH_PID="$(cat "$LOGDIR/e0_e1_orchestrator.pid")"
echo "$(date '+%F %T %Z') WAIT_FOR_E0_E1 pid=$ORCH_PID"
while true; do
  snapshot="$("$PY" run_manifest_queue.py --manifest "$MANIFEST" --stage e1 --gpu-count 2 --gpu-index 0 --status)"
  if printf '%s\n' "$snapshot" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); c=d["counts"]; n=d["stage_runs"]; sys.exit(0 if n==270 and c["completed_reported"]==270 and all(c[k]==0 for k in ("failed","incomplete_output","not_started","other","running_reported")) else (3 if c["failed"] else 1))'; then
    echo "$(date '+%F %T %Z') E0_E1_VERIFIED_COMPLETE"
    break
  else
    rc=$?
    if (( rc == 3 )); then echo "$(date '+%F %T %Z') ABORT_FAILED_RUN_IN_E0_E1"; exit 3; fi
    if ! kill -0 "$ORCH_PID" 2>/dev/null; then echo "$(date '+%F %T %Z') ABORT_ORCHESTRATOR_EXITED_BEFORE_E0_E1_COMPLETE"; exit 1; fi
    sleep 60
  fi
done

echo "$(date '+%F %T %Z') STAGE_START sensitivity"
pids=()
for gpu in 0 1; do
  ( export TORCH_HOME=/root/autodl-tmp/jiim-torch-cache; exec "$PY" -u run_manifest_queue.py --manifest "$MANIFEST" --stage sensitivity --gpu-count 2 --gpu-index "$gpu" --execute ) >"$LOGDIR/sensitivity_gpu${gpu}_unattended.log" 2>&1 &
  pids+=("$!")
  echo "$(date '+%F %T %Z') LAUNCHED sensitivity gpu=$gpu pid=${pids[-1]}"
done
rc=0
for pid in "${pids[@]}"; do if wait "$pid"; then echo "$(date '+%F %T %Z') QUEUE_EXIT sensitivity pid=$pid code=0"; else rc=$?; echo "$(date '+%F %T %Z') QUEUE_EXIT sensitivity pid=$pid code=$rc"; fi; done
if (( rc != 0 )); then echo "$(date '+%F %T %Z') SENSITIVITY_QUEUE_ERROR"; exit "$rc"; fi
snapshot="$("$PY" run_manifest_queue.py --manifest "$MANIFEST" --stage sensitivity --gpu-count 2 --gpu-index 0 --status)"
printf '%s\n' "$snapshot"
printf '%s\n' "$snapshot" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); c=d["counts"]; assert d["stage_runs"]==30 and c["completed_reported"]==30 and all(c[k]==0 for k in ("failed","incomplete_output","not_started","other","running_reported")), "sensitivity stage is not fully complete"'
echo "$(date '+%F %T %Z') ALL_300_RUNS_COMPLETE"
