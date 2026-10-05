#!/usr/bin/env python3
"""One human-started serial queue per GPU for a sealed E1 manifest.

Preview is the default. --execute requires the sealed manifest and target
environment, never retries a failed run, and stops on the first bad output.
Create runs/STOP_AFTER_CURRENT to pause after the current verified run, or use
--max-new-runs for bounded batches. --status is read-only operational progress.
Launch one instance for each distinct --gpu-index on the same host.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from audit_completed_manifest import expected_test_units, ordered_rows, read_csv, verify_run
from dispatch_manifest_run import current_weight_hash
from train_extension_run import HERE, ROOT, validate_manifest_row, validate_target_environment


def stop_file() -> Path:
    """A shared, user-created marker checked only between complete runs."""
    return HERE / "runs" / "STOP_AFTER_CURRENT"


def progress_snapshot(plan: list[dict[str, str]], gpu_count: int) -> dict:
    """Read-only operational progress; completion here is not a formal audit."""
    counts = {key: 0 for key in ("not_started", "running_reported", "completed_reported", "failed", "incomplete_output", "other")}
    active = []
    for position, row in enumerate(plan):
        output = ROOT / row["output_dir"]
        status_path = output / "run_status.json"
        if not status_path.is_file():
            counts["incomplete_output" if output.exists() else "not_started"] += 1
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            counts["other"] += 1
            continue
        category = {"completed": "completed_reported", "running": "running_reported", "failed": "failed"}.get(status, "other")
        counts[category] += 1
        if category == "running_reported":
            history_path = output / "history.csv"
            epoch = 0
            if history_path.is_file():
                with history_path.open(newline="", encoding="utf-8") as handle:
                    epoch = sum(1 for _ in csv.DictReader(handle))
            active.append({"run_id": row["run_id"], "assigned_gpu_index": position % gpu_count, "epochs_finished": epoch})
    disk = shutil.disk_usage(HERE)
    return {
        "stage_runs": len(plan), "gpu_count": gpu_count, "counts": counts, "active": active,
        "data_disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "stop_file_present": stop_file().exists(),
        "note": "Statuses come from run_status.json; running may be stale and completed still needs formal audit",
    }


def validate_manifest(manifest: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Empty manifest")
    first = rows[0]
    validate_manifest_row(SimpleNamespace(
        manifest=manifest, technical_smoke=False, run_id=first["run_id"],
        dataset=first["dataset"], model=first["model"], variant=first["variant"],
        training_seed=int(first["training_seed"]), split_file=ROOT / first["split_file"],
        dataset_index=ROOT / first["dataset_index_file"], output_dir=ROOT / first["output_dir"],
    ))


def visible_gpu_token(gpu_index: int, gpu_count: int) -> str:
    """Preserve a hosting platform's existing GPU mask, including UUID masks."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return str(gpu_index)
    tokens = [token.strip() for token in mask.split(",")]
    if len(tokens) < gpu_count or any(not token for token in tokens):
        raise RuntimeError(f"Existing CUDA_VISIBLE_DEVICES mask is incompatible with {gpu_count} GPUs")
    return tokens[gpu_index]


def execute_queue(manifest: Path, assigned: list[dict[str, str]], gpu_index: int, gpu_count: int,
                  max_new_runs: int | None = None) -> None:
    validate_target_environment()
    if torch.cuda.device_count() < gpu_count:
        raise RuntimeError(f"Requested {gpu_count} GPUs but only {torch.cuda.device_count()} visible")
    gpu_token = visible_gpu_token(gpu_index, gpu_count)
    lock_dir = HERE / "runs" / "queue_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"gpu_{gpu_index}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another queue already owns GPU index {gpu_index}") from exc
        weight_hashes = {}
        test_units = {}
        new_runs = 0
        for position, row in enumerate(assigned, 1):
            if stop_file().exists():
                print(f"[{position}/{len(assigned)}] PAUSED_AFTER_CURRENT stop_file={stop_file()}", flush=True)
                return
            output = ROOT / row["output_dir"]
            key = (row["dataset"], row["split_file"])
            if key not in test_units:
                test_units[key] = expected_test_units(ROOT / row["split_file"], row["dataset"])
            if output.exists():
                if row["model"] not in weight_hashes:
                    weight_hashes[row["model"]] = current_weight_hash(row["model"])
                verify_run(row, manifest, test_units[key], weight_hashes[row["model"]])
                print(f"[{position}/{len(assigned)}] VERIFIED_EXISTING {row['run_id']}", flush=True)
                continue
            if stop_file().exists():
                print(f"[{position}/{len(assigned)}] PAUSED_BEFORE_START stop_file={stop_file()}", flush=True)
                return
            print(f"[{position}/{len(assigned)}] START gpu={gpu_index} {row['run_id']}", flush=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_token
            command = [sys.executable, str(HERE / "dispatch_manifest_run.py"), "--manifest", str(manifest.resolve()), "--run-id", row["run_id"]]
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode:
                raise RuntimeError(f"Run failed; queue stopped without retry: {row['run_id']} exit={result.returncode}")
            if row["model"] not in weight_hashes:
                weight_hashes[row["model"]] = current_weight_hash(row["model"])
            verify_run(row, manifest, test_units[key], weight_hashes[row["model"]])
            print(f"[{position}/{len(assigned)}] VERIFIED_NEW {row['run_id']}", flush=True)
            new_runs += 1
            if max_new_runs is not None and new_runs >= max_new_runs:
                print(f"[{position}/{len(assigned)}] BATCH_LIMIT_REACHED new_runs={new_runs}", flush=True)
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("e0", "e1", "sensitivity", "all"), default="e0")
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run the assigned queue; preview is the default")
    parser.add_argument("--status", action="store_true", help="Read-only progress snapshot for the whole selected stage")
    parser.add_argument("--max-new-runs", type=int, help="Exit cleanly after this many newly completed runs on this GPU")
    args = parser.parse_args()
    if args.gpu_count < 1 or not 0 <= args.gpu_index < args.gpu_count:
        raise ValueError("GPU index must be within 0 .. gpu_count-1")
    if args.execute and args.status:
        parser.error("--execute and --status are mutually exclusive")
    if args.max_new_runs is not None and (not args.execute or args.max_new_runs < 1):
        parser.error("--max-new-runs requires --execute and a positive integer")
    rows = read_csv(args.manifest)
    validate_manifest(args.manifest, rows)
    plan = ordered_rows(rows, args.stage)
    if args.status:
        print(json.dumps(progress_snapshot(plan, args.gpu_count), indent=2, sort_keys=True), flush=True)
        return
    assigned = [row for index, row in enumerate(plan) if index % args.gpu_count == args.gpu_index]
    print(json.dumps({
        "mode": "execute" if args.execute else "preview_only", "stage": args.stage,
        "gpu_index": args.gpu_index, "gpu_count": args.gpu_count,
        "stage_runs": len(plan), "assigned_runs": len(assigned),
        "first_run_id": assigned[0]["run_id"] if assigned else None,
        "last_run_id": assigned[-1]["run_id"] if assigned else None,
    }, indent=2, sort_keys=True), flush=True)
    if args.execute:
        execute_queue(args.manifest, assigned, args.gpu_index, args.gpu_count, args.max_new_runs)


if __name__ == "__main__":
    main()
