#!/usr/bin/env python3
"""Run exactly one sealed manifest row, one process on one visible GPU.

This is intentionally a small dispatch wrapper, not a scheduler. The training
runner validates the seal, full matrix, hashes, environment and CUDA again.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from torchvision import models

from train_extension_run import DESIGN, ENVIRONMENT_TARGET, sha256, validate_manifest_row, validate_target_environment
from preprocessing import VARIANTS
from protocol_amendment import AMENDMENT, SCOPE_DECISION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RUNNER = HERE / "train_extension_run.py"
PREPROCESSING_PATH = HERE / "preprocessing.py"


def current_weight_hash(model_name: str) -> str:
    weight = getattr(models.get_model_weights(model_name), "IMAGENET1K_V1")
    filename = Path(urlparse(weight.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weight missing before retry: {path}")
    return sha256(path)


def verify_retry_candidate(output_dir: Path, manifest: Path, row: dict[str, str]) -> dict:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"No failed run directory to retry: {output_dir}")
    if output_dir.parent.resolve() != (HERE / "runs").resolve() or output_dir.name != row["run_id"]:
        raise ValueError("Retry target must be the canonical sealed run directory")
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("run_id") != row["run_id"] or status.get("status") != "failed":
        raise ValueError("Only an explicitly failed run may be retried")
    old = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "run_manifest_sha256": sha256(manifest),
        "design_lock_sha256": sha256(DESIGN),
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "swin_sensitivity_decision_sha256": sha256(SCOPE_DECISION),
        "variant_sha256": sha256(VARIANTS[row["variant"]]),
        "script_sha256": sha256(RUNNER),
        "preprocessing_sha256": sha256(PREPROCESSING_PATH),
        "environment_target_sha256": sha256(ENVIRONMENT_TARGET),
        "split_sha256": row["split_sha256"],
        "dataset_index_sha256": row["dataset_index_sha256"],
        "pretrained_weight_sha256": current_weight_hash(row["model"]),
    }
    if old.get("run_id") != row["run_id"] or old.get("technical_smoke") is not False:
        raise ValueError("Failed run identity/type differs from manifest")
    for key in ("dataset", "model", "variant"):
        if old.get(key) != row[key]:
            raise ValueError(f"Failed run identity differs from manifest: {key}")
    if old.get("training_seed") != int(row["training_seed"]):
        raise ValueError("Failed run identity differs from manifest: training_seed")
    for key, value in expected_hashes.items():
        if old.get(key) != value:
            raise ValueError(f"Failed run protocol/input changed: {key}; make a versioned amendment instead")
    return {"run_id": row["run_id"], "previous_status": status["status"], "manifest_sha256": expected_hashes["run_manifest_sha256"]}


def archive_failed_attempt(output_dir: Path, manifest: Path, row: dict[str, str]) -> Path:
    audit = verify_retry_candidate(output_dir, manifest, row)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = HERE / "runs" / "failed_attempts" / row["run_id"] / f"attempt_{stamp}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_dir.rename(destination)
    audit.update({"archived_utc": datetime.now(timezone.utc).isoformat(), "original_output_dir": str(output_dir), "archived_attempt_dir": str(destination)})
    (destination / "retry_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def preflight_retry_manifest(manifest: Path, row: dict[str, str]) -> None:
    """Validate the current seal and input files before moving failed evidence."""
    validate_manifest_row(SimpleNamespace(
        manifest=manifest, technical_smoke=False, run_id=row["run_id"],
        dataset=row["dataset"], model=row["model"], variant=row["variant"],
        training_seed=int(row["training_seed"]),
        split_file=ROOT / row["split_file"],
        dataset_index=ROOT / row["dataset_index_file"],
        output_dir=ROOT / row["output_dir"],
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly archive and retry one failed run under identical protocol hashes")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        matches = [r for r in csv.DictReader(handle) if r["run_id"] == args.run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for run ID {args.run_id}")
    row = matches[0]
    for key in ("split_file", "dataset_index_file", "output_dir"):
        if Path(row[key]).is_absolute() or ".." in Path(row[key]).parts:
            raise ValueError(f"Unsafe manifest path: {key}")
    output_dir = ROOT / row["output_dir"]
    if args.retry_failed:
        preflight_retry_manifest(args.manifest, row)
        audit = verify_retry_candidate(output_dir, args.manifest, row)
        print(f"EXPLICIT_RETRY {audit['run_id']} status={audit['previous_status']}", flush=True)
        if not args.dry_run:
            validate_target_environment()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; failed attempt was not archived")
            archived = archive_failed_attempt(output_dir, args.manifest, row)
            print(f"ARCHIVED_FAILED_ATTEMPT {archived}", flush=True)
    command = [
        sys.executable, str(RUNNER),
        "--dataset", row["dataset"],
        "--model", row["model"],
        "--variant", row["variant"],
        "--split-file", str(ROOT / row["split_file"]),
        "--dataset-index", str(ROOT / row["dataset_index_file"]),
        "--run-id", row["run_id"],
        "--training-seed", row["training_seed"],
        "--output-dir", str(ROOT / row["output_dir"]),
        "--manifest", str(args.manifest.resolve()),
    ]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
