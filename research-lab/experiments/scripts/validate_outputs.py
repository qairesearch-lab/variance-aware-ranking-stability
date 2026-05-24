#!/usr/bin/env python3
"""Validate fixed-schema per-run outputs for the registered workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FILES = [
    "run_config.json",
    "metrics.json",
    "predictions.csv",
    "history.csv",
    "checkpoint_manifest.json",
    "run_status.json",
]
RUN_STATUS_ENUM = {"completed", "failed", "rerun_completed"}
JOIN_KEYS = [
    "run_id",
    "dataset",
    "split_id",
    "split_seed",
    "training_seed",
    "model",
    "checkpoint_policy",
    "config_hash",
    "split_hash",
    "analysis_role",
]
PREDICTION_REQUIRED = JOIN_KEYS + ["sample_id", "relative_path", "label_true", "label_pred"]
METRIC_REQUIRED = JOIN_KEYS + [
    "metric_source",
    "accuracy",
    "balanced_accuracy",
    "roc_auc_macro_ovr",
    "pr_auc_macro",
    "brier_score_multiclass",
    "calibration_intercept",
    "calibration_slope",
]
HISTORY_REQUIRED = ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy", "learning_rate"]
CHECKPOINT_REQUIRED = ["run_id", "checkpoint_policy", "checkpoint_used_for_evaluation", "checkpoints", "mode"]
RUN_STATUS_REQUIRED = ["run_id", "status", "execution_mode", "started_at_utc", "ended_at_utc", "message"]


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run_output(output_dir: Path, manifest_row: dict[str, str] | None = None) -> list[str]:
    errors: list[str] = []
    for filename in REQUIRED_FILES:
        if not (output_dir / filename).exists():
            errors.append(f"Missing required file: {output_dir / filename}")
    if errors:
        return errors

    run_config = load_json(output_dir / "run_config.json")
    metrics = load_json(output_dir / "metrics.json")
    predictions = read_csv(output_dir / "predictions.csv")
    history = read_csv(output_dir / "history.csv")
    checkpoint_manifest = load_json(output_dir / "checkpoint_manifest.json")
    run_status = load_json(output_dir / "run_status.json")
    for field in RUN_STATUS_REQUIRED:
        if field not in run_status:
            errors.append(f"Missing run_status.json field {field}: {output_dir}")
    if run_status.get("status") not in RUN_STATUS_ENUM:
        errors.append(f"Invalid run_status.json status {run_status.get('status')}: {output_dir}")
    if run_status.get("status") == "failed":
        return errors

    if not predictions:
        errors.append(f"predictions.csv has no data rows: {output_dir}")
    else:
        prediction_columns = set(predictions[0].keys())
        for column in PREDICTION_REQUIRED:
            if column not in prediction_columns:
                errors.append(f"Missing predictions.csv column {column}: {output_dir}")
        probability_columns = [col for col in predictions[0].keys() if col.startswith("prob_class_")]
        if not probability_columns:
            errors.append(f"No probability columns found in predictions.csv: {output_dir}")
        for index, prediction in enumerate(predictions[:25]):
            for column in PREDICTION_REQUIRED + probability_columns:
                if prediction.get(column) in {None, ""}:
                    errors.append(f"Empty predictions.csv field {column} row {index + 1}: {output_dir}")
            probability_sum = sum(float(prediction[col]) for col in probability_columns)
            if abs(probability_sum - 1.0) > 1e-3:
                errors.append(f"Prediction probabilities do not sum to 1 in row {index + 1}: {output_dir}")

    for field in METRIC_REQUIRED:
        if field not in metrics:
            errors.append(f"Missing metrics.json field {field}: {output_dir}")
    for field in JOIN_KEYS:
        if field in metrics and metrics[field] in {None, ""}:
            errors.append(f"Empty metrics.json field {field}: {output_dir}")
    if not history:
        errors.append(f"history.csv has no data rows: {output_dir}")
    else:
        history_columns = set(history[0].keys())
        for column in HISTORY_REQUIRED:
            if column not in history_columns:
                errors.append(f"Missing history.csv column {column}: {output_dir}")
        for index, row in enumerate(history[:25]):
            for column in HISTORY_REQUIRED:
                if row.get(column) in {None, ""}:
                    errors.append(f"Empty history.csv field {column} row {index + 1}: {output_dir}")
    for field in CHECKPOINT_REQUIRED:
        if field not in checkpoint_manifest:
            errors.append(f"Missing checkpoint_manifest.json field {field}: {output_dir}")
    if checkpoint_manifest.get("run_id") != run_status.get("run_id"):
        errors.append(f"checkpoint_manifest/run_status run_id mismatch: {output_dir}")
    if run_config.get("manifest_row", {}).get("run_id") != run_status.get("run_id"):
        errors.append(f"run_config/run_status run_id mismatch: {output_dir}")

    if manifest_row:
        for key in JOIN_KEYS:
            expected = str(manifest_row[key])
            if str(metrics.get(key)) != expected:
                errors.append(f"metrics.json {key} mismatch for {manifest_row['run_id']}")
            for prediction in predictions[:10]:
                if str(prediction.get(key)) != expected:
                    errors.append(f"predictions.csv {key} mismatch for {manifest_row['run_id']}")
                    break
    return errors


def resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--output-dir", action="append")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    manifest_rows = read_manifest(manifest_path)

    targets: list[tuple[Path, dict[str, str] | None]] = []
    if args.run_id:
        for run_id in args.run_id:
            if run_id not in manifest_rows:
                raise SystemExit(f"Run id not found in manifest: {run_id}")
            targets.append((resolve_output_dir(manifest_rows[run_id]["output_dir"]), manifest_rows[run_id]))
    if args.output_dir:
        for output_dir in args.output_dir:
            targets.append((resolve_output_dir(output_dir), None))
    if not targets:
        targets = [(resolve_output_dir(row["output_dir"]), row) for row in manifest_rows.values()]

    errors: list[str] = []
    for output_dir, row in targets:
        errors.extend(validate_run_output(output_dir, row))
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "checked": len(targets), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)
    print("RUN_OUTPUT_SCHEMA_VALIDATION_OK")


if __name__ == "__main__":
    main()
