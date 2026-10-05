#!/usr/bin/env python3
"""Collect per-run metrics, metadata, and predictions into analysis tables."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from analysis_utils import (
    display_path,
    file_metadata,
    load_json,
    read_csv,
    require_analysis_role,
    resolve_path,
    runtime_metadata,
    write_csv,
    write_json,
)


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
REQUIRED_MANIFEST_COLUMNS = JOIN_KEYS + ["output_dir", "status"]


def collect(manifest: Path, output_dir: Path, expected_analysis_role: str) -> dict[str, object]:
    manifest_rows = read_csv(manifest)
    require_analysis_role(manifest_rows, expected_analysis_role, manifest)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    errors: list[str] = []
    manifest_status_counts: Counter[str] = Counter()
    run_status_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    prediction_rows_by_dataset: Counter[str] = Counter()

    columns = set(manifest_rows[0].keys()) if manifest_rows else set()
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in columns]
    if missing:
        raise SystemExit(f"Missing manifest columns in {display_path(manifest)}: {', '.join(missing)}")

    for row in manifest_rows:
        manifest_status_counts[row.get("status", "")] += 1
        run_dir = resolve_path(row["output_dir"])
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        run_status_path = run_dir / "run_status.json"
        if not metrics_path.exists() or not predictions_path.exists() or not run_status_path.exists():
            errors.append(f"Missing output files for {row['run_id']}: {display_path(run_dir)}")
            continue

        metrics = load_json(metrics_path)
        run_status = load_json(run_status_path)
        actual_status = run_status.get("status", "")
        run_status_counts[actual_status] += 1
        if actual_status != "completed":
            errors.append(f"Run is not completed: {row['run_id']} status={actual_status}")

        metric_row = {key: row[key] for key in JOIN_KEYS}
        for key, value in metrics.items():
            if key not in metric_row and not isinstance(value, (dict, list)):
                metric_row[key] = value
        metric_row["run_status"] = actual_status
        metric_row["execution_mode"] = run_status.get("execution_mode")
        metrics_rows.append(metric_row)
        dataset_counts[row["dataset"]] += 1

        predictions = read_csv(predictions_path)
        prediction_rows.extend(predictions)
        prediction_rows_by_dataset[row["dataset"]] += len(predictions)

    if errors:
        raise SystemExit("; ".join(errors))

    metrics_table = output_dir / "run_metrics_table.csv"
    predictions_table = output_dir / "predictions_table.csv"
    write_csv(metrics_table, metrics_rows)
    write_csv(predictions_table, prediction_rows)

    summary = {
        **runtime_metadata(Path(__file__).name),
        "manifest": display_path(manifest),
        "output_dir": display_path(output_dir),
        "expected_analysis_role": expected_analysis_role,
        "manifest_rows": len(manifest_rows),
        "runs_collected": len(metrics_rows),
        "prediction_rows": len(prediction_rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "prediction_rows_by_dataset": dict(sorted(prediction_rows_by_dataset.items())),
        "manifest_status_counts": dict(sorted(manifest_status_counts.items())),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "status_note": (
            "Manifest status records the locked pre-execution state; per-run run_status.json "
            "records actual execution state and is authoritative for analysis readiness."
        ),
        "outputs": {
            "metrics_table": file_metadata(metrics_table),
            "predictions_table": file_metadata(predictions_table),
        },
    }
    write_json(output_dir / "collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-analysis-role", default="primary_analysis")
    args = parser.parse_args()
    manifest = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    summary = collect(manifest, output_dir, args.expected_analysis_role)
    print(summary)
    print("COLLECT_RUN_OUTPUTS_OK")


if __name__ == "__main__":
    main()
