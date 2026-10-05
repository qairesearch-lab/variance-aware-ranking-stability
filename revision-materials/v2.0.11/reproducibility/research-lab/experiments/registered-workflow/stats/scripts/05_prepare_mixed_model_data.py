#!/usr/bin/env python3
"""Prepare fixed-schema data for the registered mixed-effects model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from analysis_utils import (
    display_path,
    parse_metric,
    read_csv,
    require_analysis_role,
    require_columns,
    require_completed,
    resolve_path,
    runtime_metadata,
    write_csv,
    write_json,
)


FIELDS = ["metric_value", "metric", "model", "split", "seed", "checkpoint_policy", "dataset", "run_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--expected-analysis-role", default="primary_analysis")
    args = parser.parse_args()
    metrics_table = resolve_path(args.metrics_table)
    output_dir = resolve_path(args.output_dir)
    rows = read_csv(metrics_table)
    require_columns(
        rows,
        ["run_id", "dataset", "split_id", "training_seed", "model", "checkpoint_policy", args.metric],
        metrics_table,
    )
    require_analysis_role(rows, args.expected_analysis_role, metrics_table)
    require_completed(rows, metrics_table)

    output_rows: list[dict[str, object]] = []
    dataset_policy_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        metric_value = parse_metric(row.get(args.metric), args.metric, row["run_id"])
        output_rows.append(
            {
                "metric_value": metric_value,
                "metric": args.metric,
                "model": row["model"],
                "split": row["split_id"],
                "seed": row["training_seed"],
                "checkpoint_policy": row["checkpoint_policy"],
                "dataset": row["dataset"],
                "run_id": row["run_id"],
            }
        )
        dataset_policy_counts[(row["dataset"], row["checkpoint_policy"])] += 1

    output_path = output_dir / "mixed_model_input.csv"
    write_csv(output_path, output_rows, FIELDS)

    stratified_dir = output_dir / "mixed_model_inputs_by_dataset_checkpoint"
    stratified_outputs = []
    for (dataset, checkpoint_policy), _ in sorted(dataset_policy_counts.items()):
        subset = [
            row for row in output_rows
            if row["dataset"] == dataset and row["checkpoint_policy"] == checkpoint_policy
        ]
        subset_path = stratified_dir / f"mixed_model_input_{dataset}_{checkpoint_policy}.csv"
        write_csv(subset_path, subset, FIELDS)
        stratified_outputs.append({"dataset": dataset, "checkpoint_policy": checkpoint_policy, "path": display_path(subset_path), "rows": len(subset)})

    summary = {
        **runtime_metadata(Path(__file__).name),
        "metrics_table": display_path(metrics_table),
        "output_dir": display_path(output_dir),
        "metric": args.metric,
        "expected_analysis_role": args.expected_analysis_role,
        "rows": len(output_rows),
        "mixed_model_input": display_path(output_path),
        "checkpoint_policy_handling": (
            "Primary mixed-effects analysis is stratified by dataset and checkpoint_policy; "
            "the SAP formula skeleton is fit within each stratum."
        ),
        "formula_skeleton": "metric_value ~ model + (1 | split) + (1 | seed) + (1 | model:split)",
        "stratified_inputs": stratified_outputs,
    }
    write_json(output_dir / "mixed_model_input_summary.json", summary)
    print(summary)
    print("PREPARE_MIXED_MODEL_DATA_OK")


if __name__ == "__main__":
    main()
