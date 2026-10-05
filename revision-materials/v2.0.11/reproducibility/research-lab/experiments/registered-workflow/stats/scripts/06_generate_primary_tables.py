#!/usr/bin/env python3
"""Generate primary analysis summary tables from prepared statistical outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

from analysis_utils import (
    display_path,
    parse_metric,
    read_csv,
    resolve_path,
    runtime_metadata,
    write_csv,
    write_json,
)


def copy_rows(source: Path, destination: Path) -> int:
    rows = read_csv(source) if source.exists() else []
    if rows:
        write_csv(destination, rows)
    return len(rows)


def performance_summary(metrics_table: Path, metric: str) -> list[dict[str, object]]:
    rows = read_csv(metrics_table)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["checkpoint_policy"], row["model"])
        grouped[key].append(parse_metric(row.get(metric), metric, row["run_id"]))

    output = []
    for (dataset, checkpoint_policy, model), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "checkpoint_policy": checkpoint_policy,
                "model": model,
                "metric": metric,
                "n_runs": len(values),
                "mean": mean(values),
                "sd": pstdev(values) if len(values) > 1 else 0.0,
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-output-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument("--model-objects-dir", default="research-lab/experiments/registered-workflow/stats/model-objects")
    parser.add_argument("--metric", default="balanced_accuracy")
    args = parser.parse_args()

    analysis_output_dir = resolve_path(args.analysis_output_dir)
    tables_dir = resolve_path(args.tables_dir)
    model_objects_dir = resolve_path(args.model_objects_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_table = analysis_output_dir / "run_metrics_table.csv"
    performance_rows = performance_summary(metrics_table, args.metric)
    performance_path = tables_dir / "primary_performance_summary.csv"
    write_csv(performance_path, performance_rows)

    copied = {
        "ranking_stability_summary.csv": copy_rows(
            analysis_output_dir / "ranking_stability_table.csv",
            tables_dir / "ranking_stability_summary.csv",
        ),
        "rank_distribution_summary.csv": copy_rows(
            analysis_output_dir / "rank_distribution_table.csv",
            tables_dir / "rank_distribution_summary.csv",
        ),
        "selection_probability_summary.csv": copy_rows(
            analysis_output_dir / "selection_probability_table.csv",
            tables_dir / "selection_probability_summary.csv",
        ),
        "paired_bootstrap_comparison_summary.csv": copy_rows(
            analysis_output_dir / "paired_bootstrap_comparison_table.csv",
            tables_dir / "paired_bootstrap_comparison_summary.csv",
        ),
        "mixed_effects_variance_attribution_summary.csv": copy_rows(
            model_objects_dir / "mixed_effects_variance_components.csv",
            tables_dir / "mixed_effects_variance_attribution_summary.csv",
        ),
        "mixed_effects_model_status_summary.csv": copy_rows(
            model_objects_dir / "mixed_effects_model_status.csv",
            tables_dir / "mixed_effects_model_status_summary.csv",
        ),
    }

    manifest = {
        **runtime_metadata(Path(__file__).name),
        "analysis_output_dir": display_path(analysis_output_dir),
        "tables_dir": display_path(tables_dir),
        "model_objects_dir": display_path(model_objects_dir),
        "metric": args.metric,
        "primary_performance_summary_rows": len(performance_rows),
        "copied_table_rows": copied,
        "available_analysis_outputs": sorted(path.name for path in analysis_output_dir.glob("*") if path.is_file()),
        "available_model_outputs": sorted(path.name for path in model_objects_dir.glob("*") if path.is_file()),
        "status": "completed",
    }
    write_json(tables_dir / "primary_analysis_audit_manifest.json", manifest)
    print(manifest)
    print("GENERATE_PRIMARY_TABLES_OK")


if __name__ == "__main__":
    main()
