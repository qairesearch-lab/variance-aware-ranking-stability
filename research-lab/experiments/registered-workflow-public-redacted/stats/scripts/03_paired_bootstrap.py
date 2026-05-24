#!/usr/bin/env python3
"""Run paired bootstrap summaries for model comparisons."""

from __future__ import annotations

import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

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


CONTEXT_KEYS = ["dataset", "split_id", "training_seed", "checkpoint_policy", "analysis_role"]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(differences: list[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=431512)
    parser.add_argument("--expected-analysis-role", default="primary_analysis")
    args = parser.parse_args()

    metrics_table = resolve_path(args.metrics_table)
    output_dir = resolve_path(args.output_dir)
    rows = read_csv(metrics_table)
    require_columns(rows, CONTEXT_KEYS + ["model", "run_id", args.metric], metrics_table)
    require_analysis_role(rows, args.expected_analysis_role, metrics_table)
    require_completed(rows, metrics_table)

    grouped: dict[tuple[str, str], dict[tuple[str, str, str], dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        dataset = row["dataset"]
        checkpoint_policy = row["checkpoint_policy"]
        context = (row["split_id"], row["training_seed"], row["analysis_role"])
        grouped[(dataset, checkpoint_policy)][context][row["model"]] = parse_metric(
            row.get(args.metric), args.metric, row["run_id"]
        )

    output_rows: list[dict[str, object]] = []
    for (dataset, checkpoint_policy), contexts in sorted(grouped.items()):
        models = sorted({model for metrics in contexts.values() for model in metrics})
        for model_a, model_b in itertools.combinations(models, 2):
            differences = [
                metrics[model_a] - metrics[model_b]
                for metrics in contexts.values()
                if model_a in metrics and model_b in metrics
            ]
            if not differences:
                continue
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + len(output_rows),
            )
            n = len(differences)
            a_better = sum(1 for value in differences if value > 0)
            b_better = sum(1 for value in differences if value < 0)
            ties = n - a_better - b_better
            output_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_policy": checkpoint_policy,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": args.metric,
                    "paired_units": n,
                    "mean_difference_model_a_minus_b": mean(differences),
                    "median_difference_model_a_minus_b": median(differences),
                    "sd_difference_model_a_minus_b": pstdev(differences) if n > 1 else 0.0,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "model_a_better_count": a_better,
                    "model_b_better_count": b_better,
                    "tie_count": ties,
                    "model_a_better_probability": a_better / n,
                    "model_b_better_probability": b_better / n,
                    "tie_probability": ties / n,
                    "direction_consistency": max(a_better, b_better, ties) / n,
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "bootstrap_seed": args.bootstrap_seed + len(output_rows),
                }
            )

    output_path = output_dir / "paired_bootstrap_comparison_table.csv"
    write_csv(output_path, output_rows)
    summary = {
        **runtime_metadata(Path(__file__).name),
        "metrics_table": display_path(metrics_table),
        "output_dir": display_path(output_dir),
        "metric": args.metric,
        "expected_analysis_role": args.expected_analysis_role,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed_base": args.bootstrap_seed,
        "comparison_rows": len(output_rows),
        "status": "completed",
    }
    write_json(output_dir / "paired_bootstrap_summary.json", summary)
    print(summary)
    print("PAIRED_BOOTSTRAP_OK")


if __name__ == "__main__":
    main()
