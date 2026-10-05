#!/usr/bin/env python3
"""Estimate probability of being top-ranked from per-context model rankings."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analysis_utils import (
    display_path,
    read_csv,
    require_analysis_role,
    require_columns,
    resolve_path,
    runtime_metadata,
    write_csv,
    write_json,
)


CONTEXT_KEYS = ["dataset", "split_id", "training_seed", "checkpoint_policy", "analysis_role"]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_probability(indicators: list[int], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(indicators)
    estimates = []
    for _ in range(resamples):
        sample = [indicators[rng.randrange(n)] for _ in range(n)]
        estimates.append(mean(sample))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ranking-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-frequency-table")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260512)
    parser.add_argument("--expected-analysis-role", default="primary_analysis")
    args = parser.parse_args()

    ranking_table = resolve_path(args.model_ranking_table)
    output_dir = resolve_path(args.output_dir)
    rows = read_csv(ranking_table)
    require_columns(rows, CONTEXT_KEYS + ["model", "is_top_ranked"], ranking_table)
    require_analysis_role(rows, args.expected_analysis_role, ranking_table)

    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["checkpoint_policy"], row["model"])
        grouped[key].append(int(row["is_top_ranked"]))

    output_rows: list[dict[str, object]] = []
    for (dataset, checkpoint_policy, model), indicators in sorted(grouped.items()):
        seed = args.bootstrap_seed + len(output_rows)
        ci_low, ci_high = bootstrap_probability(indicators, args.bootstrap_resamples, seed)
        top_count = sum(indicators)
        denominator = len(indicators)
        output_rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": checkpoint_policy,
                "model": model,
                "top_ranked_count": top_count,
                "context_denominator": denominator,
                "selection_probability": top_count / denominator if denominator else 0.0,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": seed,
            }
        )

    frequency_check = None
    if args.selection_frequency_table:
        frequency_table = resolve_path(args.selection_frequency_table)
        frequency_rows = read_csv(frequency_table)
        frequency_by_key = {
            (row["dataset"], row["checkpoint_policy"], row["model"]): float(row["selection_frequency"])
            for row in frequency_rows
        }
        mismatches = []
        for row in output_rows:
            key = (str(row["dataset"]), str(row["checkpoint_policy"]), str(row["model"]))
            if key in frequency_by_key and abs(float(row["selection_probability"]) - frequency_by_key[key]) > 1e-12:
                mismatches.append("|".join(key))
        frequency_check = {
            "selection_frequency_table": display_path(frequency_table),
            "checked_rows": len(frequency_rows),
            "mismatches": mismatches,
        }

    write_csv(output_dir / "selection_probability_table.csv", output_rows)
    summary = {
        **runtime_metadata(Path(__file__).name),
        "model_ranking_table": display_path(ranking_table),
        "output_dir": display_path(output_dir),
        "expected_analysis_role": args.expected_analysis_role,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed_base": args.bootstrap_seed,
        "probability_rows": len(output_rows),
        "frequency_cross_check": frequency_check,
        "status": "completed",
    }
    write_json(output_dir / "selection_probability_summary.json", summary)
    print(summary)
    print("PROBABILITY_OF_BEING_BEST_OK")


if __name__ == "__main__":
    main()
