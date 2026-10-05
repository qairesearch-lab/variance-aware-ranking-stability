#!/usr/bin/env python3
"""Compute top two set stability from existing RQ evidence outputs.

This script derives a supplement-only top two set stability table from
`top_two_context_gap_table.csv`. It does not rerun training or alter the
registered primary analysis matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
    )
    return parser.parse_args()


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def set_label(values: set[str]) -> str:
    return "|".join(sorted(values))


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    source_path = tables_dir / "top_two_context_gap_table.csv"
    detail_path = tables_dir / "top_two_set_stability_detail.csv"
    summary_path = tables_dir / "top_two_set_stability_summary.csv"
    manifest_path = tables_dir / "top_two_set_stability_summary.json"

    source = pd.read_csv(source_path)
    required = {
        "dataset",
        "checkpoint_policy",
        "split_id",
        "training_seed",
        "top_model",
        "runner_up_model",
        "reference_top_model",
        "reference_runner_up_model",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise SystemExit(f"{source_path} missing required columns: {missing}")

    detail_rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        context_set = {str(row["top_model"]), str(row["runner_up_model"])}
        reference_set = {str(row["reference_top_model"]), str(row["reference_runner_up_model"])}
        intersection = context_set & reference_set
        union = context_set | reference_set
        jaccard = len(intersection) / len(union) if union else math.nan
        exact_set_agreement = context_set == reference_set
        exact_ordered_agreement = (
            str(row["top_model"]) == str(row["reference_top_model"])
            and str(row["runner_up_model"]) == str(row["reference_runner_up_model"])
        )
        detail_rows.append(
            {
                "dataset": row["dataset"],
                "checkpoint_policy": row["checkpoint_policy"],
                "split_id": row["split_id"],
                "training_seed": row["training_seed"],
                "context_top_two_set": set_label(context_set),
                "reference_top_two_set": set_label(reference_set),
                "context_top_model": row["top_model"],
                "context_runner_up_model": row["runner_up_model"],
                "reference_top_model": row["reference_top_model"],
                "reference_runner_up_model": row["reference_runner_up_model"],
                "top_two_set_jaccard": jaccard,
                "exact_top_two_set_agreement": exact_set_agreement,
                "exact_ordered_top_two_agreement": exact_ordered_agreement,
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, object]] = []
    for (dataset, policy), sub in detail.groupby(["dataset", "checkpoint_policy"], sort=True):
        n = len(sub)
        exact_count = int(sub["exact_top_two_set_agreement"].sum())
        ordered_count = int(sub["exact_ordered_top_two_agreement"].sum())
        exact_low, exact_high = wilson_ci(exact_count, n)
        ordered_low, ordered_high = wilson_ci(ordered_count, n)
        set_counts = sub["context_top_two_set"].value_counts().sort_index().to_dict()
        summary_rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "contexts": n,
                "reference_top_two_set": str(sub["reference_top_two_set"].iloc[0]),
                "exact_top_two_set_agreement_count": exact_count,
                "exact_top_two_set_agreement_rate": exact_count / n if n else math.nan,
                "exact_top_two_set_agreement_ci_low": exact_low,
                "exact_top_two_set_agreement_ci_high": exact_high,
                "exact_ordered_top_two_agreement_count": ordered_count,
                "exact_ordered_top_two_agreement_rate": ordered_count / n if n else math.nan,
                "exact_ordered_top_two_agreement_ci_low": ordered_low,
                "exact_ordered_top_two_agreement_ci_high": ordered_high,
                "mean_top_two_set_jaccard": float(sub["top_two_set_jaccard"].mean()),
                "median_top_two_set_jaccard": float(sub["top_two_set_jaccard"].median()),
                "min_top_two_set_jaccard": float(sub["top_two_set_jaccard"].min()),
                "context_top_two_set_counts": json.dumps(set_counts, sort_keys=True),
                "interpretation": (
                    "Exact top two set agreement ignores first versus second order; "
                    "exact ordered agreement requires both the set and order to match."
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "source": str(source_path),
        "outputs": {
            "top_two_set_stability_summary.csv": int(len(summary)),
            "top_two_set_stability_detail.csv": int(len(detail)),
        },
        "method": (
            "For each single split seed context, the unordered set containing the top ranked "
            "and second ranked models was compared with the unordered top two set under the "
            "full repeated evaluation reference. Jaccard similarity was |intersection|/|union|. "
            "Exact set agreement ignored order, while exact ordered agreement required both "
            "first and second ranks to match."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
