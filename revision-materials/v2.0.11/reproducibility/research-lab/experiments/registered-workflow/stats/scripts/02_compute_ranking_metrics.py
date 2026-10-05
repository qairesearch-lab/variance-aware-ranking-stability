#!/usr/bin/env python3
"""Compute model-ranking summaries and selection-frequency tables."""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
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


def rank_rows(rows: list[dict[str, str]], metric: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contexts: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        context = tuple(row[key] for key in CONTEXT_KEYS)
        contexts[context].append(row)

    ranking_rows: list[dict[str, object]] = []
    context_top_rows: list[dict[str, object]] = []
    for context, items in sorted(contexts.items()):
        dataset, split_id, training_seed, checkpoint_policy, analysis_role = context
        scored = [
            (parse_metric(item.get(metric), metric, item["run_id"]), item)
            for item in items
        ]
        sorted_items = sorted(scored, key=lambda pair: (-pair[0], pair[1]["model"]))
        top_model = sorted_items[0][1]["model"] if sorted_items else ""
        context_top_rows.append(
            {
                "dataset": dataset,
                "split_id": split_id,
                "training_seed": training_seed,
                "checkpoint_policy": checkpoint_policy,
                "analysis_role": analysis_role,
                "top_model": top_model,
                "metric": metric,
                "models_ranked": len(sorted_items),
            }
        )
        for rank, (metric_value, item) in enumerate(sorted_items, start=1):
            ranking_rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "training_seed": training_seed,
                    "checkpoint_policy": checkpoint_policy,
                    "analysis_role": analysis_role,
                    "model": item["model"],
                    "metric": metric,
                    "metric_value": metric_value,
                    "rank": rank,
                    "is_top_ranked": int(rank == 1),
                }
            )
    return ranking_rows, context_top_rows


def selection_frequency(ranking_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    top_counts: Counter[tuple[str, str, str]] = Counter()
    denominators: Counter[tuple[str, str]] = Counter()
    seen_contexts: set[tuple[str, str, str, str]] = set()
    for row in ranking_rows:
        context = (str(row["dataset"]), str(row["checkpoint_policy"]), str(row["split_id"]), str(row["training_seed"]))
        if context not in seen_contexts:
            denominators[(str(row["dataset"]), str(row["checkpoint_policy"]))] += 1
            seen_contexts.add(context)
        if int(row["is_top_ranked"]) == 1:
            top_counts[(str(row["dataset"]), str(row["checkpoint_policy"]), str(row["model"]))] += 1

    output = []
    for (dataset, checkpoint_policy, model), count in sorted(top_counts.items()):
        denominator = denominators[(dataset, checkpoint_policy)]
        output.append(
            {
                "dataset": dataset,
                "checkpoint_policy": checkpoint_policy,
                "model": model,
                "selection_count": count,
                "selection_denominator": denominator,
                "selection_frequency": count / denominator if denominator else 0.0,
            }
        )
    return output


def rank_distribution(ranking_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in ranking_rows:
        grouped[(str(row["dataset"]), str(row["checkpoint_policy"]), str(row["model"]))].append(int(row["rank"]))
    output = []
    for (dataset, checkpoint_policy, model), ranks in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "checkpoint_policy": checkpoint_policy,
                "model": model,
                "rank_count": len(ranks),
                "rank_mean": mean(ranks),
                "rank_sd": pstdev(ranks) if len(ranks) > 1 else 0.0,
                "rank_median": median(ranks),
                "rank_min": min(ranks),
                "rank_max": max(ranks),
            }
        )
    return output


def ranking_stability(
    ranking_rows: list[dict[str, object]], context_top_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference: dict[tuple[str, str], list[str]] = {}
    metric_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ranking_rows:
        metric_values[(str(row["dataset"]), str(row["checkpoint_policy"]), str(row["model"]))].append(float(row["metric_value"]))
    for dataset, checkpoint_policy in sorted({(key[0], key[1]) for key in metric_values}):
        models = [
            (mean(values), model)
            for (dset, policy, model), values in metric_values.items()
            if dset == dataset and policy == checkpoint_policy
        ]
        reference[(dataset, checkpoint_policy)] = [model for _, model in sorted(models, key=lambda item: (-item[0], item[1]))]

    grouped_tops: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in context_top_rows:
        grouped_tops[(str(row["dataset"]), str(row["checkpoint_policy"]))].append(str(row["top_model"]))

    stability_rows = []
    for (dataset, checkpoint_policy), top_models in sorted(grouped_tops.items()):
        ref_order = reference[(dataset, checkpoint_policy)]
        ref_top = ref_order[0]
        stable_count = sum(1 for model in top_models if model == ref_top)
        denominator = len(top_models)
        stability_rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": checkpoint_policy,
                "reference_ordering": "|".join(ref_order),
                "reference_top_model": ref_top,
                "contexts": denominator,
                "selection_frequency_reference_top": stable_count / denominator if denominator else 0.0,
                "ranking_flip_rate": 1.0 - (stable_count / denominator if denominator else 0.0),
            }
        )

    rank_by_context: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
    for row in ranking_rows:
        context = (str(row["dataset"]), str(row["checkpoint_policy"]), str(row["split_id"]), str(row["training_seed"]))
        rank_by_context[context][str(row["model"])] = int(row["rank"])
    pairwise_rows = []
    for (dataset, checkpoint_policy), ref_order in sorted(reference.items()):
        relevant_contexts = [
            ranks for (dset, policy, _, _), ranks in rank_by_context.items()
            if dset == dataset and policy == checkpoint_policy
        ]
        for model_a, model_b in itertools.combinations(ref_order, 2):
            a_above = sum(1 for ranks in relevant_contexts if ranks[model_a] < ranks[model_b])
            b_above = sum(1 for ranks in relevant_contexts if ranks[model_b] < ranks[model_a])
            denominator = len(relevant_contexts)
            pairwise_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_policy": checkpoint_policy,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_above_model_b": a_above,
                    "model_b_above_model_a": b_above,
                    "contexts": denominator,
                    "pairwise_switch_rate_against_reference": b_above / denominator if denominator else 0.0,
                }
            )
    return stability_rows, pairwise_rows


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
    require_columns(rows, CONTEXT_KEYS + ["model", "run_id", args.metric], metrics_table)
    require_analysis_role(rows, args.expected_analysis_role, metrics_table)
    require_completed(rows, metrics_table)

    ranking_rows, context_top_rows = rank_rows(rows, args.metric)
    selection_rows = selection_frequency(ranking_rows)
    rank_distribution_rows = rank_distribution(ranking_rows)
    stability_rows, pairwise_rows = ranking_stability(ranking_rows, context_top_rows)

    write_csv(output_dir / "model_ranking_table.csv", ranking_rows)
    write_csv(output_dir / "context_top_model_table.csv", context_top_rows)
    write_csv(output_dir / "selection_frequency_table.csv", selection_rows)
    write_csv(output_dir / "rank_distribution_table.csv", rank_distribution_rows)
    write_csv(output_dir / "ranking_stability_table.csv", stability_rows)
    write_csv(output_dir / "pairwise_rank_switch_table.csv", pairwise_rows)

    summary = {
        **runtime_metadata(Path(__file__).name),
        "metrics_table": display_path(metrics_table),
        "output_dir": display_path(output_dir),
        "metric": args.metric,
        "expected_analysis_role": args.expected_analysis_role,
        "metric_rows": len(rows),
        "ranking_rows": len(ranking_rows),
        "selection_rows": len(selection_rows),
        "rank_distribution_rows": len(rank_distribution_rows),
        "ranking_stability_rows": len(stability_rows),
        "pairwise_rank_switch_rows": len(pairwise_rows),
    }
    write_json(output_dir / "ranking_metrics_summary.json", summary)
    print(summary)
    print("COMPUTE_RANKING_METRICS_OK")


if __name__ == "__main__":
    main()
