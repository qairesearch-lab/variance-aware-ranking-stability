#!/usr/bin/env python3
"""Generate additional quantitative evidence for RQ1-RQ3.

This script derives manuscript-strengthening workflow-methodology analyses from
the completed 800-run primary outputs. It does not rerun training or alter the
registered primary analysis matrix.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cervical-rq-evidence")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import display_path, resolve_path, runtime_metadata, write_json


MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
RESOURCE_BUDGETS = [(1, 1), (1, 3), (3, 1), (3, 3), (5, 3), (10, 5)]
REQUIRED_SELECTION_CI_COLUMNS = {
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_resamples",
    "bootstrap_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-output-dir",
        default="research-lab/experiments/registered-workflow/outputs/primary/analysis_io",
    )
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
    )
    parser.add_argument(
        "--figures-dir",
        default="research-lab/experiments/registered-workflow/stats/figures",
    )
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=431331)
    return parser.parse_args()


def ordered_models(models: list[str]) -> list[str]:
    known = [model for model in MODEL_ORDER if model in models]
    unknown = sorted(model for model in models if model not in known)
    return known + unknown


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def shannon_entropy(counts: list[int], denominator_models: int) -> tuple[float, float, float]:
    total = sum(counts)
    if total == 0 or denominator_models <= 1:
        return 0.0, 0.0, 1.0
    probabilities = [count / total for count in counts if count > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    normalized = entropy / math.log(denominator_models)
    effective = math.exp(entropy)
    return entropy, normalized, effective


def top_model_from_subset(df: pd.DataFrame, metric: str) -> tuple[str, str, float, pd.Series]:
    means = df.groupby("model")[metric].mean()
    means = means.reindex(ordered_models(list(means.index)))
    means = means.sort_values(ascending=False, kind="mergesort")
    top_model = str(means.index[0])
    runner_up = str(means.index[1])
    gap = float(means.iloc[0] - means.iloc[1])
    return top_model, runner_up, gap, means


def rank_series_from_means(means: pd.Series) -> pd.Series:
    return means.rank(ascending=False, method="min")


def validate_inputs(
    run_metrics: pd.DataFrame,
    model_ranking: pd.DataFrame,
    selection_probability: pd.DataFrame,
    metric: str,
) -> dict[str, object]:
    missing_selection_cols = sorted(REQUIRED_SELECTION_CI_COLUMNS - set(selection_probability.columns))
    if missing_selection_cols:
        raise SystemExit(f"selection_probability_summary.csv missing CI columns: {missing_selection_cols}")
    if metric not in run_metrics.columns:
        raise SystemExit(f"run_metrics_table.csv missing metric column: {metric}")
    if "metric_value" not in model_ranking.columns:
        raise SystemExit("model_ranking_table.csv missing metric_value column")
    if set(run_metrics["analysis_role"]) != {"primary_analysis"}:
        raise SystemExit("run_metrics_table.csv contains non-primary analysis roles")
    if set(run_metrics["run_status"]) != {"completed"}:
        raise SystemExit("run_metrics_table.csv contains non-completed runs")

    calibration_metrics = ["calibration_intercept", "calibration_slope"]
    calibration = {}
    for name in calibration_metrics:
        if name in run_metrics.columns:
            missing = int(run_metrics[name].isna().sum())
            calibration[name] = {
                "rows": int(len(run_metrics)),
                "missing_rows": missing,
                "non_missing_rows": int(len(run_metrics) - missing),
                "excluded_from_rq_enhancement": True,
            }
        else:
            calibration[name] = {
                "rows": int(len(run_metrics)),
                "missing_rows": None,
                "non_missing_rows": None,
                "excluded_from_rq_enhancement": True,
                "note": "column_not_found",
            }

    return {
        "selection_probability_ci_columns_present": True,
        "selection_probability_ci_columns": sorted(REQUIRED_SELECTION_CI_COLUMNS),
        "calibration_missingness": calibration,
        "model_ranking_table_role": "ignored_primary_intermediate_output",
    }


def build_calibration_missingness_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["calibration_intercept", "calibration_slope"]:
        if metric in run_metrics.columns:
            missing = int(run_metrics[metric].isna().sum())
            non_missing = int(len(run_metrics) - missing)
        else:
            missing = None
            non_missing = None
        rows.append(
            {
                "metric": metric,
                "rows": int(len(run_metrics)),
                "missing_rows": missing,
                "non_missing_rows": non_missing,
                "excluded_from_rq_enhancement": True,
                "interpretation": (
                    "Unavailable for current primary interpretation; RQ evidence enhancement uses balanced_accuracy, "
                    "ranking, paired uncertainty, and workflow stability outputs."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_workflow_interval_summary(
    workflow_aggregate: pd.DataFrame,
    selection_probability: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in workflow_aggregate.iterrows():
        n = int(row["units"])
        matching = int(row["matching_units"])
        changed = int(row["changed_units"])
        for target, count, estimate in [
            ("match_rate", matching, matching / n if n else math.nan),
            ("flip_rate", changed, changed / n if n else math.nan),
        ]:
            low, high = wilson_ci(count, n)
            rows.append(
                {
                    "interval_target": target,
                    "evidence_level": row["evidence_level"],
                    "dataset": row["dataset"],
                    "checkpoint_policy": row["checkpoint_policy"],
                    "model": "",
                    "count": count,
                    "denominator": n,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_method": "wilson_95",
                    "source_table": "workflow_factor_comparison_aggregate.csv",
                    "bootstrap_resamples": "",
                    "bootstrap_seed": "",
                }
            )

    for _, row in selection_probability.iterrows():
        rows.append(
            {
                "interval_target": "selection_probability",
                "evidence_level": "repeated_split_seed_context",
                "dataset": row["dataset"],
                "checkpoint_policy": row["checkpoint_policy"],
                "model": row["model"],
                "count": int(row["top_ranked_count"]),
                "denominator": int(row["context_denominator"]),
                "estimate": float(row["selection_probability"]),
                "ci_low": float(row["bootstrap_ci_low"]),
                "ci_high": float(row["bootstrap_ci_high"]),
                "ci_method": "existing_bootstrap_95",
                "source_table": "selection_probability_summary.csv",
                "bootstrap_resamples": int(row["bootstrap_resamples"]),
                "bootstrap_seed": int(row["bootstrap_seed"]),
            }
        )
    return pd.DataFrame(rows)


def build_top_model_entropy(model_ranking: pd.DataFrame) -> pd.DataFrame:
    top = model_ranking[model_ranking["is_top_ranked"].astype(int) == 1].copy()
    model_count = model_ranking.groupby(["dataset", "checkpoint_policy"])["model"].nunique().to_dict()
    rows = []
    for (dataset, policy), sub in top.groupby(["dataset", "checkpoint_policy"]):
        counts = Counter(sub["model"])
        models = ordered_models(list(model_ranking[(model_ranking["dataset"] == dataset) & (model_ranking["checkpoint_policy"] == policy)]["model"].unique()))
        count_values = [int(counts.get(model, 0)) for model in models]
        entropy, normalized, effective = shannon_entropy(count_values, int(model_count[(dataset, policy)]))
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "contexts": int(len(sub)),
                "candidate_models": int(model_count[(dataset, policy)]),
                "top_model_counts": json.dumps(dict(zip(models, count_values)), sort_keys=True),
                "shannon_entropy": entropy,
                "normalized_entropy": normalized,
                "effective_number_of_top_models": effective,
                "interpretation": "Higher entropy indicates more dispersed top-model determination across repeated workflow contexts.",
            }
        )
    return pd.DataFrame(rows)


def build_top_two_gap_tables(model_ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_cols = ["dataset", "checkpoint_policy", "split_id", "training_seed"]
    context_rows = []
    sign_rows = []
    full_means = (
        model_ranking.groupby(["dataset", "checkpoint_policy", "model"], as_index=False)["metric_value"].mean()
    )
    for (dataset, policy), sub in model_ranking.groupby(["dataset", "checkpoint_policy"]):
        full = full_means[(full_means["dataset"] == dataset) & (full_means["checkpoint_policy"] == policy)]
        full = full.sort_values(["metric_value", "model"], ascending=[False, True]).reset_index(drop=True)
        ref_top = str(full.loc[0, "model"])
        ref_runner = str(full.loc[1, "model"])
        sign_counts = {"positive": 0, "negative": 0, "tie": 0}

        for key, context in sub.groupby(context_cols):
            context = context.sort_values(["metric_value", "model"], ascending=[False, True]).reset_index(drop=True)
            top_model = str(context.loc[0, "model"])
            runner = str(context.loc[1, "model"])
            gap = float(context.loc[0, "metric_value"] - context.loc[1, "metric_value"])
            ref_top_value = float(context.loc[context["model"] == ref_top, "metric_value"].iloc[0])
            ref_runner_value = float(context.loc[context["model"] == ref_runner, "metric_value"].iloc[0])
            ref_diff = ref_top_value - ref_runner_value
            if ref_diff > 0:
                sign_counts["positive"] += 1
            elif ref_diff < 0:
                sign_counts["negative"] += 1
            else:
                sign_counts["tie"] += 1
            context_rows.append(
                {
                    "dataset": key[0],
                    "checkpoint_policy": key[1],
                    "split_id": key[2],
                    "training_seed": key[3],
                    "top_model": top_model,
                    "runner_up_model": runner,
                    "top_two_ordered_pair": f"{top_model}>{runner}",
                    "top1_top2_gap": gap,
                    "reference_top_model": ref_top,
                    "reference_runner_up_model": ref_runner,
                    "reference_pair_difference": ref_diff,
                }
            )

        n = sum(sign_counts.values())
        majority = max(sign_counts.values()) if n else 0
        sign_rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "reference_top_model": ref_top,
                "reference_runner_up_model": ref_runner,
                "reference_pair_positive_count": sign_counts["positive"],
                "reference_pair_negative_count": sign_counts["negative"],
                "reference_pair_tie_count": sign_counts["tie"],
                "reference_pair_sign_switch_frequency": 1 - majority / n if n else math.nan,
            }
        )

    context_table = pd.DataFrame(context_rows)
    summaries = []
    for (dataset, policy), sub in context_table.groupby(["dataset", "checkpoint_policy"]):
        q1, q3 = np.quantile(sub["top1_top2_gap"], [0.25, 0.75])
        pair_counts = sub["top_two_ordered_pair"].value_counts()
        sign = next(row for row in sign_rows if row["dataset"] == dataset and row["checkpoint_policy"] == policy)
        summaries.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "contexts": int(len(sub)),
                "gap_mean": float(sub["top1_top2_gap"].mean()),
                "gap_sd": float(sub["top1_top2_gap"].std(ddof=1)),
                "gap_median": float(sub["top1_top2_gap"].median()),
                "gap_iqr": float(q3 - q1),
                "gap_min": float(sub["top1_top2_gap"].min()),
                "gap_max": float(sub["top1_top2_gap"].max()),
                "dominant_top_two_ordered_pair": str(pair_counts.index[0]),
                "dominant_top_two_ordered_pair_count": int(pair_counts.iloc[0]),
                "top_two_identity_switch_frequency": float(1 - pair_counts.iloc[0] / len(sub)),
                **sign,
            }
        )
    return context_table, pd.DataFrame(summaries)


def enumerate_or_sample_combinations(
    splits: list[str],
    seeds: list[int],
    split_n: int,
    seed_n: int,
    max_resamples: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[tuple[str, ...], tuple[int, ...]]], bool]:
    split_combos = list(itertools.combinations(splits, split_n))
    seed_combos = list(itertools.combinations(seeds, seed_n))
    total = len(split_combos) * len(seed_combos)
    if total <= max_resamples:
        return [(s, e) for s in split_combos for e in seed_combos], True
    samples = []
    seen = set()
    while len(samples) < max_resamples:
        s = tuple(sorted(rng.choice(splits, size=split_n, replace=False).tolist()))
        e = tuple(sorted(int(x) for x in rng.choice(seeds, size=seed_n, replace=False).tolist()))
        key = (s, e)
        if key not in seen:
            seen.add(key)
            samples.append(key)
    return samples, False


def build_subsampling_stability(
    run_metrics: pd.DataFrame,
    metric: str,
    max_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    detail_rows = []
    summary_rows = []
    for (dataset, policy), sub in run_metrics.groupby(["dataset", "checkpoint_policy"]):
        splits = sorted(sub["split_id"].astype(str).unique())
        seeds = sorted(int(x) for x in sub["training_seed"].unique())
        full_top, full_runner, full_gap, full_means = top_model_from_subset(sub, metric)
        full_ranks = rank_series_from_means(full_means)
        for split_n, seed_n in RESOURCE_BUDGETS:
            combos, exhaustive = enumerate_or_sample_combinations(splits, seeds, split_n, seed_n, max_resamples, rng)
            top_models = []
            recovered = []
            rank_corrs = []
            gaps = []
            for sample_index, (split_combo, seed_combo) in enumerate(combos, start=1):
                sampled = sub[
                    sub["split_id"].astype(str).isin(split_combo) & sub["training_seed"].astype(int).isin(seed_combo)
                ]
                top_model, runner, gap, means = top_model_from_subset(sampled, metric)
                sample_ranks = rank_series_from_means(means.reindex(full_ranks.index))
                corr = spearmanr(full_ranks.to_numpy(), sample_ranks.to_numpy()).correlation
                if math.isnan(corr):
                    corr = 0.0
                is_recovered = top_model == full_top
                top_models.append(top_model)
                recovered.append(is_recovered)
                rank_corrs.append(float(corr))
                gaps.append(float(gap))
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint_policy": policy,
                        "split_count": split_n,
                        "seed_count": seed_n,
                        "resource_budget": f"{split_n}_splits_x_{seed_n}_seeds",
                        "sample_index": sample_index,
                        "selected_splits": ";".join(split_combo),
                        "selected_seeds": ";".join(str(x) for x in seed_combo),
                        "top_model": top_model,
                        "runner_up_model": runner,
                        "full_reference_top_model": full_top,
                        "recovered_full_reference_top_model": is_recovered,
                        "spearman_rank_correlation_with_full_reference": float(corr),
                        "top_two_gap": float(gap),
                    }
                )

            counts = Counter(top_models)
            models = ordered_models(list(full_ranks.index))
            entropy, normalized, effective = shannon_entropy([counts.get(model, 0) for model in models], len(models))
            success = int(sum(recovered))
            is_full_reference_budget = split_n == len(splits) and seed_n == len(seeds)
            if is_full_reference_budget:
                # The full repeated-evaluation grid is compared with itself; recovery is deterministic.
                low, high = 1.0, 1.0
            else:
                low, high = wilson_ci(success, len(recovered))
            gap_low, gap_high = np.quantile(gaps, [0.025, 0.975])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "split_count": split_n,
                    "seed_count": seed_n,
                    "resource_budget": f"{split_n}_splits_x_{seed_n}_seeds",
                    "sample_count": len(recovered),
                    "exhaustive_sampling": exhaustive,
                    "full_reference_self_comparison": is_full_reference_budget,
                    "full_reference_top_model": full_top,
                    "recovered_full_reference_probability": float(np.mean(recovered)),
                    "recovered_full_reference_ci_low": low,
                    "recovered_full_reference_ci_high": high,
                    "top_model_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                    "selection_entropy": entropy,
                    "normalized_selection_entropy": normalized,
                    "effective_number_of_top_models": effective,
                    "ranking_correlation_mean": float(np.mean(rank_corrs)),
                    "ranking_correlation_sd": float(np.std(rank_corrs, ddof=1)) if len(rank_corrs) > 1 else 0.0,
                    "ranking_correlation_median": float(np.median(rank_corrs)),
                    "top_two_gap_mean": float(np.mean(gaps)),
                    "top_two_gap_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
                    "top_two_gap_ci_low": float(gap_low),
                    "top_two_gap_ci_high": float(gap_high),
                    "top_two_gap_ci_width": float(gap_high - gap_low),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def save_workflow_interval_figure(intervals: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "workflow_match_flip_intervals.png"
    sub = intervals[
        (intervals["interval_target"].isin(["match_rate", "flip_rate"]))
        & (intervals["evidence_level"] != "full_repeated_split_seed_workflow")
    ].copy()
    sub["label"] = sub["dataset"] + " / " + sub["checkpoint_policy"] + " / " + sub["evidence_level"].str.replace("_", " ")
    targets = ["match_rate", "flip_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, target in zip(axes, targets):
        s = sub[sub["interval_target"] == target].reset_index(drop=True)
        y = np.arange(len(s))
        x = s["estimate"].to_numpy()
        xerr = np.vstack([x - s["ci_low"].to_numpy(), s["ci_high"].to_numpy() - x])
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#4C78A8", ecolor="#333333", capsize=3)
        ax.set_title(target.replace("_", " ").title())
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
        if target == "match_rate":
            ax.set_yticks(y)
            ax.set_yticklabels(s["label"])
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
    fig.suptitle("Workflow stability intervals", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_entropy_figure(entropy: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "top_model_entropy_by_workflow_context.png"
    labels = entropy["dataset"] + " / " + entropy["checkpoint_policy"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, entropy["normalized_entropy"], color="#B279A2")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized top-model entropy")
    ax.set_title("Selection dispersion across repeated workflow contexts")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_top_two_gap_figure(context_gaps: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "top_two_gap_distribution_by_workflow_context.png"
    context_gaps = context_gaps.copy()
    context_gaps["stratum"] = context_gaps["dataset"] + " / " + context_gaps["checkpoint_policy"]
    strata = list(context_gaps["stratum"].drop_duplicates())
    data = [context_gaps[context_gaps["stratum"] == stratum]["top1_top2_gap"].to_numpy() for stratum in strata]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=strata, showfliers=True)
    ax.set_ylabel("Top1 - Top2 balanced accuracy gap")
    ax.set_title("Top-two model-gap instability")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_subsampling_figure(summary: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "subsampling_resource_stability_curves.png"
    metrics = [
        ("recovered_full_reference_probability", "Recovered reference top-model probability"),
        ("normalized_selection_entropy", "Normalized selection entropy"),
        ("ranking_correlation_mean", "Mean rank correlation with full workflow"),
    ]
    summary = summary.copy()
    summary["budget_label"] = summary["split_count"].astype(str) + "x" + summary["seed_count"].astype(str)
    budget_order = [f"{s}x{e}" for s, e in RESOURCE_BUDGETS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for (dataset, policy), sub in summary.groupby(["dataset", "checkpoint_policy"]):
            sub = sub.set_index("budget_label").reindex(budget_order).reset_index()
            ax.plot(sub["budget_label"], sub[metric], marker="o", label=f"{dataset}/{policy}")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Resource-stability analysis from existing split x seed runs", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    analysis_output_dir = resolve_path(args.analysis_output_dir)
    tables_dir = resolve_path(args.tables_dir)
    figures_dir = resolve_path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_metrics = pd.read_csv(analysis_output_dir / "run_metrics_table.csv")
    model_ranking = pd.read_csv(analysis_output_dir / "model_ranking_table.csv")
    selection_probability = pd.read_csv(tables_dir / "selection_probability_summary.csv")
    workflow_aggregate = pd.read_csv(tables_dir / "workflow_factor_comparison_aggregate.csv")

    validation = validate_inputs(run_metrics, model_ranking, selection_probability, args.metric)

    calibration = build_calibration_missingness_table(run_metrics)
    workflow_intervals = build_workflow_interval_summary(workflow_aggregate, selection_probability)
    entropy = build_top_model_entropy(model_ranking)
    context_gaps, gap_summary = build_top_two_gap_tables(model_ranking)
    subsampling_detail, subsampling_summary = build_subsampling_stability(
        run_metrics, args.metric, args.resamples, args.seed
    )

    outputs = {
        "calibration_missingness_summary.csv": calibration,
        "workflow_stability_interval_summary.csv": workflow_intervals,
        "top_model_entropy_summary.csv": entropy,
        "top_two_context_gap_table.csv": context_gaps,
        "top_two_gap_instability_summary.csv": gap_summary,
        "subsampling_resource_stability_detail.csv": subsampling_detail,
        "subsampling_resource_stability_summary.csv": subsampling_summary,
    }
    output_paths = {}
    for filename, table in outputs.items():
        path = tables_dir / filename
        table.to_csv(path, index=False)
        output_paths[filename] = path

    figure_paths = [
        save_workflow_interval_figure(workflow_intervals, figures_dir),
        save_entropy_figure(entropy, figures_dir),
        save_top_two_gap_figure(context_gaps, figures_dir),
        save_subsampling_figure(subsampling_summary, figures_dir),
    ]

    summary = {
        **runtime_metadata(Path(__file__).name),
        "status": "completed",
        "metric": args.metric,
        "resamples": args.resamples,
        "resampling_seed": args.seed,
        "analysis_output_dir": display_path(analysis_output_dir),
        "tables_dir": display_path(tables_dir),
        "figures_dir": display_path(figures_dir),
        "validation": validation,
        "row_counts": {filename: int(table.shape[0]) for filename, table in outputs.items()},
        "figures": [display_path(path) for path in figure_paths],
        "interpretation_boundary": (
            "RQ evidence enhancement outputs are derived from existing 800 primary runs; "
            "they do not rerun training or alter the OSF-registered analysis matrix."
        ),
        "results_file_update_status": "not_updated_pending_user_review",
    }
    write_json(tables_dir / "rq_evidence_enhancement_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
