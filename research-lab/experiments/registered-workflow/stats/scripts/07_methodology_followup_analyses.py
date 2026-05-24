#!/usr/bin/env python3
"""Generate methodology-oriented follow-up tables and figures.

These outputs interpret the completed primary analysis as a workflow-methodology
study rather than as a model competition. They are derived from the frozen
800-run primary outputs and do not change the registered primary pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cervical-methodology")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import display_path, resolve_path, runtime_metadata, write_json


BOOTSTRAP_SEED_BASE = 2026051300
MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-output-dir",
        default="research-lab/experiments/registered-workflow/outputs/primary/analysis_io",
        help="Directory containing primary analysis intermediate outputs.",
    )
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
        help="Directory for lightweight follow-up tables.",
    )
    parser.add_argument(
        "--figures-dir",
        default="research-lab/experiments/registered-workflow/stats/figures",
        help="Directory for manuscript-oriented figures.",
    )
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def ordered_models(models: list[str]) -> list[str]:
    known = [model for model in MODEL_ORDER if model in models]
    unknown = sorted(model for model in models if model not in known)
    return known + unknown


def top_model_table(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols + ["model"], as_index=False)[metric].mean()
    grouped = grouped.sort_values(group_cols + [metric, "model"], ascending=[True] * len(group_cols) + [False, True])
    return grouped.groupby(group_cols, as_index=False).head(1).rename(columns={metric: "top_metric_value"})


def metric_margin_table(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols + ["model"], as_index=False)[metric].mean()
    rows = []
    for key, sub in grouped.groupby(group_cols):
        sub = sub.sort_values([metric, "model"], ascending=[False, True]).reset_index(drop=True)
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "top_model": sub.loc[0, "model"],
                "top_metric_value": sub.loc[0, metric],
                "runner_up_model": sub.loc[1, "model"] if len(sub) > 1 else "",
                "runner_up_metric_value": sub.loc[1, metric] if len(sub) > 1 else np.nan,
                "top_margin": sub.loc[0, metric] - sub.loc[1, metric] if len(sub) > 1 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_workflow_comparison(run_metrics: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratum_cols = ["dataset", "checkpoint_policy"]
    full_top = top_model_table(run_metrics, stratum_cols, metric)
    full_reference = full_top[stratum_cols + ["model"]].rename(columns={"model": "full_repeated_reference_top_model"})

    evidence_specs = [
        ("single_split_seed_context", ["dataset", "checkpoint_policy", "split_id", "training_seed"]),
        ("split_aggregated_across_seeds", ["dataset", "checkpoint_policy", "split_id"]),
        ("seed_aggregated_across_splits", ["dataset", "checkpoint_policy", "training_seed"]),
        ("full_repeated_split_seed_workflow", ["dataset", "checkpoint_policy"]),
    ]
    frames = []
    for evidence_level, group_cols in evidence_specs:
        top = top_model_table(run_metrics, group_cols, metric)
        top = top.merge(full_reference, on=stratum_cols, how="left")
        top.insert(0, "evidence_level", evidence_level)
        top["top_model"] = top["model"]
        top["matches_full_repeated_reference"] = top["top_model"] == top["full_repeated_reference_top_model"]
        top["conclusion_change_indicator"] = (~top["matches_full_repeated_reference"]).astype(int)
        top["unit_id"] = top.apply(
            lambda row: "|".join(str(row[col]) for col in group_cols if col not in stratum_cols) or "full",
            axis=1,
        )
        frames.append(
            top[
                [
                    "evidence_level",
                    "dataset",
                    "checkpoint_policy",
                    "unit_id",
                    "top_model",
                    "top_metric_value",
                    "full_repeated_reference_top_model",
                    "matches_full_repeated_reference",
                    "conclusion_change_indicator",
                ]
            ]
        )

    workflow = pd.concat(frames, ignore_index=True)
    summary = (
        workflow.groupby(["evidence_level", "dataset", "checkpoint_policy"], as_index=False)
        .agg(
            units=("unit_id", "count"),
            matching_units=("matches_full_repeated_reference", "sum"),
            changed_units=("conclusion_change_indicator", "sum"),
        )
        .assign(match_rate=lambda df: df["matching_units"] / df["units"])
    )
    return workflow, summary


def build_fragility_examples(run_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = ["dataset", "checkpoint_policy", "split_id", "training_seed"]
    contexts = metric_margin_table(run_metrics, group_cols, metric)
    full_reference = top_model_table(run_metrics, ["dataset", "checkpoint_policy"], metric)[
        ["dataset", "checkpoint_policy", "model"]
    ].rename(columns={"model": "full_repeated_reference_top_model"})
    contexts = contexts.merge(full_reference, on=["dataset", "checkpoint_policy"], how="left")
    contexts["matches_full_repeated_reference"] = contexts["top_model"] == contexts["full_repeated_reference_top_model"]

    examples = []
    for (dataset, policy), sub in contexts.groupby(["dataset", "checkpoint_policy"]):
        ordered = sub.sort_values(["split_id", "training_seed"]).reset_index(drop=True)
        first = ordered.iloc[0].copy()
        first["selection_rule"] = "first_split_first_seed"
        examples.append(first)

        median_margin = sub["top_margin"].median()
        median = sub.iloc[(sub["top_margin"] - median_margin).abs().argsort().iloc[0]].copy()
        median["selection_rule"] = "median_top_margin_context"
        examples.append(median)

        disagreed = sub[~sub["matches_full_repeated_reference"]]
        if len(disagreed) > 0:
            max_disagreement = disagreed.sort_values("top_margin", ascending=False).iloc[0].copy()
            max_disagreement["selection_rule"] = "largest_margin_disagreement_context"
            examples.append(max_disagreement)
        else:
            closest = sub.sort_values("top_margin", ascending=True).iloc[0].copy()
            closest["selection_rule"] = "smallest_margin_no_disagreement_available"
            examples.append(closest)

    columns = [
        "selection_rule",
        "dataset",
        "checkpoint_policy",
        "split_id",
        "training_seed",
        "top_model",
        "top_metric_value",
        "runner_up_model",
        "runner_up_metric_value",
        "top_margin",
        "full_repeated_reference_top_model",
        "matches_full_repeated_reference",
    ]
    return pd.DataFrame(examples)[columns]


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    boot = rng.choice(values, size=(resamples, n), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def build_checkpoint_policy_tables(
    run_metrics: pd.DataFrame, metric: str, resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, ((dataset, model), sub) in enumerate(run_metrics.groupby(["dataset", "model"])):
        wide = sub.pivot_table(index=["split_id", "training_seed"], columns="checkpoint_policy", values=metric)
        wide = wide.dropna(subset=["A", "B"])
        diff = (wide["B"] - wide["A"]).to_numpy()
        ci_low, ci_high = bootstrap_ci(diff, resamples, BOOTSTRAP_SEED_BASE + i)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "paired_units": len(diff),
                "metric": metric,
                "mean_difference_B_minus_A": float(np.mean(diff)),
                "median_difference_B_minus_A": float(np.median(diff)),
                "sd_difference_B_minus_A": float(np.std(diff, ddof=1)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "policy_B_better_count": int(np.sum(diff > 0)),
                "policy_A_better_count": int(np.sum(diff < 0)),
                "tie_count": int(np.sum(diff == 0)),
                "policy_B_better_probability": float(np.mean(diff > 0)),
                "policy_A_better_probability": float(np.mean(diff < 0)),
                "direction_consistency": float(max(np.mean(diff > 0), np.mean(diff < 0), np.mean(diff == 0))),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": BOOTSTRAP_SEED_BASE + i,
                "analysis_role": "supplementary_checkpoint_policy_sensitivity",
            }
        )

    paired = pd.DataFrame(rows)

    top_contexts = metric_margin_table(run_metrics, ["dataset", "checkpoint_policy", "split_id", "training_seed"], metric)
    shift_rows = []
    for dataset, sub in top_contexts.groupby("dataset"):
        a = sub[sub["checkpoint_policy"] == "A"][
            ["split_id", "training_seed", "top_model"]
        ].rename(columns={"top_model": "top_model_policy_A"})
        b = sub[sub["checkpoint_policy"] == "B"][
            ["split_id", "training_seed", "top_model"]
        ].rename(columns={"top_model": "top_model_policy_B"})
        joined = a.merge(b, on=["split_id", "training_seed"], how="inner")
        joined["same_top_model"] = joined["top_model_policy_A"] == joined["top_model_policy_B"]
        transitions = (
            joined.groupby(["top_model_policy_A", "top_model_policy_B"], as_index=False)
            .agg(contexts=("same_top_model", "count"))
            .sort_values(["contexts", "top_model_policy_A", "top_model_policy_B"], ascending=[False, True, True])
        )
        for _, row in transitions.iterrows():
            shift_rows.append(
                {
                    "dataset": dataset,
                    "top_model_policy_A": row["top_model_policy_A"],
                    "top_model_policy_B": row["top_model_policy_B"],
                    "contexts": int(row["contexts"]),
                    "context_denominator": len(joined),
                    "transition_probability": float(row["contexts"] / len(joined)),
                    "same_top_model_transition": row["top_model_policy_A"] == row["top_model_policy_B"],
                }
            )
    shifts = pd.DataFrame(shift_rows)
    return paired, shifts


def build_multiplicity_record(pairwise: pd.DataFrame, checkpoint_paired: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison_family": "model_pairwise_within_dataset_checkpoint_policy",
                "source_table": "paired_bootstrap_comparison_summary.csv",
                "family_definition": "Six model-pair comparisons within each dataset x checkpoint_policy stratum.",
                "strata_or_groups": pairwise[["dataset", "checkpoint_policy"]].drop_duplicates().shape[0],
                "comparisons_per_family": 6,
                "total_comparisons": len(pairwise),
                "primary_interpretation": "estimation_oriented_effect_size_ci_direction_consistency",
                "multiplicity_strategy": (
                    "No p-value adjustment applied because the table is interpreted as secondary estimation evidence; "
                    "if dichotomous inferential claims are made, apply Holm/FWER or Benjamini-Hochberg/FDR within the declared family."
                ),
                "formal_claim_allowed_without_adjustment": False,
            },
            {
                "comparison_family": "checkpoint_policy_B_minus_A_within_dataset_model",
                "source_table": "checkpoint_policy_paired_comparison_table.csv",
                "family_definition": "Policy B minus A paired comparison for each dataset x model.",
                "strata_or_groups": checkpoint_paired[["dataset", "model"]].drop_duplicates().shape[0],
                "comparisons_per_family": 1,
                "total_comparisons": len(checkpoint_paired),
                "primary_interpretation": "supplementary_workflow_factor_sensitivity",
                "multiplicity_strategy": (
                    "Report effect sizes, bootstrap CIs, and direction consistency; avoid unadjusted dichotomous significance claims."
                ),
                "formal_claim_allowed_without_adjustment": False,
            },
        ]
    )


def build_experimental_workflow_matrix(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in run_metrics.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "runs": int(len(sub)),
                "splits": int(sub["split_id"].nunique()),
                "split_ids": ";".join(sorted(sub["split_id"].astype(str).unique())),
                "training_seeds": ";".join(str(seed) for seed in sorted(sub["training_seed"].unique())),
                "models": ";".join(ordered_models(sorted(sub["model"].unique()))),
                "checkpoint_policies": ";".join(sorted(sub["checkpoint_policy"].unique())),
                "analysis_role": "primary_analysis",
                "execution_mode": ";".join(sorted(sub["execution_mode"].unique())),
                "primary_metric": "balanced_accuracy",
                "workflow_role": "proof_of_concept_methodology_matrix",
            }
        )
    rows.append(
        {
            "dataset": "all_primary_datasets",
            "runs": int(len(run_metrics)),
            "splits": int(run_metrics["split_id"].nunique()),
            "split_ids": ";".join(sorted(run_metrics["split_id"].astype(str).unique())),
            "training_seeds": ";".join(str(seed) for seed in sorted(run_metrics["training_seed"].unique())),
            "models": ";".join(ordered_models(sorted(run_metrics["model"].unique()))),
            "checkpoint_policies": ";".join(sorted(run_metrics["checkpoint_policy"].unique())),
            "analysis_role": "primary_analysis",
            "execution_mode": ";".join(sorted(run_metrics["execution_mode"].unique())),
            "primary_metric": "balanced_accuracy",
            "workflow_role": "combined_primary_methodology_matrix",
        }
    )
    return pd.DataFrame(rows)


def build_manuscript_table_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "manuscript_table": "Table 1",
                "title": "Experimental workflow matrix",
                "source_file": "experimental_workflow_matrix.csv",
                "role": "main",
            },
            {
                "manuscript_table": "Table 2",
                "title": "Workflow-context ranking stability summary",
                "source_file": "workflow_factor_comparison_aggregate.csv",
                "role": "main",
            },
            {
                "manuscript_table": "Table 3",
                "title": "Selection probability / probability of being top-ranked",
                "source_file": "selection_probability_summary.csv",
                "role": "main",
            },
            {
                "manuscript_table": "Table 4",
                "title": "Checkpoint policy paired comparison",
                "source_file": "checkpoint_policy_paired_comparison_table.csv",
                "role": "main_or_supplementary",
            },
            {
                "manuscript_table": "Table 5",
                "title": "Mixed-effects variance attribution",
                "source_file": "mixed_effects_variance_attribution_summary.csv",
                "role": "main_or_supplementary",
            },
            {
                "manuscript_table": "Supplementary Table S1",
                "title": "Primary performance summary",
                "source_file": "primary_performance_summary.csv",
                "role": "supplementary",
            },
            {
                "manuscript_table": "Supplementary Table S2",
                "title": "All pairwise paired bootstrap comparisons",
                "source_file": "paired_bootstrap_comparison_summary.csv",
                "role": "supplementary",
            },
            {
                "manuscript_table": "Supplementary Table S3",
                "title": "Per-model rank distributions",
                "source_file": "rank_distribution_summary.csv",
                "role": "supplementary",
            },
            {
                "manuscript_table": "Supplementary Table S4",
                "title": "Sensitivity analyses for alternative model specifications",
                "source_file": "mixed_effects_sensitivity_model_status.csv",
                "role": "supplementary",
            },
            {
                "manuscript_table": "Supplementary Table S5",
                "title": "Multiplicity/comparison-family record",
                "source_file": "multiplicity_comparison_family_record.csv",
                "role": "supplementary_if_formal_claims_are_made",
            },
        ]
    )


def save_selection_probability_figure(selection: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "selection_probability_by_workflow_context.png"
    datasets = list(selection["dataset"].drop_duplicates())
    policies = list(selection["checkpoint_policy"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(policies), figsize=(11, 6), sharey=True)
    if len(datasets) == 1:
        axes = np.array([axes])
    for r, dataset in enumerate(datasets):
        for c, policy in enumerate(policies):
            ax = axes[r, c]
            sub = selection[(selection["dataset"] == dataset) & (selection["checkpoint_policy"] == policy)].copy()
            models = ordered_models(sub["model"].tolist())
            sub = sub.set_index("model").loc[models].reset_index()
            ax.bar(sub["model"], sub["selection_probability"], color="#4C78A8")
            yerr_low = sub["selection_probability"] - sub["bootstrap_ci_low"]
            yerr_high = sub["bootstrap_ci_high"] - sub["selection_probability"]
            ax.errorbar(sub["model"], sub["selection_probability"], yerr=[yerr_low, yerr_high], fmt="none", color="#222222", capsize=3)
            ax.set_title(f"{dataset} / policy {policy}")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Top-ranked probability across repeated workflow contexts", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_ranking_flip_figure(stability: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "ranking_flip_rate_by_workflow_context.png"
    labels = stability["dataset"] + " / " + stability["checkpoint_policy"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, stability["ranking_flip_rate"], color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ranking flip rate")
    ax.set_title("Single-context benchmark conclusion fragility")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_paired_uncertainty_figure(pairwise: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "paired_uncertainty_resnet18_vs_resnet50.png"
    sub = pairwise[
        (
            ((pairwise["model_a"] == "resnet18") & (pairwise["model_b"] == "resnet50"))
            | ((pairwise["model_a"] == "resnet50") & (pairwise["model_b"] == "resnet18"))
        )
    ].copy()
    if sub.empty:
        return path
    signs = np.where(sub["model_a"] == "resnet18", 1.0, -1.0)
    sub["difference_resnet18_minus_resnet50"] = signs * sub["mean_difference_model_a_minus_b"]
    sub["ci_low_resnet18_minus_resnet50"] = np.where(
        signs > 0, sub["bootstrap_ci_low"], -sub["bootstrap_ci_high"]
    )
    sub["ci_high_resnet18_minus_resnet50"] = np.where(
        signs > 0, sub["bootstrap_ci_high"], -sub["bootstrap_ci_low"]
    )
    sub["label"] = sub["dataset"] + " / policy " + sub["checkpoint_policy"]
    sub = sub.sort_values(["dataset", "checkpoint_policy"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(sub))
    x = sub["difference_resnet18_minus_resnet50"].to_numpy()
    xerr = np.vstack([x - sub["ci_low_resnet18_minus_resnet50"], sub["ci_high_resnet18_minus_resnet50"] - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#54A24B", ecolor="#333333", capsize=3)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("Balanced accuracy difference: ResNet18 minus ResNet50")
    ax.set_title("Paired uncertainty for adjacent top-tier models")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_variance_figure(variance: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "mixed_effects_variance_attribution.png"
    data = variance.copy()
    data["stratum"] = data["dataset"] + " / " + data["checkpoint_policy"]
    component_order = ["split", "model:split", "Residual"]
    strata = list(data["stratum"].drop_duplicates())
    pivot = data.pivot_table(index="stratum", columns="grp", values="variance_proportion", fill_value=0).reindex(strata)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(pivot))
    colors = {"split": "#4C78A8", "model:split": "#E45756", "Residual": "#72B7B2"}
    for component in component_order:
        if component not in pivot.columns:
            continue
        values = pivot[component].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=component, color=colors.get(component))
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance proportion")
    ax.set_title("Exploratory mixed-effects variance attribution")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
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
    pairwise = pd.read_csv(tables_dir / "paired_bootstrap_comparison_summary.csv")
    selection = pd.read_csv(tables_dir / "selection_probability_summary.csv")
    stability = pd.read_csv(tables_dir / "ranking_stability_summary.csv")
    variance = pd.read_csv(tables_dir / "mixed_effects_variance_attribution_summary.csv")

    if args.metric not in run_metrics.columns:
        raise SystemExit(f"Missing metric column: {args.metric}")
    if set(run_metrics["analysis_role"]) != {"primary_analysis"}:
        raise SystemExit("run_metrics_table.csv contains non-primary analysis roles")
    if set(run_metrics["run_status"]) != {"completed"}:
        raise SystemExit("run_metrics_table.csv contains non-completed runs")

    workflow, workflow_summary = build_workflow_comparison(run_metrics, args.metric)
    fragility = build_fragility_examples(run_metrics, args.metric)
    checkpoint_paired, checkpoint_shifts = build_checkpoint_policy_tables(
        run_metrics, args.metric, args.bootstrap_resamples
    )
    multiplicity = build_multiplicity_record(pairwise, checkpoint_paired)
    workflow_matrix = build_experimental_workflow_matrix(run_metrics)
    table_inventory = build_manuscript_table_inventory()

    output_paths = {}
    for name, table in [
        ("experimental_workflow_matrix.csv", workflow_matrix),
        ("workflow_factor_comparison_summary.csv", workflow),
        ("workflow_factor_comparison_aggregate.csv", workflow_summary),
        ("single_context_fragility_examples.csv", fragility),
        ("checkpoint_policy_paired_comparison_table.csv", checkpoint_paired),
        ("checkpoint_policy_ranking_shift_summary.csv", checkpoint_shifts),
        ("multiplicity_comparison_family_record.csv", multiplicity),
        ("manuscript_table_inventory.csv", table_inventory),
    ]:
        path = tables_dir / name
        table.to_csv(path, index=False)
        output_paths[name] = path

    figure_paths = [
        save_selection_probability_figure(selection, figures_dir),
        save_ranking_flip_figure(stability, figures_dir),
        save_paired_uncertainty_figure(pairwise, figures_dir),
        save_variance_figure(variance, figures_dir),
    ]

    summary = {
        **runtime_metadata(Path(__file__).name),
        "status": "completed",
        "metric": args.metric,
        "analysis_output_dir": display_path(analysis_output_dir),
        "tables_dir": display_path(tables_dir),
        "figures_dir": display_path(figures_dir),
        "row_counts": {
            name: int(pd.read_csv(path).shape[0]) for name, path in output_paths.items()
        },
        "figures": [display_path(path) for path in figure_paths],
        "interpretation_boundary": (
            "Follow-up outputs are derived methodology-oriented summaries. They do not change the registered "
            "primary analysis matrix or the primary outcome construct."
        ),
    }
    write_json(tables_dir / "methodology_followup_analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
