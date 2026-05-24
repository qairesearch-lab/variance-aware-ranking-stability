#!/usr/bin/env python3
"""Generate reviewer-requested revision analyses from frozen primary outputs.

These analyses are derived from the completed 800-run primary matrix. They do
not rerun training, regenerate splits, or alter the registered workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cervical-review-revision")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis_utils import display_path, resolve_path, runtime_metadata, write_json


MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
METRIC = "balanced_accuracy"
PERFORMANCE_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "roc_auc_macro_ovr",
    "pr_auc_macro",
    "brier_score_multiclass",
]
RUN_KEYS = ["dataset", "checkpoint_policy", "split_id", "training_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-output-dir",
        default="research-lab/experiments/registered-workflow/outputs/primary/analysis_io",
    )
    parser.add_argument(
        "--runs-dir",
        default="research-lab/experiments/registered-workflow/runs/primary",
    )
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
    )
    parser.add_argument(
        "--figures-dir",
        default="research-lab/experiments/registered-workflow/stats/figures",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=431510)
    return parser.parse_args()


def ordered_models(models: list[str]) -> list[str]:
    known = [model for model in MODEL_ORDER if model in models]
    unknown = sorted(model for model in models if model not in known)
    return known + unknown


def top_model_from_means(means: pd.Series) -> tuple[str, pd.Series]:
    ordered = means.reindex(ordered_models(list(means.index))).sort_values(ascending=False, kind="mergesort")
    return str(ordered.index[0]), ordered


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def build_paired_context_bootstrap(run_metrics: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for stratum_index, ((dataset, policy), sub) in enumerate(run_metrics.groupby(["dataset", "checkpoint_policy"])):
        context_matrix = (
            sub.assign(context_id=sub["split_id"].astype(str) + "|" + sub["training_seed"].astype(int).astype(str))
            .pivot_table(index="context_id", columns="model", values=METRIC, aggfunc="mean")
            .reindex(columns=ordered_models(sorted(sub["model"].astype(str).unique())))
            .sort_index()
        )
        values = context_matrix.to_numpy(dtype=float)
        full_means = context_matrix.mean(axis=0)
        full_means = pd.Series(full_means, index=context_matrix.columns)
        full_top, full_ordered = top_model_from_means(full_means)
        models = list(full_ordered.index)
        top_counts = Counter()
        model_positions = [list(context_matrix.columns).index(model) for model in models]
        for _ in range(resamples):
            sample_indices = rng.integers(0, values.shape[0], size=values.shape[0])
            means = values[sample_indices, :].mean(axis=0)
            ordered_pairs = sorted(
                ((means[position], model) for position, model in zip(model_positions, models)),
                key=lambda item: (-item[0], item[1]),
            )
            top_counts[ordered_pairs[0][1]] += 1
        for model in models:
            ci_low, ci_high = wilson_ci(int(top_counts[model]), resamples)
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "model": model,
                    "full_reference_top_model": full_top,
                    "full_reference_rank": int(list(models).index(model) + 1),
                    "bootstrap_top_count": int(top_counts[model]),
                    "bootstrap_replicates": resamples,
                    "paired_context_bootstrap_top_probability": top_counts[model] / resamples,
                    "monte_carlo_ci_low": ci_low,
                    "monte_carlo_ci_high": ci_high,
                    "interval_type": "monte_carlo_wilson_95",
                    "bootstrap_seed": seed + stratum_index,
                    "resampling_unit": "split_seed_context",
                    "pairing_preserved": True,
                    "model_level_mean_recomputed_each_replicate": True,
                    "tie_breaker": "alphabetical_after_descending_metric",
                }
            )
    return pd.DataFrame(rows)


def build_reference_sensitivity(run_metrics: pd.DataFrame, model_ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, policy), sub in run_metrics.groupby(["dataset", "checkpoint_policy"]):
        full_means = sub.groupby("model")[METRIC].mean()
        mean_top, ordered = top_model_from_means(full_means)
        context_tops = (
            model_ranking[
                (model_ranking["dataset"] == dataset)
                & (model_ranking["checkpoint_policy"] == policy)
                & (model_ranking["is_top_ranked"].astype(int) == 1)
            ]["model"]
            .astype(str)
            .tolist()
        )
        counts = Counter(context_tops)
        max_count = max(counts.values())
        majority_winners = sorted([model for model, count in counts.items() if count == max_count])
        majority_top = majority_winners[0]
        leave_one = []
        for split_id in sorted(sub["split_id"].astype(str).unique()):
            loo = sub[sub["split_id"].astype(str) != split_id]
            loo_top, _ = top_model_from_means(loo.groupby("model")[METRIC].mean())
            leave_one.append(loo_top)
        leave_counts = Counter(leave_one)
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "mean_balanced_accuracy_reference_top": mean_top,
                "mean_reference_ordering": "|".join(ordered.index),
                "context_majority_vote_top_alphabetical_tie": majority_top,
                "context_majority_vote_winners_shared": "|".join(majority_winners),
                "context_top_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                "majority_vote_changes_reference_top": majority_top != mean_top,
                "majority_vote_top_count_tie": len(majority_winners) > 1,
                "leave_one_split_reference_top_counts": json.dumps(dict(sorted(leave_counts.items())), sort_keys=True),
                "leave_one_split_all_match_mean_reference": all(model == mean_top for model in leave_one),
                "leave_one_split_changed_splits": int(sum(model != mean_top for model in leave_one)),
                "interpretation": "Alternative references are sensitivity checks; none is an oracle truth.",
            }
        )
    return pd.DataFrame(rows)


def build_tie_audit(run_metrics: pd.DataFrame, model_ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    sensitivity_rows = []
    for (dataset, policy), sub in run_metrics.groupby(["dataset", "checkpoint_policy"]):
        exact_context_ties = 0
        exact_top_ties = 0
        shared_top_match_counts = []
        alphabetical_match_counts = []
        full_means = sub.groupby("model")[METRIC].mean()
        mean_top, _ = top_model_from_means(full_means)
        full_reference_mean_tie = int(full_means.value_counts().max() > 1)
        full_reference_top_tie = int((full_means == full_means.max()).sum() > 1)
        for _, context in sub.groupby(["split_id", "training_seed"]):
            values = context.set_index("model")[METRIC]
            if values.duplicated(keep=False).any():
                exact_context_ties += 1
            if (values == values.max()).sum() > 1:
                exact_top_ties += 1
            top_models = sorted(values[values == values.max()].index.astype(str).tolist())
            shared_top_match_counts.append(mean_top in top_models)
            alphabetical_match_counts.append(top_models[0] == mean_top)
        top_counts = Counter(
            model_ranking[
                (model_ranking["dataset"] == dataset)
                & (model_ranking["checkpoint_policy"] == policy)
                & (model_ranking["is_top_ranked"].astype(int) == 1)
            ]["model"].astype(str)
        )
        max_top_count = max(top_counts.values())
        top_count_tie_models = sorted([model for model, count in top_counts.items() if count == max_top_count])
        audit_rows.extend(
            [
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_type": "exact_metric_tie_within_context",
                    "tie_count": exact_context_ties,
                    "denominator": 50,
                    "affected_models_or_note": "any equal balanced_accuracy values within a split-seed context",
                },
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_type": "exact_metric_top_tie_within_context",
                    "tie_count": exact_top_ties,
                    "denominator": 50,
                    "affected_models_or_note": "multiple models share context-level maximum balanced_accuracy",
                },
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_type": "full_reference_mean_balanced_accuracy_tie",
                    "tie_count": full_reference_mean_tie,
                    "denominator": 1,
                    "affected_models_or_note": "any duplicated model-level full-reference mean",
                },
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_type": "full_reference_top_mean_tie",
                    "tie_count": full_reference_top_tie,
                    "denominator": 1,
                    "affected_models_or_note": "multiple models share full-reference maximum mean",
                },
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_type": "top_count_tie",
                    "tie_count": int(len(top_count_tie_models) > 1),
                    "denominator": 1,
                    "affected_models_or_note": "|".join(top_count_tie_models),
                },
            ]
        )
        for method, matches in [
            ("alphabetical_top_tie_breaker", alphabetical_match_counts),
            ("shared_top_counting", shared_top_match_counts),
        ]:
            successes = int(sum(matches))
            low, high = wilson_ci(successes, len(matches))
            sensitivity_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_policy": policy,
                    "tie_handling_method": method,
                    "reference_top_model": mean_top,
                    "match_count": successes,
                    "context_denominator": len(matches),
                    "match_rate": successes / len(matches),
                    "match_ci_low": low,
                    "match_ci_high": high,
                    "disagreement_rate": 1 - successes / len(matches),
                    "note": "Random tie-breaker simulation was not run because no context-level exact top ties were observed."
                    if exact_top_ties == 0
                    else "Context-level exact top ties observed; random tie-breaker simulation should be considered.",
                }
            )
    return pd.DataFrame(audit_rows), pd.DataFrame(sensitivity_rows)


def build_seed_variability(run_metrics: pd.DataFrame) -> pd.DataFrame:
    context = (
        run_metrics.groupby(["dataset", "checkpoint_policy", "model", "split_id"], as_index=False)[METRIC]
        .agg(seed_count="count", seed_sd="std", seed_range=lambda x: float(np.max(x) - np.min(x)))
    )
    rows = []
    for (dataset, policy, model), sub in context.groupby(["dataset", "checkpoint_policy", "model"]):
        q1_sd, q3_sd = np.quantile(sub["seed_sd"], [0.25, 0.75])
        q1_range, q3_range = np.quantile(sub["seed_range"], [0.25, 0.75])
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "model": model,
                "split_count": int(len(sub)),
                "seed_count_per_split": int(sub["seed_count"].median()),
                "seed_sd_median": float(sub["seed_sd"].median()),
                "seed_sd_iqr": float(q3_sd - q1_sd),
                "seed_sd_max": float(sub["seed_sd"].max()),
                "seed_range_median": float(sub["seed_range"].median()),
                "seed_range_iqr": float(q3_range - q1_range),
                "seed_range_max": float(sub["seed_range"].max()),
            }
        )
    return pd.DataFrame(rows)


def hodges_lehmann_one_sample(values: np.ndarray) -> float:
    walsh = []
    for i in range(len(values)):
        for j in range(i, len(values)):
            walsh.append((values[i] + values[j]) / 2)
    return float(np.median(walsh))


def rank_biserial_from_wilcoxon(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    total = n * (n + 1) / 2
    return (pos - neg) / total


def build_checkpoint_effects(run_metrics: pd.DataFrame, checkpoint_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, base in checkpoint_table.iterrows():
        dataset = base["dataset"]
        model = base["model"]
        sub = run_metrics[(run_metrics["dataset"] == dataset) & (run_metrics["model"] == model)]
        wide = sub.pivot_table(index=["split_id", "training_seed"], columns="checkpoint_policy", values=METRIC)
        wide = wide.dropna(subset=["A", "B"])
        diff = (wide["B"] - wide["A"]).to_numpy(dtype=float)
        sd = float(np.std(diff, ddof=1))
        dz = float(np.mean(diff) / sd) if sd > 0 else math.nan
        try:
            p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = math.nan
        row = base.to_dict()
        row.update(
            {
                "paired_cohens_dz": dz,
                "hodges_lehmann_difference": hodges_lehmann_one_sample(diff),
                "wilcoxon_signed_rank_p_value_exploratory_unadjusted": p_value,
                "rank_biserial_correlation": rank_biserial_from_wilcoxon(diff),
                "effect_size_note": "Paired Cohen's d_z is the preferred standardized effect size for the paired B-minus-A design.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def selected_epoch_rows(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for manifest in sorted(runs_dir.glob("run_*/checkpoint_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": data.get("run_id", manifest.parent.name),
                "checkpoint_policy": data.get("checkpoint_policy"),
                "checkpoint_used_for_evaluation": data.get("checkpoint_used_for_evaluation"),
                "selected_epoch": data.get("selected_epoch"),
            }
        )
    return pd.DataFrame(rows)


def build_epoch_distribution(run_metrics: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    epochs = selected_epoch_rows(runs_dir)
    joined = run_metrics.merge(epochs, on=["run_id", "checkpoint_policy"], how="left")
    rows = []
    for (dataset, policy, model), sub in joined.groupby(["dataset", "checkpoint_policy", "model"]):
        values = pd.to_numeric(sub["selected_epoch"], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        q1, q3 = np.quantile(values, [0.25, 0.75])
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "model": model,
                "run_count": int(len(values)),
                "selected_epoch_mean": float(np.mean(values)),
                "selected_epoch_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "selected_epoch_median": float(np.median(values)),
                "selected_epoch_iqr": float(q3 - q1),
                "selected_epoch_min": int(np.min(values)),
                "selected_epoch_max": int(np.max(values)),
                "early_stopped_before_60_count": int(np.sum(values < 60)),
            }
        )
    return pd.DataFrame(rows)


def build_model_level_performance(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = run_metrics.groupby(["dataset", "checkpoint_policy", "model"], as_index=False)
    summary = grouped[PERFORMANCE_METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    for (dataset, policy), sub in summary.groupby(["dataset", "checkpoint_policy"]):
        sub = sub.copy()
        sub["rank"] = sub[f"{METRIC}_mean"].rank(ascending=False, method="first").astype(int)
        top_value = float(sub[f"{METRIC}_mean"].max())
        top_two = sub.sort_values([f"{METRIC}_mean", "model"], ascending=[False, True]).head(2)
        top_two_gap = float(top_two.iloc[0][f"{METRIC}_mean"] - top_two.iloc[1][f"{METRIC}_mean"])
        sub["distance_to_top_model"] = top_value - sub[f"{METRIC}_mean"]
        sub["top_two_gap_in_stratum"] = top_two_gap
        rows.append(sub)
    return (
        pd.concat(rows, ignore_index=True)
        .drop(columns=["index"], errors="ignore")
        .sort_values(["dataset", "checkpoint_policy", "rank", "model"])
    )


def _load_training_env(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        pkg_versions = config.get("package_versions", {})
        device_info = config.get("device_info", {})
        return {
            "torch_version": pkg_versions.get("torch", ""),
            "torchvision_version": pkg_versions.get("torchvision", ""),
            "python_version": pkg_versions.get("python", ""),
            "cuda_version": device_info.get("cuda_version", ""),
            "gpu_models": ", ".join(device_info.get("gpu_models", [])) or "",
            "hostname": "not_recorded",
        }
    return {}


def build_parameter_environment_table() -> pd.DataFrame:
    rows = []
    runs_dir = REPO_ROOT / "research-lab/experiments/registered-workflow/runs/primary"
    run_dirs = sorted(list(runs_dir.glob("run_*")))
    
    training_envs = {}
    if len(run_dirs) >= 400:
        training_envs["sipakmed"] = _load_training_env(run_dirs[0])
        training_envs["organamnist"] = _load_training_env(run_dirs[400])

    try:
        import torch
        import torchvision
        from torch import nn
        from torchvision import models

        class_counts = {"sipakmed": 5, "organamnist": 11}
        scopes = {
            "resnet18": "layer4 + fc",
            "resnet50": "layer4 + fc",
            "densenet121": "features.denseblock4 + features.norm5 + classifier",
            "efficientnet_b0": "features[-1] + classifier",
        }
        for model_name in MODEL_ORDER:
            for dataset, classes in class_counts.items():
                model = models.get_model(model_name, weights=None)
                if model_name.startswith("resnet"):
                    model.fc = nn.Linear(model.fc.in_features, classes)
                elif model_name == "densenet121":
                    model.classifier = nn.Linear(model.classifier.in_features, classes)
                elif model_name == "efficientnet_b0":
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
                for parameter in model.parameters():
                    parameter.requires_grad = False
                if model_name.startswith("resnet"):
                    for parameter in model.layer4.parameters():
                        parameter.requires_grad = True
                    for parameter in model.fc.parameters():
                        parameter.requires_grad = True
                elif model_name == "densenet121":
                    for parameter in model.features.denseblock4.parameters():
                        parameter.requires_grad = True
                    for parameter in model.features.norm5.parameters():
                        parameter.requires_grad = True
                    for parameter in model.classifier.parameters():
                        parameter.requires_grad = True
                elif model_name == "efficientnet_b0":
                    for parameter in model.features[-1].parameters():
                        parameter.requires_grad = True
                    for parameter in model.classifier.parameters():
                        parameter.requires_grad = True
                total = sum(parameter.numel() for parameter in model.parameters())
                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                
                train_env = training_envs.get(dataset, {})
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "class_count": classes,
                        "torchvision_call": f"torchvision.models.{model_name}",
                        "weights_identifier": "IMAGENET1K_V1",
                        "trainable_scope": scopes[model_name],
                        "total_parameters_after_classifier_replacement": int(total),
                        "trainable_parameters": int(trainable),
                        "trainable_parameter_fraction": trainable / total,
                        "torch_version_analysis_host": torch.__version__,
                        "torchvision_version_analysis_host": torchvision.__version__,
                        "python_version_analysis_host": platform.python_version(),
                        "torch_version_training_host": train_env.get("torch_version", ""),
                        "torchvision_version_training_host": train_env.get("torchvision_version", ""),
                        "python_version_training_host": train_env.get("python_version", ""),
                        "cuda_version_training_host": train_env.get("cuda_version", ""),
                        "gpu_model_training_host": train_env.get("gpu_models", ""),
                    }
                )
    except Exception as exc:  # pragma: no cover - records environment failure.
        rows.append(
            {
                "dataset": "",
                "model": "",
                "class_count": "",
                "torchvision_call": "",
                "weights_identifier": "IMAGENET1K_V1",
                "trainable_scope": "",
                "total_parameters_after_classifier_replacement": "",
                "trainable_parameters": "",
                "trainable_parameter_fraction": "",
                "torch_version_analysis_host": "",
                "torchvision_version_analysis_host": "",
                "python_version_analysis_host": platform.python_version(),
                "torch_version_training_host": "",
                "torchvision_version_training_host": "",
                "python_version_training_host": "",
                "cuda_version_training_host": "",
                "gpu_model_training_host": "",
                "error": repr(exc),
            }
        )
    return pd.DataFrame(rows)


def save_seed_variability_figure(seed_summary: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "seed_level_variability_by_model.png"
    data = seed_summary.copy()
    data["stratum_model"] = data["dataset"] + "/" + data["checkpoint_policy"] + "/" + data["model"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(data["stratum_model"], data["seed_sd_median"], color="#4C78A8")
    ax.set_ylabel("Median within-split seed SD, balanced accuracy")
    ax.set_title("Seed-level variability across fixed splits")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_checkpoint_difference_figure(effects: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "checkpoint_policy_paired_effects.png"
    data = effects.copy()
    data["label"] = data["dataset"] + "/" + data["model"]
    y = np.arange(len(data))
    x = data["mean_difference_B_minus_A"].to_numpy()
    xerr = np.vstack([x - data["bootstrap_ci_low"].to_numpy(), data["bootstrap_ci_high"].to_numpy() - x])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#F58518", ecolor="#333333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Policy B - policy A balanced accuracy")
    ax.set_title("Paired checkpoint-policy differences")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_matrix_figure(figures_dir: Path) -> Path:
    path = figures_dir / "experimental_matrix_flow.png"
    labels = [
        ("Datasets", "2"),
        ("Models", "4"),
        ("Splits", "10"),
        ("Training seeds", "5"),
        ("Checkpoint\nselection rules", "2"),
        ("Primary runs", "800"),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(labels))
    for i, (x, (name, value)) in enumerate(zip(xs, labels)):
        ax.text(x, 0.58, value, ha="center", va="center", fontsize=16, fontweight="bold")
        ax.text(x, 0.32, name, ha="center", va="center", fontsize=8.5, linespacing=1.2)
        ax.add_patch(plt.Rectangle((x - 0.065, 0.22), 0.13, 0.48, fill=False, linewidth=1.4, edgecolor="#4C78A8"))
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 0.48), xytext=(x + 0.08, 0.48), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("Prespecified repeated-evaluation workflow matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    analysis_output_dir = resolve_path(args.analysis_output_dir)
    runs_dir = resolve_path(args.runs_dir)
    tables_dir = resolve_path(args.tables_dir)
    figures_dir = resolve_path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_metrics = pd.read_csv(analysis_output_dir / "run_metrics_table.csv")
    model_ranking = pd.read_csv(analysis_output_dir / "model_ranking_table.csv")
    checkpoint_base = pd.read_csv(tables_dir / "checkpoint_policy_paired_comparison_table.csv")

    paired_bootstrap = build_paired_context_bootstrap(run_metrics, args.bootstrap_resamples, args.bootstrap_seed)
    reference_sensitivity = build_reference_sensitivity(run_metrics, model_ranking)
    tie_audit, tie_sensitivity = build_tie_audit(run_metrics, model_ranking)
    seed_variability = build_seed_variability(run_metrics)
    checkpoint_effects = build_checkpoint_effects(run_metrics, checkpoint_base)
    epoch_distribution = build_epoch_distribution(run_metrics, runs_dir)
    model_performance = build_model_level_performance(run_metrics)
    parameter_environment = build_parameter_environment_table()

    outputs = {
        "paired_context_bootstrap_selection_probability.csv": paired_bootstrap,
        "reference_definition_sensitivity.csv": reference_sensitivity,
        "tie_handling_audit.csv": tie_audit,
        "tie_handling_sensitivity_summary.csv": tie_sensitivity,
        "seed_variability_summary.csv": seed_variability,
        "checkpoint_policy_paired_effects.csv": checkpoint_effects,
        "checkpoint_policy_epoch_distribution.csv": epoch_distribution,
        "model_level_performance_table.csv": model_performance,
        "model_parameter_environment_summary.csv": parameter_environment,
    }
    for filename, frame in outputs.items():
        frame.to_csv(tables_dir / filename, index=False)

    figure_paths = [
        save_matrix_figure(figures_dir),
        save_seed_variability_figure(seed_variability, figures_dir),
        save_checkpoint_difference_figure(checkpoint_effects, figures_dir),
    ]

    summary = {
        **runtime_metadata(Path(__file__).name),
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_output_dir": display_path(analysis_output_dir),
        "runs_dir": display_path(runs_dir),
        "tables_dir": display_path(tables_dir),
        "figures_dir": display_path(figures_dir),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "row_counts": {filename: int(frame.shape[0]) for filename, frame in outputs.items()},
        "figures": [display_path(path) for path in figure_paths],
        "interpretation_boundary": (
            "Revision analyses are derived from existing primary outputs and checkpoint manifests; "
            "no new training run, split generation, or registered matrix change was performed."
        ),
        "deferred_author_judgment_items": [
            "title removal of 800-run",
            "calibration ECE/MCE reanalysis",
            "transformer or ConvNeXt extension",
            "third dataset extension",
            "Bayesian mixed model",
            "checkpoint B-prime step-matched sensitivity",
            "Kendall W or ICC supplementary indices",
        ],
    }
    write_json(tables_dir / "review_revision_analyses_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("REVIEW_REVISION_ANALYSES_OK")


if __name__ == "__main__":
    main()
