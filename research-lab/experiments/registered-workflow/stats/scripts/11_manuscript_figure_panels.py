#!/usr/bin/env python3
"""Create main-text manuscript figures using harmonized terminology."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_LABELS = {
    "organamnist": "OrganAMNIST",
    "sipakmed": "SIPaKMeD",
}

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
}

MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
STRATUM_ORDER = [
    ("organamnist", "A"),
    ("organamnist", "B"),
    ("sipakmed", "A"),
    ("sipakmed", "B"),
]
STRATUM_COLORS = {
    ("organamnist", "A"): "#3B6EA8",
    ("organamnist", "B"): "#D28E2A",
    ("sipakmed", "A"): "#2D8C5B",
    ("sipakmed", "B"): "#B54A4A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
        help="Directory containing statistical output tables.",
    )
    parser.add_argument(
        "--figures-dir",
        default="research-lab/experiments/registered-workflow/stats/figures",
        help="Directory for manuscript figures.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def stratum_label(dataset: str, checkpoint_rule: str) -> str:
    return f"{DATASET_LABELS.get(dataset, dataset)} / rule {checkpoint_rule}"


def save_figure(fig: plt.Figure, path_base: Path) -> list[Path]:
    outputs = [
        path_base.with_suffix(".png"),
        path_base.with_suffix(".svg"),
        path_base.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def make_selection_frequency_figure(selection: pd.DataFrame, figures_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    axes = axes.ravel()

    for ax, (dataset, rule) in zip(axes, STRATUM_ORDER):
        sub = (
            selection[
                (selection["dataset"] == dataset)
                & (selection["checkpoint_policy"] == rule)
            ]
            .set_index("model")
            .reindex(MODEL_ORDER)
            .reset_index()
        )
        values = sub["selection_probability"].fillna(0.0).to_numpy(dtype=float)
        low = sub["bootstrap_ci_low"].to_numpy(dtype=float)
        high = sub["bootstrap_ci_high"].to_numpy(dtype=float)
        low = np.where(np.isnan(low), values, low)
        high = np.where(np.isnan(high), values, high)
        x = np.arange(len(MODEL_ORDER))
        colors = ["#4E79A7" if v > 0 else "#D9D9D9" for v in values]
        ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.4)
        ax.errorbar(
            x,
            values,
            yerr=[np.maximum(values - low, 0), np.maximum(high - values, 0)],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
        )
        ax.set_title(stratum_label(dataset, rule), fontsize=9, pad=6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=35, ha="right")
        ax.set_ylabel("Selection frequency" if ax in (axes[0], axes[2]) else "")
        ax.grid(axis="x", visible=False)

    fig.suptitle("Top-ranked model selection frequency across single split-seed contexts", fontsize=11, y=1.02)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_selection_frequency")


def make_subsampling_recovery_figure(subsampling: pd.DataFrame, figures_dir: Path) -> list[Path]:
    budget_order = [
        "1_splits_x_1_seeds",
        "1_splits_x_3_seeds",
        "3_splits_x_1_seeds",
        "3_splits_x_3_seeds",
        "5_splits_x_3_seeds",
        "10_splits_x_5_seeds",
    ]
    budget_labels = ["1x1", "1x3", "3x1", "3x3", "5x3", "10x5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(budget_order))

    for dataset, rule in STRATUM_ORDER:
        sub = (
            subsampling[
                (subsampling["dataset"] == dataset)
                & (subsampling["checkpoint_policy"] == rule)
            ]
            .set_index("resource_budget")
            .reindex(budget_order)
            .reset_index()
        )
        y = sub["recovered_full_reference_probability"].to_numpy(dtype=float)
        lo = sub["recovered_full_reference_ci_low"].to_numpy(dtype=float)
        hi = sub["recovered_full_reference_ci_high"].to_numpy(dtype=float)
        color = STRATUM_COLORS[(dataset, rule)]
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=stratum_label(dataset, rule))
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

    ax.set_title("Subsampling recovery of the full repeated-evaluation reference", fontsize=11, pad=8)
    ax.set_ylabel("Subsampling recovery rate")
    ax.set_xlabel("Repeated-evaluation budget (splits x seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure3_subsampling_recovery")


def make_top_two_margin_figure(
    context_gaps: pd.DataFrame,
    model_level: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    groups = []
    labels = []
    full_reference_margins = []
    for dataset, rule in STRATUM_ORDER:
        sub = context_gaps[
            (context_gaps["dataset"] == dataset)
            & (context_gaps["checkpoint_policy"] == rule)
        ]
        groups.append(sub["top1_top2_gap"].to_numpy(dtype=float))
        labels.append(stratum_label(dataset, rule))
        ml = model_level[
            (model_level["dataset"] == dataset)
            & (model_level["checkpoint_policy"] == rule)
            & (model_level["rank"] == 1)
        ]
        full_reference_margins.append(float(ml.iloc[0]["top_two_gap_in_stratum"]))

    positions = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 1.1},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, key in zip(bp["boxes"], STRATUM_ORDER):
        patch.set_facecolor(STRATUM_COLORS[key])
        patch.set_alpha(0.28)

    ax.scatter(
        positions,
        full_reference_margins,
        marker="D",
        s=42,
        color="#111111",
        label="Full-reference margin",
        zorder=3,
    )
    ax.set_title("Distribution of top-two balanced-accuracy margins", fontsize=11, pad=8)
    ax.set_ylabel("Top-two margin")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure2_top_two_margin")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(tables_dir / "selection_probability_summary.csv")
    subsampling = pd.read_csv(tables_dir / "subsampling_resource_stability_summary.csv")
    context_gaps = pd.read_csv(tables_dir / "top_two_context_gap_table.csv")
    model_level = pd.read_csv(tables_dir / "model_level_performance_table.csv")

    outputs = []
    outputs.extend(make_selection_frequency_figure(selection, figures_dir))
    outputs.extend(make_subsampling_recovery_figure(subsampling, figures_dir))
    outputs.extend(make_top_two_margin_figure(context_gaps, model_level, figures_dir))

    print("MANUSCRIPT_FIGURES_OK")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
