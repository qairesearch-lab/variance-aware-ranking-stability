"""Summarize observed GPU time and conditional stability from completed runs.

This operational secondary analysis uses only completed primary-run timestamps
and the existing exhaustive subsampling outputs. It does not rerun training or
alter the prespecified 800-run analysis matrix.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = WORKFLOW_ROOT / "runs" / "primary"
TABLES_DIR = WORKFLOW_ROOT / "stats" / "tables"
FIGURES_DIR = WORKFLOW_ROOT / "stats" / "figures"
MODELS = ("resnet18", "resnet50", "densenet121", "efficientnet_b0")
BUDGET_LABELS = {(1, 1): "1x1", (1, 3): "1x3", (3, 1): "3x1", (3, 3): "3x3", (5, 3): "5x3", (10, 5): "10x5"}
STRATUM_ORDER = (("organamnist", "A"), ("organamnist", "B"), ("sipakmed", "A"), ("sipakmed", "B"))
STRATUM_TITLES = {
    ("organamnist", "A"): "OrganAMNIST / A",
    ("organamnist", "B"): "OrganAMNIST / B",
    ("sipakmed", "A"): "SIPaKMeD / A",
    ("sipakmed", "B"): "SIPaKMeD / B",
}
STRATUM_COLORS = {
    ("organamnist", "A"): "#35618D",
    ("organamnist", "B"): "#A84E4E",
    ("sipakmed", "A"): "#36836E",
    ("sipakmed", "B"): "#9A6B2F",
}


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_selected(value: str, cast):
    return [cast(item) for item in str(value).split(";") if item]


def collect_run_durations() -> tuple[dict[tuple[str, str, str, int, str], float], pd.DataFrame]:
    durations: dict[tuple[str, str, str, int, str], float] = {}
    records: list[dict[str, object]] = []

    for status_path in sorted(RUNS_DIR.glob("run_*/run_status.json")):
        config_path = status_path.with_name("run_config.json")
        status = json.loads(status_path.read_text())
        config = json.loads(config_path.read_text())
        manifest = config["manifest_row"]
        seconds = (parse_timestamp(status["ended_at_utc"]) - parse_timestamp(status["started_at_utc"])).total_seconds()
        key = (
            manifest["dataset"],
            manifest["checkpoint_policy"],
            manifest["split_id"],
            int(manifest["training_seed"]),
            manifest["model"],
        )
        durations[key] = seconds
        records.append(
            {
                "run_id": status["run_id"],
                "dataset": manifest["dataset"],
                "checkpoint_policy": manifest["checkpoint_policy"],
                "model": manifest["model"],
                "duration_seconds": seconds,
                "gpu_model": "; ".join(config["device_info"]["gpu_models"]),
            }
        )

    if len(durations) != 800:
        raise ValueError(f"Expected 800 completed primary runs, found {len(durations)}")
    return durations, pd.DataFrame(records)


def calculate_costs(durations: dict[tuple[str, str, str, int, str], float]) -> pd.DataFrame:
    detail = pd.read_csv(TABLES_DIR / "subsampling_resource_stability_detail.csv")
    detail["observed_gpu_hours"] = detail.apply(
        lambda row: sum(
            durations[(row.dataset, row.checkpoint_policy, split_id, seed, model)]
            for split_id in parse_selected(row.selected_splits, str)
            for seed in parse_selected(row.selected_seeds, int)
            for model in MODELS
        )
        / 3600,
        axis=1,
    )

    group_columns = ["dataset", "checkpoint_policy", "split_count", "seed_count", "resource_budget"]
    summary = (
        detail.groupby(group_columns, as_index=False)
        .agg(
            exhaustive_budget_selections=("sample_index", "count"),
            mean_observed_gpu_hours=("observed_gpu_hours", "mean"),
            median_observed_gpu_hours=("observed_gpu_hours", "median"),
            p25_observed_gpu_hours=("observed_gpu_hours", lambda values: values.quantile(0.25)),
            p75_observed_gpu_hours=("observed_gpu_hours", lambda values: values.quantile(0.75)),
        )
    )
    summary["model_runs_per_budget"] = 4 * summary["split_count"] * summary["seed_count"]

    stability = pd.read_csv(TABLES_DIR / "subsampling_resource_stability_summary.csv")
    stability = stability[
        group_columns
        + [
            "full_reference_self_comparison",
            "recovered_full_reference_probability",
            "recovered_full_reference_ci_low",
            "recovered_full_reference_ci_high",
            "ranking_correlation_mean",
            "ranking_correlation_sd",
        ]
    ]
    return summary.merge(stability, on=group_columns, validate="one_to_one")


def plot_tradeoff(cost_summary: pd.DataFrame, output_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)

    for axis, stratum in zip(axes.flat, STRATUM_ORDER):
        data = cost_summary.loc[
            (cost_summary["dataset"] == stratum[0]) & (cost_summary["checkpoint_policy"] == stratum[1])
        ].sort_values(["split_count", "seed_count"])
        color = STRATUM_COLORS[stratum]
        non_reference = data.loc[~data["full_reference_self_comparison"]]

        axis.plot(
            non_reference["mean_observed_gpu_hours"],
            non_reference["recovered_full_reference_probability"],
            color=color,
            linewidth=1.3,
            zorder=2,
        )
        axis.errorbar(
            non_reference["mean_observed_gpu_hours"],
            non_reference["recovered_full_reference_probability"],
            yerr=[
                non_reference["recovered_full_reference_probability"] - non_reference["recovered_full_reference_ci_low"],
                non_reference["recovered_full_reference_ci_high"] - non_reference["recovered_full_reference_probability"],
            ],
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=4.6,
            capsize=1.8,
            elinewidth=0.8,
            zorder=3,
        )
        reference = data.loc[data["full_reference_self_comparison"]]
        axis.scatter(
            reference["mean_observed_gpu_hours"],
            reference["recovered_full_reference_probability"],
            facecolors="white",
            edgecolors=color,
            marker="s",
            linewidths=1.1,
            s=30,
            zorder=4,
        )

        for row in data.itertuples():
            label = BUDGET_LABELS[(row.split_count, row.seed_count)]
            if label in {"1x1", "3x3", "5x3", "10x5"}:
                axis.annotate(
                    label,
                    (row.mean_observed_gpu_hours, row.recovered_full_reference_probability),
                    xytext=(3, -9 if label != "10x5" else 5),
                    textcoords="offset points",
                    color="#303030",
                    fontsize=6,
                )

        axis.set_title(STRATUM_TITLES[stratum], fontsize=8, fontweight="bold", loc="left")
        axis.set_xscale("log")
        axis.set_xlim(0.25, 150)
        axis.set_ylim(0.35, 1.05)
        axis.set_yticks([0.4, 0.6, 0.8, 1.0])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.tick_params(labelsize=6, length=2.5)

    fig.supxlabel("Mean observed GPU hours per four-model stratum (log scale)", y=0.075, fontsize=7)
    fig.supylabel("Conditional recovery of the completed-grid empirical comparator", x=0.012, fontsize=7)
    fig.tight_layout(rect=(0.055, 0.12, 1, 0.98))
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    durations, run_records = collect_run_durations()
    cost_summary = calculate_costs(durations)
    cost_summary = cost_summary.sort_values(["dataset", "checkpoint_policy", "split_count", "seed_count"])
    cost_summary.to_csv(TABLES_DIR / "observed_compute_cost_stability_summary.csv", index=False)

    run_records.to_csv(TABLES_DIR / "observed_primary_run_duration_summary.csv", index=False)
    manifest = {
        "analysis_type": "operational secondary analysis derived from completed primary-run timestamps",
        "completed_primary_runs": int(len(run_records)),
        "total_observed_gpu_hours": float(run_records["duration_seconds"].sum() / 3600),
        "source_tables": [
            "subsampling_resource_stability_detail.csv",
            "subsampling_resource_stability_summary.csv",
        ],
        "outputs": [
            "observed_compute_cost_stability_summary.csv",
            "observed_primary_run_duration_summary.csv",
            "observed_compute_cost_stability_tradeoff.svg",
            "observed_compute_cost_stability_tradeoff.pdf",
            "observed_compute_cost_stability_tradeoff.png",
            "observed_compute_cost_stability_tradeoff.tiff",
        ],
        "interpretation_boundary": "Observed GPU hours describe the completed hardware-specific workflow and do not benchmark GPU hardware performance.",
    }
    (TABLES_DIR / "observed_compute_cost_stability_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_tradeoff(cost_summary, FIGURES_DIR / "observed_compute_cost_stability_tradeoff")
    print("OBSERVED_COMPUTE_COST_STABILITY_OK")


if __name__ == "__main__":
    main()
