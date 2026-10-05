#!/usr/bin/env python3
"""Generate lightweight reviewer-response supplement tables."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def read_split(dataset: str, splits_dir: Path) -> pd.DataFrame:
    path = splits_dir / dataset / "split_01.csv"
    if not path.exists():
        raise SystemExit(f"Missing split file: {path}")
    return pd.read_csv(path)


def build_dataset_feature_summary(splits_dir: Path) -> pd.DataFrame:
    rows = []
    metadata = {
        "sipakmed": {
            "dataset_name": "SIPaKMeD",
            "dataset_version": "original release, ICIP 2018",
            "official_split_available": False,
            "official_split_used_for_primary_analysis": False,
            "primary_split_policy": "generated_image_level_repeated_random_holdout",
            "group_id_available": False,
            "split_unit": "image",
            "image_level_limitation": "No patient/slide/group identifiers available; repeated random holdout is image-level.",
        },
        "organamnist": {
            "dataset_name": "OrganAMNIST",
            "dataset_version": "MedMNIST v2, local medmnist package 3.0.2",
            "official_split_available": True,
            "official_split_used_for_primary_analysis": False,
            "primary_split_policy": "generated_image_level_repeated_random_holdout",
            "group_id_available": False,
            "split_unit": "image",
            "image_level_limitation": "No patient/group identifiers available; official split recorded for traceability only.",
        },
    }
    for dataset, meta in metadata.items():
        split = read_split(dataset, splits_dir)
        class_counts = split.groupby("label")["sample_id"].nunique().sort_index()
        subset_counts = split.groupby("subset")["sample_id"].nunique().to_dict()
        rows.append(
            {
                "dataset": dataset,
                **meta,
                "sample_count": int(split["sample_id"].nunique()),
                "class_count": int(class_counts.shape[0]),
                "class_distribution_by_label": json.dumps(
                    {str(k): int(v) for k, v in class_counts.items()}, sort_keys=True
                ),
                "split_01_subset_counts": json.dumps(
                    {str(k): int(v) for k, v in sorted(subset_counts.items())}, sort_keys=True
                ),
                "interpretation_role": (
                    "Dataset features are used to contextualize cross-dataset stability differences; "
                    "they are not treated as causal explanations of ranking instability."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_performance_gap_boundary(tables_dir: Path) -> pd.DataFrame:
    top_two = pd.read_csv(tables_dir / "top_two_gap_instability_summary.csv")
    pairwise = pd.read_csv(tables_dir / "paired_bootstrap_comparison_summary.csv")
    rows = []
    for _, row in top_two.iterrows():
        dataset = row["dataset"]
        policy = row["checkpoint_policy"]
        sub = pairwise[(pairwise["dataset"] == dataset) & (pairwise["checkpoint_policy"] == policy)].copy()
        sub["abs_mean_difference"] = sub["mean_difference_model_a_minus_b"].abs()
        extreme = sub.sort_values(["abs_mean_difference", "model_a", "model_b"], ascending=[False, True, True]).iloc[0]
        reference_pair = sub[
            (
                (sub["model_a"] == row["reference_top_model"])
                & (sub["model_b"] == row["reference_runner_up_model"])
            )
            | (
                (sub["model_b"] == row["reference_top_model"])
                & (sub["model_a"] == row["reference_runner_up_model"])
            )
        ]
        ref_direction_consistency = float(reference_pair.iloc[0]["direction_consistency"]) if not reference_pair.empty else float("nan")
        extreme_gap = float(extreme["abs_mean_difference"])
        top_two_gap = float(row["gap_mean"])
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_policy": policy,
                "contexts": int(row["contexts"]),
                "reference_top_model": row["reference_top_model"],
                "reference_runner_up_model": row["reference_runner_up_model"],
                "top_two_gap_mean": top_two_gap,
                "top_two_gap_sd": float(row["gap_sd"]),
                "top_two_gap_median": float(row["gap_median"]),
                "top_two_gap_min": float(row["gap_min"]),
                "top_two_gap_max": float(row["gap_max"]),
                "reference_pair_sign_switch_frequency": float(row["reference_pair_sign_switch_frequency"]),
                "top_two_identity_switch_frequency": float(row["top_two_identity_switch_frequency"]),
                "reference_pair_direction_consistency": ref_direction_consistency,
                "largest_pairwise_abs_mean_difference": extreme_gap,
                "largest_gap_model_pair": f"{extreme['model_a']} vs {extreme['model_b']}",
                "top_two_gap_to_largest_gap_ratio": top_two_gap / extreme_gap if extreme_gap else float("nan"),
                "boundary_interpretation": (
                    "Small top-two margins support caution against top-ranked model claims; "
                    "larger non-top pair gaps show this is a boundary about close candidate comparisons, "
                    "not a universal claim that every model ordering is equally unstable."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--tables-dir", required=True)
    args = parser.parse_args()

    splits_dir = resolve_path(args.splits_dir)
    tables_dir = resolve_path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_features = build_dataset_feature_summary(splits_dir)
    performance_gaps = build_performance_gap_boundary(tables_dir)

    outputs = {
        "dataset_feature_comparison_summary.csv": dataset_features,
        "performance_gap_boundary_summary.csv": performance_gaps,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables_dir / name, index=False)

    summary = {
        "script": Path(__file__).name,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "splits_dir": display_path(splits_dir),
        "tables_dir": display_path(tables_dir),
        "outputs": {name: int(len(frame)) for name, frame in outputs.items()},
        "interpretation_boundary": (
            "Reviewer-response supplement tables are derived from frozen split definitions and existing 800-run analysis outputs; "
            "they do not add training runs or change the registered analysis matrix."
        ),
    }
    (tables_dir / "review_response_supplement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("REVIEW_RESPONSE_SUPPLEMENT_OK")


if __name__ == "__main__":
    main()
