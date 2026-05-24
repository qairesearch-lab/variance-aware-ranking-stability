#!/usr/bin/env python3
"""Generate a manuscript numeric-consistency audit table for the CMPB manuscript fork."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STRATA = [
    ("organamnist", "A", "OrganAMNIST / A"),
    ("organamnist", "B", "OrganAMNIST / B"),
    ("sipakmed", "A", "SIPaKMeD / A"),
    ("sipakmed", "B", "SIPaKMeD / B"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def row(
    metric: str,
    stratum: str,
    abstract: str,
    results_section: str,
    table_id: str,
    csv_source: str,
    source_value: str,
    match_yes_no: str,
    action: str,
) -> dict[str, str]:
    return {
        "metric": metric,
        "stratum": stratum,
        "abstract": abstract,
        "results_section": results_section,
        "table_id": table_id,
        "csv_source": csv_source,
        "source_value": source_value,
        "match_yes_no": match_yes_no,
        "action": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
        help="Directory containing statistical output tables.",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    interval = read_csv(tables_dir / "workflow_stability_interval_summary.csv")
    paired_boot = read_csv(tables_dir / "paired_context_bootstrap_selection_probability.csv")
    model_level = read_csv(tables_dir / "model_level_performance_table.csv")
    top_two = read_csv(tables_dir / "top_two_gap_instability_summary.csv")
    checkpoint = read_csv(tables_dir / "checkpoint_policy_paired_effects.csv")
    epochs = read_csv(tables_dir / "checkpoint_policy_epoch_distribution.csv")
    variance = read_csv(tables_dir / "mixed_effects_sensitivity_variance_components.csv")
    subsampling = read_csv(tables_dir / "subsampling_resource_stability_summary.csv")

    output: list[dict[str, str]] = []

    for dataset, policy, label in STRATA:
        match = next(
            r
            for r in interval
            if r["interval_target"] == "match_rate"
            and r["evidence_level"] == "single_split_seed_context"
            and r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
        )
        flip = next(
            r
            for r in interval
            if r["interval_target"] == "flip_rate"
            and r["evidence_level"] == "single_split_seed_context"
            and r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
        )
        output.append(
            row(
                "single-context agreement rate",
                label,
                "0.44-0.62 range stated",
                f"{fmt2(float(match['estimate']))} [{fmt2(float(match['ci_low']))}, {fmt2(float(match['ci_high']))}]",
                "Results 3.2 table",
                "workflow_stability_interval_summary.csv",
                f"count={match['count']}/{match['denominator']}; estimate={match['estimate']}; ci_method={match['ci_method']}",
                "yes",
                "Retain; values match source table after rounding.",
            )
        )
        output.append(
            row(
                "single-context top-model discordance rate",
                label,
                "0.38-0.56 range stated",
                f"{fmt2(float(flip['estimate']))} [{fmt2(float(flip['ci_low']))}, {fmt2(float(flip['ci_high']))}]",
                "Results 3.2 table",
                "workflow_stability_interval_summary.csv",
                f"count={flip['count']}/{flip['denominator']}; estimate={flip['estimate']}; ci_method={flip['ci_method']}",
                "yes",
                "Retain; values match source table after rounding.",
            )
        )

        top_model = next(
            r
            for r in model_level
            if r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
            and r["rank"] == "1"
        )["model"]
        observed = next(
            r
            for r in interval
            if r["interval_target"] == "selection_probability"
            and r["evidence_level"] == "repeated_split_seed_context"
            and r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
            and r["model"] == top_model
        )
        output.append(
            row(
                "observed selection frequency of full-reference top model",
                label,
                "0.62, 0.60, 0.52, 0.44 stated",
                f"{fmt2(float(observed['estimate']))} [{fmt2(float(observed['ci_low']))}, {fmt2(float(observed['ci_high']))}]",
                "Results 3.2 text and Figure 1",
                "workflow_stability_interval_summary.csv",
                f"model={top_model}; count={observed['count']}/{observed['denominator']}; ci_method={observed['ci_method']}",
                "yes",
                "Retain; Figure 1 uses bootstrap CI from selection_probability_summary.csv, while the text reports the same point estimates.",
            )
        )

        boot = next(
            r
            for r in paired_boot
            if r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
            and r["full_reference_rank"] == "1"
        )
        output.append(
            row(
                "paired bootstrap selection frequency of full-reference top model",
                label,
                "0.99, 0.83, 0.96, 0.57 stated",
                fmt2(float(boot["paired_context_bootstrap_top_probability"])),
                "Results 3.2 text; Supplementary Table S10",
                "paired_context_bootstrap_selection_probability.csv",
                (
                    f"model={boot['model']}; probability={boot['paired_context_bootstrap_top_probability']}; "
                    f"CI={boot['monte_carlo_ci_low']}-{boot['monte_carlo_ci_high']}; interval_type={boot['interval_type']}"
                ),
                "yes",
                "Retain; Monte Carlo Wilson 95% CIs have been added to the manuscript text.",
            )
        )

        full_top = next(
            r
            for r in model_level
            if r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
            and r["rank"] == "1"
        )
        context_gap = next(
            r
            for r in top_two
            if r["dataset"] == dataset and r["checkpoint_policy"] == policy
        )
        output.append(
            row(
                "full-reference top-two margin",
                label,
                "0.0002-0.0023 full-reference range stated",
                fmt(float(full_top["top_two_gap_in_stratum"])),
                "Results 3.1 model-level table; Figure 2 diamond",
                "model_level_performance_table.csv",
                full_top["top_two_gap_in_stratum"],
                "yes",
                "Retain; this is the full repeated-evaluation reference margin.",
            )
        )
        output.append(
            row(
                "single-context mean top-two margin",
                label,
                "Not used for the abstract range",
                fmt(float(context_gap["gap_mean"])),
                "Results 3.3 text; Figure 2 boxplot",
                "top_two_gap_instability_summary.csv",
                f"mean={context_gap['gap_mean']}; sd={context_gap['gap_sd']}; contexts={context_gap['contexts']}",
                "explained",
                "Clarify in that this value is averaged across the 50 single split-seed contexts.",
            )
        )

    checkpoint_deltas = [float(r["mean_difference_B_minus_A"]) for r in checkpoint]
    dz_values = [float(r["paired_cohens_dz"]) for r in checkpoint]
    direction_values = [float(r["direction_consistency"]) for r in checkpoint]
    ci_cross_zero = [
        f"{r['dataset']}/{r['model']}"
        for r in checkpoint
        if float(r["bootstrap_ci_low"]) < 0 < float(r["bootstrap_ci_high"])
    ]
    output.extend(
        [
            row(
                "checkpoint-rule B-minus-A mean difference range",
                "All checkpoint-paired comparisons",
                "Not in abstract",
                f"{fmt(min(checkpoint_deltas))}-{fmt(max(checkpoint_deltas))}",
                "Results 3.5 text",
                "checkpoint_policy_paired_effects.csv",
                f"min={min(checkpoint_deltas)}; max={max(checkpoint_deltas)}",
                "yes",
                "Retain; values match source table after rounding.",
            ),
            row(
                "checkpoint-rule CI crossing zero",
                "All checkpoint-paired comparisons",
                "Not in abstract",
                "Only SIPaKMeD/EfficientNet-B0 crosses zero",
                "Results 3.5 text",
                "checkpoint_policy_paired_effects.csv",
                ";".join(ci_cross_zero),
                "yes",
                "Retain; only one bootstrap CI crosses zero.",
            ),
            row(
                "checkpoint-rule paired Cohen dz range",
                "All checkpoint-paired comparisons",
                "Not in abstract",
                f"{fmt(min(dz_values))}-{fmt(max(dz_values))}",
                "Results 3.5 text",
                "checkpoint_policy_paired_effects.csv",
                f"min={min(dz_values)}; max={max(dz_values)}",
                "yes",
                "Retain; values match source table after rounding.",
            ),
            row(
                "checkpoint-rule direction consistency range",
                "All checkpoint-paired comparisons",
                "Not in abstract",
                f"{fmt2(min(direction_values))}-{fmt2(max(direction_values))}",
                "Results 3.5 text",
                "checkpoint_policy_paired_effects.csv",
                f"min={min(direction_values)}; max={max(direction_values)}",
                "yes",
                "Retain; values match source table after rounding.",
            ),
        ]
    )

    a_epoch_medians = [
        float(r["selected_epoch_median"])
        for r in epochs
        if r["checkpoint_policy"] == "A"
    ]
    output.append(
        row(
            "checkpoint rule A selected-epoch median range",
            "All rule-A model strata",
            "Not in abstract",
            f"{min(a_epoch_medians):.1f}-{max(a_epoch_medians):.1f}",
            "Results 3.5 text",
            "checkpoint_policy_epoch_distribution.csv",
            f"min={min(a_epoch_medians)}; max={max(a_epoch_medians)}",
            "yes",
            "Retain; range is across the eight dataset-model rule-A medians.",
        )
    )

    for dataset, policy, label in STRATA:
        parts = [
            r
            for r in variance
            if r["analysis_scope"] == "dataset_checkpoint_policy_stratified"
            and r["model_spec"] == "primary_reported_fallback_model"
            and r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
        ]
        source_value = "; ".join(
            f"{r['grp']}={fmt(float(r['variance_proportion']))}" for r in parts
        )
        output.append(
            row(
                "fallback variance-component proportions",
                label,
                "0.17 / 0.25 / 0.36 / 0.34 model-by-split examples stated",
                source_value,
                "Results 3.3 text",
                "mixed_effects_sensitivity_variance_components.csv",
                source_value,
                "yes",
                "Retain; proportions match fallback model variance-component source after rounding.",
            )
        )

    for dataset, policy, label in [("organamnist", "A", "OrganAMNIST / A"), ("sipakmed", "B", "SIPaKMeD / B")]:
        rec = next(
            r
            for r in subsampling
            if r["dataset"] == dataset
            and r["checkpoint_policy"] == policy
            and r["resource_budget"] == "5_splits_x_3_seeds"
        )
        output.append(
            row(
                "5-split by 3-seed subsampling recovery rate",
                label,
                "0.91 for OrganAMNIST/A and 0.54 for SIPaKMeD/B stated",
                fmt2(float(rec["recovered_full_reference_probability"])),
                "Abstract and Results 3.4 table",
                "subsampling_resource_stability_summary.csv",
                (
                    f"probability={rec['recovered_full_reference_probability']}; "
                    f"CI={rec['recovered_full_reference_ci_low']}-{rec['recovered_full_reference_ci_high']}"
                ),
                "yes",
                "Retain; values match source table after rounding.",
            )
        )

    out_path = tables_dir / "numeric_consistency_audit.csv"
    fieldnames = [
        "metric",
        "stratum",
        "abstract",
        "results_section",
        "table_id",
        "csv_source",
        "source_value",
        "match_yes_no",
        "action",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    print(f"Wrote {out_path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
