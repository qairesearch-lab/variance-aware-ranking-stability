#!/usr/bin/env python3
"""Seal the author-confirmed 300-run manifest after final data/split audit.

The main E1 analysis remains 270 runs; the other 30 are Swin sensitivity.
No model outcome is read. Every split is checked for group leakage and against
the full index before dispatch commands are written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from train_extension_run import HERE, ROOT, SPLIT_COLUMNS, sha256, read_split, write_csv
from protocol_amendment import AMENDMENT, SCOPE_DECISION, expected_images, load_scope_decision, validate_final_audit

DESIGN = HERE / "configs" / "extension_protocol_design_locked_v0.3.yaml"
COLUMNS = (
    "run_id", "dataset", "model", "variant", "split_id", "training_seed",
    "split_file", "split_sha256", "dataset_index_file", "dataset_index_sha256", "output_dir",
)
DATASETS = ("isic2019", "mura")
CNN = ("resnet18", "resnet50", "densenet121", "efficientnet_b0")
SEEDS = (42, 52, 62)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def check_index(path: Path, dataset: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(SPLIT_COLUMNS) - {"subset"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid dataset index columns: {path}")
        rows = list(reader)
    expected = expected_images(dataset)
    if len(rows) != expected or any(r["dataset"] != dataset for r in rows):
        raise ValueError(f"Unexpected final index size/dataset: {path}")
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != expected or "" in by_id:
        raise ValueError(f"Repeated sample IDs in final index: {path}")
    return by_id


def check_split_matches_index(partitions: dict[str, list[dict[str, str]]], index_by_id: dict[str, dict[str, str]], dataset: str, split_id: str) -> None:
    """Require the split to preserve every authoritative index field, not just IDs."""
    split_by_id = {row["sample_id"]: row for rows in partitions.values() for row in rows}
    if split_by_id.keys() != index_by_id.keys():
        raise ValueError(f"Split does not cover exact final index: {dataset}/{split_id}")
    for sample_id, split_row in split_by_id.items():
        index_row = index_by_id[sample_id]
        for field in SPLIT_COLUMNS:
            if field != "subset" and split_row[field] != index_row[field]:
                raise ValueError(f"Split/index mismatch: {dataset}/{split_id}/{sample_id}/{field}")


def check_audit(path: Path, index_paths: dict[str, Path]) -> None:
    audit = json.loads(path.read_text(encoding="utf-8"))
    validate_final_audit(audit, sha256)
    hashes = audit.get("dataset_index_sha256", {})
    if hashes != {dataset: sha256(index_paths[dataset]) for dataset in DATASETS}:
        raise ValueError("Audit does not seal the current dataset indices")
    if audit.get("near_duplicate_role") != "descriptive_risk_screen_only_no_manual_same_source_adjudication":
        raise ValueError("pHash candidates must be descriptive only")


def make_rows(split_root: Path, index_root: Path, audit_file: Path):
    design = yaml.safe_load(DESIGN.read_text(encoding="utf-8"))
    if design["author_decisions"]["target_matrix"]["selected"] != "E1":
        raise ValueError("Design lock no longer selects E1")
    load_scope_decision()
    index_paths = {dataset: index_root / f"{dataset}_dataset_index.csv" for dataset in DATASETS}
    index_by_id = {dataset: check_index(path, dataset) for dataset, path in index_paths.items()}
    check_audit(audit_file, index_paths)
    rows = []
    split_hashes = {}
    for dataset in DATASETS:
        for split_number in range(1, 11):
            split_id = f"split_{split_number:02d}"
            split_path = split_root / dataset / f"{split_id}.csv"
            partitions = read_split(split_path, dataset, technical_smoke=False)
            check_split_matches_index(partitions, index_by_id[dataset], dataset, split_id)
            split_hash = sha256(split_path)
            split_hashes[(dataset, split_id)] = split_hash
            models = CNN + (("swin_t",) if split_number <= 5 else ())
            for model in models:
                for seed in SEEDS:
                    run_id = f"jiim_ext_{dataset}_{split_id}_seed_{seed}_{model}_A"
                    rows.append({
                        "run_id": run_id, "dataset": dataset, "model": model,
                        "variant": "shared", "split_id": split_id,
                        "training_seed": str(seed), "split_file": repo_relative(split_path),
                        "split_sha256": split_hash,
                        "dataset_index_file": repo_relative(index_paths[dataset]),
                        "dataset_index_sha256": sha256(index_paths[dataset]),
                        "output_dir": repo_relative(HERE / "runs" / run_id),
                    })
            if split_number <= 5:
                for seed in SEEDS:
                    run_id = f"jiim_ext_{dataset}_{split_id}_seed_{seed}_swin_t_B"
                    rows.append({
                        "run_id": run_id, "dataset": dataset, "model": "swin_t",
                        "variant": "swin_weight_eval", "split_id": split_id,
                        "training_seed": str(seed), "split_file": repo_relative(split_path),
                        "split_sha256": split_hash,
                        "dataset_index_file": repo_relative(index_paths[dataset]),
                        "dataset_index_sha256": sha256(index_paths[dataset]),
                        "output_dir": repo_relative(HERE / "runs" / run_id),
                    })
    expected = 300
    if len(rows) != expected or len({r["run_id"] for r in rows}) != expected:
        raise AssertionError(f"Expected {expected} unique runs, found {len(rows)}")
    return rows, split_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Will not overwrite sealed manifest: {args.output}")
    rows, split_hashes = make_rows(args.split_root, args.index_root, args.audit_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, list(COLUMNS), rows)
    seal = {
        "status": "sealed_after_data_quality_audit",
        "design_lock_sha256": sha256(DESIGN),
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "swin_sensitivity_decision_sha256": sha256(SCOPE_DECISION),
        "audit_sha256": sha256(args.audit_file),
        "manifest_sha256": sha256(args.output),
        "run_count": len(rows),
        "swin_weight_eval_run_count": sum(r["variant"] == "swin_weight_eval" for r in rows),
        "split_hashes": {f"{dataset}/{split_id}": value for (dataset, split_id), value in split_hashes.items()},
    }
    seal_path = args.output.with_suffix(".seal.json")
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Sealed {len(rows)} runs: {args.output}")


if __name__ == "__main__":
    main()
