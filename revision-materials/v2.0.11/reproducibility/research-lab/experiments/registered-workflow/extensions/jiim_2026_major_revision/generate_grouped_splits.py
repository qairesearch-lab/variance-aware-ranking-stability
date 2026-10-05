#!/usr/bin/env python3
"""Deterministic 70/15/15 grouped split search from the locked v0.3 design.

Formal output requires an objective exact-image data-quality audit. Diagnostic mode may use
candidate indices but writes an explicitly non-formal split and cannot create
the ten-split authority used by the run-manifest generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from protocol_amendment import AMENDMENT, expected_images, validate_final_audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATASETS = ("isic2019", "mura")
SPLIT_COLUMNS = ("dataset", "subset", "sample_id", "relative_path", "label", "group_id", "study_id", "body_region")
TARGET = np.array([0.70, 0.15, 0.15], dtype=float)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path, dataset: str, *, diagnostic_candidate: bool = False):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or any(row["dataset"] != dataset for row in rows):
        raise ValueError(f"Wrong or empty index: {path}")
    expected = expected_images(dataset, diagnostic_candidate=diagnostic_candidate)
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError(f"Index size/ID mismatch: {path}")
    if not all(row["group_id"] for row in rows):
        raise ValueError("Missing group IDs")
    return rows


def primary_units(rows, dataset: str):
    if dataset == "isic2019":
        return rows
    by_study = {}
    for row in rows:
        study_id = row["study_id"]
        if not study_id:
            raise ValueError("MURA study ID is missing")
        if study_id in by_study:
            first = by_study[study_id]
            if (first["label"], first["group_id"], first["body_region"]) != (row["label"], row["group_id"], row["body_region"]):
                raise ValueError(f"Conflicting MURA study metadata: {study_id}")
        else:
            by_study[study_id] = row
    if len(by_study) != 14656:
        raise ValueError("Unexpected number of MURA primary study units")
    return list(by_study.values())


def balance_score(unit_labels, unit_regions, partitions, dataset: str):
    n = len(unit_labels)
    fractions = np.array([len(p) / n for p in partitions])
    if np.max(np.abs(fractions - TARGET)) > 0.03:
        return None
    classes = sorted(set(unit_labels))
    if any(set(unit_labels[p]) != set(classes) for p in partitions):
        return None
    all_regions = sorted(set(unit_regions)) if dataset == "mura" else []
    if all_regions and any(set(unit_regions[p]) != set(all_regions) for p in partitions):
        return None
    overall_term = float(np.mean(np.abs(fractions - TARGET)))
    class_terms = []
    for label in classes:
        total = int(np.sum(unit_labels == label))
        class_terms.extend(abs(int(np.sum(unit_labels[p] == label)) / total - target) for p, target in zip(partitions, TARGET))
    class_term = float(np.mean(class_terms))
    region_term = 0.0
    if all_regions:
        region_terms = []
        for region in all_regions:
            total = int(np.sum(unit_regions == region))
            region_terms.extend(abs(int(np.sum(unit_regions[p] == region)) / total - target) for p, target in zip(partitions, TARGET))
        region_term = float(np.mean(region_terms))
    return 0.5 * overall_term + 0.4 * class_term + (0.1 * region_term if dataset == "mura" else 0.0)


def choose_split(units, dataset: str, split_seed: int):
    groups = np.array([row["group_id"] for row in units])
    labels = np.array([int(row["label"]) for row in units])
    regions = np.array([row["body_region"] for row in units])
    dummy = np.zeros(len(units), dtype=np.uint8)
    best = None
    valid_candidates = 0
    for candidate_index in range(512):
        candidate_seed = split_seed * 1000 + candidate_index
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=candidate_seed)
        train_val, test = next(first.split(dummy, labels, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.17647058823529413, random_state=candidate_seed + 500000)
        train_local, val_local = next(second.split(dummy[train_val], labels[train_val], groups[train_val]))
        train, validation = train_val[train_local], train_val[val_local]
        partitions = (train, validation, test)
        score = balance_score(labels, regions, partitions, dataset)
        if score is None:
            continue
        valid_candidates += 1
        if best is None or score < best[0] - 1e-15 or (abs(score - best[0]) <= 1e-15 and candidate_index < best[1]):
            best = (score, candidate_index, partitions)
    if best is None:
        raise RuntimeError(f"No valid candidate for {dataset} seed={split_seed}; amend protocol before trying again")
    score, candidate_index, partitions = best
    group_subset = {}
    for subset, indexes in zip(("train", "validation", "test"), partitions):
        for group_id in groups[indexes]:
            old = group_subset.setdefault(group_id, subset)
            if old != subset:
                raise AssertionError("Group leakage in selected split")
    return group_subset, {
        "score": score, "candidate_index": candidate_index,
        "valid_candidate_count": valid_candidates,
        "primary_unit_counts": {name: len(indexes) for name, indexes in zip(("train", "validation", "test"), partitions)},
        "group_counts": dict(Counter(group_subset.values())),
    }


def write_split(path: Path, rows, group_subset):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset"], "subset": group_subset[row["group_id"]],
                "sample_id": row["sample_id"], "relative_path": row["relative_path"],
                "label": row["label"], "group_id": row["group_id"],
                "study_id": row["study_id"], "body_region": row["body_region"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--diagnostic-candidate", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Will not overwrite split output: {args.output_root}")
    if args.diagnostic_candidate:
        if args.audit_file is not None:
            raise ValueError("Diagnostic candidate mode cannot use a final audit")
        split_numbers = (1,)
        suffix = "_dataset_index_candidate.csv"
    else:
        if args.audit_file is None:
            raise ValueError("Formal split generation requires an objective data-quality audit")
        audit = json.loads(args.audit_file.read_text(encoding="utf-8"))
        validate_final_audit(audit, sha256)
        split_numbers = tuple(range(1, 11))
        suffix = "_dataset_index.csv"
    index_paths = {dataset: args.index_root / f"{dataset}{suffix}" for dataset in DATASETS}
    if not args.diagnostic_candidate:
        expected = {dataset: sha256(index_paths[dataset]) for dataset in DATASETS}
        if audit.get("dataset_index_sha256") != expected:
            raise ValueError("Audit/index hash mismatch")
    prepared = {}
    near_risk_pairs = {}
    for dataset in DATASETS:
        rows = load_rows(index_paths[dataset], dataset, diagnostic_candidate=args.diagnostic_candidate)
        prepared[dataset] = (rows, primary_units(rows, dataset))
        if not args.diagnostic_candidate:
            near_path = args.index_root / f"{dataset}_near_risk_pairs.csv"
            if audit.get("near_risk_file_sha256", {}).get(dataset) != sha256(near_path):
                raise ValueError(f"Near-risk candidate file hash mismatch: {dataset}")
            with near_path.open("r", newline="", encoding="utf-8") as handle:
                near_risk_pairs[dataset] = list(csv.DictReader(handle))
            if len(near_risk_pairs[dataset]) != audit["near_pairs_by_dataset"][dataset]:
                raise ValueError(f"Near-risk candidate count mismatch: {dataset}")
    args.output_root.mkdir(parents=True)
    records = []
    for dataset, (rows, units) in prepared.items():
        group_by_sample = {row["sample_id"]: row["group_id"] for row in rows}
        dataset_dir = args.output_root / dataset
        dataset_dir.mkdir()
        for split_number in split_numbers:
            split_id = f"split_{split_number:02d}"
            seed = 1000 + split_number
            group_subset, audit_record = choose_split(units, dataset, seed)
            if not args.diagnostic_candidate:
                crossing = Counter()
                for pair in near_risk_pairs[dataset]:
                    left = group_subset[group_by_sample[pair["sample_a"]]]
                    right = group_subset[group_by_sample[pair["sample_b"]]]
                    if left != right:
                        crossing["/".join(sorted((left, right)))] += 1
                audit_record["near_risk_cross_partition_pair_count"] = sum(crossing.values())
                audit_record["near_risk_cross_partition_pair_types"] = dict(sorted(crossing.items()))
                audit_record["near_risk_candidate_pair_count"] = len(near_risk_pairs[dataset])
            split_path = dataset_dir / f"{split_id}.csv"
            write_split(split_path, rows, group_subset)
            records.append({
                "dataset": dataset, "split_id": split_id, "split_seed": seed,
                "split_sha256": sha256(split_path), "index_sha256": sha256(index_paths[dataset]),
                **audit_record,
            })
            print(f"{dataset}/{split_id}: candidate={audit_record['candidate_index']} score={audit_record['score']:.8f}", flush=True)
    payload = {
        "status": "diagnostic_not_for_formal_training" if args.diagnostic_candidate else "generated_after_objective_data_quality_audit",
        "protocol_candidate_count": 512,
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "records": records,
    }
    (args.output_root / "split_generation_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
