#!/usr/bin/env python3
import csv
import hashlib
from pathlib import Path

import yaml


DATASET_ORDER = ["sipakmed", "organamnist"]
SPLIT_COLUMNS = [
    "dataset",
    "split_id",
    "sample_id",
    "relative_path",
    "label",
    "subset",
    "split_seed",
]
HASH_COLUMNS = ["dataset", "split_id", "split_seed", "path", "rows", "sha256"]


def fail(message):
    raise RuntimeError(message)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def expected_split_metadata(frozen_dir):
    split_plan = load_yaml(frozen_dir / "split_plan.yaml")
    datasets = load_yaml(frozen_dir / "datasets.yaml")

    split_ids = split_plan["split_design"]["split_ids"]
    expected_splits = {
        item["id"]: int(item["generation_seed"])
        for item in split_ids
    }
    expected_rows = {
        item["dataset_id"]: int(item["expected_samples"])
        for item in datasets["primary_datasets"]
        if item["dataset_id"] in DATASET_ORDER
    }
    return expected_splits, expected_rows


def read_split_csv(split_path):
    with open(split_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SPLIT_COLUMNS:
            fail(f"{split_path} columns differ from frozen split schema: {reader.fieldnames}")
        rows = list(reader)
    return rows


def validate_split_file(split_path, dataset, split_id, expected_seed, expected_row_count):
    rows = read_split_csv(split_path)
    if len(rows) != expected_row_count:
        fail(f"{split_path} row count mismatch: {len(rows)} != {expected_row_count}")
    datasets = {row["dataset"] for row in rows}
    split_ids = {row["split_id"] for row in rows}
    split_seeds = {int(row["split_seed"]) for row in rows}
    if datasets != {dataset}:
        fail(f"{split_path} dataset column mismatch: {datasets}")
    if split_ids != {split_id}:
        fail(f"{split_path} split_id column mismatch: {split_ids}")
    if split_seeds != {expected_seed}:
        fail(f"{split_path} split_seed mismatch: {split_seeds} != {expected_seed}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        fail(f"{split_path} contains duplicate sample_id values")
    if len({row["relative_path"] for row in rows}) != len(rows):
        fail(f"{split_path} contains duplicate relative_path values")
    return len(rows)


def generate_split_hashes(splits_dir, output_path, frozen_dir):
    expected_splits, expected_rows = expected_split_metadata(frozen_dir)
    hash_rows = []

    for dataset in DATASET_ORDER:
        dataset_dir = splits_dir / dataset
        if not dataset_dir.is_dir():
            fail(f"Missing split directory: {dataset_dir}")
        for split_id, expected_seed in expected_splits.items():
            split_path = dataset_dir / f"{split_id}.csv"
            if not split_path.exists():
                fail(f"Missing split file: {split_path}")
            row_count = validate_split_file(
                split_path,
                dataset,
                split_id,
                expected_seed,
                expected_rows[dataset],
            )
            hash_rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "split_seed": expected_seed,
                    "path": Path("splits", dataset, f"{split_id}.csv").as_posix(),
                    "rows": row_count,
                    "sha256": compute_sha256(split_path),
                }
            )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HASH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(hash_rows)

    print(f"Generated split hashes manifest: {output_path}")
    print(f"Total split hashes: {len(hash_rows)}")


def main():
    script_dir = Path(__file__).resolve().parent
    workflow_splits_dir = script_dir.parent / "registered-workflow" / "splits"
    splits_dir = workflow_splits_dir if workflow_splits_dir.exists() else script_dir / "splits"
    output_path = splits_dir / "split_hashes.generated.csv"
    frozen_dir = script_dir.parent / "registered-workflow" / "configs" / "frozen"
    generate_split_hashes(splits_dir, output_path, frozen_dir)


if __name__ == "__main__":
    main()
