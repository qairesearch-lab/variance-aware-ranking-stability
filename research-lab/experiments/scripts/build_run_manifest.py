#!/usr/bin/env python3
"""Generate the primary-analysis run manifest draft and validation report."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
CONFIG_DIR = EXPERIMENTS_DIR / "registered-workflow" / "configs" / "frozen"
SPLIT_HASHES_PATH = EXPERIMENTS_DIR / "registered-workflow" / "splits" / "split_hashes.csv"
OUTPUT_PATH = SCRIPT_DIR / "run-manifests" / "primary_run_manifest_draft.csv"
AUDIT_PATH = SCRIPT_DIR / "run-manifests" / "primary_run_manifest_audit.json"

REQUIRED_COLUMNS = [
    "run_id",
    "dataset",
    "split_id",
    "split_seed",
    "training_seed",
    "model",
    "checkpoint_policy",
    "config_hash",
    "split_hash",
    "output_dir",
    "status",
    "analysis_role",
]

CONFIG_HASH_FILES = [
    "analysis_pipeline.yaml",
    "checkpoint_policies.yaml",
    "datasets.yaml",
    "models.yaml",
    "random_seeds.yaml",
    "split_plan.yaml",
    "training_common.yaml",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def calculate_config_hash(config_dir: Path) -> str:
    """Hash the frozen execution config bundle used by training and analysis."""
    hasher = hashlib.sha256()
    for relative_name in CONFIG_HASH_FILES:
        path = config_dir / relative_name
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen config file: {path}")
        data = path.read_bytes()
        hasher.update(relative_name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_split_hashes(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_rows(path)
    expected_columns = ["dataset", "split_id", "split_seed", "path", "rows", "sha256"]
    if rows and list(rows[0].keys()) != expected_columns:
        raise ValueError(f"Unexpected split hash schema in {path}: {list(rows[0].keys())}")

    split_hashes: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["split_id"])
        if key in split_hashes:
            raise ValueError(f"Duplicate split hash key: {key}")
        split_hashes[key] = row
    return split_hashes


def expected_matrix() -> dict[str, object]:
    datasets_config = load_yaml(CONFIG_DIR / "datasets.yaml")
    models_config = load_yaml(CONFIG_DIR / "models.yaml")
    seeds_config = load_yaml(CONFIG_DIR / "random_seeds.yaml")
    policies_config = load_yaml(CONFIG_DIR / "checkpoint_policies.yaml")
    split_plan = load_yaml(CONFIG_DIR / "split_plan.yaml")
    run_manifest_config = load_yaml(CONFIG_DIR / "run_manifest.yaml")

    datasets = [item["dataset_id"] for item in datasets_config["primary_datasets"] if item["primary_matrix"]]
    models = [item["name"] for item in models_config["model_pool"]]
    split_ids = [item["id"] for item in split_plan["split_design"]["split_ids"]]
    split_seed_by_id = {
        item["id"]: str(item["generation_seed"]) for item in split_plan["split_design"]["split_ids"]
    }
    training_seeds = [str(seed) for seed in seeds_config["training_seeds"]["values"]]
    checkpoint_policies = list(policies_config["primary_checkpoint_policies"].keys())

    frozen_matrix = run_manifest_config["experiment_matrix"]
    expected_total = int(run_manifest_config["total_runs"])

    if datasets != frozen_matrix["datasets"]:
        raise ValueError(f"Dataset matrix mismatch: {datasets} != {frozen_matrix['datasets']}")
    if models != frozen_matrix["models"]:
        raise ValueError(f"Model matrix mismatch: {models} != {frozen_matrix['models']}")
    if split_ids != frozen_matrix["splits"]:
        raise ValueError(f"Split matrix mismatch: {split_ids} != {frozen_matrix['splits']}")
    if [int(seed) for seed in training_seeds] != frozen_matrix["training_seeds"]:
        raise ValueError("Training seed matrix mismatch")
    if checkpoint_policies != frozen_matrix["checkpoint_policies"]:
        raise ValueError("Checkpoint policy matrix mismatch")

    return {
        "datasets": datasets,
        "models": models,
        "split_ids": split_ids,
        "split_seed_by_id": split_seed_by_id,
        "training_seeds": training_seeds,
        "checkpoint_policies": checkpoint_policies,
        "expected_total": expected_total,
    }


def build_rows(matrix: dict[str, object], split_hashes: dict[tuple[str, str], dict[str, str]], config_hash: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_seed_by_id = matrix["split_seed_by_id"]

    combinations = product(
        matrix["datasets"],
        matrix["split_ids"],
        matrix["training_seeds"],
        matrix["models"],
        matrix["checkpoint_policies"],
    )
    for index, (dataset, split_id, training_seed, model, checkpoint_policy) in enumerate(combinations, start=1):
        split_key = (dataset, split_id)
        if split_key not in split_hashes:
            raise ValueError(f"Missing split hash for {dataset} {split_id}")

        split_hash_row = split_hashes[split_key]
        split_seed = split_seed_by_id[split_id]
        if split_hash_row["split_seed"] != split_seed:
            raise ValueError(f"Split seed mismatch for {dataset} {split_id}")

        run_id = f"run_{index:04d}"
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "split_id": split_id,
                "split_seed": split_seed,
                "training_seed": training_seed,
                "model": model,
                "checkpoint_policy": checkpoint_policy,
                "config_hash": config_hash,
                "split_hash": split_hash_row["sha256"],
                "output_dir": f"research-lab/experiments/registered-workflow/runs/primary/{run_id}",
                "status": "pending",
                "analysis_role": "primary_analysis",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(rows: list[dict[str, str]], matrix: dict[str, object], split_hashes: dict[tuple[str, str], dict[str, str]], config_hash: str) -> dict[str, object]:
    errors: list[str] = []
    expected_total = matrix["expected_total"]

    if len(rows) != expected_total:
        errors.append(f"Expected {expected_total} rows, found {len(rows)}")

    seen_run_ids = set()
    seen_conditions = set()
    expected_conditions = set(
        product(
            matrix["datasets"],
            matrix["split_ids"],
            matrix["training_seeds"],
            matrix["models"],
            matrix["checkpoint_policies"],
        )
    )

    for index, row in enumerate(rows, start=1):
        expected_run_id = f"run_{index:04d}"
        if row["run_id"] != expected_run_id:
            errors.append(f"Run id order mismatch at row {index}: {row['run_id']} != {expected_run_id}")
        if row["run_id"] in seen_run_ids:
            errors.append(f"Duplicate run_id: {row['run_id']}")
        seen_run_ids.add(row["run_id"])

        condition = (
            row["dataset"],
            row["split_id"],
            row["training_seed"],
            row["model"],
            row["checkpoint_policy"],
        )
        if condition in seen_conditions:
            errors.append(f"Duplicate matrix condition: {condition}")
        seen_conditions.add(condition)

        split_key = (row["dataset"], row["split_id"])
        split_hash_row = split_hashes.get(split_key)
        if split_hash_row is None:
            errors.append(f"Missing split hash row for {split_key}")
        else:
            if row["split_seed"] != split_hash_row["split_seed"]:
                errors.append(f"split_seed mismatch for {row['run_id']}")
            if row["split_hash"] != split_hash_row["sha256"]:
                errors.append(f"split_hash mismatch for {row['run_id']}")

        if row["config_hash"] != config_hash:
            errors.append(f"config_hash mismatch for {row['run_id']}")
        if row["status"] != "pending":
            errors.append(f"status mismatch for {row['run_id']}")
        if row["analysis_role"] != "primary_analysis":
            errors.append(f"analysis_role mismatch for {row['run_id']}")
        if Path(row["output_dir"]).is_absolute():
            errors.append(f"Absolute output_dir for {row['run_id']}: {row['output_dir']}")
        path_parts = Path(row["output_dir"]).parts
        if any(part.startswith(".") for part in path_parts):
            errors.append(f"Non-public output_dir component for {row['run_id']}: {row['output_dir']}")

    missing_conditions = expected_conditions - seen_conditions
    extra_conditions = seen_conditions - expected_conditions
    if missing_conditions:
        errors.append(f"Missing matrix conditions: {len(missing_conditions)}")
    if extra_conditions:
        errors.append(f"Unexpected matrix conditions: {len(extra_conditions)}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest_path": str(OUTPUT_PATH.relative_to(EXPERIMENTS_DIR.parent.parent)),
        "row_count": len(rows),
        "expected_row_count": expected_total,
        "required_columns": REQUIRED_COLUMNS,
        "config_hash": config_hash,
        "config_hash_files": CONFIG_HASH_FILES,
        "matrix": {
            "datasets": matrix["datasets"],
            "splits": matrix["split_ids"],
            "training_seeds": matrix["training_seeds"],
            "models": matrix["models"],
            "checkpoint_policies": matrix["checkpoint_policies"],
        },
        "path_policy": "relative repository paths only",
        "output_dir_prefix": "research-lab/experiments/registered-workflow/runs/primary/",
    }


def write_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    matrix = expected_matrix()
    split_hashes = load_split_hashes(SPLIT_HASHES_PATH)
    config_hash = calculate_config_hash(CONFIG_DIR)
    rows = build_rows(matrix, split_hashes, config_hash)
    audit = audit_rows(rows, matrix, split_hashes, config_hash)
    if audit["status"] != "PASS":
        write_audit(AUDIT_PATH, audit)
        raise SystemExit(f"Primary run manifest audit failed: {AUDIT_PATH}")

    write_manifest(OUTPUT_PATH, rows)
    persisted_rows = read_csv_rows(OUTPUT_PATH)
    persisted_audit = audit_rows(persisted_rows, matrix, split_hashes, config_hash)
    write_audit(AUDIT_PATH, persisted_audit)
    if persisted_audit["status"] != "PASS":
        raise SystemExit(f"Persisted primary run manifest audit failed: {AUDIT_PATH}")

    print(f"Generated: {OUTPUT_PATH}")
    print(f"Audit: {AUDIT_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Config hash: {config_hash}")
    print("PRIMARY_RUN_MANIFEST_DRAFT_VALIDATION_OK")


if __name__ == "__main__":
    main()
