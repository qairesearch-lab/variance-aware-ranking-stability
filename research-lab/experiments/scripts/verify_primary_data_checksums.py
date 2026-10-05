#!/usr/bin/env python3
"""Verify local primary dataset files against frozen SHA-256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


DATASETS = ("sipakmed", "organamnist")
DEFAULT_MANIFESTS = {
    "sipakmed": Path("research-lab/experiments/registered-workflow/configs/frozen/checksums/sipakmed_primary_images_sha256.csv"),
    "organamnist": Path("research-lab/experiments/registered-workflow/configs/frozen/checksums/organamnist_primary_images_sha256.csv"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if list(rows[0].keys()) != ["relative_path", "sha256"]:
        raise ValueError(f"Unexpected manifest columns in {path}: {list(rows[0].keys())}")
    return rows


def verify_dataset(repo_root: Path, dataset: str, stop_after: int | None) -> dict[str, int]:
    manifest_path = repo_root / DEFAULT_MANIFESTS[dataset]
    rows = read_manifest(manifest_path)
    checked = 0
    missing = 0
    mismatched = 0
    examples: list[str] = []

    for row in rows:
        relative_path = Path(row["relative_path"])
        file_path = repo_root / relative_path
        checked += 1
        if not file_path.exists():
            missing += 1
            if len(examples) < 10:
                examples.append(f"MISSING {relative_path}")
        else:
            observed = sha256_file(file_path)
            expected = row["sha256"]
            if observed != expected:
                mismatched += 1
                if len(examples) < 10:
                    examples.append(f"MISMATCH {relative_path} expected={expected} observed={observed}")

        if stop_after is not None and checked >= stop_after:
            break

    print(f"{dataset}: checked={checked} missing={missing} mismatched={mismatched}")
    for example in examples:
        print(f"  {example}")
    return {"checked": checked, "missing": missing, "mismatched": mismatched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Dataset to verify.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Debug option: verify only the first N manifest rows per dataset.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    totals = {"checked": 0, "missing": 0, "mismatched": 0}
    for dataset in datasets:
        result = verify_dataset(repo_root, dataset, args.stop_after)
        for key in totals:
            totals[key] += result[key]

    print(f"TOTAL: checked={totals['checked']} missing={totals['missing']} mismatched={totals['mismatched']}")
    if totals["missing"] or totals["mismatched"]:
        raise SystemExit(1)
    print("PRIMARY_DATA_CHECKSUMS_OK")


if __name__ == "__main__":
    main()
