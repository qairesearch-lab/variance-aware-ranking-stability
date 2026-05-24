#!/usr/bin/env python3
"""Run schema-level smoke validation for the smoke-test manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR.parent / "registered-workflow" / "run-manifests" / "smoke_test_manifest.csv"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--sample-limit", type=int, default=24)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    rows = read_manifest(manifest)
    for row in rows:
        subprocess.run(
            [
                "python3",
                str(SCRIPT_DIR / "train_one_run.py"),
                "--manifest",
                str(manifest),
                "--run-id",
                row["run_id"],
                "--execution-mode",
                "schema_smoke",
                "--sample-limit",
                str(args.sample_limit),
            ],
            check=True,
        )
    subprocess.run(["python3", str(SCRIPT_DIR / "validate_outputs.py"), "--manifest", str(manifest)], check=True)
    print(f"SMOKE_MANIFEST_SCHEMA_RUN_OK rows={len(rows)}")


if __name__ == "__main__":
    main()
