#!/usr/bin/env python3
"""Run analysis I/O smoke validation through formal analysis-script entry points."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
STATS_SCRIPTS = EXPERIMENTS_DIR / "registered-workflow" / "stats" / "scripts"
DEFAULT_MANIFEST = EXPERIMENTS_DIR / "registered-workflow" / "run-manifests" / "smoke_test_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "smoke" / "analysis_io"
REPORT_PATH = SCRIPT_DIR / "outputs" / "smoke" / "analysis_io_smoke_report.md"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def display_command(command: list[str]) -> str:
    displayed = []
    for part in command:
        path = Path(part)
        if path.is_absolute() and path.exists():
            displayed.append(rel(path))
        else:
            displayed.append(part)
    return " ".join(displayed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            "python3",
            str(STATS_SCRIPTS / "01_collect_run_outputs.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        [
            "python3",
            str(STATS_SCRIPTS / "02_compute_ranking_metrics.py"),
            "--metrics-table",
            str(output_dir / "run_metrics_table.csv"),
            "--output-dir",
            str(output_dir),
        ],
        [
            "python3",
            str(STATS_SCRIPTS / "05_prepare_mixed_model_data.py"),
            "--metrics-table",
            str(output_dir / "run_metrics_table.csv"),
            "--output-dir",
            str(output_dir),
        ],
    ]

    results = []
    for command in commands:
        completed = run_command(command)
        results.append({"command": display_command(command), "stdout": completed.stdout.strip()})

    summary = {
        "status": "PASS",
        "manifest": rel(manifest),
        "output_dir": rel(output_dir),
        "commands": results,
        "required_outputs": [
            rel(output_dir / "run_metrics_table.csv"),
            rel(output_dir / "predictions_table.csv"),
            rel(output_dir / "model_ranking_table.csv"),
            rel(output_dir / "selection_frequency_table.csv"),
            rel(output_dir / "mixed_model_input.csv"),
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Analysis I/O Smoke Test Report\n\n"
        "Status: PASS\n\n"
        "Formal entry points exercised:\n\n"
        "- `01_collect_run_outputs.py`\n"
        "- `02_compute_ranking_metrics.py`\n"
        "- `05_prepare_mixed_model_data.py`\n\n"
        "Summary:\n\n"
        f"```json\n{json.dumps(summary, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ANALYSIS_IO_SMOKE_VALIDATION_OK")


if __name__ == "__main__":
    main()
