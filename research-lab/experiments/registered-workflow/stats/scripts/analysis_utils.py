"""Shared helpers for registered-workflow statistical analysis scripts."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[5]
ALLOWED_NULL_METRICS = {"calibration_intercept", "calibration_slope"}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(rows: list[dict[str, str]], required: Iterable[str], source: Path) -> None:
    if not rows:
        raise SystemExit(f"No data rows found: {display_path(source)}")
    columns = set(rows[0].keys())
    missing = [column for column in required if column not in columns]
    if missing:
        raise SystemExit(f"Missing required columns in {display_path(source)}: {', '.join(missing)}")


def require_analysis_role(rows: list[dict[str, str]], expected_role: str, source: Path) -> None:
    require_columns(rows, ["analysis_role"], source)
    roles = sorted({row.get("analysis_role", "") for row in rows})
    if roles != [expected_role]:
        raise SystemExit(
            f"Unexpected analysis_role values in {display_path(source)}: {roles}; expected only {expected_role}"
        )


def require_completed(rows: list[dict[str, str]], source: Path) -> None:
    require_columns(rows, ["run_status"], source)
    statuses = sorted({row.get("run_status", "") for row in rows})
    if statuses != ["completed"]:
        raise SystemExit(f"Unexpected run_status values in {display_path(source)}: {statuses}; expected completed")


def parse_metric(value: object, metric_name: str, context: str) -> float:
    if value in {None, ""}:
        if metric_name in ALLOWED_NULL_METRICS:
            raise ValueError(f"Allowed-null metric {metric_name} cannot be used as primary analysis metric: {context}")
        raise ValueError(f"Empty metric value for {metric_name}: {context}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric metric value for {metric_name}: {context} -> {value!r}") from exc


def file_metadata(path: Path) -> dict[str, object]:
    return {"path": display_path(path), "size_bytes": path.stat().st_size if path.exists() else None}


def runtime_metadata(script_name: str) -> dict[str, object]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "script": script_name,
    }
