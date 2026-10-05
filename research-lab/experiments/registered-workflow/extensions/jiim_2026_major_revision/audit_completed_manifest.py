#!/usr/bin/env python3
"""Read-only post-run audit of a sealed E1 manifest and its output artifacts.

No training is launched, retried, or repaired. A formal pass requires every
manifest row to have complete and internally consistent outputs. This tool is
intended to run on the same machine with the sealed source/weight files present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import yaml

from dispatch_manifest_run import current_weight_hash
from preprocessing import VARIANTS
from train_extension_run import DESIGN, ENVIRONMENT_TARGET, HERE, ROOT, read_split, sha256, validate_manifest_row
from protocol_amendment import AMENDMENT, SCOPE_DECISION

RUNNER = HERE / "train_extension_run.py"
PREPROCESSING = HERE / "preprocessing.py"
REQUIRED = (
    "run_config.json", "run_status.json", "history.csv", "metrics.json",
    "predictions.csv", "checkpoint_manifest.json", "timing.json",
    "checkpoints/best_validation_loss.pt",
)


def ordered_rows(rows: list[dict[str, str]], stage: str) -> list[dict[str, str]]:
    """E0 first, E1 increment next, separately tagged Swin sensitivity last."""
    def category(row):
        if row["variant"] == "swin_weight_eval":
            return 2
        return 0 if int(row["split_id"].split("_")[1]) <= 5 else 1

    maximum = {"e0": 0, "e1": 1, "sensitivity": 2, "all": 2}[stage]
    selected = [row for row in rows if (category(row) == 2 if stage == "sensitivity" else category(row) <= maximum)]
    if not selected:
        raise ValueError(f"No manifest rows for stage {stage}")
    return sorted(selected, key=lambda row: (category(row), row["dataset"], row["split_id"], row["model"], int(row["training_seed"])))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, got {actual!r}")


def expected_test_units(split_path: Path, dataset: str) -> dict[str, int]:
    partitions = read_split(split_path, dataset, technical_smoke=False)
    rows = partitions["test"]
    if dataset == "isic2019":
        return {row["sample_id"]: int(row["label"]) for row in rows}
    units = {}
    for row in rows:
        study_id, label = row["study_id"], int(row["label"])
        if study_id in units and units[study_id] != label:
            raise ValueError(f"Conflicting test study labels: {study_id}")
        units[study_id] = label
    return units


def verify_predictions(path: Path, row: dict[str, str], expected: dict[str, int], metrics: dict) -> None:
    predictions = read_csv(path)
    classes = 8 if row["dataset"] == "isic2019" else 2
    unit = "image" if row["dataset"] == "isic2019" else "study"
    ids = [item["unit_id"] for item in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Predictions do not cover the exact test units once")
    true, pred = [], []
    for item in predictions:
        require_equal(item["run_id"], row["run_id"], "Prediction run ID")
        require_equal(item["variant"], row["variant"], "Prediction variant")
        require_equal(item["dataset"], row["dataset"], "Prediction dataset")
        require_equal(item["prediction_unit"], unit, "Prediction unit")
        label_true, label_pred = int(item["label_true"]), int(item["label_pred"])
        require_equal(label_true, expected[item["unit_id"]], "Test-unit label")
        if label_pred not in range(classes):
            raise ValueError("Predicted class outside the declared label set")
        probs = [float(item[f"prob_{class_id}"]) for class_id in range(classes)]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probs):
            raise ValueError("Invalid prediction probability")
        if abs(sum(probs) - 1.0) > 1e-5 or max(range(classes), key=probs.__getitem__) != label_pred:
            raise ValueError("Prediction probability/label mismatch")
        true.append(label_true)
        pred.append(label_pred)
    require_equal(metrics["prediction_unit"], unit, "Metric prediction unit")
    require_equal(int(metrics["test_units"]), len(expected), "Metric test-unit count")
    if not math.isfinite(float(metrics["balanced_accuracy"])):
        raise ValueError("Non-finite balanced accuracy")
    if abs(float(metrics["balanced_accuracy"]) - balanced_accuracy_score(true, pred)) > 1e-10:
        raise ValueError("Reported balanced accuracy does not match predictions")
    require_equal(metrics["confusion_matrix"], confusion_matrix(true, pred, labels=list(range(classes))).tolist(), "Confusion matrix")


def verify_run(row: dict[str, str], manifest: Path, expected: dict[str, int], weight_hash: str) -> None:
    output = ROOT / row["output_dir"]
    if output.parent.resolve() != (HERE / "runs").resolve() or output.name != row["run_id"]:
        raise ValueError("Output is not the canonical formal run directory")
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run files: {missing}")
    status, config = read_json(output / "run_status.json"), read_json(output / "run_config.json")
    require_equal(status.get("status"), "completed", "Run status")
    require_equal(status.get("run_id"), row["run_id"], "Status run ID")
    require_equal(config.get("technical_smoke"), False, "Formal/technical marker")
    for key in ("run_id", "dataset", "model", "variant"):
        require_equal(config.get(key), row[key], f"Run identity {key}")
    require_equal(config.get("training_seed"), int(row["training_seed"]), "Training seed")
    expected_hashes = {
        "run_manifest_sha256": sha256(manifest),
        "design_lock_sha256": sha256(DESIGN),
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "swin_sensitivity_decision_sha256": sha256(SCOPE_DECISION),
        "variant_sha256": sha256(VARIANTS[row["variant"]]),
        "script_sha256": sha256(RUNNER),
        "preprocessing_sha256": sha256(PREPROCESSING),
        "environment_target_sha256": sha256(ENVIRONMENT_TARGET),
        "split_sha256": row["split_sha256"],
        "dataset_index_sha256": row["dataset_index_sha256"],
        "pretrained_weight_sha256": weight_hash,
    }
    for key, value in expected_hashes.items():
        require_equal(config.get(key), value, f"Run hash {key}")
    protocol_hash = hashlib.sha256(json.dumps({
        "design": expected_hashes["design_lock_sha256"],
        "data_quality_amendment": expected_hashes["data_quality_amendment_sha256"],
        "swin_sensitivity_decision": expected_hashes["swin_sensitivity_decision_sha256"],
        "variant": expected_hashes["variant_sha256"],
        "runner": expected_hashes["script_sha256"],
        "preprocessing": expected_hashes["preprocessing_sha256"],
        "environment_target": expected_hashes["environment_target_sha256"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    for key, value in {
        "protocol_hash": protocol_hash,
        "split_hash": expected_hashes["split_sha256"],
        "dataset_index_hash": expected_hashes["dataset_index_sha256"],
        "model_weight_hash": weight_hash,
        "script_hash": expected_hashes["script_sha256"],
        "environment_lock_hash": expected_hashes["environment_target_sha256"],
    }.items():
        require_equal(config.get(key), value, f"Run identity alias {key}")
    environment = config.get("environment", {})
    if environment.get("device") != "cuda" or not environment.get("gpu"):
        raise ValueError("Formal output was not recorded on CUDA GPU")
    target = yaml.safe_load(ENVIRONMENT_TARGET.read_text(encoding="utf-8"))
    for key in ("torch", "torchvision", "numpy", "pillow", "pandas", "scikit_learn", "pyyaml"):
        require_equal(environment.get(key), target[key], f"Runtime package {key}")
    if not str(environment.get("python", "")).startswith(target["python"] + "."):
        raise ValueError("Runtime Python version differs from the target")
    expected_hparams = {"batch_size": 32, "max_epochs": 60, "num_workers": 4, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "mixed_precision": False}
    for key, value in expected_hparams.items():
        require_equal(config.get("hyperparameters", {}).get(key), value, f"Hyperparameter {key}")

    history = read_csv(output / "history.csv")
    if not history or len(history) > 60 or [int(item["epoch"]) for item in history] != list(range(1, len(history) + 1)):
        raise ValueError("Invalid epoch history")
    for item in history:
        if not math.isfinite(float(item["train_loss_batch_mean"])) or not math.isfinite(float(item["validation_loss"])):
            raise ValueError("Non-finite loss in history")
    metrics, checkpoint, timing = (read_json(output / name) for name in ("metrics.json", "checkpoint_manifest.json", "timing.json"))
    for key in ("run_id", "variant", "dataset"):
        require_equal(metrics.get(key), row[key], f"Metric identity {key}")
    require_equal(metrics.get("technical_smoke_not_research_result"), False, "Metric formal marker")
    selected_epoch = int(metrics["selected_epoch"])
    if selected_epoch not in range(1, len(history) + 1) or history[selected_epoch - 1]["significant_improvement"] != "True":
        raise ValueError("Selected checkpoint is not a Rule-A improvement")
    require_equal(checkpoint.get("rule"), "A", "Checkpoint rule")
    require_equal(int(checkpoint["best_epoch"]), selected_epoch, "Checkpoint epoch")
    if abs(float(checkpoint["best_validation_loss"]) - float(history[selected_epoch - 1]["validation_loss"])) > 1e-10:
        raise ValueError("Selected checkpoint validation loss mismatch")
    if Path(checkpoint["selected_checkpoint"]).name != "best_validation_loss.pt":
        raise ValueError("Selected checkpoint filename mismatch")
    require_equal(checkpoint.get("selected_checkpoint_sha256"), sha256(output / "checkpoints" / "best_validation_loss.pt"), "Checkpoint hash")
    require_equal(int(timing["epochs_executed"]), len(history), "Epoch count")
    if not math.isfinite(float(timing["elapsed_seconds"])) or float(timing["elapsed_seconds"]) <= 0:
        raise ValueError("Invalid run duration")
    if not math.isfinite(float(metrics["test_loss"])):
        raise ValueError("Non-finite test loss")
    verify_predictions(output / "predictions.csv", row, expected, metrics)
    if row["dataset"] == "isic2019":
        if int(metrics.get("secondary_lesion_units", 0)) <= 0:
            raise ValueError("Missing secondary ISIC lesion-unit count")
        secondary_ba = float(metrics.get("secondary_lesion_balanced_accuracy", float("nan")))
        if not math.isfinite(secondary_ba) or not 0 <= secondary_ba <= 1:
            raise ValueError("Invalid secondary ISIC lesion balanced accuracy")


def audit(manifest: Path, stage: str = "all") -> dict:
    rows = read_csv(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    first = rows[0]
    validate_manifest_row(SimpleNamespace(
        manifest=manifest, technical_smoke=False, run_id=first["run_id"],
        dataset=first["dataset"], model=first["model"], variant=first["variant"],
        training_seed=int(first["training_seed"]), split_file=ROOT / first["split_file"],
        dataset_index=ROOT / first["dataset_index_file"], output_dir=ROOT / first["output_dir"],
    ))
    selected = ordered_rows(rows, stage)
    weight_hashes = {model: current_weight_hash(model) for model in {row["model"] for row in selected}}
    splits = {}
    issues = []
    checked = 0
    for row in selected:
        try:
            for field in ("split_file", "dataset_index_file", "output_dir"):
                if Path(row[field]).is_absolute() or ".." in Path(row[field]).parts:
                    raise ValueError(f"Unsafe manifest path: {field}")
            for file_field, hash_field in (("split_file", "split_sha256"), ("dataset_index_file", "dataset_index_sha256")):
                require_equal(sha256(ROOT / row[file_field]), row[hash_field], f"Manifest input {file_field}")
            key = (row["dataset"], row["split_file"])
            if key not in splits:
                splits[key] = expected_test_units(ROOT / row["split_file"], row["dataset"])
            verify_run(row, manifest, splits[key], weight_hashes[row["model"]])
            checked += 1
        except Exception as exc:
            issues.append({"run_id": row.get("run_id"), "error": f"{type(exc).__name__}: {exc}"})
    return {"status": "passed" if not issues else "failed", "stage": stage,
            "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
            "manifest_total_runs": len(rows), "expected_stage_runs": len(selected),
            "verified_runs": checked, "issues": issues}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("e0", "e1", "sensitivity", "all"), default="all")
    parser.add_argument("--report", type=Path, help="Optional new JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    result = audit(args.manifest, args.stage)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "passed":
        sys.exit(2)


if __name__ == "__main__":
    main()
