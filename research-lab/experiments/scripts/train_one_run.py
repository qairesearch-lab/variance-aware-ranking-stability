#!/usr/bin/env python3
"""Execute one manifest-defined run or emit schema-compliant smoke outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SCRIPT_VERSION = "train_one_run_public_v2"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "research-lab" / "experiments"
CONFIG_DIR = EXPERIMENTS_DIR / "registered-workflow" / "configs" / "frozen"
SPLITS_DIR = EXPERIMENTS_DIR / "registered-workflow" / "splits"
SPLIT_HASHES_PATH = SPLITS_DIR / "split_hashes.csv"
DATA_DIR = REPO_ROOT / "research-lab" / "data"
STATUS_ENUM = {"completed", "failed", "rerun_completed"}

RUN_COLUMNS = [
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

PREDICTION_BASE_COLUMNS = [
    "run_id",
    "dataset",
    "split_id",
    "split_seed",
    "training_seed",
    "model",
    "checkpoint_policy",
    "config_hash",
    "split_hash",
    "analysis_role",
    "sample_id",
    "relative_path",
    "label_true",
    "label_pred",
]
JOIN_KEYS = [
    "run_id",
    "dataset",
    "split_id",
    "split_seed",
    "training_seed",
    "model",
    "checkpoint_policy",
    "config_hash",
    "split_hash",
    "analysis_role",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_manifest_row(manifest: Path, run_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest)
    if rows and list(rows[0].keys()) != RUN_COLUMNS:
        raise ValueError(f"Unexpected manifest schema: {list(rows[0].keys())}")
    matches = [row for row in rows if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for run_id={run_id}, found {len(matches)}")
    return matches[0]


def load_split_hash_map() -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["split_id"]): row for row in read_csv_rows(SPLIT_HASHES_PATH)}


def validate_manifest_row(row: dict[str, str]) -> None:
    datasets = {
        item["dataset_id"]: item for item in load_yaml(CONFIG_DIR / "datasets.yaml")["primary_datasets"]
    }
    models = {item["name"]: item for item in load_yaml(CONFIG_DIR / "models.yaml")["model_pool"]}
    policies = load_yaml(CONFIG_DIR / "checkpoint_policies.yaml")["primary_checkpoint_policies"]
    training_seeds = {str(seed) for seed in load_yaml(CONFIG_DIR / "random_seeds.yaml")["training_seeds"]["values"]}
    split_plan = load_yaml(CONFIG_DIR / "split_plan.yaml")["split_design"]["split_ids"]
    split_seed_by_id = {item["id"]: str(item["generation_seed"]) for item in split_plan}
    split_hashes = load_split_hash_map()

    if row["dataset"] not in datasets:
        raise ValueError(f"Dataset is not in the frozen primary matrix: {row['dataset']}")
    if row["model"] not in models:
        raise ValueError(f"Model is not in the frozen model pool: {row['model']}")
    if row["checkpoint_policy"] not in policies:
        raise ValueError(f"Checkpoint policy is not frozen for primary analysis: {row['checkpoint_policy']}")
    if row["checkpoint_policy"] == "C":
        raise ValueError("Policy C is excluded from primary training")
    if row["training_seed"] not in training_seeds:
        raise ValueError(f"Training seed is not in the frozen seed set: {row['training_seed']}")
    if row["split_id"] not in split_seed_by_id:
        raise ValueError(f"Split id is not in the frozen split plan: {row['split_id']}")
    if row["split_seed"] != split_seed_by_id[row["split_id"]]:
        raise ValueError(f"Split seed mismatch for {row['run_id']}")

    split_key = (row["dataset"], row["split_id"])
    if split_key not in split_hashes:
        raise ValueError(f"Missing split hash for {split_key}")
    split_hash_row = split_hashes[split_key]
    if row["split_hash"] != split_hash_row["sha256"]:
        raise ValueError(f"Split hash mismatch for {row['run_id']}")
    if row["split_seed"] != split_hash_row["split_seed"]:
        raise ValueError(f"Split hash manifest seed mismatch for {row['run_id']}")
    if Path(row["output_dir"]).is_absolute():
        raise ValueError(f"output_dir must be repository-relative: {row['output_dir']}")


def package_versions() -> dict[str, str | None]:
    packages = ["python", "PyYAML", "numpy", "pandas", "scikit-learn", "Pillow", "torch", "torchvision"]
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages[1:]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def device_info(device: torch.device) -> dict:
    cuda_available = torch.cuda.is_available()
    gpu_models = []
    if cuda_available:
        gpu_models = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "hostname": "not_recorded",
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_models": gpu_models,
        "visible_device_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def resolved_config(row: dict[str, str]) -> dict:
    datasets = {
        item["dataset_id"]: item for item in load_yaml(CONFIG_DIR / "datasets.yaml")["primary_datasets"]
    }
    models = {item["name"]: item for item in load_yaml(CONFIG_DIR / "models.yaml")["model_pool"]}
    policies = load_yaml(CONFIG_DIR / "checkpoint_policies.yaml")["primary_checkpoint_policies"]
    training_common = load_yaml(CONFIG_DIR / "training_common.yaml")
    return {
        "dataset": datasets[row["dataset"]],
        "model": models[row["model"]],
        "checkpoint_policy": policies[row["checkpoint_policy"]],
        "training_common": training_common,
        "split_file": str(
            (SPLITS_DIR / row["dataset"] / f"{row['split_id']}.csv").relative_to(REPO_ROOT)
        ),
    }


def dataset_root(dataset_id: str) -> Path:
    if dataset_id == "sipakmed":
        return DATA_DIR / "sipakmed" / "raw"
    if dataset_id == "organamnist":
        return DATA_DIR / "organamnist"
    raise ValueError(f"Unsupported dataset: {dataset_id}")


def split_rows(row: dict[str, str], subset: str, limit: int | None = None) -> list[dict[str, str]]:
    split_file = SPLITS_DIR / row["dataset"] / f"{row['split_id']}.csv"
    rows = [item for item in read_csv_rows(split_file) if item["subset"] == subset]
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_rows_for_smoke(row: dict[str, str], limit: int) -> list[dict[str, str]]:
    return split_rows(row, "test", limit)


def class_count(row: dict[str, str]) -> int:
    datasets = {
        item["dataset_id"]: item for item in load_yaml(CONFIG_DIR / "datasets.yaml")["primary_datasets"]
    }
    return int(datasets[row["dataset"]]["classes"])


class SplitCsvImageDataset(Dataset):
    def __init__(self, row: dict[str, str], subset: str, transform: transforms.Compose, limit: int | None = None):
        self.row = row
        self.subset = subset
        self.root = dataset_root(row["dataset"])
        self.rows = split_rows(row, subset, limit)
        self.transform = transform
        if not self.rows:
            raise ValueError(f"No {subset} rows for {row['dataset']} {row['split_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        item = self.rows[index]
        relative_path = Path(item["relative_path"])
        image_path = self.root / relative_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image from split relative_path: {image_path}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), int(item["label"]), item


def build_transforms(config: dict, dataset_id: str) -> tuple[transforms.Compose, transforms.Compose]:
    training_common = config["training_common"]
    image_size = int(training_common["input"]["image_size"])
    mean = training_common["normalization"]["mean"]
    std = training_common["normalization"]["std"]
    common_train = training_common["augmentation"]["common_train"]
    dataset_override = training_common["augmentation"]["dataset_overrides"][dataset_id]

    train_ops: list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=tuple(common_train["random_resized_crop"]["scale"]),
        )
    ]
    if dataset_override["horizontal_flip"]["enabled"]:
        train_ops.append(transforms.RandomHorizontalFlip(p=float(dataset_override["horizontal_flip"]["p"])))
    if common_train["rotation"]["enabled"]:
        train_ops.append(transforms.RandomRotation(degrees=float(common_train["rotation"]["degrees"])))
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_eval_only_transforms(config: dict) -> transforms.Compose:
    training_common = config["training_common"]
    image_size = int(training_common["input"]["image_size"])
    mean = training_common["normalization"]["mean"]
    std = training_common["normalization"]["std"]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


class SeededWorkerInit:
    """Pickle-safe DataLoader worker seeding callable for Windows spawn mode."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def build_model(row: dict[str, str], config: dict, classes: int) -> nn.Module:
    model_name = row["model"]
    weight_name = config["model"]["weights"]
    weights = getattr(models.get_model_weights(model_name), weight_name)
    model = models.get_model(model_name, weights=weights)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, classes)
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, classes)
    elif model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, classes)
    else:
        raise ValueError(f"Unsupported frozen model: {model_name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    if model_name.startswith("resnet"):
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif model_name == "densenet121":
        for parameter in model.features.denseblock4.parameters():
            parameter.requires_grad = True
        for parameter in model.features.norm5.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif model_name == "efficientnet_b0":
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return model


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    timeout: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=SeededWorkerInit(seed),
        generator=generator,
        timeout=timeout if num_workers > 0 else 0,
    )


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, classes: int) -> float:
    scores = []
    for class_index in range(classes):
        scores.append(brier_score_loss((y_true == class_index).astype(int), probabilities[:, class_index]))
    return float(np.mean(scores))


def compute_metrics(y_true: list[int], y_pred: list[int], probabilities: list[list[float]], classes: int) -> dict:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    prob_array = np.asarray(probabilities)
    labels = list(range(classes))
    one_hot = np.eye(classes)[y_true_array]
    try:
        roc_auc = roc_auc_score(one_hot, prob_array, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(one_hot, prob_array, average="macro")
    except ValueError:
        pr_auc = None
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "roc_auc_macro_ovr": finite_or_none(roc_auc),
        "pr_auc_macro": finite_or_none(pr_auc),
        "brier_score_multiclass": multiclass_brier(y_true_array, prob_array, classes),
        "calibration_intercept": None,
        "calibration_slope": None,
        "per_class_metrics": classification_report(
            y_true_array,
            y_pred_array,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred_array, labels=labels).tolist(),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: int,
) -> tuple[float, dict, list[dict[str, object]]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    prediction_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels_cpu = labels.cpu().numpy()
            losses.append(float(loss.item()))
            y_true.extend(labels_cpu.tolist())
            y_pred.extend(preds.tolist())
            probabilities.extend(probs.tolist())
            for index in range(len(labels_cpu)):
                row = {key: metadata[key][index] for key in metadata}
                row["label_true"] = int(labels_cpu[index])
                row["label_pred"] = int(preds[index])
                for class_index in range(classes):
                    row[f"prob_class_{class_index}"] = round(float(probs[index][class_index]), 8)
                prediction_rows.append(row)
    metrics = compute_metrics(y_true, y_pred, probabilities, classes)
    return float(np.mean(losses)) if losses else 0.0, metrics, prediction_rows


def emit_schema_smoke_outputs(row: dict[str, str], output_dir: Path, sample_limit: int) -> None:
    now = utc_now()
    config = resolved_config(row)
    classes = class_count(row)
    test_rows = split_rows_for_smoke(row, sample_limit)
    if not test_rows:
        raise ValueError(f"No test rows found for {row['dataset']} {row['split_id']}")

    prediction_columns = PREDICTION_BASE_COLUMNS + [f"prob_class_{index}" for index in range(classes)]
    predictions: list[dict[str, object]] = []
    for index, sample in enumerate(test_rows):
        true_label = int(sample["label"])
        pred_label = true_label if index % 4 else (true_label + 1) % classes
        probabilities = {f"prob_class_{class_index}": 0.02 for class_index in range(classes)}
        probabilities[f"prob_class_{pred_label}"] = round(1.0 - 0.02 * (classes - 1), 6)
        predictions.append(
            {
                **{key: row[key] for key in RUN_COLUMNS if key not in {"output_dir", "status"}},
                "sample_id": sample["sample_id"],
                "relative_path": sample["relative_path"],
                "label_true": true_label,
                "label_pred": pred_label,
                **probabilities,
            }
        )

    correct = sum(int(item["label_true"] == item["label_pred"]) for item in predictions)
    accuracy = correct / len(predictions)
    metrics = {
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "split_id": row["split_id"],
        "split_seed": row["split_seed"],
        "training_seed": row["training_seed"],
        "model": row["model"],
        "checkpoint_policy": row["checkpoint_policy"],
        "config_hash": row["config_hash"],
        "split_hash": row["split_hash"],
        "analysis_role": row["analysis_role"],
        "metric_source": "schema_smoke",
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(accuracy, 6),
        "roc_auc_macro_ovr": None,
        "pr_auc_macro": None,
        "brier_score_multiclass": None,
        "calibration_intercept": None,
        "calibration_slope": None,
        "per_class_metrics": {},
        "confusion_matrix": [],
    }

    write_json(
        output_dir / "run_config.json",
        {
            "created_at_utc": now,
            "manifest_row": row,
            "resolved_frozen_config": config,
            "package_versions": package_versions(),
            "device_info": {
                "hostname": "not_recorded",
                "platform": platform.platform(),
                "cuda_available": None,
                "cuda_version": None,
                "cudnn_version": None,
                "gpu_models": [],
                "visible_device_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "seed_policy": {
                "training_seed": row["training_seed"],
                "dataloader_worker_seed_policy": config["training_common"]["reproducibility"][
                    "dataloader_worker_seed_policy"
                ],
                "dataloader_generator_policy": config["training_common"]["reproducibility"][
                    "dataloader_generator_policy"
                ],
                "worker_seed_formula": config["training_common"]["reproducibility"]["worker_seed_formula"],
            },
            "weights": {
                "weights_enum": config["model"]["weights"],
                "cache_location": os.environ.get("TORCH_HOME", "$TORCH_HOME") + "/hub/checkpoints/",
            },
            "git_commit": git_commit(),
        },
    )
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "predictions.csv", prediction_columns, predictions)
    write_csv(
        output_dir / "history.csv",
        ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy", "learning_rate"],
        [
            {
                "epoch": 0,
                "train_loss": 0.0,
                "validation_loss": 0.0,
                "train_accuracy": 0.0,
                "validation_accuracy": 0.0,
                "learning_rate": config["training_common"]["optimization"]["learning_rate"],
            }
        ],
    )
    write_json(
        output_dir / "checkpoint_manifest.json",
        {
            "run_id": row["run_id"],
            "checkpoint_policy": row["checkpoint_policy"],
            "checkpoint_used_for_evaluation": config["checkpoint_policy"]["checkpoint_used_for_evaluation"],
            "checkpoints": [],
            "mode": "schema_smoke",
        },
    )
    write_json(
        output_dir / "run_status.json",
        {
            "run_id": row["run_id"],
            "status": "completed",
            "execution_mode": "schema_smoke",
            "started_at_utc": now,
            "ended_at_utc": now,
            "message": "Schema-compliant smoke output emitted without primary training.",
        },
    )


def previous_status(output_dir: Path) -> str | None:
    status_path = output_dir / "run_status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return None


def write_failed_status(output_dir: Path, row: dict[str, str], mode: str, started_at: str, exc: Exception) -> None:
    write_json(
        output_dir / "run_status.json",
        {
            "run_id": row["run_id"],
            "status": "failed",
            "execution_mode": mode,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "message": str(exc),
            "failure_policy": "Execution/data/output failures may be failed; model performance must not define failure.",
        },
    )


def write_progress(output_dir: Path, row: dict[str, str], stage: str, message: str, **extra: object) -> None:
    payload = {
        "run_id": row["run_id"],
        "updated_at_utc": utc_now(),
        "stage": stage,
        "message": message,
        **extra,
    }
    write_json(output_dir / "progress.json", payload)
    extras = " ".join(f"{key}={value}" for key, value in extra.items())
    suffix = f" {extras}" if extras else ""
    print(f"[{payload['updated_at_utc']}] {stage}: {message}{suffix}", flush=True)


def emit_primary_train_outputs(
    row: dict[str, str],
    output_dir: Path,
    max_epochs_override: int | None = None,
    sample_limit: int | None = None,
    progress_every_batches: int = 50,
    allow_cpu: bool = False,
    dataloader_timeout: int = 0,
    num_workers_override: int | None = None,
    disable_pin_memory: bool = False,
    max_train_batches: int | None = None,
    sync_cuda_each_batch: bool = False,
    disable_train_augmentation: bool = False,
) -> None:
    started_at = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_progress(
        output_dir,
        row,
        "initializing",
        "Loading frozen configuration and manifest row.",
        script_version=SCRIPT_VERSION,
        script_path=str(Path(__file__).resolve()),
        python_executable=sys.executable,
    )
    config = resolved_config(row)
    training_common = config["training_common"]
    policy = config["checkpoint_policy"]
    seed = int(row["training_seed"])
    classes = class_count(row)
    previous = previous_status(output_dir)
    write_json(
        output_dir / "run_status.json",
        {
            "run_id": row["run_id"],
            "status": "running",
            "execution_mode": "primary_train",
            "started_at_utc": started_at,
            "ended_at_utc": None,
            "message": "Training run started.",
        },
    )
    try:
        write_progress(
            output_dir,
            row,
            "started",
            "Primary training started.",
            dataset=row["dataset"],
            model=row["model"],
            checkpoint_policy=row["checkpoint_policy"],
            seed=seed,
        )
        cuda_names = []
        if torch.cuda.is_available():
            cuda_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        write_progress(
            output_dir,
            row,
            "cuda_probe",
            "Checking CUDA before any data loading or model training.",
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
            cuda_device_names=cuda_names,
            allow_cpu=allow_cpu,
        )
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError(
                "CUDA is required for primary_train by default, but torch.cuda.is_available() is false. "
                f"script_version={SCRIPT_VERSION}; python_executable={sys.executable}; "
                f"torch_version={torch.__version__}; torch_cuda_version={torch.version.cuda}. "
                "Fix the PyTorch/driver environment or pass --allow-cpu only for diagnostic runs."
            )
        write_progress(output_dir, row, "seeding", "Setting Python/NumPy/PyTorch seeds.")
        seed_everything(seed, bool(training_common["reproducibility"]["cudnn_deterministic"]))
        write_progress(output_dir, row, "transforms", "Building train/eval transforms.")
        train_transform, eval_transform = build_transforms(config, row["dataset"])
        if disable_train_augmentation:
            write_progress(output_dir, row, "transforms", "Disabling train augmentation for diagnostics.")
            train_transform = build_eval_only_transforms(config)
        write_progress(output_dir, row, "dataset_build", "Building train dataset.", subset="train")
        train_dataset = SplitCsvImageDataset(row, "train", train_transform, sample_limit)
        write_progress(output_dir, row, "dataset_build", "Building validation dataset.", subset="validation")
        val_dataset = SplitCsvImageDataset(row, "validation", eval_transform, sample_limit)
        write_progress(output_dir, row, "dataset_build", "Building test dataset.", subset="test")
        test_dataset = SplitCsvImageDataset(row, "test", eval_transform, sample_limit)
        batch_size = int(training_common["optimization"]["batch_size"])
        num_workers = 0 if sample_limit is not None else int(training_common["reproducibility"]["num_workers"])
        if num_workers_override is not None:
            num_workers = int(num_workers_override)
        pin_memory = bool(training_common["reproducibility"]["pin_memory"]) and not disable_pin_memory
        write_progress(
            output_dir,
            row,
            "dataloader_build",
            "Creating DataLoaders.",
            train_samples=len(train_dataset),
            validation_samples=len(val_dataset),
            test_samples=len(test_dataset),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            dataloader_timeout=dataloader_timeout,
            max_train_batches=max_train_batches,
            sync_cuda_each_batch=sync_cuda_each_batch,
            disable_train_augmentation=disable_train_augmentation,
        )
        train_loader = make_loader(train_dataset, batch_size, True, seed, num_workers, pin_memory, dataloader_timeout)
        val_loader = make_loader(val_dataset, batch_size, False, seed, num_workers, pin_memory, dataloader_timeout)
        test_loader = make_loader(test_dataset, batch_size, False, seed, num_workers, pin_memory, dataloader_timeout)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        write_progress(
            output_dir,
            row,
            "device",
            "Resolved training device.",
            device=str(device),
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
        )
        write_progress(output_dir, row, "model_build", "Building model and loading pretrained weights.")
        model = build_model(row, config, classes).to(device)
        write_progress(output_dir, row, "model_ready", "Model moved to device.")
        criterion = nn.CrossEntropyLoss()
        write_progress(output_dir, row, "optimizer", "Creating optimizer.")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(training_common["optimization"]["learning_rate"]),
            weight_decay=float(training_common["optimization"]["weight_decay"]),
        )

        max_epochs = int(policy["max_epochs"])
        if max_epochs_override is not None:
            max_epochs = max_epochs_override
        write_progress(output_dir, row, "train_plan", "Training plan resolved.", max_epochs=max_epochs)
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history_rows: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best_validation_loss.pt"
        last_path = checkpoint_dir / "last_epoch.pt"

        for epoch in range(1, max_epochs + 1):
            write_progress(output_dir, row, "epoch_train_start", "Starting training epoch.", epoch=epoch, max_epochs=max_epochs)
            model.train()
            train_losses: list[float] = []
            train_true: list[int] = []
            train_pred: list[int] = []
            total_batches = len(train_loader)
            write_progress(
                output_dir,
                row,
                "epoch_first_batch_wait",
                "Waiting for first training batch from DataLoader.",
                epoch=epoch,
                total_batches=total_batches,
                num_workers=num_workers,
            )
            for batch_index, (images, labels, _metadata) in enumerate(train_loader, start=1):
                if batch_index == 1:
                    write_progress(
                        output_dir,
                        row,
                        "epoch_first_batch_received",
                        "Received first training batch.",
                        epoch=epoch,
                        image_shape=list(images.shape),
                        label_shape=list(labels.shape),
                    )
                if max_train_batches is not None and batch_index > max_train_batches:
                    write_progress(
                        output_dir,
                        row,
                        "diagnostic_batch_stop",
                        "Stopping epoch early because --max-train-batches was reached.",
                        epoch=epoch,
                        max_train_batches=max_train_batches,
                    )
                    break
                if progress_every_batches > 0 and (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every_batches == 0
                ):
                    write_progress(
                        output_dir,
                        row,
                        "batch_to_device_start",
                        "Moving batch to device.",
                        epoch=epoch,
                        batch=batch_index,
                        total_batches=total_batches,
                    )
                images = images.to(device)
                labels = labels.to(device)
                if sync_cuda_each_batch and device.type == "cuda":
                    torch.cuda.synchronize()
                    write_progress(
                        output_dir,
                        row,
                        "batch_to_device_done",
                        "Batch moved to device and CUDA synchronized.",
                        epoch=epoch,
                        batch=batch_index,
                    )
                if progress_every_batches > 0 and (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every_batches == 0
                ):
                    write_progress(
                        output_dir,
                        row,
                        "batch_forward_start",
                        "Starting forward pass.",
                        epoch=epoch,
                        batch=batch_index,
                        total_batches=total_batches,
                    )
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                if sync_cuda_each_batch and device.type == "cuda":
                    torch.cuda.synchronize()
                    write_progress(
                        output_dir,
                        row,
                        "batch_forward_done",
                        "Forward pass completed and CUDA synchronized.",
                        epoch=epoch,
                        batch=batch_index,
                    )
                if progress_every_batches > 0 and (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every_batches == 0
                ):
                    write_progress(
                        output_dir,
                        row,
                        "batch_backward_start",
                        "Starting backward pass.",
                        epoch=epoch,
                        batch=batch_index,
                        total_batches=total_batches,
                        loss=round(float(loss.item()), 6),
                    )
                loss.backward()
                if sync_cuda_each_batch and device.type == "cuda":
                    torch.cuda.synchronize()
                    write_progress(
                        output_dir,
                        row,
                        "batch_backward_done",
                        "Backward pass completed and CUDA synchronized.",
                        epoch=epoch,
                        batch=batch_index,
                    )
                if progress_every_batches > 0 and (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every_batches == 0
                ):
                    write_progress(
                        output_dir,
                        row,
                        "batch_optimizer_start",
                        "Starting optimizer step.",
                        epoch=epoch,
                        batch=batch_index,
                        total_batches=total_batches,
                    )
                optimizer.step()
                if sync_cuda_each_batch and device.type == "cuda":
                    torch.cuda.synchronize()
                    write_progress(
                        output_dir,
                        row,
                        "batch_optimizer_done",
                        "Optimizer step completed and CUDA synchronized.",
                        epoch=epoch,
                        batch=batch_index,
                    )
                train_losses.append(float(loss.item()))
                train_true.extend(labels.detach().cpu().numpy().tolist())
                train_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())
                if progress_every_batches > 0 and (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % progress_every_batches == 0
                ):
                    write_progress(
                        output_dir,
                        row,
                        "epoch_batch",
                        "Training batch completed.",
                        epoch=epoch,
                        max_epochs=max_epochs,
                        batch=batch_index,
                        total_batches=total_batches,
                        loss=round(float(loss.item()), 6),
                    )

            write_progress(output_dir, row, "validation_start", "Starting validation evaluation.", epoch=epoch)
            val_loss, val_metrics, _ = evaluate_model(model, val_loader, criterion, device, classes)
            train_accuracy = accuracy_score(train_true, train_pred) if train_true else 0.0
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                    "validation_loss": val_loss,
                    "train_accuracy": float(train_accuracy),
                    "validation_accuracy": val_metrics["accuracy"],
                    "learning_rate": float(training_common["optimization"]["learning_rate"]),
                }
            )
            write_progress(
                output_dir,
                row,
                "validation_done",
                "Validation evaluation completed.",
                epoch=epoch,
                train_accuracy=round(float(train_accuracy), 6),
                validation_loss=round(float(val_loss), 6),
                validation_accuracy=round(float(val_metrics["accuracy"]), 6),
            )

            if policy["selection_rule"] == "minimum_val_loss":
                improved = val_loss < best_val_loss - float(policy["min_delta"])
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    write_progress(output_dir, row, "checkpoint_save", "Saving best validation checkpoint.", epoch=epoch)
                    torch.save(model.state_dict(), best_path)
                else:
                    patience_counter += 1
                if policy["early_stopping"] and patience_counter >= int(policy["patience"]):
                    write_progress(output_dir, row, "early_stop", "Early stopping triggered.", epoch=epoch, patience=policy["patience"])
                    break

        write_progress(output_dir, row, "checkpoint_save", "Saving last epoch checkpoint.", epoch=history_rows[-1]["epoch"])
        torch.save(model.state_dict(), last_path)
        checkpoints.append({"path": str(last_path.relative_to(REPO_ROOT)), "epoch": history_rows[-1]["epoch"], "role": "last_epoch"})
        if best_path.exists():
            checkpoints.append({"path": str(best_path.relative_to(REPO_ROOT)), "epoch": best_epoch, "role": "best_validation_loss"})

        selected_checkpoint = last_path
        selected_epoch = history_rows[-1]["epoch"]
        if policy["selection_rule"] == "minimum_val_loss" and best_path.exists():
            selected_checkpoint = best_path
            selected_epoch = best_epoch
            write_progress(output_dir, row, "checkpoint_load", "Loading selected best validation checkpoint.", selected_epoch=selected_epoch)
            model.load_state_dict(torch.load(best_path, map_location=device))

        write_progress(output_dir, row, "test_start", "Starting test evaluation.", selected_epoch=selected_epoch)
        _test_loss, test_metrics, prediction_rows = evaluate_model(model, test_loader, criterion, device, classes)
        write_progress(
            output_dir,
            row,
            "test_done",
            "Test evaluation completed.",
            accuracy=round(float(test_metrics["accuracy"]), 6),
            predictions=len(prediction_rows),
        )
        prediction_columns = PREDICTION_BASE_COLUMNS + [f"prob_class_{index}" for index in range(classes)]
        joined_predictions = []
        for prediction in prediction_rows:
            joined_predictions.append(
                {
                    **{key: row[key] for key in RUN_COLUMNS if key not in {"output_dir", "status"}},
                    "sample_id": prediction["sample_id"],
                    "relative_path": prediction["relative_path"],
                    "label_true": prediction["label_true"],
                    "label_pred": prediction["label_pred"],
                    **{f"prob_class_{index}": prediction[f"prob_class_{index}"] for index in range(classes)},
                }
            )

        metrics = {
            **{key: row[key] for key in JOIN_KEYS},
            "metric_source": "primary_train",
            **test_metrics,
        }
        final_status = "rerun_completed" if previous == "failed" else "completed"
        write_progress(output_dir, row, "write_outputs", "Writing run outputs.")
        write_json(
            output_dir / "run_config.json",
            {
                "script_version": SCRIPT_VERSION,
                "script_path": str(Path(__file__).resolve()),
                "python_executable": sys.executable,
                "created_at_utc": started_at,
                "manifest_row": row,
                "resolved_frozen_config": config,
                "validation_overrides": {
                    "max_epochs": max_epochs_override,
                    "sample_limit_per_subset": sample_limit,
                    "num_workers_override": num_workers_override,
                    "disable_pin_memory": disable_pin_memory,
                    "max_train_batches": max_train_batches,
                    "sync_cuda_each_batch": sync_cuda_each_batch,
                    "disable_train_augmentation": disable_train_augmentation,
                },
                "package_versions": package_versions(),
                "device_info": device_info(device),
                "seed_policy": {
                    "training_seed": row["training_seed"],
                    "dataloader_worker_seed_policy": training_common["reproducibility"]["dataloader_worker_seed_policy"],
                    "dataloader_generator_policy": training_common["reproducibility"]["dataloader_generator_policy"],
                    "worker_seed_formula": training_common["reproducibility"]["worker_seed_formula"],
                },
                "augmentation_check": {
                    "dataset": row["dataset"],
                    "horizontal_flip_enabled": training_common["augmentation"]["dataset_overrides"][row["dataset"]][
                        "horizontal_flip"
                    ]["enabled"],
                },
                "weights": {
                    "weights_enum": config["model"]["weights"],
                    "cache_location": os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
                    + "/hub/checkpoints/",
                },
                "git_commit": git_commit(),
            },
        )
        write_json(output_dir / "metrics.json", metrics)
        write_csv(output_dir / "predictions.csv", prediction_columns, joined_predictions)
        write_csv(
            output_dir / "history.csv",
            ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy", "learning_rate"],
            history_rows,
        )
        write_json(
            output_dir / "checkpoint_manifest.json",
            {
                "run_id": row["run_id"],
                "checkpoint_policy": row["checkpoint_policy"],
                "checkpoint_used_for_evaluation": policy["checkpoint_used_for_evaluation"],
                "selected_checkpoint": str(selected_checkpoint.relative_to(REPO_ROOT)),
                "selected_epoch": selected_epoch,
                "checkpoints": checkpoints,
                "mode": "primary_train",
            },
        )
        write_json(
            output_dir / "run_status.json",
            {
                "run_id": row["run_id"],
                "status": final_status,
                "execution_mode": "primary_train",
                "started_at_utc": started_at,
                "ended_at_utc": utc_now(),
                "message": "Training run completed. Performance values do not determine run success/failure.",
            },
        )
        write_progress(output_dir, row, "done", "Training run completed.", status=final_status)
    except Exception as exc:
        write_failed_status(output_dir, row, "primary_train", started_at, exc)
        write_progress(output_dir, row, "failed", "Training run failed.", error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-mode", choices=["schema_smoke", "primary_train"], default="schema_smoke")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--progress-every-batches", type=int, default=50)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow primary_train to run on CPU for diagnostics. Formal workstation runs should not use this.",
    )
    parser.add_argument(
        "--dataloader-timeout",
        type=int,
        default=0,
        help="Seconds before a multi-worker DataLoader raises a timeout error. Use 120 for diagnostics.",
    )
    parser.add_argument(
        "--num-workers-override",
        type=int,
        default=None,
        help="Diagnostic override for DataLoader num_workers. Use 0 to isolate Windows worker hangs.",
    )
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Diagnostic option to disable DataLoader pin_memory.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Diagnostic option to stop each epoch after N training batches.",
    )
    parser.add_argument(
        "--sync-cuda-each-batch",
        action="store_true",
        help="Diagnostic option to synchronize CUDA after to_device/forward/backward/optimizer.",
    )
    parser.add_argument(
        "--disable-train-augmentation",
        action="store_true",
        help="Diagnostic option to use eval transforms for training.",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    row = find_manifest_row(manifest, args.run_id)
    validate_manifest_row(row)
    output_dir = REPO_ROOT / row["output_dir"]
    if args.execution_mode == "schema_smoke":
        emit_schema_smoke_outputs(row, output_dir, args.sample_limit or 24)
        print(f"RUN_OUTPUT_SCHEMA_READY {row['run_id']} {output_dir}")
    elif args.execution_mode == "primary_train":
        emit_primary_train_outputs(
            row,
            output_dir,
            args.max_epochs,
            args.sample_limit,
            args.progress_every_batches,
            args.allow_cpu,
            args.dataloader_timeout,
            args.num_workers_override,
            args.disable_pin_memory,
            args.max_train_batches,
            args.sync_cuda_each_batch,
            args.disable_train_augmentation,
        )
        print(f"RUN_TRAINING_COMPLETED {row['run_id']} {output_dir}")


if __name__ == "__main__":
    main()
