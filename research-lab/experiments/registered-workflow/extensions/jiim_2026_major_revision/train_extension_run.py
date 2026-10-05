#!/usr/bin/env python3
"""One-run runner for the JIIM reviewer-requested extension.

Requires a pre-audited split CSV. The runner deliberately does not synthesize
splits or silently amend frozen hyperparameters. `--technical-smoke` is for
local integration checks only; its outputs are marked non-formal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from preprocessing import DATASETS, MODELS, VARIANTS, build_model, build_transforms, decode_image, rule_a_update
from protocol_amendment import AMENDMENT, SCOPE_DECISION, load_scope_decision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DESIGN = HERE / "configs" / "extension_protocol_design_locked_v0.3.yaml"
ENVIRONMENT_TARGET = HERE / "configs" / "environment_target_v1.yaml"
SPLIT_COLUMNS = ("dataset", "subset", "sample_id", "relative_path", "label", "group_id", "study_id", "body_region")
SEEDS = (42, 52, 62)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "pandas": importlib.metadata.version("pandas"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "pyyaml": importlib.metadata.version("PyYAML"),
    }


def validate_target_environment() -> None:
    with ENVIRONMENT_TARGET.open("r", encoding="utf-8") as handle:
        expected = yaml.safe_load(handle)
    actual = runtime_versions()
    mismatches = {}
    for key in ("torch", "torchvision", "numpy", "pillow", "pandas", "scikit_learn", "pyyaml"):
        if actual[key] != expected[key]:
            mismatches[key] = {"expected": expected[key], "actual": actual[key]}
    if not actual["python"].startswith(expected["python"] + "."):
        mismatches["python"] = {"expected": expected["python"], "actual": actual["python"]}
    if mismatches:
        raise RuntimeError(f"Formal environment is not the pinned target: {mismatches}")


def validate_manifest_row(args) -> str:
    if args.manifest is None:
        if args.technical_smoke:
            return "technical_smoke_no_formal_manifest"
        raise ValueError("Formal run must be selected from a sealed run manifest")
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated run ID in manifest")
    matches = [row for row in rows if row["run_id"] == args.run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {args.run_id}")
    row = matches[0]
    if not args.technical_smoke:
        seal_path = args.manifest.with_suffix(".seal.json")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("status") != "sealed_after_data_quality_audit":
            raise ValueError("Manifest is not sealed after the data-quality audit")
        if (seal.get("manifest_sha256") != sha256(args.manifest)
                or seal.get("design_lock_sha256") != sha256(DESIGN)
                or seal.get("data_quality_amendment_sha256") != sha256(AMENDMENT)
                or seal.get("swin_sensitivity_decision_sha256") != sha256(SCOPE_DECISION)):
            raise ValueError("Manifest/design/amendment/scope hash does not match seal")
        scope = load_scope_decision()
        if seal.get("run_count") != len(rows) or len(rows) != scope["total_manifest_runs"]:
            raise ValueError("Sealed E1+B run count is invalid")
        if seal.get("swin_weight_eval_run_count") != scope["swin_weight_specific_eval_sensitivity_runs"]:
            raise ValueError("Sensitivity run count is invalid")
        expected_keys = {
            (dataset, f"split_{split_number:02d}", model, seed, "shared")
            for dataset in DATASETS
            for split_number in range(1, 11)
            for model in (("resnet18", "resnet50", "densenet121", "efficientnet_b0", "swin_t") if split_number <= 5 else ("resnet18", "resnet50", "densenet121", "efficientnet_b0"))
            for seed in SEEDS
        }
        expected_keys |= {
            (dataset, f"split_{split_number:02d}", "swin_t", seed, "swin_weight_eval")
            for dataset in DATASETS for split_number in range(1, 6) for seed in SEEDS
        }
        observed_keys = {
            (r["dataset"], r["split_id"], r["model"], int(r["training_seed"]), r["variant"])
            for r in rows
        }
        if observed_keys != expected_keys:
            raise ValueError("Sealed manifest does not match approved E1 matrix")
        for item in rows:
            split_key = f"{item['dataset']}/{item['split_id']}"
            if seal.get("split_hashes", {}).get(split_key) != item["split_sha256"]:
                raise ValueError(f"Sealed split hash mismatch: {split_key}")
            tag = "B" if item["variant"] == "swin_weight_eval" else "A"
            expected_id = f"jiim_ext_{item['dataset']}_{item['split_id']}_seed_{item['training_seed']}_{item['model']}_{tag}"
            if item["run_id"] != expected_id:
                raise ValueError(f"Unexpected run ID: {item['run_id']}")
    for column, value in (("dataset", args.dataset), ("model", args.model), ("variant", args.variant), ("training_seed", str(args.training_seed))):
        if row[column] != value:
            raise ValueError(f"Manifest mismatch: {column}")
    for column, path in (("split_file", args.split_file), ("dataset_index_file", args.dataset_index), ("output_dir", args.output_dir)):
        if Path(row[column]).is_absolute() or ".." in Path(row[column]).parts:
            raise ValueError(f"Manifest path must be repository-relative: {column}")
        if (ROOT / row[column]).resolve() != path.resolve():
            raise ValueError(f"Manifest path mismatch: {column}")
    if not args.technical_smoke and args.output_dir.parent.resolve() != (HERE / "runs").resolve():
        raise ValueError("Formal outputs must be direct children of extension runs/")
    if row["split_sha256"] != sha256(args.split_file) or row["dataset_index_sha256"] != sha256(args.dataset_index):
        raise ValueError("Manifest input hash mismatch")
    return sha256(args.manifest)


def read_split(path: Path, dataset: str, *, technical_smoke: bool) -> dict[str, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise ValueError(f"Split columns must exactly equal {SPLIT_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty split")
    partitions = {"train": [], "validation": [], "test": []}
    ids = set()
    groups_by_partition = defaultdict(set)
    expected_classes = set(range(8 if dataset == "isic2019" else 2))
    for row in rows:
        if row["dataset"] != dataset:
            raise ValueError("Dataset mismatch in split")
        if row["subset"] not in partitions:
            raise ValueError("Unexpected subset")
        if not row["sample_id"] or row["sample_id"] in ids:
            raise ValueError("Duplicate or empty sample ID")
        ids.add(row["sample_id"])
        if not row["group_id"]:
            raise ValueError("Missing split group ID")
        if dataset == "mura" and not row["study_id"]:
            raise ValueError("Missing MURA study ID")
        if int(row["label"]) not in expected_classes:
            raise ValueError("Invalid class")
        if Path(row["relative_path"]).is_absolute() or ".." in Path(row["relative_path"]).parts:
            raise ValueError("Unsafe image path")
        if not (ROOT / row["relative_path"]).is_file():
            raise FileNotFoundError(ROOT / row["relative_path"])
        partitions[row["subset"]].append(row)
        groups_by_partition[row["subset"]].add(row["group_id"])
    for subset, subset_rows in partitions.items():
        if not subset_rows:
            raise ValueError(f"Empty subset: {subset}")
        if not technical_smoke and {int(r["label"]) for r in subset_rows} != expected_classes:
            raise ValueError(f"Missing primary class in {subset}")
        if dataset == "mura" and not technical_smoke:
            if len({r["body_region"] for r in subset_rows}) != 7:
                raise ValueError(f"Missing MURA body region in {subset}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups_by_partition[left] & groups_by_partition[right]:
            raise ValueError(f"Group leakage between {left} and {right}")
    if dataset == "mura":
        study_labels = defaultdict(set)
        study_subsets = defaultdict(set)
        for row in rows:
            study_labels[row["study_id"]].add(int(row["label"]))
            study_subsets[row["study_id"]].add(row["subset"])
        if any(len(x) != 1 for x in study_labels.values()) or any(len(x) != 1 for x in study_subsets.values()):
            raise ValueError("Study label/subset conflict")
    return partitions


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, str]], dataset: str, transform):
        self.rows = rows
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(ROOT / row["relative_path"]) as source:
            image = decode_image(source, self.dataset)
        return self.transform(image), int(row["label"]), index


class WorkerSeed:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        seed = self.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def make_loader(rows, dataset, transform, seed, shuffle, workers, batch_size, device):
    return DataLoader(
        ImageRows(rows, dataset, transform), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=False,
        worker_init_fn=WorkerSeed(seed), generator=torch.Generator().manual_seed(seed),
    )


def mean_group_logits(logits: torch.Tensor, labels: torch.Tensor, rows, key: str):
    indexes = defaultdict(list)
    for i, row in enumerate(rows):
        indexes[row[key]].append(i)
    out_logits, out_labels, out_ids = [], [], []
    for group_id, idx in sorted(indexes.items()):
        group_labels = {int(labels[i]) for i in idx}
        if len(group_labels) != 1:
            raise ValueError(f"Conflicting labels within {key}={group_id}")
        out_logits.append(logits[idx].mean(dim=0))
        out_labels.append(next(iter(group_labels)))
        out_ids.append(group_id)
    return torch.stack(out_logits), torch.tensor(out_labels, dtype=torch.long), out_ids


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels, all_indexes = [], [], []
    batch_losses = []
    with torch.no_grad():
        for images, labels, indexes in loader:
            logits = model(images.to(device))
            batch_losses.append(float(nn.functional.cross_entropy(logits, labels.to(device)).item()))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_indexes.extend(int(x) for x in indexes)
    merged_logits = torch.cat(all_logits)
    merged_labels = torch.cat(all_labels)
    order = np.argsort(np.asarray(all_indexes))
    return merged_logits[order], merged_labels[order], batch_losses


def eval_loss(logits, labels, rows, dataset, batch_losses):
    if dataset == "isic2019":
        return float(np.mean(batch_losses))
    group_logits, group_labels, _ = mean_group_logits(logits, labels, rows, "study_id")
    return float(nn.functional.cross_entropy(group_logits, group_labels).item())


def prediction_rows(logits, labels, rows, dataset, run_id, variant):
    key = "study_id" if dataset == "mura" else "sample_id"
    if key == "study_id":
        logits, labels, ids = mean_group_logits(logits, labels, rows, key)
    else:
        ids = [r["sample_id"] for r in rows]
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    columns = ["run_id", "variant", "dataset", "prediction_unit", "unit_id", "label_true", "label_pred"] + [f"prob_{i}" for i in range(probabilities.shape[1])]
    output_rows = []
    for i, unit_id in enumerate(ids):
        record = {
            "run_id": run_id, "variant": variant, "dataset": dataset,
            "prediction_unit": "study" if dataset == "mura" else "image",
            "unit_id": unit_id, "label_true": int(labels[i]), "label_pred": int(predictions[i]),
        }
        record.update({f"prob_{j}": float(value) for j, value in enumerate(probabilities[i])})
        output_rows.append(record)
    return columns, output_rows


def run(args) -> None:
    if args.dataset not in DATASETS or args.model not in MODELS or args.variant not in VARIANTS:
        raise ValueError("Invalid dataset/model/variant")
    if args.variant == "swin_weight_eval" and args.model != "swin_t":
        raise ValueError("The sensitivity branch is Swin-only")
    if not args.technical_smoke:
        validate_target_environment()
        if args.training_seed not in SEEDS or args.max_epochs != 60 or args.batch_size != 32 or args.num_workers != 4:
            raise ValueError("Formal hyperparameters differ from the author-approved design")
        if not torch.cuda.is_available():
            raise RuntimeError("Formal run requires CUDA; use --technical-smoke for local checks")
    manifest_hash = validate_manifest_row(args)
    if args.output_dir.exists():
        raise FileExistsError(f"Will not overwrite an existing run: {args.output_dir}")
    if not args.dataset_index.is_file():
        raise FileNotFoundError(args.dataset_index)
    subsets = read_split(args.split_file, args.dataset, technical_smoke=args.technical_smoke)
    if args.technical_smoke:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda")
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_transform, eval_transform = build_transforms(args.dataset, args.model, args.variant)
    loaders = {
        "train": make_loader(subsets["train"], args.dataset, train_transform, args.training_seed, True, args.num_workers, args.batch_size, device),
        "validation": make_loader(subsets["validation"], args.dataset, eval_transform, args.training_seed, False, args.num_workers, args.batch_size, device),
        "test": make_loader(subsets["test"], args.dataset, eval_transform, args.training_seed, False, args.num_workers, args.batch_size, device),
    }
    classes = 8 if args.dataset == "isic2019" else 2
    model = build_model(args.model, classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-4)
    weight_enum = getattr(models.get_model_weights(args.model), "IMAGENET1K_V1")
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weight_enum.url).path).name
    if not weight_file.is_file():
        raise FileNotFoundError(weight_file)
    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    best_path = checkpoint_dir / "best_validation_loss.pt"
    start = time.perf_counter()
    identity = {
        "run_id": args.run_id, "dataset": args.dataset, "model": args.model,
        "variant": args.variant, "training_seed": args.training_seed,
        "technical_smoke": args.technical_smoke,
        "design_lock_sha256": sha256(DESIGN),
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "swin_sensitivity_decision_sha256": sha256(SCOPE_DECISION),
        "variant_sha256": sha256(VARIANTS[args.variant]),
        "split_sha256": sha256(args.split_file),
        "dataset_index_sha256": sha256(args.dataset_index),
        "script_sha256": sha256(Path(__file__)),
        "preprocessing_sha256": sha256(HERE / "preprocessing.py"),
        "environment_target_sha256": sha256(ENVIRONMENT_TARGET),
        "run_manifest_sha256": manifest_hash,
        "pretrained_weight_sha256": sha256(weight_file),
        "source_paths": {"split": str(args.split_file), "dataset_index": str(args.dataset_index)},
        "environment": {
            **runtime_versions(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "hyperparameters": {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": args.batch_size, "num_workers": args.num_workers, "max_epochs": args.max_epochs, "mixed_precision": False},
    }
    identity["protocol_hash"] = hashlib.sha256(
        json.dumps({
            "design": identity["design_lock_sha256"],
            "data_quality_amendment": identity["data_quality_amendment_sha256"],
            "swin_sensitivity_decision": identity["swin_sensitivity_decision_sha256"],
            "variant": identity["variant_sha256"],
            "runner": identity["script_sha256"],
            "preprocessing": identity["preprocessing_sha256"],
            "environment_target": identity["environment_target_sha256"],
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity["split_hash"] = identity["split_sha256"]
    identity["dataset_index_hash"] = identity["dataset_index_sha256"]
    identity["model_weight_hash"] = identity["pretrained_weight_sha256"]
    identity["script_hash"] = identity["script_sha256"]
    identity["environment_lock_hash"] = identity["environment_target_sha256"]
    write_json(args.output_dir / "run_config.json", identity)
    write_json(args.output_dir / "run_status.json", {"run_id": args.run_id, "status": "running", "started_utc": utc_now(), "technical_smoke": args.technical_smoke})
    history = []
    best_loss = math.inf
    patience_count = 0
    best_epoch = None
    try:
        for epoch in range(1, args.max_epochs + 1):
            model.train()  # same BN-running-buffer behavior as submitted CNN code
            train_losses = []
            for images, labels, _ in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().item()))
            val_logits, val_labels, val_batch_losses = collect_logits(model, loaders["validation"], device)
            validation_loss = eval_loss(val_logits, val_labels, subsets["validation"], args.dataset, val_batch_losses)
            if not math.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            improved, best_loss, patience_count, stop = rule_a_update(validation_loss, best_loss, patience_count)
            history.append({"epoch": epoch, "train_loss_batch_mean": float(np.mean(train_losses)), "validation_loss": validation_loss, "significant_improvement": improved, "patience_count": patience_count})
            write_csv(args.output_dir / "history.csv", list(history[0]), history)
            if improved:
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "run_id": args.run_id, "variant": args.variant}, best_path)
            if stop:
                break
        if best_epoch is None:
            raise RuntimeError("No Rule-A checkpoint selected")
        selected = torch.load(best_path, map_location=device, weights_only=True)
        if selected["epoch"] != best_epoch or selected["variant"] != args.variant:
            raise RuntimeError("Selected checkpoint metadata mismatch")
        model.load_state_dict(selected["model_state_dict"])
        test_logits, test_labels, test_batch_losses = collect_logits(model, loaders["test"], device)
        test_loss = eval_loss(test_logits, test_labels, subsets["test"], args.dataset, test_batch_losses)
        columns, predictions = prediction_rows(test_logits, test_labels, subsets["test"], args.dataset, args.run_id, args.variant)
        write_csv(args.output_dir / "predictions.csv", columns, predictions)
        true = [int(p["label_true"]) for p in predictions]
        pred = [int(p["label_pred"]) for p in predictions]
        metrics = {
            "run_id": args.run_id, "variant": args.variant, "dataset": args.dataset,
            "prediction_unit": "study" if args.dataset == "mura" else "image",
            "test_units": len(predictions), "test_loss": test_loss,
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "confusion_matrix": confusion_matrix(true, pred, labels=list(range(classes))).tolist(),
            "selected_epoch": best_epoch,
            "technical_smoke_not_research_result": args.technical_smoke,
        }
        if args.dataset == "isic2019":
            lesion_logits, lesion_labels, lesion_ids = mean_group_logits(test_logits, test_labels, subsets["test"], "group_id")
            lesion_pred = lesion_logits.argmax(dim=1).tolist()
            metrics["secondary_lesion_units"] = len(lesion_ids)
            metrics["secondary_lesion_balanced_accuracy"] = float(balanced_accuracy_score(lesion_labels.tolist(), lesion_pred))
        write_json(args.output_dir / "metrics.json", metrics)
        write_json(args.output_dir / "checkpoint_manifest.json", {"rule": "A", "best_epoch": best_epoch, "best_validation_loss": best_loss, "selected_checkpoint": str(best_path), "selected_checkpoint_sha256": sha256(best_path)})
        write_json(args.output_dir / "timing.json", {"elapsed_seconds": time.perf_counter() - start, "epochs_executed": len(history)})
        write_json(args.output_dir / "run_status.json", {"run_id": args.run_id, "status": "technical_smoke_completed" if args.technical_smoke else "completed", "completed_utc": utc_now(), "technical_smoke": args.technical_smoke})
    except Exception as exc:
        write_json(args.output_dir / "run_status.json", {"run_id": args.run_id, "status": "failed", "failed_utc": utc_now(), "error": repr(exc), "technical_smoke": args.technical_smoke})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
