#!/usr/bin/env python3
"""Export MedMNIST OrganAMNIST .npz into the frozen PNG/CSV layout.

Expected input keys:
  train_images, train_labels, val_images, val_labels, test_images, test_labels

Expected output layout:
  research-lab/data/organamnist/
    train/organamnist/train0_6.png
    train/organamnist.csv
    val/organamnist/val0_4.png
    val/organamnist.csv
    test/organamnist/test0_7.png
    test/organamnist.csv

The registered split files refer to these relative paths, so this export keeps
the existing split hashes, manifest, and training loader usable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")


def label_at(labels: np.ndarray, index: int) -> int:
    value = labels[index]
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    return int(value)


def image_from_array(array: np.ndarray) -> Image.Image:
    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    if image.ndim == 3 and image.shape[-1] == 1:
        return Image.fromarray(image[:, :, 0], mode="L")
    if image.ndim == 3 and image.shape[-1] == 3:
        return Image.fromarray(image, mode="RGB")
    raise ValueError(f"Unsupported image shape: {image.shape}")


def export_split(npz: np.lib.npyio.NpzFile, split: str, output_root: Path, overwrite: bool) -> tuple[int, int]:
    images = npz[f"{split}_images"]
    labels = npz[f"{split}_labels"]
    if len(images) != len(labels):
        raise ValueError(f"{split}: image/label count mismatch: {len(images)} != {len(labels)}")

    split_dir = output_root / split
    image_dir = split_dir / "organamnist"
    csv_path = split_dir / "organamnist.csv"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(len(images)):
        label = label_at(labels, index)
        filename = f"{split}{index}_{label}.png"
        image_path = image_dir / filename
        if image_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {image_path}")
        image_from_array(images[index]).save(image_path)
        rows.append([split, filename, label])

    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {csv_path}")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)

    return len(images), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        required=True,
        type=Path,
        help="Path to organamnist.npz downloaded from MedMNIST.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research-lab/data/organamnist"),
        help="Output root matching the frozen workflow data layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG/CSV files.",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)

    with np.load(args.npz) as npz:
        missing = [f"{split}_{kind}" for split in SPLITS for kind in ("images", "labels") if f"{split}_{kind}" not in npz]
        if missing:
            raise KeyError(f"Missing expected npz keys: {missing}")

        summary = {}
        for split in SPLITS:
            image_count, row_count = export_split(npz, split, args.output_root, args.overwrite)
            summary[split] = {"images": image_count, "csv_rows": row_count}

    total = sum(item["images"] for item in summary.values())
    print({"output_root": str(args.output_root), "summary": summary, "total_images": total})
    if total != 58830:
        print("WARNING: total image count is not 58830; verify MedMNIST version and dataset file.")


if __name__ == "__main__":
    main()
