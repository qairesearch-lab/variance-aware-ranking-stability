#!/usr/bin/env python3
"""Find exact and perceptually near image-duplicate candidates in source ZIPs.

This tool NEVER declares a near pair to be a true duplicate or alters index
groups. pHash candidates are retained as descriptive split-risk indicators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATASETS = ("isic2019", "mura")
FINGERPRINT_COLUMNS = (
    "dataset", "sample_id", "label", "group_id", "archive_member",
    "file_sha256", "decoded_rgb_sha256", "phash64_hex",
)
PAIR_COLUMNS = (
    "dataset", "sample_a", "sample_b", "group_a", "group_b",
    "label_a", "label_b", "phash_hamming_distance",
)


def hex_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_fingerprint(data: bytes) -> tuple[str, str, int]:
    file_hash = hex_digest(data)
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        image = source.convert("RGB")
        width, height = image.size
        pixel_hash = hashlib.sha256()
        pixel_hash.update(f"RGB:{width}:{height}:".encode("ascii"))
        pixel_hash.update(image.tobytes())
        gray = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    low_frequency = dctn(np.asarray(gray, dtype=np.float32), type=2, norm="ortho")[:8, :8]
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    bits = (low_frequency.reshape(-1) > median).astype(np.uint8)
    signature = 0
    for bit in bits:
        signature = (signature << 1) | int(bit)
    return file_hash, pixel_hash.hexdigest(), signature


def load_index(dataset: str, candidate_root: Path, limit: int | None):
    path = candidate_root / f"{dataset}_dataset_index_candidate.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    return rows


def archive_path(dataset: str) -> Path:
    if dataset == "isic2019":
        return ROOT / "research-lab" / "data" / "isic2019" / "raw" / "ISIC_2019_Training_Input.zip"
    return ROOT / "research-lab" / "data" / "mura" / "raw" / "MURA-v1.1_files.zip"


def exact_pairs(records, key: str, dataset: str):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    pairs = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a["group_id"] != b["group_id"]:
                    pairs.append({
                        "dataset": dataset, "kind": key, "digest": digest,
                        "sample_a": a["sample_id"], "sample_b": b["sample_id"],
                        "group_a": a["group_id"], "group_b": b["group_id"],
                        "label_a": a["label"], "label_b": b["label"],
                    })
    return pairs


def near_pairs(records, dataset: str, max_pairs: int):
    # Eight disjoint 8-bit bands guarantee that a <=4-bit Hamming neighbor has
    # at least one identical band (pigeonhole principle).
    buckets = defaultdict(list)
    results = []
    for current_index, current in enumerate(records):
        signature = int(current["phash64_hex"], 16)
        candidates = set()
        for band in range(8):
            band_value = (signature >> (band * 8)) & 255
            candidates.update(buckets[(band, band_value)])
        for earlier_index in candidates:
            prior = records[earlier_index]
            if prior["group_id"] == current["group_id"]:
                continue
            distance = (signature ^ int(prior["phash64_hex"], 16)).bit_count()
            if distance > 4:
                continue
            if prior["decoded_rgb_sha256"] == current["decoded_rgb_sha256"]:
                continue  # already in exact-pixel candidates
            results.append({
                "dataset": dataset, "sample_a": prior["sample_id"],
                "sample_b": current["sample_id"], "group_a": prior["group_id"],
                "group_b": current["group_id"], "label_a": prior["label"],
                "label_b": current["label"], "phash_hamming_distance": distance,
            })
            if len(results) > max_pairs:
                raise RuntimeError(f"Near-pair count exceeded {max_pairs}; review candidate-search scalability before continuing")
        for band in range(8):
            band_value = (signature >> (band * 8)) & 255
            buckets[(band, band_value)].append(current_index)
    return results


def write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index-root", type=Path, default=HERE / "candidate_indices")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--max-near-pairs", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Will not overwrite prior audit: {args.output_root}")
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        raise ValueError("Limit must be positive")
    args.output_root.mkdir(parents=True)
    summary = {"status": "candidate_screen_only_no_identity_adjudication", "limit_per_dataset": args.limit_per_dataset, "datasets": {}}
    for dataset in DATASETS:
        rows = load_index(dataset, args.candidate_index_root, args.limit_per_dataset)
        fingerprints = []
        with zipfile.ZipFile(archive_path(dataset)) as archive:
            for i, row in enumerate(rows, 1):
                data = archive.read(row["archive_member"])
                file_hash, pixel_hash, signature = image_fingerprint(data)
                fingerprints.append({
                    "dataset": dataset, "sample_id": row["sample_id"],
                    "label": row["label"], "group_id": row["group_id"],
                    "archive_member": row["archive_member"],
                    "file_sha256": file_hash,
                    "decoded_rgb_sha256": pixel_hash,
                    "phash64_hex": f"{signature:016x}",
                })
                if i % 1000 == 0:
                    print(f"{dataset}: fingerprinted {i}/{len(rows)}", flush=True)
        prefix = args.output_root / dataset
        write_csv(prefix.with_name(f"{dataset}_fingerprints.csv"), FINGERPRINT_COLUMNS, fingerprints)
        exact_file = exact_pairs(fingerprints, "file_sha256", dataset)
        exact_pixel = exact_pairs(fingerprints, "decoded_rgb_sha256", dataset)
        near = near_pairs(fingerprints, dataset, args.max_near_pairs)
        exact_columns = ("dataset", "kind", "digest", "sample_a", "sample_b", "group_a", "group_b", "label_a", "label_b")
        write_csv(prefix.with_name(f"{dataset}_exact_pairs.csv"), exact_columns, exact_file + exact_pixel)
        write_csv(prefix.with_name(f"{dataset}_near_pairs.csv"), PAIR_COLUMNS, near)
        summary["datasets"][dataset] = {
            "images_fingerprinted": len(fingerprints),
            "cross_group_exact_file_pairs": len(exact_file),
            "cross_group_exact_pixel_pairs": len(exact_pixel),
            "cross_group_near_phash_pairs_excluding_exact_pixel": len(near),
            "exact_label_conflict_pairs": sum(r["label_a"] != r["label_b"] for r in exact_pixel),
            "near_label_conflict_candidates": sum(r["label_a"] != r["label_b"] for r in near),
        }
        print(dataset, json.dumps(summary["datasets"][dataset]), flush=True)
    (args.output_root / "candidate_duplicate_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
