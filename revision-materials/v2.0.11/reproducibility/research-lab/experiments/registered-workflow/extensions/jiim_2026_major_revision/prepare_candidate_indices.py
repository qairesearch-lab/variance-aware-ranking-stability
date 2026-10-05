#!/usr/bin/env python3
"""Metadata/ZIP-membership index, before duplicate adjudication and split freeze.

No source archive is extracted or modified. Outputs are explicitly candidate
indices and cannot be used as a substitute for the image-duplicate audit.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "candidate_indices"
INDEX_COLUMNS = (
    "dataset", "sample_id", "relative_path", "label", "group_id",
    "study_id", "body_region", "archive_member", "source_partition",
)
ISIC_CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def index_isic() -> dict:
    data = ROOT / "research-lab" / "data" / "isic2019"
    metadata = read_dict_csv(data / "metadata" / "ISIC_2019_Training_Metadata.csv")
    labels = read_dict_csv(data / "metadata" / "ISIC_2019_Training_GroundTruth.csv")
    metadata_by_id = {r["image"]: r for r in metadata}
    if len(metadata_by_id) != len(metadata) or len(labels) != 25331:
        raise ValueError("ISIC metadata/ground-truth IDs are duplicated or count changed")
    zip_path = data / "raw" / "ISIC_2019_Training_Input.zip"
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
    rows = []
    excluded = Counter()
    lesion_labels = defaultdict(set)
    for entry in labels:
        image_id = entry["image"]
        if image_id not in metadata_by_id:
            raise ValueError(f"Missing ISIC metadata: {image_id}")
        positives = [i for i, cls in enumerate(ISIC_CLASSES) if float(entry[cls]) == 1.0]
        if len(positives) != 1 or float(entry["UNK"]) != 0.0:
            raise ValueError(f"Invalid known ISIC class: {image_id}")
        label = positives[0]
        lesion_id = metadata_by_id[image_id]["lesion_id"].strip()
        if not lesion_id:
            excluded[ISIC_CLASSES[label]] += 1
            continue
        member = f"ISIC_2019_Training_Input/{image_id}.jpg"
        # Official archive also has some *_downsampled JPEG names. Resolve
        # strictly by image_id and fail if neither is present.
        if member not in members:
            member = f"ISIC_2019_Training_Input/{image_id}_downsampled.jpg"
        if member not in members:
            raise ValueError(f"No archive image for ISIC ID {image_id}")
        lesion_labels[lesion_id].add(label)
        rows.append({
            "dataset": "isic2019", "sample_id": image_id,
            "relative_path": f"research-lab/data/isic2019/extracted/{member}",
            "label": str(label), "group_id": lesion_id, "study_id": "",
            "body_region": "", "archive_member": member, "source_partition": "official_train",
        })
    if len(rows) != 23247 or sum(excluded.values()) != 2084:
        raise ValueError("ISIC primary cohort does not match the author-approved counts")
    if len(lesion_labels) != 11847:
        raise ValueError("Unexpected ISIC lesion group count")
    write_index(OUT / "isic2019_dataset_index_candidate.csv", rows)
    return {
        "images_in_ground_truth": len(labels), "primary_known_lesion_images": len(rows),
        "excluded_missing_lesion_id": sum(excluded.values()),
        "excluded_class_counts": dict(excluded), "lesion_groups_before_duplicate_merges": len(lesion_labels),
        "lesion_ids_with_conflicting_labels": sum(len(classes) > 1 for classes in lesion_labels.values()),
        "status": "candidate_before_image_duplicate_adjudication",
    }


def read_headerless(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def index_mura() -> dict:
    data = ROOT / "research-lab" / "data" / "mura"
    metadata = data / "metadata"
    studies = {}
    for partition in ("train", "valid"):
        for path, label in read_headerless(metadata / f"{partition}_labeled_studies.csv"):
            clean_path = path.removeprefix("MURA-v1.1/").rstrip("/")
            if clean_path in studies:
                raise ValueError(f"Repeated MURA study: {clean_path}")
            if label not in ("0", "1"):
                raise ValueError(f"Invalid MURA label: {label}")
            studies[clean_path] = label
    archive = data / "raw" / "MURA-v1.1_files.zip"
    with zipfile.ZipFile(archive) as handle:
        members = set(handle.namelist())
    rows = []
    for partition in ("train", "valid"):
        for (source_path,) in read_headerless(metadata / f"{partition}_image_paths.csv"):
            member = source_path.removeprefix("MURA-v1.1/")
            if any(component.startswith("._") for component in Path(member).parts):
                continue
            parts = Path(member).parts
            if len(parts) != 5 or parts[0] != partition:
                raise ValueError(f"Unexpected MURA path: {member}")
            study_id = "/".join(parts[:4])
            if study_id not in studies or member not in members:
                raise ValueError(f"MURA image/study missing: {member}")
            rows.append({
                "dataset": "mura", "sample_id": member,
                "relative_path": f"research-lab/data/mura/extracted/{member}",
                "label": studies[study_id], "group_id": parts[2],
                "study_id": study_id, "body_region": parts[1],
                "archive_member": member, "source_partition": f"official_{partition}",
            })
    if len(rows) != 40005 or len(studies) != 14656:
        raise ValueError("MURA corpus count differs from design lock")
    patients = {r["group_id"] for r in rows}
    if len(patients) != 11967:
        raise ValueError("Unexpected MURA patient count")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("MURA duplicate image ID")
    write_index(OUT / "mura_dataset_index_candidate.csv", rows)
    return {
        "valid_images": len(rows), "patients": len(patients), "studies": len(studies),
        "body_regions": dict(Counter(r["body_region"] for r in rows)),
        "official_partition_counts": dict(Counter(r["source_partition"] for r in rows)),
        "status": "candidate_before_image_duplicate_adjudication",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"isic2019": index_isic(), "mura": index_mura()}
    (OUT / "candidate_index_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
