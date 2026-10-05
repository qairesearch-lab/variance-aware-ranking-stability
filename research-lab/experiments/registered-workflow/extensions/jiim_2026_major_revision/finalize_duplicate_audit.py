#!/usr/bin/env python3
"""Seal objective exact-image links and retain pHash candidates as risk flags.

Published labels and IDs remain authoritative. pHash candidates do not alter
the cohort, labels or split groups; no image-identity annotation is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from protocol_amendment import AMENDMENT, load_amendment

HERE = Path(__file__).resolve().parent
DATASETS = ("isic2019", "mura")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class UnionFind:
    def __init__(self, group_ids):
        self.parent = {group_id: group_id for group_id in group_ids}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            canonical = min(root_a, root_b)
            self.parent[root_a] = canonical
            self.parent[root_b] = canonical


def pair_key(row):
    return tuple(sorted((row["sample_a"], row["sample_b"])))


def candidate_id(dataset: str, row) -> str:
    joined = ":".join((dataset, *pair_key(row)))
    return f"{dataset}_{hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]}"


def finalize(candidate_root: Path, index_root: Path, output_root: Path):
    if output_root.exists():
        raise FileExistsError(output_root)
    candidate_summary = json.loads((candidate_root / "candidate_duplicate_audit.json").read_text(encoding="utf-8"))
    if candidate_summary.get("limit_per_dataset") is not None:
        raise ValueError("Cannot finalize a limited/sample audit")
    amendment = load_amendment()
    isic_decision = amendment["isic2019"]
    screen = amendment["near_duplicate_screen"]
    for dataset in DATASETS:
        near_candidates = read_csv(candidate_root / f"{dataset}_near_pairs.csv")
        if len(near_candidates) != screen["detected_candidate_pairs_before_exclusion"][dataset]:
            raise ValueError(f"Near-pair candidate count changed: {dataset}")
        if any(int(pair["phash_hamming_distance"]) > screen["maximum_hamming_distance_inclusive"] for pair in near_candidates):
            raise ValueError(f"Near-pair distance exceeds approved pHash threshold: {dataset}")
    final_indices = {}
    merge_records = []
    excluded_records = []
    exact_counts = {}
    near_counts = {}
    candidate_hashes = {}
    for dataset in DATASETS:
        index_path = index_root / f"{dataset}_dataset_index_candidate.csv"
        rows = read_csv(index_path)
        if len(rows) != candidate_summary["datasets"][dataset]["images_fingerprinted"]:
            raise ValueError(f"Fingerprint coverage mismatch: {dataset}")
        if dataset == "isic2019":
            if len(rows) != isic_decision["source_known_lesion_id_images"] or len({r["group_id"] for r in rows}) != isic_decision["source_known_lesion_id_groups"]:
                raise ValueError("ISIC source cohort differs from approved amendment")
            excluded = [r for r in rows if r["group_id"] in set(isic_decision["excluded_group_ids"])]
            if (len(excluded) != isic_decision["excluded_image_count"]
                    or {r["sample_id"] for r in excluded} != set(isic_decision["excluded_sample_ids"])
                    or {r["group_id"] for r in excluded} != set(isic_decision["excluded_group_ids"])):
                raise ValueError("ISIC complete-lesion exclusion does not match approved samples")
            label_counts = {name: sum(r["label"] == str(label) for r in excluded)
                            for name, label in (("MEL", 0), ("NV", 1))}
            if label_counts != isic_decision["excluded_original_label_counts"]:
                raise ValueError("ISIC exclusion label composition changed")
            excluded_records = excluded
            rows = [r for r in rows if r["group_id"] not in set(isic_decision["excluded_group_ids"])]
            if len(rows) != isic_decision["expected_images_after_exclusion"]:
                raise ValueError("ISIC remaining cohort size mismatch")
        by_sample = {r["sample_id"]: r for r in rows}
        uf = UnionFind({r["group_id"] for r in rows})
        fingerprints_path = candidate_root / f"{dataset}_fingerprints.csv"
        exact_path = candidate_root / f"{dataset}_exact_pairs.csv"
        near_path = candidate_root / f"{dataset}_near_pairs.csv"
        fingerprints = read_csv(fingerprints_path)
        if len(fingerprints) != candidate_summary["datasets"][dataset]["images_fingerprinted"]:
            raise ValueError(f"Fingerprint file coverage mismatch: {dataset}")
        fingerprint_by_sample = {r["sample_id"]: r for r in fingerprints}
        if len(fingerprint_by_sample) != len(fingerprints):
            raise ValueError(f"Repeated fingerprint sample ID: {dataset}")
        source_by_sample = {r["sample_id"]: r for r in read_csv(index_path)}
        if set(fingerprint_by_sample) != set(source_by_sample):
            raise ValueError(f"Fingerprint/index sample mismatch: {dataset}")
        for fingerprint in fingerprints:
            source = source_by_sample[fingerprint["sample_id"]]
            if any(fingerprint[field] != source[field] for field in ("dataset", "label", "group_id", "archive_member")):
                raise ValueError(f"Fingerprint/index metadata mismatch: {dataset}/{fingerprint['sample_id']}")
        exact = read_csv(exact_path)
        seen_exact = set()
        for pair in exact:
            first, second = pair_key(pair)
            if first not in fingerprint_by_sample or second not in fingerprint_by_sample or first == second:
                raise ValueError(f"Exact pair has unknown samples: {dataset}/{pair_key(pair)}")
            kind = pair["kind"]
            if kind not in ("file_sha256", "decoded_rgb_sha256"):
                raise ValueError(f"Unknown exact hash kind: {dataset}/{kind}")
            a, b = fingerprint_by_sample[pair["sample_a"]], fingerprint_by_sample[pair["sample_b"]]
            if (a[kind] != b[kind] or pair["digest"] != a[kind]
                    or pair["label_a"] != a["label"] or pair["label_b"] != b["label"]):
                raise ValueError(f"Unverified exact-image pair: {dataset}/{pair_key(pair)}")
            if (pair["group_a"], pair["group_b"]) != (a["group_id"], b["group_id"]) or a["group_id"] == b["group_id"]:
                raise ValueError(f"Exact pair group metadata mismatch: {dataset}/{pair_key(pair)}")
            key = (kind, first, second)
            if key in seen_exact:
                raise ValueError(f"Repeated exact pair: {dataset}/{key}")
            seen_exact.add(key)
            pair_in_index = [sample in by_sample for sample in pair_key(pair)]
            if pair_in_index == [False, False] and dataset == "isic2019" and set(pair_key(pair)).issubset(set(isic_decision["excluded_sample_ids"])):
                continue
            if pair_in_index != [True, True]:
                raise ValueError(f"Exact pair has an unapproved exclusion: {dataset}/{pair_key(pair)}")
            if pair["label_a"] != pair["label_b"]:
                raise ValueError(f"Exact duplicate with label conflict: {dataset}/{pair_key(pair)}")
            if pair["sample_a"] not in by_sample or pair["sample_b"] not in by_sample:
                raise ValueError("Exact pair absent from index")
            uf.union(pair["group_a"], pair["group_b"])
        # Ensure every cross-ID identical file/pixel pair was represented.
        expected_exact = set()
        for kind in ("file_sha256", "decoded_rgb_sha256"):
            by_digest = {}
            for fingerprint in fingerprints:
                by_digest.setdefault(fingerprint[kind], []).append(fingerprint)
            for members in by_digest.values():
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        if a["group_id"] != b["group_id"]:
                            expected_exact.add((kind, *sorted((a["sample_id"], b["sample_id"]))))
        if seen_exact != expected_exact:
            raise ValueError(f"Exact-image candidate file is incomplete or inconsistent: {dataset}")
        exact_counts[dataset] = len({pair_key(pair) for pair in exact if all(sample in by_sample for sample in pair_key(pair))})
        near = read_csv(near_path)
        if dataset == "isic2019":
            touching = [pair for pair in near if pair["sample_a"] not in by_sample or pair["sample_b"] not in by_sample]
            if len(touching) != screen["candidate_pairs_touching_excluded_isic_groups"]:
                raise ValueError("Near-pair candidates touch excluded ISIC groups; review protocol needs amendment")
        seen_near = set()
        for pair in near:
            key = pair_key(pair)
            if key in seen_near:
                raise ValueError(f"Repeated pHash candidate pair: {dataset}/{key}")
            seen_near.add(key)
            if any(pair[sample_key] not in by_sample for sample_key in ("sample_a", "sample_b")):
                raise ValueError(f"pHash candidate outside final cohort: {dataset}/{key}")
            if any(pair[f"group_{side}"] != by_sample[pair[f"sample_{side}"]]["group_id"] or
                   pair[f"label_{side}"] != by_sample[pair[f"sample_{side}"]]["label"] for side in ("a", "b")):
                raise ValueError(f"pHash candidate metadata mismatch: {dataset}/{key}")
            a, b = fingerprint_by_sample[pair["sample_a"]], fingerprint_by_sample[pair["sample_b"]]
            distance = (int(a["phash64_hex"], 16) ^ int(b["phash64_hex"], 16)).bit_count()
            if (pair["dataset"] != dataset or pair["group_a"] == pair["group_b"]
                    or distance != int(pair["phash_hamming_distance"])
                    or a["decoded_rgb_sha256"] == b["decoded_rgb_sha256"]):
                raise ValueError(f"pHash candidate does not match fingerprints: {dataset}/{key}")
        near_counts[dataset] = len(near)
        candidate_hashes[dataset] = {name: sha256(path) for name, path in (("fingerprints", fingerprints_path), ("exact_pairs", exact_path), ("near_pairs", near_path))}
        original_groups = {r["group_id"] for r in rows}
        mapping = {group_id: uf.find(group_id) for group_id in original_groups}
        # An exact-image component may join several groups. Do not rewrite labels.
        labels_by_merged_group = {}
        for row in rows:
            merged = mapping[row["group_id"]]
            previous = labels_by_merged_group.setdefault(merged, row["label"])
            if dataset == "isic2019" and previous != row["label"]:
                raise ValueError(f"Merged ISIC lesion group has conflicting labels: {merged}")
            row["group_id"] = merged
        final_indices[dataset] = (list(rows[0]), rows)
        for original, merged in sorted(mapping.items()):
            if original != merged:
                merge_records.append({"dataset": dataset, "original_group_id": original, "merged_group_id": merged})
    output_root.mkdir(parents=True)
    hashes = {}
    for dataset, (columns, rows) in final_indices.items():
        path = output_root / f"{dataset}_dataset_index.csv"
        write_csv(path, columns, rows)
        hashes[dataset] = sha256(path)
        near = read_csv(candidate_root / f"{dataset}_near_pairs.csv")
        write_csv(output_root / f"{dataset}_near_risk_pairs.csv", list(near[0]), near)
    write_csv(output_root / "group_merges.csv", ("dataset", "original_group_id", "merged_group_id"), merge_records)
    write_csv(output_root / "isic2019_excluded_records.csv", list(excluded_records[0]), excluded_records)
    seal = {
        "status": "objective_data_quality_passed",
        "data_quality_policy_id": amendment["amendment_id"],
        "dataset_index_sha256": hashes,
        "source_candidate_audit_sha256": sha256(candidate_root / "candidate_duplicate_audit.json"),
        "source_candidate_file_sha256": candidate_hashes,
        "near_risk_file_sha256": {dataset: sha256(output_root / f"{dataset}_near_risk_pairs.csv") for dataset in DATASETS},
        "near_duplicate_role": screen["decision_rule"],
        "exact_duplicate_group_merges_complete": True,
        "exact_cross_group_pairs_in_final_cohort": exact_counts,
        "near_pairs_screened": sum(near_counts.values()),
        "near_pairs_by_dataset": near_counts,
        "groups_reassigned": len(merge_records),
        "data_quality_amendment_sha256": sha256(AMENDMENT),
        "isic2019_excluded_group_ids": isic_decision["excluded_group_ids"],
        "isic2019_excluded_image_count": len(excluded_records),
        "isic2019_excluded_records_sha256": sha256(output_root / "isic2019_excluded_records.csv"),
        "near_duplicate_phash_max_distance_inclusive": screen["maximum_hamming_distance_inclusive"],
    }
    (output_root / "data_quality_audit.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(seal, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=HERE / "candidate_indices")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.candidate_root, args.index_root, args.output_root)


if __name__ == "__main__":
    main()
