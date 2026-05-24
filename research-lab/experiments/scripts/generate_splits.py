#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


SPLIT_COLUMNS = [
    "dataset",
    "split_id",
    "sample_id",
    "relative_path",
    "label",
    "subset",
    "split_seed",
]

SIPAKMED_LABEL_MAP = {
    "im_Parabasal": 0,
    "im_Superficial-Intermediate": 1,
    "im_Koilocytotic": 2,
    "im_Dyskeratotic": 3,
    "im_Metaplastic": 4,
}

ORGANAMNIST_OFFICIAL_SPLITS = ["train", "val", "test"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_manifest(path):
    text = Path(path).read_text(encoding="utf-8")
    manifest = {}
    for block in re.findall(r"```yaml\n(.*?)\n```", text, flags=re.DOTALL):
        data = yaml.safe_load(block)
        if isinstance(data, dict) and "dataset_id" in data:
            manifest[data["dataset_id"]] = data
    return manifest


def fail(message):
    raise RuntimeError(message)


def get_dataset_config(datasets_config, dataset_id):
    for dataset in datasets_config.get("primary_datasets", []):
        if dataset.get("dataset_id") == dataset_id:
            return dataset
    fail(f"Missing dataset in datasets.yaml: {dataset_id}")


def expected_split_plan(split_plan_config):
    design = split_plan_config["split_design"]
    split_ids = design["split_ids"]
    expected_ids = [item["id"] for item in split_ids]
    expected_seeds = [item["generation_seed"] for item in split_ids]
    return design, expected_ids, expected_seeds


def validate_frozen_config(datasets_config, seeds_config, split_plan_config, dataset_manifest):
    design, expected_ids, expected_seeds = expected_split_plan(split_plan_config)
    seed_values = seeds_config["split_generation_seeds"]["values"]
    if seed_values != expected_seeds:
        fail(f"random_seeds.yaml split seeds differ from split_plan.yaml: {seed_values} != {expected_seeds}")
    if design["type"] != "repeated_random_holdout":
        fail(f"Unexpected split design: {design['type']}")
    if design["number_of_splits"] != 10 or len(expected_ids) != 10:
        fail("Frozen split plan must define exactly 10 splits")
    if design["split_ratio"] != {"train": 0.70, "validation": 0.15, "test": 0.15}:
        fail(f"Unexpected split ratio: {design['split_ratio']}")
    if design["stratification"] != "class_label":
        fail(f"Unexpected stratification: {design['stratification']}")

    for dataset_id in ["sipakmed", "organamnist"]:
        dataset_config = get_dataset_config(datasets_config, dataset_id)
        manifest = dataset_manifest.get(dataset_id)
        if not manifest:
            fail(f"Missing dataset_manifest.md YAML block for {dataset_id}")
        if dataset_config["expected_samples"] != manifest["expected_samples"]:
            fail(f"{dataset_id} expected sample count differs between datasets.yaml and dataset_manifest.md")
        if dataset_config["grouping_policy"]["group_id_available"] or manifest["group_id_available"]:
            fail(f"{dataset_id} unexpectedly has group IDs; image-level split would violate frozen policy")
        if manifest["primary_split_policy"] != "generated_image_level_repeated_random_holdout":
            fail(f"{dataset_id} manifest split policy is not generated repeated random holdout")

    organ_config = get_dataset_config(datasets_config, "organamnist")
    organ_manifest = dataset_manifest["organamnist"]
    if organ_config["official_split_used_for_primary_analysis"]:
        fail("datasets.yaml says OrganAMNIST official split is used for primary analysis")
    if organ_manifest["official_split_used_for_primary_analysis"]:
        fail("dataset_manifest.md says OrganAMNIST official split is used for primary analysis")

    return expected_ids, expected_seeds


def build_sipakmed_index(data_dir):
    samples = []
    data_dir = Path(data_dir)
    for class_dir_name in sorted(SIPAKMED_LABEL_MAP):
        class_dir = data_dir / class_dir_name / "CROPPED"
        if not class_dir.is_dir():
            fail(f"Missing SIPaKMeD CROPPED directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.bmp")):
            relative_path = Path(class_dir_name) / "CROPPED" / image_path.name
            samples.append(
                {
                    "sample_id": f"sipakmed_{len(samples)}",
                    "relative_path": relative_path.as_posix(),
                    "label": SIPAKMED_LABEL_MAP[class_dir_name],
                }
            )
    return pd.DataFrame(samples)


def build_organamnist_index(data_dir):
    samples = []
    data_dir = Path(data_dir)
    for split in ORGANAMNIST_OFFICIAL_SPLITS:
        split_dir = data_dir / split / "organamnist"
        csv_path = data_dir / split / "organamnist.csv"
        if not csv_path.exists():
            fail(f"Missing OrganAMNIST CSV: {csv_path}")
        if not split_dir.is_dir():
            fail(f"Missing OrganAMNIST image directory: {split_dir}")
        df = pd.read_csv(csv_path, header=None, names=["source_split", "filename", "label"])
        for row in df.itertuples(index=False):
            image_path = split_dir / row.filename
            if not image_path.exists():
                fail(f"OrganAMNIST CSV references missing PNG: {image_path}")
            samples.append(
                {
                    "sample_id": f"organamnist_{len(samples)}",
                    "relative_path": Path(split, "organamnist", row.filename).as_posix(),
                    "label": int(row.label),
                }
            )
    return pd.DataFrame(samples)


def validate_sample_index(df, dataset_id, expected_samples, expected_classes):
    if len(df) != expected_samples:
        fail(f"{dataset_id} sample count mismatch: {len(df)} != {expected_samples}")
    if df["sample_id"].duplicated().any():
        fail(f"{dataset_id} sample_id values are not unique")
    if df["relative_path"].duplicated().any():
        fail(f"{dataset_id} relative_path values are not unique")
    observed_classes = sorted(df["label"].unique().tolist())
    if observed_classes != list(range(expected_classes)):
        fail(f"{dataset_id} label set mismatch: {observed_classes} != {list(range(expected_classes))}")


def subset_class_counts(split_df, subset):
    counts = (
        split_df[split_df["subset"] == subset]["label"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    return {str(k): v for k, v in counts.items()}


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_split_df(split_df, dataset_id, split_id, split_seed, expected_samples, allowed_subsets):
    if len(split_df) != expected_samples:
        fail(f"{dataset_id} {split_id} row count mismatch: {len(split_df)} != {expected_samples}")
    if split_df["sample_id"].duplicated().any():
        fail(f"{dataset_id} {split_id} has duplicated sample_id values")
    if split_df["relative_path"].duplicated().any():
        fail(f"{dataset_id} {split_id} has duplicated relative_path values")
    if set(split_df["subset"]) != set(allowed_subsets):
        fail(f"{dataset_id} {split_id} subset values differ from split_plan.yaml")
    if set(split_df["split_seed"].astype(int)) != {int(split_seed)}:
        fail(f"{dataset_id} {split_id} split_seed mismatch")
    if set(split_df["split_id"]) != {split_id}:
        fail(f"{dataset_id} {split_id} split_id mismatch")
    if set(split_df["dataset"]) != {dataset_id}:
        fail(f"{dataset_id} {split_id} dataset column mismatch")


def generate_splits(df, dataset_id, split_ids, split_seeds, output_dir, split_ratio, expected_samples, allowed_subsets):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = []
    test_size = split_ratio["test"]
    val_size = split_ratio["validation"]
    for split_id, seed in zip(split_ids, split_seeds):
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=seed,
        )
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val["label"],
            random_state=seed,
        )

        train = train.copy()
        val = val.copy()
        test = test.copy()
        train["subset"] = "train"
        val["subset"] = "validation"
        test["subset"] = "test"

        split_df = pd.concat([train, val, test], ignore_index=True)
        split_df["dataset"] = dataset_id
        split_df["split_id"] = split_id
        split_df["split_seed"] = seed
        split_df = split_df[SPLIT_COLUMNS]
        validate_split_df(split_df, dataset_id, split_id, seed, expected_samples, allowed_subsets)

        output_path = output_dir / f"{split_id}.csv"
        split_df.to_csv(output_path, index=False, lineterminator="\n", encoding="utf-8")
        splits.append(split_df)
        print(f"Generated {output_path}")
    return splits


def generate_audit_summary(splits, dataset_id, output_path):
    audit_rows = []
    for split_df in splits:
        subset_counts = split_df["subset"].value_counts().to_dict()
        class_counts = split_df["label"].value_counts().sort_index().astype(int).to_dict()
        class_counts = {str(k): v for k, v in class_counts.items()}
        train_counts = subset_class_counts(split_df, "train")
        validation_counts = subset_class_counts(split_df, "validation")
        test_counts = subset_class_counts(split_df, "test")
        warnings = []
        if not train_counts or not validation_counts or not test_counts:
            warnings.append("missing_subset_class_counts")
        audit_rows.append(
            {
                "dataset": dataset_id,
                "split_id": split_df["split_id"].iloc[0],
                "split_seed": int(split_df["split_seed"].iloc[0]),
                "total_samples": len(split_df),
                "train_count": int(subset_counts.get("train", 0)),
                "validation_count": int(subset_counts.get("validation", 0)),
                "test_count": int(subset_counts.get("test", 0)),
                "class_counts": stable_json(class_counts),
                "train_class_counts": stable_json(train_counts),
                "validation_class_counts": stable_json(validation_counts),
                "test_class_counts": stable_json(test_counts),
                "warnings": ";".join(warnings),
            }
        )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Generated audit summary: {output_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    frozen_dir = script_dir.parent / "registered-workflow" / "configs" / "frozen"
    data_dir = script_dir.parent.parent / "data"
    output_base = script_dir / "splits"

    datasets_config = load_yaml(frozen_dir / "datasets.yaml")
    seeds_config = load_yaml(frozen_dir / "random_seeds.yaml")
    split_plan_config = load_yaml(frozen_dir / "split_plan.yaml")
    dataset_manifest = load_dataset_manifest(frozen_dir / "dataset_manifest.md")
    split_ids, split_seeds = validate_frozen_config(
        datasets_config, seeds_config, split_plan_config, dataset_manifest
    )
    split_ratio = split_plan_config["split_design"]["split_ratio"]
    allowed_subsets = split_plan_config["split_file_schema"]["allowed_subset_values"]

    print("Processing SIPaKMeD...")
    sipakmed_config = get_dataset_config(datasets_config, "sipakmed")
    sipakmed_df = build_sipakmed_index(data_dir / "sipakmed" / "raw")
    validate_sample_index(
        sipakmed_df,
        "sipakmed",
        sipakmed_config["expected_samples"],
        sipakmed_config["classes"],
    )
    print(f"Found {len(sipakmed_df)} samples for SIPaKMeD")
    sipakmed_splits = generate_splits(
        sipakmed_df,
        "sipakmed",
        split_ids,
        split_seeds,
        output_base / "sipakmed",
        split_ratio,
        sipakmed_config["expected_samples"],
        allowed_subsets,
    )
    generate_audit_summary(sipakmed_splits, "sipakmed", output_base / "sipakmed_audit_summary.csv")

    print("Processing OrganAMNIST...")
    organ_config = get_dataset_config(datasets_config, "organamnist")
    organamnist_df = build_organamnist_index(data_dir / "organamnist")
    validate_sample_index(
        organamnist_df,
        "organamnist",
        organ_config["expected_samples"],
        organ_config["classes"],
    )
    print(f"Found {len(organamnist_df)} samples for OrganAMNIST")
    organamnist_splits = generate_splits(
        organamnist_df,
        "organamnist",
        split_ids,
        split_seeds,
        output_base / "organamnist",
        split_ratio,
        organ_config["expected_samples"],
        allowed_subsets,
    )
    generate_audit_summary(organamnist_splits, "organamnist", output_base / "organamnist_audit_summary.csv")

    total_audit_path = output_base / "split_audit_summary.csv"
    total_audit = pd.concat(
        [
            pd.read_csv(output_base / "sipakmed_audit_summary.csv"),
            pd.read_csv(output_base / "organamnist_audit_summary.csv"),
        ],
        ignore_index=True,
    )
    total_audit.to_csv(total_audit_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Generated total audit summary: {total_audit_path}")


if __name__ == "__main__":
    main()
