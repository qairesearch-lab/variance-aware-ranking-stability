"""Author-confirmed data-quality amendment shared by formal pipeline stages."""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "configs" / "data_quality_amendment_v0.6.yaml"
SCOPE_DECISION = HERE / "configs" / "swin_sensitivity_decision_v0.5.yaml"
ANALYSIS_SCOPE_DECISION = HERE / "configs" / "analysis_scope_decision_v0.7.yaml"


def load_amendment():
    value = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    if value.get("status") != "AUTHOR_CONFIRMED_2026_09_28_OBJECTIVE_AUDIT":
        raise ValueError("Data-quality amendment status changed")
    if value.get("base_design_lock") != "extension_protocol_design_locked_v0.3.yaml":
        raise ValueError("Data-quality amendment refers to a different design lock")
    if value["near_duplicate_screen"]["maximum_hamming_distance_inclusive"] != 4:
        raise ValueError("Near-duplicate threshold is not the author-confirmed pHash <= 4")
    if value["near_duplicate_screen"]["decision_rule"] != "descriptive_risk_screen_only_no_manual_same_source_adjudication":
        raise ValueError("Near-duplicate policy changed")
    if value["near_duplicate_screen"]["alter_labels_inclusion_or_split_groups_from_phash"] is not False:
        raise ValueError("pHash candidates must not alter the formal cohort or split groups")
    return value


def load_scope_decision():
    value = yaml.safe_load(SCOPE_DECISION.read_text(encoding="utf-8"))
    if (value.get("status") != "AUTHOR_CONFIRMED_BEFORE_FORMAL_RUNS"
            or value.get("primary_shared_preprocessing_runs") != 270
            or value.get("swin_weight_specific_eval_sensitivity_runs") != 30
            or value.get("total_manifest_runs") != 300
            or value.get("include_sensitivity_in_initial_manifest") is not True):
        raise ValueError("Swin sensitivity scope differs from the author-confirmed 300-run decision")
    return value


def load_analysis_scope_decision(sha256):
    """Validate the post-training scope against the already sealed inputs."""
    value = yaml.safe_load(ANALYSIS_SCOPE_DECISION.read_text(encoding="utf-8"))
    manifest = HERE / "run_manifests" / "jiim_extension_300_v0.6.csv"
    if (value.get("status") != "AUTHOR_CONFIRMED_BEFORE_FORMAL_OUTCOMES"
            or value.get("applies_to_manifest") != manifest.name
            or value.get("applies_to_manifest_sha256") != sha256(manifest)
            or value.get("applies_to_data_quality_amendment") != AMENDMENT.name
            or value.get("applies_to_data_quality_amendment_sha256") != sha256(AMENDMENT)
            or value.get("total_sealed_extension_runs") != load_scope_decision()["total_manifest_runs"]):
        raise ValueError("Analysis scope is not bound to the sealed 300-run inputs")
    screen = value["phash_candidate_screen"]
    if (screen.get("max_hamming_distance_inclusive") != 4
            or screen.get("role") != "descriptive_cross_partition_risk_reporting_only"
            or screen.get("remove_test_images_or_studies_by_phash") is not False
            or screen.get("rerank_on_phash_filtered_test_set") is not False
            or screen.get("additional_gpu_runs") != 0):
        raise ValueError("pHash-filtered test analysis is not author-approved")
    return value


def expected_images(dataset: str, *, diagnostic_candidate: bool = False) -> int:
    if dataset == "mura":
        return 40005
    if dataset != "isic2019":
        raise ValueError(f"Unknown dataset: {dataset}")
    isic = load_amendment()["isic2019"]
    return isic["source_known_lesion_id_images" if diagnostic_candidate else "expected_images_after_exclusion"]


def validate_final_audit(audit: dict, sha256) -> None:
    amendment = load_amendment()
    if audit.get("status") != "objective_data_quality_passed":
        raise ValueError("Data-quality audit is not the objective-policy audit")
    if audit.get("data_quality_policy_id") != amendment["amendment_id"]:
        raise ValueError("Data-quality audit policy ID mismatch")
    if audit.get("data_quality_amendment_sha256") != sha256(AMENDMENT):
        raise ValueError("Data-quality audit/amendment hash mismatch")
    if audit.get("isic2019_excluded_group_ids") != amendment["isic2019"]["excluded_group_ids"]:
        raise ValueError("Data-quality audit does not record the approved lesion exclusions")
    if audit.get("isic2019_excluded_image_count") != amendment["isic2019"]["excluded_image_count"]:
        raise ValueError("Data-quality audit exclusion count mismatch")
    if audit.get("near_duplicate_phash_max_distance_inclusive") != 4:
        raise ValueError("Data-quality audit pHash threshold mismatch")
    if audit.get("near_duplicate_role") != amendment["near_duplicate_screen"]["decision_rule"]:
        raise ValueError("Data-quality audit pHash role mismatch")
    if audit.get("exact_duplicate_group_merges_complete") is not True:
        raise ValueError("Exact-image split group links are incomplete")
    if audit.get("near_pairs_screened") != amendment["near_duplicate_screen"]["detected_candidate_pairs_before_exclusion"]["total"]:
        raise ValueError("Near-candidate screen count mismatch")
