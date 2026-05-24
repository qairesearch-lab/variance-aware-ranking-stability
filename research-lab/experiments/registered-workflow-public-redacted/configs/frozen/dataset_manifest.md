# Dataset Manifest

Status: frozen schema; dataset facts verified before formal execution
Frozen status: protocol freeze
Scope: Primary analysis datasets in `datasets.yaml`

## Purpose

This file freezes the field schema for dataset sources and version records. Public release copies keep dataset source, version, expected sample counts, split policy, and integrity summaries without redistributing original image data.

## Primary Analysis Datasets

| dataset_id | Dataset | Version | Primary Analysis Status | Source |
|------------|---------|---------|-------------------------|--------|
| sipakmed | SIPaKMeD | original release, ICIP 2018 | included | https://www.cs.uoi.gr/~marina/sipakmed.html |
| organamnist | OrganAMNIST | MedMNIST v2 | included | https://github.com/MedMNIST/MedMNIST |

## Deferred Datasets

| dataset_id | Dataset | Current Status | Primary Analysis Handling |
|------------|---------|---------------|---------------------------|
| autopet | FDG-PET-CT-Lesions / autoPET | future extension | not included in the current primary run manifest |

## SIPaKMeD Manifest Fields

```yaml
dataset_id: sipakmed
name: SIPaKMeD
version: original_release_icip_2018
source_url: https://www.cs.uoi.gr/~marina/sipakmed.html
download_date: recorded_in_project_archive
raw_data_local_path: ../../../../data/sipakmed/raw
processed_data_local_path: ../../../../data/sipakmed/processed
primary_image_glob: ../../../../data/sipakmed/raw/im_*/CROPPED/*.bmp
file_hashes:
  raw_archives_sha256:
    im_Dyskeratotic.7z: a8768bc03c9e814063ff946b7ef7548067e16b760168267d2b12e95a66955cce
    im_Koilocytotic.7z: 653578e878e3cc63a8510cd7ec26d918637d9881e913e29a5bb205d0d9cfb70d
    im_Metaplastic.7z: 97f1badc5f606f127d2af0d18a008ebd41a2b60f44785cb332a6eaa33004d2af
    im_Parabasal.7z: 13e9c29f016a5b83731e038cfe518671a7815df28836dee706acba6ecd2369a8
    im_Superficial-Intermediate.7z: 3b73e406df14d60b1bd9c8ac3b01b594e70bb51ba32fb080141fc6a7f489a599
  primary_image_relative_path_list_sha256: d64cc5fda6cf8f6681f8c347da3c7c76316592b67d3ac1a8ecfcafd7844b8cb8
  primary_image_content_sha256_manifest: not_included_in_public_github_package
expected_samples: 4049
observed_samples_after_download: 4049
observed_primary_images:
  Dyskeratotic: 813
  Koilocytotic: 825
  Metaplastic: 793
  Parabasal: 787
  Superficial-Intermediate: 831
official_nonprimary_cell_cluster_images:
  parent_im_folder_bmp_files: 966
  note: >
    These 966 BMP files are official SIPaKMeD cell cluster images. They are
    part of the released dataset, but are not included in the current primary
    image-classification analysis, which is restricted to isolated single-cell
    CROPPED images.
excluded_annotation_files:
  dat_files: 16196
classes:
  - Dyskeratotic
  - Koilocytotic
  - Metaplastic
  - Parabasal
  - Superficial-Intermediate
group_id_available: false
grouping_limitation_if_any: image-level split only (no patient/slide/group identifiers in dataset files)
official_split_available: false
primary_split_policy: generated_image_level_repeated_random_holdout
reorganized_date: recorded_in_project_archive
reorganization_note: >
  Data folders im_* are expected under data/sipakmed/raw/.
  raw/ contains all original data (including .bmp and .dat files, kept unchanged).
  processed/ is for derived data only (index, audit tables, stats, checksum manifests, etc.).
  split files are stored separately under registered-workflow/splits/.
  Training code MUST NOT directly scan raw/ directory.
  The canonical primary-analysis image set is restricted to isolated single-cell images in raw/im_*/CROPPED/*.bmp,
  totaling 4049 images. The 966 official cell cluster images are retained for
  traceability but excluded from the current primary analysis.
```

## OrganAMNIST Manifest Fields

```yaml
dataset_id: organamnist
name: OrganAMNIST
version: medmnist_v2
source_url: https://github.com/MedMNIST/MedMNIST
download_method: medmnist_python_package
medmnist_package_version: 3.0.2
download_date: recorded_in_project_archive
raw_data_local_path: ../../../../data/organamnist
processed_data_local_path: ../../../../data/organamnist
file_hashes:
  train_csv_sha256: 8a03cb31b92c2e1c28f5f34e212c27e0190f6848a94db5aa5427a410d9cbc33c
  val_csv_sha256: b75ab075c2062941fa81929658f421527a6836f3b3f0496455911a2d1b28dcad
  test_csv_sha256: 1df015e2a15b9801ae8df189a7e44171208ac03bdb8546b77c1d57b7b6e50be4
  primary_image_relative_path_list_sha256: d1bee8a6b058e4f241306a7f5f2742fb7d065209cf1b596bf3d6f8953b581ae4
  primary_image_content_sha256_manifest: not_included_in_public_github_package
registered_or_prior_expected_samples: 58850
current_medmnist_info_expected_samples: 58830
expected_samples: 58830
observed_samples_after_download: 58830
sample_count_difference_vs_registered_or_prior_expected: -20
sample_count_difference_percent_vs_registered_or_prior_expected: 0.03
sample_count_difference_explanation: >
  The 20-sample difference is explained by the MedMNIST v2.2.4 release note,
  which states that a small number of blank samples were removed in
  OrganAMNIST and related Organ/Vessel datasets. The registered/prior expected
  count (58,850) reflects the earlier count used during dataset selection.
  The current local medmnist package (v3.0.2) reports INFO['organamnist']
  n_samples as train=34561, val=6491, test=17778, total=58830. The
  exported CSV rows and PNG images match this current package-level metadata,
  so the difference is treated as an upstream dataset correction/version
  difference rather than a local download or export error.
sample_count_difference_sources:
  - MedMNIST README release notes: v2.2.4 removed a small number of blank samples in OrganAMNIST.
  - https://github.com/MedMNIST/MedMNIST
  - local medmnist 3.0.2 INFO['organamnist']['n_samples']: train=34561, val=6491, test=17778
observed_split_samples:
  train: 34561
  val: 6491
  test: 17778
observed_image_files:
  train: 34561
  val: 6491
  test: 17778
classes: 11
class_labels:
  0: bladder
  1: femur-l
  2: femur-r
  3: heart
  4: kidney-l
  5: kidney-r
  6: liver
  7: lung-l
  8: lung-r
  9: pancreas
  10: spleen
group_id_available: false
grouping_limitation_if_any: image-level split only (no patient/group identifiers in dataset)
official_split_available: true
official_split_recorded_for_traceability: true
official_split_used_for_primary_analysis: false
official_split_local_record:
  train_csv: ../../../../data/organamnist/train/organamnist.csv
  val_csv: ../../../../data/organamnist/val/organamnist.csv
  test_csv: ../../../../data/organamnist/test/organamnist.csv
primary_split_policy: generated_image_level_repeated_random_holdout
```

<br />

<br />

## Completion Status and Governance

The dataset manifest completion rules have been executed for the current primary-analysis datasets.

- Repository-relative paths, observed sample counts, class mappings, and integrity summaries are recorded for SIPaKMeD and OrganAMNIST.
- Per-image checksum manifests are not included in the public GitHub package.
- No patient, case, slide, volume, or other group IDs are available for either primary dataset; the image-level split limitation is therefore explicitly documented.
- OrganAMNIST official split files are recorded for traceability only and are not used for the primary analysis.
- The primary analysis will use generated image-level repeated random holdout splits as specified in `split_plan.yaml`.
- Any future replacement of data sources, version changes, changes to the primary analysis dataset list, or changes to the split policy must be recorded through an amendment or deviation record before use.
