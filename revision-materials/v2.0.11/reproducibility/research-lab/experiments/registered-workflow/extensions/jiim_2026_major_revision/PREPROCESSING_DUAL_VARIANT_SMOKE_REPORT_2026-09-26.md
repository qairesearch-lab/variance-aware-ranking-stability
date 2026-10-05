# JIIM revision: Swin-T preprocessing branches, local smoke report

> Historical technical report. Its 2026-09-26 statement that human near-duplicate adjudication remains a formal gate was superseded on 2026-09-28 by `OBJECTIVE_DATA_QUALITY_HANDOFF_v0.6.md`. The preprocessing smoke observations are not formal training results.

Date: 2026-09-26. Scope: technical/local validation of the two requested
preprocessing branches; **not a formal result and not permission to launch the
270-run matrix yet**.

## Frozen interpretation

| Branch | Train augmentation | Validation/test | Research role |
|---|---|---|---|
| A: `shared_dataset_transform_v1` | ISIC or MURA dataset-specific recipe, identical across all five candidates within that dataset | Direct bilinear 224×224 for all five; MURA square-padded first | Primary model-pool analysis. This is the author-approved D3 design choice. |
| B: `swin_weight_eval_v1` | Exactly the same training augmentation as A | Swin-T only: `Swin_T_Weights.IMAGENET1K_V1.transforms()` (232 short-edge bicubic, 224 center crop, ImageNet normalization); MURA square-padded first. Four CNNs stay as in A. | Predeclared preprocessing sensitivity, **not** a pure architecture effect. |

Both branches use the same explicit Swin-T V1 pretrained weights, 224 model
input, model fine-tuning scope, optimizer, checkpoint Rule A, group splits and
training seeds. Because B changes the **validation** loss trajectory, B needs
its own Rule-A selected checkpoint; taking A's selected checkpoint and changing
only the test transform would not answer the same question. If B is run across
all five matched splits × three seeds × two datasets, it adds **30 Swin runs**
to the author-approved E1's 270 (300 total), without changing the 270 primary
runs. Analysis should pair A/B by dataset, split, and seed and label this a
pipeline sensitivity. Do not combine A and B Swin results as exchangeable
replicates or describe their difference as an architecture-only effect.
For non-square ISIC images, A's direct 224×224 evaluation resize changes the
aspect ratio, while B's short-edge resize preserves aspect ratio and center
crop may remove peripheral content. That is a substantive image-treatment
difference to disclose; it is not just an interpolation-kernel switch.

## Implementation and local evidence

- Configs: `configs/preprocessing_shared_v1.yaml` and
  `configs/preprocessing_swin_weight_eval_v1.yaml`; implementation:
  `preprocessing.py`; executable technical smoke:
  `pilot/preprocessing_dual_variant_smoke.py`.
- Local result JSON:
  `pilot/results/preprocessing_dual_variant_smoke_20260926.json`.
- 12/12 cases passed with actual cached ISIC/MURA images and official
  ImageNet-1K V1 pretrained weights: 2 datasets × (four CNNs + Swin A + Swin B).
  Each case completed a short backward/optimizer step, validation loss,
  Rule-A checkpoint save/select/reload, and finite test loss. Swin cases ran
  two short epochs. MURA validation/test used study-mean logits; the synthetic
  two-image study and Rule-A threshold/patience checks passed.
- Main ISIC train (three random seeds) and eval tensors exactly matched the
  released submission script's SIPaKMeD transform implementation. This tests
  image-operation parity, **not** dataset equivalence. The source script and
  training-config hashes are recorded in the JSON.
- Swin A/B training tensors were exactly equal under a fixed seed; the four
  CNN training and evaluation tensors were exactly equal across branch configs.
  Swin evaluation tensors differed (mean absolute normalized-tensor difference
  0.1442 on the selected ISIC image and 0.0922 on the selected MURA image), as
  expected. These are *input diagnostics*, not accuracy differences.
- Local machine: M5/MPS, Python 3.12, PyTorch 2.8.0, torchvision 0.23.0.
  The JSON records source/config/weight hashes and per-case elapsed time.
  The official torchvision 0.19 Swin V1 documentation independently states
  the same 232-bicubic/224-center-crop/normalization recipe:
  https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.swin_t.html
- The single-run entry point `train_extension_run.py` passed **nine** additional
  tiny end-to-end technical runs (both Swin branches on both datasets and
  ResNet-18 controls on both datasets, plus two identity/environment checks).
  It produced `run_config.json`,
  `run_status.json`, `history.csv`, selected checkpoint + manifest,
  `predictions.csv`, `metrics.json`, and `timing.json`. All are under
  `pilot/results/technical_runner/` and are **not** research outcomes.
  One run exercised a one-row manifest/hash check.
- A wrong local package environment was deliberately rejected before creating
  a formal run directory. Formal mode now requires the target environment,
  CUDA, fixed batch/epoch/worker/seed settings, a 270/300-row sealed manifest
  with the exact approved matrix, and matching split/index hashes; it refuses
  to overwrite any existing run directory. Four synthetic manifest-contract
  unit tests passed, including tampering and wrong-matrix rejection. The
  run config now carries each identity field named in the design lock.
- `dispatch_manifest_run.py` accepts a single sealed run ID and has an
  **explicit manual** `--retry-failed` path. It rechecks the sealed manifest,
  current split/index hashes, target environment, CUDA, model-weight file and
  all old-run identity hashes *before* moving a failed attempt into
  `runs/failed_attempts/<run_id>/`. Completed runs cannot be retried or
  overwritten; no automatic retry is scheduled. Four retry-contract unit
  tests passed. This is only a local safety check, not a completed formal
  CUDA retry.
- The formal manifest generator now checks every split row against the final
  dataset index for dataset, image path, label, split group, study ID, and body
  region; matching only the sample-ID set is insufficient. A synthetic
  tampering test rejects each altered field, and both current candidate
  indices pass the strengthened input-schema read. The combined manifest and
  retry contract suite was **9/9 passing** at that gate. The formal 270/300-row manifest is
  still not generated because the data-quality decision gate is open.
- A read-only `audit_completed_manifest.py` now checks a sealed matrix after
  training: exact run identity and source/weight/environment hashes, CUDA
  package versions, required files, Rule-A checkpoint hash/selection, test
  unit identities and labels, probability normalization, recalculated primary
  balanced accuracy/confusion matrix, secondary ISIC metric presence, epoch
  history, and timing. It can audit E0 (150), primary E1 (270), sensitivity
  (30), or all (270/300) **within one unchanging manifest**. It has passed
  synthetic valid/tampered/missing-artifact tests; no formal CUDA outputs exist
  yet for a real full-matrix audit.
- `run_manifest_queue.py` provides a human-started, preview-by-default,
  one-serial-queue-per-GPU handoff for E0→E1→optional B. Assignment is
  deterministic/disjoint across card indices, respects an existing host GPU
  mask, verifies completed outputs before skipping, and stops at the first
  failure without automatic retry. The combined contract suite is now
  **14/14 passing**. The queue itself has not been exercised on 4090 hardware.
  A 300-row manifest must be chosen *before* the first formal A run if B will
  be part of that manifest; changing a 270-row manifest to 300 afterwards
  would invalidate every A run's recorded manifest hash.
- `prepare_candidate_indices.py` matched ZIP members to metadata without
  extracting archives. ISIC yielded 23,247 primary images, 11,847 lesion
  groups, 2,084 excluded missing-ID images (NV 1,549; MEL 337; BKL 198), and
  no conflicting labels within an ID. MURA yielded 40,005 valid images,
  11,967 patients, 14,656 studies. Candidate files are under
  `candidate_indices/`; they are not final split authorities. Both original
  ZIP SHA-256 values matched the approved design lock.
- `generate_extension_manifest.py` encodes the approved E1 geometry (270 runs)
  and optional separately tagged B sensitivity (30 more), but refuses to seal
  a manifest until a final index, all ten image-present/group-valid splits,
  and an adjudicated duplicate-audit record with matching hashes exist.
- `generate_grouped_splits.py` executed its full 512-candidate search for one
  explicitly **diagnostic** split per dataset. ISIC split_01 selected candidate
  247: 16,261/3,518/3,468 images; MURA selected candidate 194:
  10,257/2,206/2,193 study units. Both had zero cross-partition group overlap,
  all classes in every partition, and all seven MURA regions in every partition.
  These output files are marked `diagnostic_not_for_formal_training` and must
  not be relabeled or reused as final splits before duplicate adjudication.
- The full pixel/file-hash and pHash≤4 candidate scan completed across both
  archives. It found **ISIC: 10 cross-group byte-identical pairs, 1,150
  near-pHash pairs; MURA: 4 byte-identical pairs, 1,870 near-pHash pairs**.
  The exact-pixel counts equal the byte-identical counts. These are image-pair
  counts; the exact-pair CSV also repeats byte-identical pairs under the pixel
  hash method, so it must not be counted twice. The MURA exact pairs have no
  label conflict; one crosses the publisher's original train/valid boundary.
  Crucially, two ISIC byte-identical pairs carry contradictory MEL/NV labels
  across `BCN_0003560` and `BCN_0000237` (two images in each group). This
  prevents automatic merging/final split freeze. A proposed *pre-outcome* data
  quality amendment is to exclude both complete lesion groups (four images,
  2 MEL and 2 NV), changing the primary image count from 23,247 to 23,243;
  this requires the author's explicit approval. No model performance was used
  to discover or choose the amendment. This same pair of conflicts was
  discussed on the [ISIC 2019 Forum](https://forum.isic-archive.com/t/a-list-of-duplicate-images-in-the-training-set/1141),
  where a participant confirmed the images correspond to MEL. An alternative
  transparent amendment would relabel the two NV records as MEL and merge the
  group IDs, retaining 23,247 images; it changes the published challenge
  ground truth and is **not** applied without author approval. Original CSV
  age/sex metadata also disagrees between these exact-image pairs, which
  reinforces the case for a cautious exclusion policy.
- `finalize_duplicate_audit.py` prepares a **blinded** decision sheet for the
  3,020 near-pHash pairs (opaque candidate ID and distance, but no original
  image path, diagnosis, patient or lesion ID) and will refuse to seal final indices until every candidate is reviewed
  with reviewer identity/time. The review sheet is
  `candidate_indices/duplicate_candidate_audit_full_v1/near_pair_blinded_review_template.csv`.
  An offline visual gallery of all 3,020 pairs is at
  `candidate_indices/duplicate_candidate_audit_full_v1/blinded_review_gallery_v2/index.html`;
  it contains 2,434 unique, aspect-preserving local thumbnails and no remote
  scripts. A first gallery version was discarded because MURA filenames
  disclosed `positive/negative`; the current HTML and exported CSV omit those
  filenames, and a static leak check and JavaScript syntax check passed.
  A change from pHash≤4 to a narrower candidate threshold is a protocol
  amendment, not an implementation detail; the detected ≤2 subset contains
  539 pairs (ISIC 140, MURA 399). A candidate count alone is not evidence
  that a pair depicts the same lesion/patient.

## What this confirms — and does not

**Confirmed locally:** the two image-treatment branches are explicit,
machine-readable, non-overlapping in their scientific role, faithful to the
submitted ISIC image-operation rule where applicable, and functional with
pretrained models, grouped MURA validation, and checkpoint reload. The smoke
script supports `--device cuda` for the later server check.

**Not yet confirmed for formal AutoDL execution:** human near-duplicate adjudication,
final dataset indices and ten full group-disjoint split files
with hashes, the runner at full size/batch 32 on CUDA in the pinned target
Python 3.10/PyTorch 2.4.1/torchvision 0.19.1 environment, and at least one
complete Rule-A 4090 run per dataset. The runner/output schema has passed
only tiny local MPS integration checks. The 12 short cases' elapsed times
do not estimate full training time. The 270-run E1 remains the target, but do
not call the *entire extension protocol* `FROZEN` until those gates pass.
**2026-09-27 addendum:** the author approved exclusion of both complete ISIC
lesion IDs (four images, 23,243 remaining) and pHash Hamming distance ≤4.
The separate v0.4 data-quality amendment and formal split/manifest count gates
now encode this decision without changing the hashed v0.3 design. The earlier
paragraph above records the proposal as it stood on 2026-09-26; it is no
longer an open author choice. Human review of 3,020 candidates is still open.
The v0.3 design text abbreviates torchvision as `0.19.1`, while the candidate
CUDA environment file pins the build string `0.19.1+cu121`. Runtime validation
uses the latter exact string. Reconcile this notation in the next versioned
protocol lock before the first formal 4090 run; do not silently edit an
already hashed design file.

## Reviewer-facing interpretation

Report A as the primary comparison because each candidate within each dataset
received the same geometry, size and normalization; explicitly disclose that
Swin's pretrained inference recipe was not used in A. Report B as a *paired,
Swin-only preprocessing sensitivity* if its 30 formal runs are funded: whether
the Swin model-pool conclusion changes when its official V1 inference recipe
is used for both validation and test. State that B tests the combined deployed
pipeline, not an isolated architecture effect. This directly anticipates
reviewer concerns about preprocessing fairness and pretrained-weight mismatch.
