# JIIM reviewer-requested extension: protocol freeze decisions v0.2

> Historical design record. The 2026-09-28 pre-outcome data-quality amendment `configs/data_quality_amendment_v0.6.yaml` supersedes the human near-duplicate confirmation and related handoff gate below. See `OBJECTIVE_DATA_QUALITY_HANDOFF_v0.6.md`; do not run the old manual-review procedure. Other confirmed design decisions and the original 800 runs are unchanged.

Status: **author design decisions confirmed on 2026-09-26; implementation not yet
validated and formal runs not yet permitted**. The original OSF-frozen 800-run
study remains untouched. Machine-readable design lock:
[`configs/extension_protocol_design_locked_v0.3.yaml`](configs/extension_protocol_design_locked_v0.3.yaml).
Design-lock YAML SHA-256:
`e8b6a2e5ed9845a753d370fc27ae07d4b740e3c10e7801fe40d16199fcc582be`.

## Evidence base and purpose

- Original decision letter: `Journal of Imaging Informatics in Medicine  大修意见.docx`
  (6 September 2026; manuscript JDIM-D-26-02658).
- Submitted reproducibility release: GitHub `v1.0-submission`, commit
  `2cc72ccdb0fafe0269a8c890891a779032beac9c`, also backed up in `github/`.
- The supplementary experiment addresses R1.1 (harder datasets / margin), R1.3
  and R2.5 (architecture-pool sensitivity), and R2.2 (grouped evaluation).
  It does **not** replace the original 800-run reanalysis required by the editor,
  R2.3, R2.7, or R3.5.
- The original training script and public script differ only in version label and
  machine-name redaction. The frozen training/model/split/seed/checkpoint YAMLs
  and primary manifest match byte-for-byte. Use that release as the comparator,
  not the post-run local `analysis_pipeline.yaml` status edit.

## Author decisions — confirmed in this chat on 2026-09-26

| ID | Choice | Recommendation | Reason and cost |
|---|---|---|---|
| D1 | ISIC primary cohort: 23,247 images with known lesion ID, or all 25,331 with 2,084 missing IDs treated as singleton groups? | **Known-ID cohort for primary** | A singleton is not evidence of a unique lesion; known-only makes the grouped primary claim more defensible. The 2,084 omitted images are 8.2% of the archive and occur only in NV (1,549), MEL (337), and BKL (198). Report exclusions and compare class composition; do not imply a full-cohort performance sensitivity without additional training. All-image primary preserves size but leaves residual cross-partition lesion leakage unresolvable. Neither choice changes the number of runs. |
| D2 | Target training matrix | **E1 = 270 runs**, E0 = 150-run technical/resource floor | E1 has 10 split clusters for the shared four CNNs, whereas E0 has 5. Swin-T stays on the paired first 5 splits × 3 seeds. E2 = 430 runs may be added only after a predeclared resource gate, not after inspecting outcomes. |
| D3 | Swin-T input recipe: shared dataset transform or its official pretrained-weight transform? | **Shared transform across all five models within each dataset** | This better isolates model-pool heterogeneity. Swin-T's official 0.19.1 inference recipe uses bicubic 232 resize and 224 center crop; applying it only to Swin would confound architecture with preprocessing. A shared direct 224 pipeline may be suboptimal for its pretrained recipe, which must be disclosed. |

The author chose all three recommended options: D1 known-ID cohort, D2 E1 as
target with an outcome-independent E0 resource floor, and D3 shared transform
within each dataset. These design choices are locked; they may not be silently
changed during implementation. Record the final protocol SHA-256 in the
subsequent executable handoff package.

## Recommended fixed technical settings (no author decision unless objected)

1. **Data/endpoint.** ISIC uses the eight known ground-truth classes, one image
   per primary observation, lesion groups for splitting; lesion-level mean-logit
   results are secondary. MURA combines the released train/valid corpus for new
   repeated holdouts, splits by patient, trains on individual images, selects
   checkpoints using equal-weight study-level cross-entropy of mean logits, and
   reports study-level normal/abnormal balanced accuracy. It does not claim
   fracture diagnosis or use a hidden competition test.
2. **Split rule.** Ten fixed seeds 1001–1010, target 70/15/15, with group
   disjointness and all primary classes in each partition as hard constraints.
   MURA must also have all seven body regions per partition. The candidate
   protocol predeclares a deterministic multi-candidate group-shuffle search;
   the generated index, ten CSVs, audit, and hashes become the final split
   authority. Exact/near-duplicate decisions must be made before those hashes
   are sealed. If a hard constraint is impossible, stop and amend openly.
3. **Images.** ISIC matches the submitted SIPaKMeD transform order: RGB,
   RandomResizedCrop 224 with area scale 0.8–1.0, horizontal flip 0.5, rotation
   ±10°, then ImageNet normalization; direct 224 resize for validation/test.
   MURA uses grayscale repeated to RGB, centered black square padding before
   bilinear 224 resize, then rotation ±10° during training; no random crop or
   horizontal flip. Under the D3 recommendation every candidate model sees the
   *same* transform within a dataset. The MURA exception is predeclared for anatomy/laterality and must be
   disclosed as a cross-dataset methodological difference.
4. **Interpolation trap.** The submitted script's `RandomRotation(10)` omitted
   interpolation and therefore used torchvision's `NEAREST` default, despite
   the generic frozen YAML saying `bilinear`. Keep NEAREST for the new rotation
   to match executed code; set bilinear explicitly for crop/resize. Do not
   silently "repair" the original protocol during extension.
5. **Models/optimization.** The common pool is ResNet-18, ResNet-50,
   DenseNet-121, EfficientNet-B0. Swin-T provides a fourth architecture family
   in the expanded pool, with official `IMAGENET1K_V1` weights, last feature
   stage + norm + head trainable. All five models use the same 224 input,
   unweighted cross-entropy, AdamW, LR 1e-4, decay 1e-4, batch 32, no scheduler,
   no AMP, no model-specific tuning, and Rule A only. The original code's
   `model.train()` behavior updates frozen BatchNorm running buffers; preserve
   that behavior for CNN comparison rather than introducing a silent change.
6. **Rule A fidelity.** Up to 60 epochs; improvement means validation loss
   `< best_loss - 0.001`, patience 7. ISIC uses the original script's batch-mean
   validation loss; MURA necessarily uses study-mean-logit cross-entropy so
   checkpoint selection matches its primary evaluation unit. This is an
   explicit endpoint-driven exception, not an identical loss implementation.
7. **Analysis.** Compare the four shared CNNs under Rule A at the same
   split/seed budget (5×3 if E0; 10×3 if E1). Pair models **within** each
   dataset/context; different datasets are not paired samples. Report absolute
   balanced accuracy and uncertainty, margin-versus-instability, grouped
   leakage audit, and Swin pool sensitivity only on its 5×3 matched contexts.
   Never claim that cross-dataset differences isolate dataset difficulty:
   grouping, endpoint, modality, and preprocessing also change. Report even
   if ISIC or MURA proves saturated; do not swap datasets after seeing results.
8. **Environment and dispatch.** Prefer one pinned extension environment close
   to original SIPaKMeD runs: Python 3.10, PyTorch 2.4.1+cu121,
   torchvision 0.19.1. `swin_t` with `IMAGENET1K_V1` exists in that version.
   The submitted public `requirements.txt` is not an exact lock and its
   `numpy>=2` conflicts with the original SIPaKMeD run's NumPy 1.26.4; create
   a separate tested extension lock, never claim the old runs shared one
   environment (OrganAMNIST used torchvision 0.25.0). On a multi-4090 server,
   dispatch independent run IDs, one process per GPU; do not convert this to
   DDP. A uniform worker count of 4 is the candidate default; record it.

## Freeze and handoff gates

The candidate YAML is **not** a command to rent GPUs. Before it can be marked
`FROZEN` and handed to AutoDL, all of these must be true:

1. D1, D2, and D3 are answered and reflected in the YAML (complete).
2. Source ZIP/metadata hashes verified; exact- and near-duplicate decisions
   recorded; indices and ten grouped splits generated, class/group audits pass,
   and output SHA-256 manifest is sealed.
3. A formal extension runner and manifest validator implement this config;
   MURA study aggregation and ISIC lesion aggregation have unit tests. The old
   `train_one_run.py` currently rejects both new datasets and Swin-T.
4. Two datasets × five models pass small schema/finite-loss smoke checks.
   Then at least one complete Rule-A CUDA run per dataset validates checkpoint,
   prediction unit, output schema, memory, and wall-clock. A short M5/MPS
   timing script with random weights is **not** evidence of formal parity.
5. The exact Python/package/CUDA/driver/weight and code hashes are recorded.
   A completed run must retain manifest row, dataset/split/protocol hashes,
   loss history, selected checkpoint, predictions, metrics, and timings. Any
   OOM or inability to satisfy batch 32 triggers a documented amendment before
   a restart; gradient accumulation is not assumed identical because BatchNorm
   and optimizer trajectories can differ.

Only after those checks should a freeze record name the final protocol hash,
index/split hashes, manifest hashes, software lock, and exact handoff commands.
Once formal outcomes have been viewed, change scope or hyperparameters only
with a versioned amendment; preserve failed and superseded run records.
