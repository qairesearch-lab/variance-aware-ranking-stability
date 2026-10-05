#!/usr/bin/env python3
"""Reanalyze the reviewer-requested 300-run extension against the original 800-run grid.

The extension remains a post hoc reviewer-requested analysis. This script does
not retrain models, regenerate splits, or alter the sealed manifest.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[6]
EXT = ROOT / "research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision"
STATS = ROOT / "research-lab/experiments/registered-workflow/stats"
OUT = EXT / "analysis/results_v0.1"
MANIFEST = EXT / "run_manifests/jiim_extension_300_v0.6.csv"
SEAL = EXT / "run_manifests/jiim_extension_300_v0.6.seal.json"
OLD_METRICS = ROOT / "research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv"
OLD_MANIFEST = ROOT / "research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv"
MODELS = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
ALL_MODELS = MODELS + ["swin_t"]
TIE_MODELS = sorted(ALL_MODELS)
N_BOOT = 10_000
SEED = 20260926


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def top_and_margin(values: dict[str, float], model_order=MODELS):
    ordered = sorted(values, key=lambda m: (-values[m], m))
    if len(ordered) < 2:
        return ordered[0], math.nan, ordered
    return ordered[0], values[ordered[0]] - values[ordered[1]], ordered


def ba_from_labels(true: list[int], pred: list[int], n_classes: int) -> float:
    recall = []
    for c in range(n_classes):
        n = sum(t == c for t in true)
        if n:
            recall.append(sum(t == c and p == c for t, p in zip(true, pred)) / n)
    return float(sum(recall) / len(recall)) if recall else math.nan


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def audit_and_collect():
    manifest_bytes_hash = sha(MANIFEST)
    seal = read_json(SEAL)
    if manifest_bytes_hash != seal["manifest_sha256"]:
        raise ValueError("Manifest SHA-256 does not match its frozen seal")
    rows = read_csv(MANIFEST)
    if len(rows) != 300 or len({r["run_id"] for r in rows}) != 300:
        raise ValueError("Manifest must contain 300 unique runs")

    # Frozen split evidence and output roots are checked without requiring the
    # original AutoDL image cache or a CUDA runtime on this analysis host.
    for row in rows:
        split = ROOT / row["split_file"]
        index = ROOT / row["dataset_index_file"]
        if sha(split) != row["split_sha256"] or sha(index) != row["dataset_index_sha256"]:
            raise ValueError(f"Frozen split/index hash mismatch: {row['run_id']}")

    records, audit = [], []
    expected_test_cache = {}
    env_signatures = set()
    runner = EXT / "train_extension_run.py"
    preprocessing = EXT / "preprocessing.py"
    design = EXT / "configs/extension_protocol_design_locked_v0.3.yaml"
    amendment = EXT / "configs/data_quality_amendment_v0.6.yaml"
    swin_decision = EXT / "configs/swin_sensitivity_decision_v0.5.yaml"
    environment_target = EXT / "configs/environment_target_v1.yaml"
    predictions_by_context = defaultdict(dict)
    expected_counts = Counter()
    for row in rows:
        run_id = row["run_id"]
        out = ROOT / row["output_dir"]
        cfg, status = read_json(out / "run_config.json"), read_json(out / "run_status.json")
        metrics = read_json(out / "metrics.json")
        checkpoint = read_json(out / "checkpoint_manifest.json")
        timing = read_json(out / "timing.json")
        if status.get("status") != "completed" or status.get("run_id") != run_id:
            raise ValueError(f"Non-completed or mismatched run status: {run_id}")
        for key, value in (("run_id", run_id), ("dataset", row["dataset"]), ("model", row["model"]), ("variant", row["variant"])):
            if cfg.get(key) != value:
                raise ValueError(f"Run identity mismatch {key}: {run_id}")
        for key, value in (("run_id", run_id), ("dataset", row["dataset"]), ("variant", row["variant"])):
            if metrics.get(key) != value:
                raise ValueError(f"Metric identity mismatch {key}: {run_id}")
        if int(cfg.get("training_seed", -1)) != int(row["training_seed"]):
            raise ValueError(f"Seed mismatch: {run_id}")
        expected_config_hashes = {
            "run_manifest_sha256": manifest_bytes_hash,
            "design_lock_sha256": sha(design),
            "data_quality_amendment_sha256": sha(amendment),
            "swin_sensitivity_decision_sha256": sha(swin_decision),
            "variant_sha256": sha(EXT / "configs" / ("preprocessing_shared_v1.yaml" if row["variant"] == "shared" else "preprocessing_swin_weight_eval_v1.yaml")),
            "script_sha256": sha(runner),
            "preprocessing_sha256": sha(preprocessing),
            "environment_target_sha256": sha(environment_target),
            "split_sha256": row["split_sha256"],
            "dataset_index_sha256": row["dataset_index_sha256"],
        }
        for key, expected in expected_config_hashes.items():
            if cfg.get(key) != expected:
                raise ValueError(f"Frozen execution provenance mismatch {key}: {run_id}")
        env = cfg.get("environment", {})
        env_signatures.add(tuple(sorted(env.items())))
        hp = cfg.get("hyperparameters", {})
        expected_hp = {"batch_size": 32, "learning_rate": 0.0001, "max_epochs": 60, "mixed_precision": False,
                       "num_workers": 4, "optimizer": "AdamW", "weight_decay": 0.0001}
        if hp != expected_hp or env.get("device") != "cuda" or env.get("gpu") != "NVIDIA GeForce RTX 4090 D":
            raise ValueError(f"Training environment/hyperparameters differ from the frozen target: {run_id}")
        if checkpoint.get("rule") != "A" or int(checkpoint.get("best_epoch", -1)) != int(metrics["selected_epoch"]):
            raise ValueError(f"Checkpoint selection metadata mismatch: {run_id}")
        model_file = out / "checkpoints/best_validation_loss.pt"
        if not model_file.is_file() or sha(model_file) != checkpoint.get("selected_checkpoint_sha256"):
            raise ValueError(f"Checkpoint file hash mismatch: {run_id}")
        history = read_csv(out / "history.csv")
        if not history or int(timing["epochs_executed"]) != len(history) or int(metrics["selected_epoch"]) > len(history):
            raise ValueError(f"History/timing mismatch: {run_id}")

        pred_rows = read_csv(out / "predictions.csv")
        ids = [p["unit_id"] for p in pred_rows]
        if len(ids) != len(set(ids)) or not pred_rows:
            raise ValueError(f"Duplicate or empty prediction units: {run_id}")
        split_key = (row["dataset"], row["split_id"])
        if split_key not in expected_test_cache:
            split_rows = read_csv(ROOT / row["split_file"])
            test_rows = [x for x in split_rows if x["subset"] == "test"]
            if row["dataset"] == "isic2019":
                expected_test_cache[split_key] = {x["sample_id"]: int(x["label"]) for x in test_rows}
            else:
                expected = {}
                for x in test_rows:
                    if x["study_id"] in expected and expected[x["study_id"]] != int(x["label"]):
                        raise ValueError(f"Conflicting MURA labels in split study: {x['study_id']}")
                    expected[x["study_id"]] = int(x["label"])
                expected_test_cache[split_key] = expected
        expected_units = expected_test_cache[split_key]
        if set(ids) != set(expected_units):
            raise ValueError(f"Predictions do not cover the exact sealed test units: {run_id}")
        y, yh = [], []
        region_y, region_yh = defaultdict(list), defaultdict(list)
        for p in pred_rows:
            if p["run_id"] != run_id or p["variant"] != row["variant"] or p["dataset"] != row["dataset"]:
                raise ValueError(f"Prediction identity mismatch: {run_id}")
            truth, pred = int(p["label_true"]), int(p["label_pred"])
            if truth != expected_units[p["unit_id"]]:
                raise ValueError(f"Prediction label does not match sealed split: {run_id}")
            probs = [float(p[f"prob_{i}"]) for i in range(8 if row["dataset"] == "isic2019" else 2)]
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in probs) or abs(sum(probs) - 1) > 1e-5 or int(np.argmax(probs)) != pred:
                raise ValueError(f"Invalid probability/prediction row: {run_id}")
            y.append(truth)
            yh.append(pred)
            if row["dataset"] == "mura":
                region = p["unit_id"].split("/")[1]
                region_y[region].append(truth)
                region_yh[region].append(pred)
        recomputed = ba_from_labels(y, yh, 8 if row["dataset"] == "isic2019" else 2)
        if abs(recomputed - float(metrics["balanced_accuracy"])) > 1e-10:
            raise ValueError(f"BA does not reproduce from predictions: {run_id}")
        context = (row["dataset"], row["split_id"], int(row["training_seed"]), row["variant"])
        predictions_by_context[context][row["model"]] = (set(ids), {i: int(p["label_true"]) for i, p in zip(ids, pred_rows)})
        records.append({
            "run_id": run_id, "dataset": row["dataset"], "model": row["model"], "variant": row["variant"],
            "split_id": row["split_id"], "training_seed": int(row["training_seed"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "secondary_lesion_balanced_accuracy": float(metrics.get("secondary_lesion_balanced_accuracy", math.nan)),
            "test_units": int(metrics["test_units"]), "selected_epoch": int(metrics["selected_epoch"]),
            "elapsed_seconds": float(timing["elapsed_seconds"]),
            "checkpoint_sha256": checkpoint["selected_checkpoint_sha256"],
            "mura_region_ba": {k: ba_from_labels(region_y[k], region_yh[k], 2) for k in region_y},
            "mura_region_n": {k: len(region_y[k]) for k in region_y},
        })
        expected_counts[(row["dataset"], row["variant"])] += 1
        audit.append({"run_id": run_id, "status": "passed", "predictions": len(ids), "metrics_BA_reproduced": True, "checkpoint_sha256_verified": True})

    if expected_counts != Counter({("isic2019", "shared"): 135, ("mura", "shared"): 135, ("isic2019", "swin_weight_eval"): 15, ("mura", "swin_weight_eval"): 15}):
        raise ValueError(f"Unexpected manifest distribution: {expected_counts}")
    # Same held-out contexts must use identical test units and labels across models.
    for context, model_data in predictions_by_context.items():
        values = list(model_data.values())
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f"Paired models do not share identical test units/labels: {context}")
    if len(env_signatures) != 1:
        raise ValueError("The 300 runs do not share one recorded execution environment")
    return rows, pd.DataFrame(records), pd.DataFrame(audit)


def attach_original(metrics: pd.DataFrame):
    prior_table = pd.read_csv(OLD_METRICS)
    prior_table = prior_table[(prior_table["checkpoint_policy"] == "A") & (prior_table["model"].isin(MODELS)) & (prior_table["analysis_role"] == "primary_analysis")]
    manifest_rows = read_csv(OLD_MANIFEST)
    source_rows, audit = [], []
    for row in manifest_rows:
        if row["checkpoint_policy"] != "A" or row["analysis_role"] != "primary_analysis" or row["model"] not in MODELS:
            continue
        run_dir = ROOT / row["output_dir"]
        status = read_json(run_dir / "run_status.json")
        result = read_json(run_dir / "metrics.json")
        if status.get("status") != "completed" or status.get("run_id") != row["run_id"]:
            raise ValueError(f"Original Rule-A run not completed: {row['run_id']}")
        for key in ("run_id", "dataset", "model", "split_id", "training_seed", "checkpoint_policy", "analysis_role", "config_hash", "split_hash"):
            if str(result.get(key)) != str(row[key]):
                raise ValueError(f"Original raw metric provenance mismatch {key}: {row['run_id']}")
        source_rows.append({"run_id":row["run_id"],"dataset":row["dataset"],"model":row["model"],
                            "split_id":row["split_id"],"training_seed":int(row["training_seed"]),
                            "balanced_accuracy":float(result["balanced_accuracy"])})
    old = pd.DataFrame(source_rows)
    if len(old) != 400 or len(prior_table) != 400 or old.run_id.nunique() != 400:
        raise ValueError("Expected 400 completed original Rule-A CNN runs")
    cached = prior_table.set_index("run_id")["balanced_accuracy"]
    for row in source_rows:
        cached_value = float(cached.loc[row["run_id"]])
        difference = abs(cached_value - row["balanced_accuracy"])
        if difference > 1e-12:
            raise ValueError(f"Original collected table differs from raw run metric: {row['run_id']}")
        audit.append({"run_id":row["run_id"],"status":"passed","raw_metrics_match_collected_table":True,
                      "absolute_BA_difference":difference})
    old["variant"] = "original_primary_A"
    old["secondary_lesion_balanced_accuracy"] = np.nan
    old["test_units"] = np.nan
    old["selected_epoch"] = np.nan
    old["elapsed_seconds"] = np.nan
    old["checkpoint_sha256"] = ""
    ext = metrics[metrics["variant"] == "shared"].copy()
    return pd.concat([old, ext], ignore_index=True, sort=False),pd.DataFrame(audit)


def build_contexts(all_metrics: pd.DataFrame):
    rows = []
    primary = all_metrics[all_metrics["model"].isin(MODELS)].copy()
    for (dataset, split, seed), group in primary.groupby(["dataset", "split_id", "training_seed"]):
        vals = dict(zip(group["model"], group["balanced_accuracy"]))
        if set(vals) != set(MODELS):
            continue
        winner, margin, order = top_and_margin(vals)
        rows.append({"dataset": dataset, "split_id": str(split), "training_seed": int(seed), "winner": winner,
                     "margin": margin, "top2": "|".join(order[:2]), "model_values": vals})
    ctx = pd.DataFrame(rows)
    # Leave-one-split-out reference: no held-out split contributes to its comparator.
    discordance = []
    for dataset, d in ctx.groupby("dataset"):
        for split in d["split_id"].unique():
            ref_data = primary[(primary["dataset"] == dataset) & (primary["split_id"] != split)]
            ref_means = ref_data.groupby("model")["balanced_accuracy"].mean().to_dict()
            ref_top, _, _ = top_and_margin(ref_means)
            for idx in d.index[d["split_id"] == split]:
                discordance.append((idx, ref_top, int(ctx.loc[idx, "winner"] != ref_top)))
    ctx["lso_reference_top"] = ""
    ctx["lso_discordant"] = 0
    for idx, ref, disc in discordance:
        ctx.loc[idx, "lso_reference_top"] = ref
        ctx.loc[idx, "lso_discordant"] = disc
    return primary, ctx


def metric_tensor(d: pd.DataFrame):
    splits = sorted(d.split_id.astype(str).unique())
    seeds = sorted(int(v) for v in d.training_seed.unique())
    wide = d.pivot_table(index=["split_id", "training_seed"], columns="model", values="balanced_accuracy")
    wide = wide.reindex(columns=TIE_MODELS[:4])
    arr = np.stack([np.stack([wide.loc[(s, seed)].to_numpy(float) for seed in seeds]) for s in splits])
    return splits, seeds, arr


def bootstrap_model_summaries(d: pd.DataFrame, rng: np.random.Generator, n_boot=N_BOOT):
    splits, seeds, arr = metric_tensor(d)
    s_n, k_n, m_n = arr.shape
    split_idx = rng.integers(0, s_n, size=(n_boot, s_n))
    seed_idx = rng.integers(0, k_n, size=(n_boot, s_n, k_n))
    sample = arr[split_idx[:, :, None], seed_idx]
    model_means = sample.mean(axis=(1, 2))
    top_idx = np.argmax(model_means, axis=1)
    probs = {m: float(np.mean(top_idx == j)) for j, m in enumerate(TIE_MODELS[:4])}
    per_context_winners = np.argmax(sample, axis=-1)
    context_frequencies = np.stack([(per_context_winners == j).mean(axis=(1,2)) for j in range(m_n)], axis=1)
    return {m: model_means[:, j] for j, m in enumerate(TIE_MODELS[:4])}, probs, context_frequencies


def bootstrap_lso_agreement(d: pd.DataFrame, rng: np.random.Generator, n_boot=N_BOOT):
    splits, seeds, arr = metric_tensor(d)
    s_n, k_n, _ = arr.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, s_n, size=s_n)
        seed_samples = rng.integers(0, k_n, size=(s_n, k_n))
        match = 0
        for i in range(s_n):
            other = [j for j in range(s_n) if j != i]
            ref_values = np.stack([arr[chosen[j], seed_samples[j]] for j in other]).mean(axis=(0, 1))
            ref_top = int(np.argmax(ref_values))
            held_values = arr[chosen[i], seed_samples[i]]
            match += int(np.sum(np.argmax(held_values, axis=1) == ref_top))
        out[b] = match / (s_n * k_n)
    return out


def make_dataset_outputs(primary: pd.DataFrame, contexts: pd.DataFrame):
    perf, stability = [], []
    rng = np.random.default_rng(SEED)
    for dataset, d in primary.groupby("dataset"):
        cd = contexts[contexts["dataset"] == dataset]
        model_boot, top_probs, freq_boot = bootstrap_model_summaries(d, rng)
        for model in MODELS:
            md = d[d["model"] == model].copy()
            reps = model_boot[model]
            model_idx = TIE_MODELS[:4].index(model)
            top_prob = top_probs[model]
            vals = md["balanced_accuracy"].astype(float).to_numpy()
            ranks = []
            for _, context in d.groupby(["split_id", "training_seed"]):
                order = top_and_margin(dict(zip(context.model, context.balanced_accuracy)))[2]
                ranks.append(order.index(model) + 1)
            perf.append({"dataset": dataset, "model": model, "contexts": len(md), "mean_BA": vals.mean(),
                         "SD_BA": vals.std(ddof=1), "split_seed_bootstrap_mean_CI_low": percentile(reps, .025),
                         "split_seed_bootstrap_mean_CI_high": percentile(reps, .975),
                         "observed_selection_frequency": float((cd.winner == model).mean()),
                         "context_selection_frequency_bootstrap_CI_low": percentile(freq_boot[:, model_idx], .025),
                         "context_selection_frequency_bootstrap_CI_high": percentile(freq_boot[:, model_idx], .975),
                         "hierarchical_bootstrap_top_probability": top_prob,
                         "mean_rank": float(np.mean(ranks)), "rank_SD": float(np.std(ranks, ddof=1)),
                         "secondary_lesion_BA_mean": float(md.secondary_lesion_balanced_accuracy.mean()) if md.secondary_lesion_balanced_accuracy.notna().any() else math.nan,
                         "mean_selected_epoch": float(md.selected_epoch.mean()) if md.selected_epoch.notna().any() else math.nan,
                         "mean_wall_clock_seconds": float(md.elapsed_seconds.mean()) if md.elapsed_seconds.notna().any() else math.nan})
        split_n = cd.split_id.nunique()
        agreement = 1 - float(cd.lso_discordant.mean())
        boot = bootstrap_lso_agreement(d, rng)
        # Full-grid in-sample comparator is retained only as a descriptive sensitivity.
        means = d.groupby("model").balanced_accuracy.mean().to_dict()
        full_top = top_and_margin(means)[0]
        entropy_counts = Counter(cd.winner)
        entropy = -sum((n/len(cd))*math.log(n/len(cd)) for n in entropy_counts.values()) / math.log(len(MODELS))
        stability.append({"dataset": dataset, "splits": split_n, "seeds_per_split": d.training_seed.nunique(),
                          "contexts": len(cd),
                          "leave_one_split_out_agreement": agreement,
                          "leave_one_split_out_discordance": 1-agreement,
                          "split_then_seed_hierarchical_bootstrap_agreement_CI_low": percentile(boot,.025),
                          "split_then_seed_hierarchical_bootstrap_agreement_CI_high": percentile(boot,.975),
                          "mean_top_two_margin": float(cd.margin.mean()), "median_top_two_margin": float(cd.margin.median()),
                          "contexts_margin_below_0.01": int((cd.margin < .01).sum()),
                          "selection_entropy_normalized": entropy,
                          "full_grid_in_sample_agreement_descriptive": float((cd.winner == full_top).mean()),
                          "bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED})
    return pd.DataFrame(perf), pd.DataFrame(stability)


def logistic_coef(x: np.ndarray, y: np.ndarray) -> float:
    """Small ridge-stabilized IRLS; return the last coefficient for x margin."""
    n, p = x.shape
    beta = np.zeros(p)
    ridge = np.eye(p) * 1e-8
    ridge[0, 0] = 0.0
    for _ in range(60):
        eta = np.clip(x @ beta, -25, 25)
        prob = 1 / (1 + np.exp(-eta))
        w = np.clip(prob * (1-prob), 1e-8, None)
        h = x.T @ (w[:, None] * x) + ridge
        step = np.linalg.solve(h, x.T @ (y-prob) - ridge @ beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[-1])


def margin_model(contexts: pd.DataFrame):
    ds_order = sorted(contexts.dataset.unique())
    log_margin = np.log10(contexts.margin.to_numpy(float) + 1e-6)
    sd = float(log_margin.std(ddof=1))
    z = (log_margin - float(log_margin.mean())) / sd
    y = contexts.lso_discordant.to_numpy(float)
    ds = contexts.dataset.to_numpy()
    x = np.column_stack([np.ones(len(contexts))] + [(ds == name).astype(float) for name in ds_order[1:]] + [-z])
    beta = logistic_coef(x, y)

    # Cluster bootstrap at dataset x split; recompute each split's reference in
    # every replicate so held-out contexts never define their own comparator.
    groups = {(d, s): g for (d, s), g in contexts.groupby(["dataset", "split_id"])}
    rng = np.random.default_rng(SEED + 1)
    boots = []
    # Reconstitute compact split x seed x model tensors from context metric maps.
    tensors = {}
    for dataset in ds_order:
        cd = contexts[contexts.dataset == dataset]
        ds = sorted(cd.split_id.unique())
        ks = sorted(cd.training_seed.unique())
        tensor = np.empty((len(ds), len(ks), len(MODELS)))
        ctx_index = cd.set_index(["split_id", "training_seed"])
        for i, split in enumerate(ds):
            for j, seed in enumerate(ks):
                vals = ctx_index.loc[(split, seed), "model_values"]
                tensor[i,j,:] = [vals[m] for m in TIE_MODELS[:4]]
        tensors[dataset] = (ds, ks, tensor)
    for _ in range(N_BOOT):
        yb, zb, db = [], [], []
        for dataset in ds_order:
            ds, ks, tensor = tensors[dataset]
            ns, nk, _ = tensor.shape
            chosen = rng.integers(0, ns, size=ns)
            seed_sample = rng.integers(0, nk, size=(ns, nk))
            for i in range(ns):
                ref = np.stack([tensor[chosen[j], seed_sample[j]] for j in range(ns) if j != i]).mean(axis=(0,1))
                ref_top = int(np.argmax(ref))
                held = tensor[chosen[i], seed_sample[i]]
                held_winners = np.argmax(held, axis=1)
                held_sorted = np.sort(held, axis=1)
                held_margins = held_sorted[:, -1] - held_sorted[:, -2]
                yb.extend((held_winners != ref_top).astype(int).tolist())
                zb.extend(np.log10(held_margins + 1e-6).tolist())
                db.extend([dataset] * nk)
        l = np.asarray(zb)
        zz = (l - float(log_margin.mean())) / sd
        dd = np.asarray(db)
        xx = np.column_stack([np.ones(len(yb))] + [(dd == name).astype(float) for name in ds_order[1:]] + [-zz])
        try:
            boots.append(logistic_coef(xx, np.asarray(yb,float)))
        except np.linalg.LinAlgError:
            continue
    or_point = math.exp(beta)
    ci = [math.exp(percentile(boots, .025)), math.exp(percentile(boots, .975))] if boots else [math.nan, math.nan]
    return pd.DataFrame([{"term":"one_SD_decrease_in_log10_top_two_margin", "odds_ratio":or_point,
                          "cluster_bootstrap_95CI_low":ci[0], "cluster_bootstrap_95CI_high":ci[1],
                          "clusters":len(groups), "contexts":len(contexts), "dataset_fixed_effects":"included",
                          "cluster_bootstrap_replicates":len(boots), "epsilon":1e-6,
                          "interpretation":"associational; discordance is defined against leave-one-split-out comparator"}])


def crossfit_summary(primary: pd.DataFrame):
    rows = []
    for dataset in ["isic2019", "mura"]:
        d = primary[(primary.dataset == dataset) & primary.model.isin(MODELS)]
        splits = sorted(d.split_id.unique())
        matches, total, block_scores = 0, 0, []
        for first in itertools.combinations(splits, 5):
            a, b = set(first), set(splits)-set(first)
            for discovery, reference in ((a,b),(b,a)):
                ref = d[d.split_id.isin(reference)].groupby("model").balanced_accuracy.mean().to_dict()
                ref_top = top_and_margin(ref)[0]
                disc = d[d.split_id.isin(discovery)]
                local = []
                for _, c in disc.groupby(["split_id","training_seed"]):
                    winner = top_and_margin(dict(zip(c.model,c.balanced_accuracy)))[0]
                    local.append(int(winner == ref_top))
                score=float(np.mean(local)); block_scores.append(score)
                matches += sum(local); total += len(local)
        rows.append({"dataset":dataset,"independent_split_blocks":"5 discovery / 5 reference; all 252 partitions, both directions",
                     "cross_fitted_contexts_scored":total,"cross_fitted_agreement":matches/total,
                     "min_partition_agreement":min(block_scores),"max_partition_agreement":max(block_scores),
                     "note":"exact finite-grid descriptive estimate; 5-split block reference"})
    return pd.DataFrame(rows)


def paired_model_differences(primary: pd.DataFrame):
    rows = []
    rng = np.random.default_rng(SEED + 10)
    for dataset in ["isic2019", "mura"]:
        d = primary[(primary.dataset == dataset) & primary.model.isin(MODELS)]
        wide = d.pivot_table(index=["split_id", "training_seed"], columns="model", values="balanced_accuracy").reindex(columns=MODELS)
        raw_rows = []
        for i, first in enumerate(MODELS):
            for second in MODELS[i+1:]:
                per_context = (wide[first] - wide[second]).rename("difference").reset_index()
                per_split = per_context.groupby("split_id")["difference"].mean().to_numpy(float)
                estimate = float(per_split.mean())
                boots = np.array([rng.choice(per_split, size=len(per_split), replace=True).mean() for _ in range(N_BOOT)])
                signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(per_split))))
                p = float(np.mean(np.abs(signs @ per_split / len(per_split)) >= abs(estimate) - 1e-15))
                raw_rows.append({"dataset":dataset,"model_1":first,"model_2":second,"paired_contexts":len(per_context),
                                 "split_clusters":len(per_split),"mean_paired_BA_difference_model1_minus_model2":estimate,
                                 "split_cluster_bootstrap_95CI_low":percentile(boots,.025),
                                 "split_cluster_bootstrap_95CI_high":percentile(boots,.975),
                                 "exact_split_block_sign_flip_p_raw":p,"permutation_assignments":len(signs),
                                 "effect_size_note":"paired BA difference; not a model-superiority claim"})
        order = sorted(range(len(raw_rows)), key=lambda j: raw_rows[j]["exact_split_block_sign_flip_p_raw"])
        adj, running = [1.0] * len(raw_rows), 0.0
        for rank, j in enumerate(order):
            running = max(running, min(1.0, (len(order)-rank) * raw_rows[j]["exact_split_block_sign_flip_p_raw"]))
            adj[j] = running
        for j, row in enumerate(raw_rows):
            row["holm_adjusted_p_within_dataset"] = adj[j]
            row["multiplicity_family"] = f"{dataset}: six paired four-CNN comparisons"
            rows.append(row)
    return pd.DataFrame(rows)


def model_pool_and_transform(all_ext: pd.DataFrame):
    rows, transform_rows = [], []
    for dataset in ["isic2019", "mura"]:
        shared = all_ext[(all_ext.dataset == dataset) & (all_ext.variant == "shared")]
        subset = shared[shared.split_id.isin([f"split_{i:02d}" for i in range(1,6)])]
        index = ["split_id","training_seed"]
        wide = subset.pivot_table(index=index, columns="model", values="balanced_accuracy")
        a_four=[]; a_five=[]
        for _, r in wide.iterrows():
            v4={m:float(r[m]) for m in MODELS}; v5={**v4,"swin_t":float(r["swin_t"])}
            a_four.append(top_and_margin(v4))
            a_five.append(top_and_margin(v5,ALL_MODELS))
        switch=[x[0]!=y[0] for x,y in zip(a_four,a_five)]
        rows.append({"dataset":dataset,"paired_contexts":len(wide),
                     "adding_shared_transform_Swin_changes_context_winner_fraction":float(np.mean(switch)),
                     "four_CNN_top_two_margin_mean":float(np.mean([x[1] for x in a_four])),
                     "five_model_top_two_margin_mean":float(np.mean([x[1] for x in a_five])),
                     "swin_shared_observed_selection_frequency":float(np.mean([x[0]=="swin_t" for x in a_five])),
                     "interpretation":"paired candidate-pool sensitivity only; not a model-superiority claim"})

        # B changes only Swin's preprocessing recipe; compare the same 15 contexts.
        a = all_ext[(all_ext.dataset==dataset)&(all_ext.model=="swin_t")&(all_ext.variant=="shared")&(all_ext.split_id.isin([f"split_{i:02d}" for i in range(1,6)]))]
        b = all_ext[(all_ext.dataset==dataset)&(all_ext.model=="swin_t")&(all_ext.variant=="swin_weight_eval")]
        pairs=a.merge(b,on=["dataset","model","split_id","training_seed"],suffixes=("_shared","_weight_specific"))
        delta=pairs.balanced_accuracy_weight_specific-pairs.balanced_accuracy_shared
        by_split=pairs.assign(delta=delta).groupby("split_id").delta.mean().to_numpy()
        rng=np.random.default_rng(SEED+2)
        boot=np.array([rng.choice(by_split,size=len(by_split),replace=True).mean() for _ in range(N_BOOT)])
        transform_rows.append({"dataset":dataset,"paired_contexts":len(pairs),
                     "mean_BA_difference_weight_specific_minus_shared":float(delta.mean()),
                     "split_cluster_bootstrap_95CI_low":percentile(boot,.025),
                     "split_cluster_bootstrap_95CI_high":percentile(boot,.975),
                     "fraction_contexts_weight_specific_BA_higher":float((delta>0).mean()),
                     "interpretation":"same Swin-T weights and architecture; sensitivity to evaluation input recipe"})
    return pd.DataFrame(rows),pd.DataFrame(transform_rows)


def mura_regions(metrics: pd.DataFrame):
    rows=[]
    d=metrics[(metrics.dataset=="mura")&(metrics.variant=="shared")&(metrics.model.isin(MODELS))]
    for region in sorted(set(k for item in d.mura_region_ba for k in item)):
        for model in MODELS:
            subset=d[d.model==model]
            vals=[]; splits=[]; n=[]
            for _, r in subset.iterrows():
                if region in r.mura_region_ba:
                    vals.append(r.mura_region_ba[region]); splits.append(r.split_id); n.append(r.mura_region_n[region])
            unique_splits=sorted(set(splits)); split_means=np.array([np.mean([v for v,s in zip(vals,splits) if s==sp]) for sp in unique_splits])
            rng=np.random.default_rng(SEED+3+MODELS.index(model))
            boot=np.array([rng.choice(split_means,size=len(split_means),replace=True).mean() for _ in range(N_BOOT)])
            rows.append({"body_region":region,"model":model,"contexts":len(vals),"mean_test_studies_per_context":float(np.mean(n)),
                         "mean_BA":float(np.mean(vals)),"SD_context_BA":float(np.std(vals,ddof=1)),
                         "split_cluster_bootstrap_CI_low":percentile(boot,.025),"split_cluster_bootstrap_CI_high":percentile(boot,.975)})
    frame=pd.DataFrame(rows)
    summaries=[]
    for region,g in frame.groupby("body_region"):
        top,margin,_=top_and_margin(dict(zip(g.model,g.mean_BA)))
        summaries.append({"body_region":region,"model_with_highest_mean_BA_descriptive":top,"top_two_mean_BA_gap":margin,
                          "study_units_per_context_mean":float(g.mean_test_studies_per_context.iloc[0]),"models":len(g),
                          "interpretation":"descriptive subgroup; no multiplicity-adjusted significance claim"})
    return frame,pd.DataFrame(summaries)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest, ext, audit = audit_and_collect()
    all_metrics,original_audit=attach_original(ext)
    primary, contexts=build_contexts(all_metrics)
    performance, stability=make_dataset_outputs(primary,contexts)
    margin=margin_model(contexts)
    crossfit=crossfit_summary(primary)
    model_pool,transform_sensitivity=model_pool_and_transform(ext)
    mura_perf,mura_summary=mura_regions(ext)
    paired_differences=paired_model_differences(primary)

    tables={
        "run_integrity_audit.csv":audit,
        "original_rule_A_raw_metrics_audit.csv":original_audit,
        "extension_run_metrics.csv":ext.drop(columns=["mura_region_ba","mura_region_n"]),
        "four_dataset_context_stability.csv":contexts.drop(columns=["model_values"]),
        "four_dataset_model_performance_and_selection.csv":performance,
        "four_dataset_ranking_stability_summary.csv":stability,
        "extension_paired_model_differences.csv":paired_differences,
        "margin_discordance_logistic_model.csv":margin,
        "isic_mura_split_block_crossfit.csv":crossfit,
        "swin_candidate_pool_and_input_recipe_sensitivity.csv":model_pool,
        "swin_preprocessing_sensitivity_paired.csv":transform_sensitivity,
        "mura_body_region_model_performance.csv":mura_perf,
        "mura_body_region_stability_summary.csv":mura_summary,
    }
    for name,frame in tables.items():
        frame.to_csv(OUT/name,index=False)

    summary={
        "analysis_id":"JIIM_revision_extension_reanalysis_v0.1",
        "created_local_date":"2026-10-03",
        "status":"completed",
        "registered_extension_role":"reviewer-requested post hoc extension; original 800-run matrix unchanged",
        "manifest_sha256":sha(MANIFEST),"manifest_rows":len(manifest),"integrity_passed_runs":len(audit),
        "primary_extension_runs":270,"Swin_preprocessing_sensitivity_runs":30,
        "datasets_in_comparison":["organamnist","sipakmed","isic2019","mura"],
        "common_model_pool":MODELS,"primary_metric":"balanced_accuracy",
        "resampling":"split-first, seed-second paired hierarchical bootstrap",
        "bootstrap_replicates":N_BOOT,"bootstrap_seed":SEED,
        "margin_model":"discordance ~ standardized[-log10(top-two BA margin + 1e-6)] + dataset fixed effects; dataset-by-split cluster bootstrap CI",
        "interpretation_boundary":"Descriptive benchmark-stage robustness across these four datasets; does not establish causal dataset-difficulty effects, universal medical-image generalization, clinical/patient-level generalization, or a best architecture.",
        "outputs":{name:len(frame) for name,frame in tables.items()},
        "preliminary_dataset_summary":stability.drop(columns=["completed_grid_top_model"],errors="ignore").to_dict(orient="records"),
        "preliminary_margin_model":margin.to_dict(orient="records"),
        "preliminary_swin_sensitivity":model_pool.to_dict(orient="records"),
        "preliminary_swin_preprocessing_sensitivity":transform_sensitivity.to_dict(orient="records"),
    }
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True),encoding="utf-8")
    report=["# JIIM 修订后扩展实验：初步统计分析", "", "本报告仅分析冻结的审稿后扩展与原始 800-run Rule-A 对照子集；扩展实验是 post hoc reviewer-requested extension，不并入原预注册矩阵。研究问题是原有“排名/选择不稳定”结论在新增数据集与评估情境中是否仍成立。绝对 balanced accuracy 仅用于描述任务饱和程度，不以模型性能排序寻找优胜架构。", "", "## 数据完整性", "", f"- 封存 manifest：300 行；SHA-256 与 seal 一致。", f"- 输出审计：{len(audit)}/300 通过运行状态、执行方案/环境哈希、冻结测试单位与标签、BA 复算、概率和 checkpoint 哈希核验；原始 800-run Rule-A 中 400 个配对比较 run 也与此前采集表逐项一致。", "- 主扩展：270 次 shared-preprocessing Rule-A runs；另 30 次 Swin 权重专属预处理敏感性 runs。", "- AutoDL 记录的训练环境一致：Python 3.10.21、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、RTX 4090 D；batch size 32、最多 60 epochs、AdamW。", "- 每个 run 的 balanced accuracy 均从保存的预测重新复算；相同 split × seed 的模型预测覆盖相同测试单位与标签。", "", "## 主要结果", "", "### 四数据集的排名稳定性", "", "| 数据集 | splits × seeds | LSO agreement (95% CI) | LSO discordance | 平均 top-two BA gap | gap <0.01 contexts |", "|---|---:|---:|---:|---:|---:|"]
    for r in stability.to_dict(orient="records"):
        report.append(f"| {r['dataset']} | {r['splits']} × {r['seeds_per_split']} | {r['leave_one_split_out_agreement']:.3f} ({r['split_then_seed_hierarchical_bootstrap_agreement_CI_low']:.3f}–{r['split_then_seed_hierarchical_bootstrap_agreement_CI_high']:.3f}) | {r['leave_one_split_out_discordance']:.3f} | {r['mean_top_two_margin']:.4f} | {r['contexts_margin_below_0.01']} / {r['contexts']} |")
    m=margin.iloc[0]
    report += ["", "### Margin 与排名不一致", "", f"跨四个数据集、以 dataset fixed effects 调整的 split-cluster bootstrap logistic model：top-two BA gap 每减少 1 SD，leave-one-split-out comparator discordance OR = {m.odds_ratio:.3f}（95% cluster-bootstrap CI {m.cluster_bootstrap_95CI_low:.3f}–{m.cluster_bootstrap_95CI_high:.3f}；{int(m.clusters)} dataset×split clusters，{int(m.contexts)} contexts）。该关系是关联性分析；discordance 本身由排名反转定义，不能解读为因果效应。", "", "### 独立 split-block 参照", "", "| 数据集 | 5/5 split-block cross-fitted agreement | scored contexts |", "|---|---:|---:|"]
    for r in crossfit.to_dict(orient="records"):
        report.append(f"| {r['dataset']} | {r['cross_fitted_agreement']:.3f} (partition range {r['min_partition_agreement']:.3f}–{r['max_partition_agreement']:.3f}) | {r['cross_fitted_contexts_scored']} |")
    report += ["", "### 候选池敏感性", "", "四 CNN 与扩展候选池只在相同 5 splits × 3 seeds contexts 配对比较。下表报告扩充候选池后选择身份改变的比例；它衡量结论对候选池的敏感性。Swin B 与 Swin A 比较只反映 Swin 权重专属输入预处理变化，不能归因于架构本身。", "", "| 数据集 | 配对 contexts | 扩充候选池后选择身份改变率 |", "|---|---:|---:|"]
    for r in model_pool.to_dict(orient="records"):
        report.append(f"| {r['dataset']} | {r['paired_contexts']} | {r['adding_shared_transform_Swin_changes_context_winner_fraction']:.3f} |")
    report += ["", "### 推断性补充（非模型优劣分析）", "", "补充表列出每个新增数据集内四 CNN 的六组配对 balanced-accuracy 差异、split-cluster bootstrap 95% CI、split-block exact sign-flip p 值及 Holm 校正。这些统计是回应审稿人推断性要求的辅助结果，不用于寻找或宣称最优架构。详见 `extension_paired_model_differences.csv`。", "", "Swin 权重专属输入 recipe 与 shared recipe 的配对估计和区间单独见 `swin_preprocessing_sensitivity_paired.csv`。"]
    report.extend(["", "### 初步结论", "", "扩展实验对原结论提供的是有边界、依赖数据集的支持。原 OrganAMNIST 与 SIPaKMeD 的平均 BA 分别约为 0.985、0.958，显示明显饱和；新增 ISIC 2019、MURA 的平均 BA 约为 0.471、0.781，整体较不饱和。因此新增实验检验原发现能否离开饱和任务，而非比较架构谁更强。", "", "在共同四 CNN 候选池下，ISIC 的 LSO agreement 与 split-block cross-fit agreement 均约为 0.90，显示此评估情境中的选择相对稳定；MURA 分别为 0.60 与 0.491，仍有较多不一致。新增数据并未复现统一方向：ISIC 不支持‘所有任务都不稳定’，MURA 则表明不稳定现象并未局限于原有饱和数据集。最稳妥的结论是排名/选择稳定性受数据集和评估情境影响，原结论不能无条件推广到所有医学影像 benchmark。", "", f"近零 margin 是否解释不稳定仍未解决：合并模型 OR={m.odds_ratio:.2f}，95% cluster-bootstrap CI {m.cluster_bootstrap_95CI_low:.2f}–{m.cluster_bootstrap_95CI_high:.2f}，包含 1；40 个 dataset×split clusters 的证据不足以支持稳健关联。Swin 扩充候选池后，15 个配对 contexts 中 winner 身份改变比例为 ISIC 40%、MURA 80%；这只说明候选池会影响选择结论，是稳定性分析的一部分，不表示任何模型更优。", "", "两个新增数据集不足以证明医学影像任务总体或患者层级临床泛化。绝对 BA、每模型选择频率、配对统计及 MURA 七部位描述统计均放在补充 CSV 中供审稿核查，不作为主结论。MURA subgroup 仅作估计与不确定性描述。数据质量审计中的 pHash 候选只按冻结方案作风险披露，未据此筛除测试样本或重排。", "", "## 输出表", ""])
    report.extend([f"- `{name}`：{len(frame)} rows" for name,frame in tables.items()])
    report.append("")
    (OUT/"preliminary_conclusions_zh.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps({"status":"completed","output_dir":str(OUT),"summary":summary},ensure_ascii=False,indent=2,allow_nan=True))


if __name__ == "__main__":
    main()
