"""Complete the six primary agreement intervals under the adopted crossed scheme.

User-authorized on 2026-10-06. Existing B1 outputs and raw runs remain read-only.
This does not compute an outer uncertainty interval for bootstrap frequency q.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
OUT = HERE / "crossed/agreement"
OLD = HERE / "results"
SOURCE = HERE / "crossed_core.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("jiim_b1_readonly", SOURCE)
    b1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = b1
    spec.loader.exec_module(b1)
    manifest = json.loads((OLD / "B1_analysis_manifest.json").read_text())
    paths = [SOURCE, Path(__file__), OLD / "B1_analysis_manifest.json",
             OLD / "B1_run_inputs.csv", OLD / "D02_ranking_stability.csv",
             OLD / "D02_resampling_design_sensitivity.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for path in paths[3:]:
        assert sha(path) == manifest["output_hashes"][path.name], path
    assert sha(SOURCE) == manifest["source_hashes"][str(SOURCE.relative_to(ROOT))]
    runs = pd.read_csv(paths[3], float_precision="round_trip")
    old_stability = pd.read_csv(paths[4], float_precision="round_trip").set_index("stratum")
    old_crossed = pd.read_csv(paths[5], float_precision="round_trip")
    raw_hashes = {}
    summaries, reference_counts, audit = [], [], []
    for dataset, policy, stage in [("organamnist", "A", "original"),
                                  ("organamnist", "B", "original"),
                                  ("sipakmed", "A", "original"),
                                  ("sipakmed", "B", "original"),
                                  ("isic2019", "A", "extension"),
                                  ("mura", "A", "extension")]:
        data = runs[(runs.dataset == dataset) & (runs.checkpoint_policy == policy)
                    & (runs.stage == stage) & runs.model.isin(b1.MODELS)]
        if stage == "extension":
            data = data[data.variant == "shared"]
        for row in data.itertuples():
            path = ROOT / row.metric_source_path
            actual_hash = sha(path)
            assert actual_hash == manifest["source_hashes"][row.metric_source_path], path
            metric = json.loads(path.read_text())
            assert float(metric["balanced_accuracy"]) == row.balanced_accuracy, row.run_id
            raw_hashes[row.metric_source_path] = actual_hash
        key = f"{dataset}_{policy}_cnn4_all{5 if stage == 'original' else 3}"
        s = b1.Stratum(key, dataset, policy, "primary_" + stage, data, b1.MODELS)
        obs = b1.observed(s)
        rng_seed = b1.keyseed(key + ":crossed")
        sp, se = b1.draw_indices(s, np.random.default_rng(rng_seed),
                                 mode="crossed", tag="agreement completion")
        assert np.array_equal(se, np.broadcast_to(se[:, :1], se.shape))
        sample = s.arr[sp[:, :, None], se]
        means = sample.mean(axis=(1, 2))
        reference = means.argmax(axis=-1)
        winners = sample.argmax(axis=-1)
        agreement = (winners == reference[:, None, None]).mean(axis=(1, 2))
        discordance = 1 - agreement
        low, high = b1.interval(agreement)
        dlow, dhigh = 1 - high, 1 - low
        # Re-express the statistic through model winner counts for all draws.
        counts = np.stack([(winners == j).sum(axis=(1, 2))
                           for j in range(len(s.models))], axis=1)
        alternative = counts[np.arange(b1.NBOOT), reference] / (len(s.splits) * len(s.seeds))
        assert np.array_equal(agreement, alternative)
        # Scalar direct indexing confirms ordering and the moving reference.
        for draw in range(32):
            grid = np.array([[s.arr[int(sp[draw, i]), int(se[draw, i, k])]
                              for k in range(len(s.seeds))] for i in range(len(s.splits))])
            scalar_ref = grid.mean(axis=(0, 1)).argmax()
            scalar_agreement = (grid.argmax(axis=-1) == scalar_ref).mean()
            assert scalar_ref == reference[draw] and scalar_agreement == agreement[draw]
        q_values = (reference[:, None] == np.arange(len(s.models))[None, :]).mean(axis=0)
        historic = old_crossed[(old_crossed.stratum == key) & (old_crossed.resampling == "crossed")]
        for j, model in enumerate(s.models):
            old_q = historic[historic.model == model].iloc[0].paired_bootstrap_selection_frequency
            assert abs(q_values[j] - old_q) < 1e-14, (key, model)
            reference_counts.append(dict(stratum=key, model=model,
                                         reference_draw_count=int((reference == j).sum()),
                                         q=float(q_values[j])))
        old = old_stability.loc[key]
        assert obs["full_agreement"] == old.full_grid_in_sample_agreement
        assert s.models[obs["full_reference"]] == old.reference_top_model
        summaries.append({**s.info(), "reference_top_model":s.models[obs["full_reference"]],
            "agreement":obs["full_agreement"], "agreement_CI_low":float(low),
            "agreement_CI_high":float(high), "discordance":1-obs["full_agreement"],
            "discordance_CI_low":float(dlow), "discordance_CI_high":float(dhigh),
            "q_reproduced":float(q_values[obs["full_reference"]]),
            "nested_agreement_CI_low":old.full_grid_agreement_CI_low,
            "nested_agreement_CI_high":old.full_grid_agreement_CI_high,
            "lower_endpoint_change":float(low-old.full_grid_agreement_CI_low),
            "upper_endpoint_change":float(high-old.full_grid_agreement_CI_high),
            "reference_switch_draws":int((reference != obs["full_reference"]).sum()),
            "bootstrap_replicates":b1.NBOOT, "resampling":"paired crossed split-seed",
            "reference_rule":"re-estimated from resampled mean BA in every draw",
            "CI_method":"95% percentile; NumPy quantile linear interpolation",
            "rng_seed":str(rng_seed)})
        np.savez_compressed(OUT / (key + "_draws.npz"),
                            split_indices=sp, global_seed_indices=se[:, 0],
                            reference_indices=reference, agreement=agreement,
                            discordance=discordance, model_ids=np.array(s.models))
        audit.append(dict(stratum=key, status="PASS", raw_metrics=len(data),
                          crossed_global_seed_vector="PASS", q_replay_all_models="EXACT_COUNTS",
                          observed_reference_and_point="UNCHANGED",
                          alternative_statistic_all_10000_draws="EXACT",
                          scalar_direct_indexing_draws=32))
        print(f"{key}: {obs['full_agreement']:.3f} [{low:.6f}, {high:.6f}]; q replay exact", flush=True)
    assert len(raw_hashes) == 1040
    for name, rows in [("Crossed_Agreement_Summary.csv", summaries),
                       ("Crossed_Reference_Draw_Counts.csv", reference_counts),
                       ("Crossed_Agreement_Computation_Audit.csv", audit)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.17g")
    (OUT / "Raw_Metric_Source_Hashes.json").write_text(
        json.dumps(raw_hashes, indent=2, ensure_ascii=False)+"\n")
    # Confirm every read-only analysis input and raw source is unchanged at completion.
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    assert all(sha(ROOT / p) == h for p, h in raw_hashes.items())
    result = dict(analysis_id="JIIM_crossed_agreement_completion",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  status="COMPUTED_AND_SOURCE_CHECKED", authorization="Author: 好的，补齐。",
                  scope="six primary strata; agreement and complementary discordance only",
                  training_runs_added=0, raw_metric_inputs=1040, strata=6,
                  resampling="paired crossed split-seed", bootstrap_replicates=b1.NBOOT,
                  base_seed=b1.BASE_SEED, seed_derivation="SHA256(stratum:crossed) first8bytes big endian XOR base_seed",
                  reference_reestimated_every_draw=True, degenerate_draws_rejected=0,
                  tie_rule="alphabetically sorted fixed candidate IDs; first argmax",
                  q_outer_data_interval_computed=False, python=platform.python_version(),
                  executable=sys.executable, numpy=np.__version__, pandas=pd.__version__,
                  elapsed_seconds=time.perf_counter()-started, input_hashes=hashes,
                  output_hashes={p.name:sha(p) for p in sorted(OUT.iterdir())},
                  verification="Same-agent arithmetic/source checks; not independent-person review",
                  application_state="New statistical materials; manuscript/Word integration pending")
    (OUT / "Crossed_Agreement_Manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(f"Completed six strata in {result['elapsed_seconds']:.2f}s; raw inputs unchanged.", flush=True)


if __name__ == "__main__":
    main()
