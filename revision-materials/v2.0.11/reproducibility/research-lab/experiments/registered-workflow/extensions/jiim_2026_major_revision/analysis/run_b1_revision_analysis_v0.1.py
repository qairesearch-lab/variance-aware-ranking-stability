"""Versioned B1 revision computations; raw run outputs and earlier analyses are read only."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ROOT = HERE.parents[5]
OUT = HERE / 'results_v0.3'
WS = ROOT / 'research-lab/reports/JIIM_Revision_Workspace'
SPEC = WS / '04_B1_Statistical_Analysis/B1_Analysis_Specification_v0.1.md'
MODELS = sorted(['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0'])
COMMON = [42, 52, 62]
NBOOT = 10_000
NREF = 2_000
BASE_SEED = 2026100301
SOURCE_HASHES = {}
DRAW_AUDIT = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path)
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def write(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, float_format='%.17g')
    return frame


def keyseed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') ^ BASE_SEED


def interval(values):
    return np.quantile(values, [.025, .975], axis=0)


def ranks(values):
    return np.argsort(np.argsort(-values, axis=-1, kind='stable'), axis=-1, kind='stable') + 1


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def signflip(difference):
    n = len(difference)
    signs = np.asarray(list(itertools.product([-1., 1.], repeat=n)))
    null = signs @ difference / n
    return float(np.mean(np.abs(null) >= abs(difference.mean()) - 1e-15)), len(signs)


def load_runs():
    records = []
    old_manifest = pd.read_csv(source(ROOT / 'research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv'))
    ext_manifest = pd.read_csv(source(EXT / 'run_manifests/jiim_extension_300_v0.6.csv'))
    old_cache = pd.read_csv(source(ROOT / 'research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv')).set_index('run_id')
    ext_cache = pd.read_csv(source(HERE / 'results_v0.1/extension_run_metrics.csv')).set_index('run_id')
    for stage, manifest, cache in [('original', old_manifest, old_cache), ('extension', ext_manifest, ext_cache)]:
        for row in manifest.to_dict('records'):
            run = ROOT / row['output_dir'] if stage == 'original' else EXT / 'runs' / row['run_id']
            cfg = read_json(run / 'run_config.json')
            metric = read_json(run / 'metrics.json')
            status = read_json(run / 'run_status.json')
            assert status['status'] in ('completed', 'rerun_completed')
            assert status['run_id'] == metric['run_id'] == cfg['run_id'] if stage == 'extension' else status['run_id'] == metric['run_id'] == cfg['manifest_row']['run_id']
            assert metric['dataset'] == row['dataset']
            assert abs(float(metric['balanced_accuracy']) - float(cache.loc[row['run_id'], 'balanced_accuracy'])) < 1e-12
            if stage == 'original':
                for field in ['model', 'split_id', 'training_seed', 'checkpoint_policy', 'split_hash', 'config_hash']:
                    assert str(metric[field]) == str(row[field])
                variant, unit = 'original', 'image'
            else:
                assert not cfg.get('technical_smoke', True)
                assert cfg['model'] == row['model'] and cfg['variant'] == row['variant']
                assert int(cfg['training_seed']) == int(row['training_seed'])
                variant, unit = row['variant'], metric['prediction_unit']
            records.append(dict(run_id=row['run_id'], stage=stage, dataset=row['dataset'],
                split_id=row['split_id'], training_seed=int(row['training_seed']), model=row['model'],
                checkpoint_policy=row['checkpoint_policy'] if stage == 'original' else 'A',
                variant=variant, endpoint_unit=unit, balanced_accuracy=float(metric['balanced_accuracy']),
                accuracy=float(metric.get('accuracy', np.nan)), selected_epoch=metric.get('selected_epoch'),
                metric_source_path=str((run / 'metrics.json').relative_to(ROOT))))
    result = pd.DataFrame(records)
    assert len(result) == result.run_id.nunique() == 1100
    assert Counter(result.stage) == Counter(original=800, extension=300)
    write('B1_run_inputs_v0.1.csv', result)
    source(HERE / 'input_audit_v0.1/B0_input_audit_manifest_v0.1.json')
    source(HERE / 'input_audit_v0.1/D05_original_common_three_seeds_input_v0.1.csv')
    return result


@dataclass
class Stratum:
    key: str
    dataset: str
    policy: str
    role: str
    data: pd.DataFrame
    models: list[str]

    def __post_init__(self):
        self.models = sorted(self.models)
        assert not self.data.duplicated(['split_id', 'training_seed', 'model']).any()
        self.splits = sorted(self.data.split_id.unique())
        self.seeds = sorted(self.data.training_seed.unique())
        wide = self.data.pivot(index=['split_id', 'training_seed'], columns='model', values='balanced_accuracy')
        self.arr = np.asarray([[wide.loc[(s, k), self.models].to_numpy(float) for k in self.seeds] for s in self.splits])
        assert self.arr.shape == (len(self.splits), len(self.seeds), len(self.models))
        assert len(self.data) == self.arr.size and np.isfinite(self.arr).all()

    def info(self):
        return dict(stratum=self.key, dataset=self.dataset, checkpoint_policy=self.policy, role=self.role,
                    endpoint_unit=self.data.endpoint_unit.iloc[0], splits=len(self.splits),
                    seeds=len(self.seeds), models=len(self.models), contexts=len(self.splits)*len(self.seeds),
                    runs=len(self.data), candidate_pool='|'.join(self.models), seed_ids='|'.join(map(str,self.seeds)))


def make_strata(runs):
    strata = []
    for ds in ['organamnist', 'sipakmed']:
        for policy in ['A', 'B']:
            data = runs[(runs.dataset == ds) & (runs.checkpoint_policy == policy)]
            strata.append(Stratum(f'{ds}_{policy}_cnn4_all5', ds, policy, 'primary_original', data, MODELS))
            common = data[data.training_seed.isin(COMMON)]
            strata.append(Stratum(f'{ds}_{policy}_cnn4_common3', ds, policy, 'common_seed_sensitivity', common, MODELS))
    for ds in ['isic2019', 'mura']:
        data = runs[(runs.dataset == ds) & (runs.variant == 'shared') & runs.model.isin(MODELS)]
        strata.append(Stratum(f'{ds}_A_cnn4_all3', ds, 'A', 'primary_extension', data, MODELS))
        short = data[data.split_id.isin([f'split_{s:02d}' for s in range(1, 6)])]
        strata.append(Stratum(f'{ds}_A_cnn4_pool_contexts', ds, 'A', 'candidate_pool_control', short, MODELS))
        for variant in ['shared', 'swin_weight_eval']:
            swin = runs[(runs.dataset == ds) & (runs.model == 'swin_t') & (runs.variant == variant)]
            combined = pd.concat([short, swin])
            strata.append(Stratum(f'{ds}_A_pool5_{variant}', ds, 'A', 'candidate_pool_sensitivity', combined, MODELS + ['swin_t']))
    write('B1_stratum_inventory_v0.1.csv', [s.info() for s in strata])
    assert set().union(*(set(s.data.run_id) for s in strata)) == set(runs.run_id)
    return strata


def draw_indices(s, rng, n=NBOOT, mode='nested', tag='', require_reference=False):
    ns, nk, _ = s.arr.shape
    splits = rng.integers(0, ns, (n, ns))
    bad = np.all(splits == splits[:, :1], axis=1)
    rejected = 0
    while require_reference and bad.any():
        rejected += int(bad.sum())
        splits[bad] = rng.integers(0, ns, (int(bad.sum()), ns))
        bad = np.all(splits == splits[:, :1], axis=1)
    if mode == 'nested':
        seeds = rng.integers(0, nk, (n, ns, nk))
    elif mode == 'crossed':
        seeds = np.broadcast_to(rng.integers(0, nk, (n, 1, nk)), (n, ns, nk))
    elif mode == 'split_only':
        seeds = np.broadcast_to(np.arange(nk)[None, None, :], (n, ns, nk))
    else:
        raise ValueError(mode)
    DRAW_AUDIT.append(dict(stratum=s.key, purpose=tag, resampling=mode, requested=n,
                           accepted=n, degenerate_draws_resampled=rejected))
    return splits, seeds


def valid_lso_sample(s, splits, seeds, mode, rng, tag):
    bad = np.all(splits == splits[:, :1], axis=1)
    if not bad.any():
        return s.arr[splits[:,:,None],seeds], splits
    repaired_splits = splits.copy()
    repaired_seeds = seeds.copy()
    replacement, replacement_seeds = draw_indices(s,rng,n=int(bad.sum()),mode=mode,
        tag=tag+' LSO degenerate replacements',require_reference=True)
    repaired_splits[bad]=replacement
    repaired_seeds[bad]=replacement_seeds
    DRAW_AUDIT.append(dict(stratum=s.key,purpose=tag+' LSO rejects from unconditional draws',
        resampling=mode,requested=len(splits),accepted=len(splits),degenerate_draws_resampled=int(bad.sum())))
    return s.arr[repaired_splits[:,:,None],repaired_seeds],repaired_splits


def lso_from_sample(sample, chosen):
    # b=replicate, i=held split occurrence, j=reference split occurrence.
    mask = chosen[:, :, None] != chosen[:, None, :]
    count = mask.sum(axis=2)
    assert np.all(count > 0)
    ref = np.einsum('bij,bjm->bim', mask, sample.mean(axis=2)) / count[:, :, None]
    winners = np.argmax(sample, axis=-1)
    ref_winners = np.argmax(ref, axis=-1)
    return (winners == ref_winners[:, :, None]).mean(axis=(1, 2)), ref_winners


def observed(s):
    a = s.arr
    ns, nk, nm = a.shape
    winner = np.argmax(a, axis=-1)
    reference = np.argmax(a.mean(axis=(0,1)))
    ref = (a.sum(axis=(0,1))[None, :] - a.sum(axis=1)) / ((ns-1)*nk)
    lso_ref = np.argmax(ref, axis=-1)
    margin = np.sort(a, axis=-1)[..., -1] - np.sort(a, axis=-1)[..., -2]
    counts = np.bincount(winner.ravel(), minlength=nm)
    nonzero = counts[counts > 0] / winner.size
    rank = ranks(a)
    rank_sums = rank.sum(axis=(0,1))
    w = 12*np.sum((rank_sums - ns*nk*(nm+1)/2)**2) / ((ns*nk)**2*(nm**3-nm))
    return dict(winner=winner, full_reference=reference, lso_reference=lso_ref,
                margin=margin, frequency=counts/winner.size, rank=rank,
                full_agreement=float((winner == reference).mean()),
                lso_agreement=float((winner == lso_ref[:, None]).mean()),
                entropy=float(-np.sum(nonzero*np.log(nonzero))/np.log(nm)), kendall_w=float(w))


def analyze_stratum(s):
    obs = observed(s)
    rng = np.random.default_rng(keyseed(s.key + ':nested'))
    sp, se = draw_indices(s, rng, tag='D02 summaries')
    sample = s.arr[sp[:, :, None], se]
    means = sample.mean(axis=(1,2))
    winners = np.argmax(sample, axis=-1)
    mean_winner = np.argmax(means, axis=-1)
    freq = np.stack([(winners == i).mean(axis=(1,2)) for i in range(len(s.models))], axis=1)
    mean_ci, freq_ci = interval(means), interval(freq)
    margin_samples = np.sort(sample, axis=-1)[..., -1] - np.sort(sample, axis=-1)[..., -2]
    full_agree_boot = (winners == mean_winner[:, None, None]).mean(axis=(1,2))
    lso_sample,lso_indices=valid_lso_sample(s,sp,se,'nested',rng,'D02')
    lso_boot, _ = lso_from_sample(lso_sample, lso_indices)
    mci = interval(margin_samples.mean(axis=(1,2)))
    model_rows = []
    for j, model in enumerate(s.models):
        boot_p = float((mean_winner == j).mean())
        model_rows.append({**s.info(), 'model':model, 'mean_BA':float(s.arr[...,j].mean()),
            'SD_context_BA':float(s.arr[...,j].std(ddof=1)), 'mean_BA_CI_low':mean_ci[0,j],
            'mean_BA_CI_high':mean_ci[1,j], 'observed_selection_frequency':obs['frequency'][j],
            'selection_frequency_CI_low':freq_ci[0,j], 'selection_frequency_CI_high':freq_ci[1,j],
            'paired_bootstrap_selection_frequency':boot_p,
            'selection_bootstrap_MC_SE':math.sqrt(boot_p*(1-boot_p)/NBOOT),
            'mean_rank':float(obs['rank'][...,j].mean()), 'rank_SD':float(obs['rank'][...,j].std(ddof=1)),
            'bootstrap_replicates':NBOOT, 'CI_method':'paired split-then-seed percentile 95%; fixed dataset',
            'zero_frequency_boundary':bool(obs['frequency'][j] == 0)})
    stability = {**s.info(), 'reference_top_model':s.models[obs['full_reference']],
        'full_grid_in_sample_agreement':obs['full_agreement'],
        'full_grid_agreement_CI_low':interval(full_agree_boot)[0], 'full_grid_agreement_CI_high':interval(full_agree_boot)[1],
        'identity_correct_LSO_agreement':obs['lso_agreement'],
        'LSO_CI_low':interval(lso_boot)[0], 'LSO_CI_high':interval(lso_boot)[1],
        'mean_context_top_two_BA_difference':float(obs['margin'].mean()),
        'median_context_top_two_BA_difference':float(np.median(obs['margin'])),
        'mean_context_difference_CI_low':mci[0], 'mean_context_difference_CI_high':mci[1],
        'full_grid_top_two_BA_difference':float(np.diff(np.sort(s.arr.mean(axis=(0,1)))[-2:])[0]),
        'normalized_selection_entropy':obs['entropy'], 'Kendall_W_deterministic_tie_resolved':obs['kendall_w'],
        'exact_context_top_ties':int(np.sum((s.arr == s.arr.max(axis=-1,keepdims=True)).sum(axis=-1) > 1)),
        'bootstrap_replicates':NBOOT}
    contexts = []
    for i, split in enumerate(s.splits):
        for k, seed in enumerate(s.seeds):
            contexts.append({**s.info(), 'split_id':split, 'training_seed':seed,
                'selected_model':s.models[obs['winner'][i,k]], 'LSO_reference_model':s.models[obs['lso_reference'][i]],
                'full_grid_agreement':int(obs['winner'][i,k] == obs['full_reference']),
                'LSO_agreement':int(obs['winner'][i,k] == obs['lso_reference'][i]),
                'top_two_BA_difference':obs['margin'][i,k]})
    pairs = []
    for i, j in itertools.combinations(range(len(s.models)), 2):
        delta = s.arr[...,i] - s.arr[...,j]
        bdelta = means[:,i] - means[:,j]
        lo, hi = interval(bdelta)
        p, assignments = signflip(delta.mean(axis=1))
        pairs.append({**s.info(), 'model_1':s.models[i], 'model_2':s.models[j],
            'mean_paired_BA_difference_model1_minus_model2':float(delta.mean()),
            'paired_CI_low':lo, 'paired_CI_high':hi, 'sign_flip_p_raw_conditional':p,
            'sign_assignments':assignments, 'minimum_two_sided_p':2/assignments,
            'multiplicity_family':s.key + ': all paired model BA differences',
            'p_value_scope':'sensitivity conditional on independent sign symmetry of split-average differences; not external-population proof'})
    adj = holm([r['sign_flip_p_raw_conditional'] for r in pairs])
    for row, p in zip(pairs, adj):
        row['holm_p_within_stratum'] = p
    resampling = []
    for mode in ['split_only', 'crossed']:
        rr = np.random.default_rng(keyseed(s.key + ':' + mode))
        ss, kk = draw_indices(s, rr, mode=mode, tag='D02 resampling sensitivity')
        sampled = s.arr[ss[:,:,None], kk]
        mm = sampled.mean(axis=(1,2))
        ww = np.argmax(sampled, axis=-1)
        lso_sample,lso_indices=valid_lso_sample(s,ss,kk,mode,rr,'D02 sensitivity')
        lso, _ = lso_from_sample(lso_sample, lso_indices)
        for j, model in enumerate(s.models):
            mlow,mhigh = interval(mm[:,j]); flow,fhigh = interval((ww == j).mean(axis=(1,2)))
            resampling.append({**s.info(), 'resampling':mode, 'model':model,
                'mean_BA_CI_low':mlow,'mean_BA_CI_high':mhigh,
                'observed_frequency_CI_low':flow,'observed_frequency_CI_high':fhigh,
                'paired_bootstrap_selection_frequency':float((np.argmax(mm,axis=-1)==j).mean()),
                'LSO_CI_low':interval(lso)[0], 'LSO_CI_high':interval(lso)[1]})
    return model_rows, stability, contexts, pairs, resampling


def batch_logistic(x, y):
    """Separately written batched Newton implementation, matching the historical tiny ridge."""
    b, _, p = x.shape
    beta = np.zeros((b,p))
    penalty = np.eye(p)*1e-8
    penalty[0,0] = 0
    active = np.ones(b, bool)
    for iteration in range(60):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        xx = x[ids]
        eta = np.clip(np.einsum('bnp,bp->bn', xx, beta[ids]), -25, 25)
        probability = 1 / (1 + np.exp(-eta))
        weight = np.maximum(probability*(1-probability), 1e-8)
        hessian = np.einsum('bnp,bn,bnq->bpq', xx, weight, xx) + penalty
        score = np.einsum('bnp,bn->bp', xx, y[ids] - probability) - beta[ids] @ penalty
        step = np.linalg.solve(hessian, score[...,None])[...,0]
        beta[ids] += step
        active[ids] = np.max(np.abs(step),axis=1) >= 1e-8
    return beta[:,-1], active


def audit_d01(strata):
    selected = sorted([s for s in strata if s.role.startswith('primary') and s.policy == 'A'], key=lambda s:s.dataset)
    old_lso = pd.read_csv(source(HERE / 'results_v0.2/d01_identity_aware_lso_intervals.csv')).set_index('dataset')
    old_margin = pd.read_csv(source(HERE / 'results_v0.2/d01_identity_aware_margin_model.csv')).iloc[0]
    source(HERE / 'reanalyze_d01_split_identity_v0.2.py')
    rng = np.random.default_rng(20261003)
    rows = []
    for s in selected:
        ns,nk,_ = s.arr.shape
        split_draws, seed_draws, rejects = [], [], 0
        for _ in range(NBOOT):
            while True:
                chosen = rng.integers(0,ns,size=ns)
                if len(set(chosen)) >= 2:
                    break
                rejects += 1
            split_draws.append(chosen)
            seed_draws.append(rng.integers(0,nk,size=(ns,nk)))
        chosen=np.asarray(split_draws); seeds=np.asarray(seed_draws)
        sample=s.arr[chosen[:,:,None], seeds]
        result,_=lso_from_sample(sample,chosen)
        lo,hi=interval(result)
        point=observed(s)['lso_agreement']
        legacy=old_lso.loc[s.dataset]
        error=max(abs(point-legacy.observed_LSO_agreement),abs(lo-legacy.identity_aware_bootstrap_CI_low),abs(hi-legacy.identity_aware_bootstrap_CI_high))
        assert error < 1e-12
        rows.append(dict(dataset=s.dataset, contexts=ns*nk, recomputed_LSO=point, replay_CI_low=lo,
            replay_CI_high=hi, legacy_max_abs_difference=error, all_same_identity_excluded=True,
            bootstrap_replicates=NBOOT, degenerate_draws_resampled=rejects,
            audit_kind='different implementation by same agent; not independent-person review'))
    write('D01_implementation_replay_v0.1.csv',rows)
    ds_order=[s.dataset for s in selected]
    point_margin=np.concatenate([observed(s)['margin'].ravel() for s in selected])
    log=np.log10(point_margin+1e-6); center=float(log.mean());scale=float(log.std(ddof=1))
    ds=np.concatenate([[s.dataset]*(s.arr.shape[0]*s.arr.shape[1]) for s in selected])
    y=np.concatenate([(observed(s)['winner']!=observed(s)['lso_reference'][:,None]).ravel() for s in selected])
    x=np.column_stack([np.ones(len(y))]+[(ds==name).astype(float) for name in ds_order[1:]]+[-(log-center)/scale])
    coef,active=batch_logistic(x[None,:,:],y[None,:])
    point_or=float(np.exp(coef[0]))
    boot_or=[]; nonconverged=0; errors=0
    for first in range(0,NBOOT,256):
        count=min(256,NBOOT-first)
        all_y=[];all_log=[];all_ds=[]
        # Generate RNG draws replicate-first to reproduce historical call order.
        draws=[[] for _ in selected];seed_draws=[[] for _ in selected]
        for b in range(count):
            for j,s in enumerate(selected):
                ns,nk,_=s.arr.shape
                while True:
                    chosen=rng.integers(0,ns,size=ns)
                    if len(set(chosen))>=2: break
                draws[j].append(chosen);seed_draws[j].append(rng.integers(0,nk,size=(ns,nk)))
        for j,s in enumerate(selected):
            chosen=np.asarray(draws[j]);seed=np.asarray(seed_draws[j]);sample=s.arr[chosen[:,:,None],seed]
            _,ref=lso_from_sample(sample,chosen)
            all_y.append((np.argmax(sample,axis=-1)!=ref[:,:,None]).reshape(count,-1))
            sorted_sample=np.sort(sample,axis=-1)
            all_log.append(np.log10(sorted_sample[...,-1]-sorted_sample[...,-2]+1e-6).reshape(count,-1))
            all_ds.extend([s.dataset]*(s.arr.shape[0]*s.arr.shape[1]))
        yy=np.concatenate(all_y,axis=1);zz=(np.concatenate(all_log,axis=1)-center)/scale
        dd=np.asarray(all_ds)
        xx=np.stack([np.ones_like(zz)]+[np.broadcast_to((dd==name).astype(float),zz.shape) for name in ds_order[1:]]+[-zz],axis=-1)
        beta,not_converged=batch_logistic(xx,yy)
        nonconverged+=int(not_converged.sum())
        boot_or.extend(np.exp(beta).tolist())
    lo,hi=interval(boot_or)
    diff=max(abs(point_or-old_margin.odds_ratio),abs(lo-old_margin.identity_aware_cluster_bootstrap_CI_low),abs(hi-old_margin.identity_aware_cluster_bootstrap_CI_high))
    assert diff < 1e-6
    result=dict(term=old_margin.term, recomputed_odds_ratio=point_or,replay_CI_low=lo,replay_CI_high=hi,
        legacy_max_abs_difference=diff,bootstrap_replicates=NBOOT,nonconverged_fits=nonconverged,
        point_nonconverged=bool(active[0]),ridge=1e-8,epsilon=1e-6,
        interpretation='associational and supplementary; fixed four datasets; not a universal threshold',
        audit_kind='different implementation by same agent; independent-person review remains pending')
    write('D01_margin_implementation_replay_v0.1.csv',[result])
    return rows,result


def policy_comparisons(strata):
    rows=[]; selection=[]
    for dataset in ['organamnist','sipakmed']:
        for role in ['primary_original','common_seed_sensitivity']:
            a=next(s for s in strata if s.dataset==dataset and s.role==role and s.policy=='A')
            b=next(s for s in strata if s.dataset==dataset and s.role==role and s.policy=='B')
            assert a.splits==b.splits and a.seeds==b.seeds and a.models==b.models
            rng=np.random.default_rng(keyseed(dataset+role+':policy'))
            sp,se=draw_indices(a,rng,tag='D02 paired A/B')
            delta=b.arr-a.arr
            samples=delta[sp[:,:,None],se].mean(axis=(1,2))
            ci=interval(samples)
            for j,model in enumerate(a.models):
                p,assignments=signflip(delta[...,j].mean(axis=1))
                rows.append({**a.info(),'model':model,'comparison':'B_minus_A',
                    'mean_paired_BA_difference':float(delta[...,j].mean()),'CI_low':ci[0,j],'CI_high':ci[1,j],
                    'sign_flip_p_raw_conditional':p,'sign_assignments':assignments,'multiplicity_family':role+': eight policy BA differences'})
            a_sample=a.arr[sp[:,:,None],se];b_sample=b.arr[sp[:,:,None],se]
            aw=np.argmax(a_sample,axis=-1);bw=np.argmax(b_sample,axis=-1)
            oa,ob=observed(a),observed(b)
            for j,model in enumerate(a.models):
                diff=(bw==j).mean(axis=(1,2))-(aw==j).mean(axis=(1,2))
                lo,hi=interval(diff)
                selection.append({**a.info(),'model':model,'frequency_difference_B_minus_A':float(ob['frequency'][j]-oa['frequency'][j]),
                                  'CI_low':lo,'CI_high':hi,'interpretation':'paired selection-frequency sensitivity, not model quality'})
            switch=(aw!=bw).mean(axis=(1,2));lo,hi=interval(switch)
            selection.append({**a.info(),'model':'ALL','context_winner_changed_fraction':float((oa['winner']!=ob['winner']).mean()),
                              'CI_low':lo,'CI_high':hi,'interpretation':'paired checkpoint-rule effect on selected identity'})
    for family in sorted({row['multiplicity_family'] for row in rows}):
        subset=[r for r in rows if r['multiplicity_family']==family]
        for row,p in zip(subset,holm([r['sign_flip_p_raw_conditional'] for r in subset])):
            row['holm_p_policy_family']=p
    write('D02_policy_B_minus_A_v0.1.csv',rows)
    write('D02_policy_selection_sensitivity_v0.1.csv',selection)
    return rows,selection


def swin_sensitivity(runs,strata):
    transform=[];pool=[]
    for ds in ['isic2019','mura']:
        a=runs[(runs.dataset==ds)&(runs.model=='swin_t')&(runs.variant=='shared')]
        b=runs[(runs.dataset==ds)&(runs.model=='swin_t')&(runs.variant=='swin_weight_eval')]
        paired=a.merge(b,on=['split_id','training_seed'],suffixes=('_shared','_weight_eval'),validate='one_to_one')
        delta=paired.assign(delta=paired.balanced_accuracy_weight_eval-paired.balanced_accuracy_shared).pivot(index='split_id',columns='training_seed',values='delta').to_numpy()
        ns,nk=delta.shape;assert (ns,nk)==(5,3)
        rng=np.random.default_rng(keyseed(ds+':swin_input'))
        sp=rng.integers(0,ns,(NBOOT,ns));se=rng.integers(0,nk,(NBOOT,ns,nk))
        boot=delta[sp[:,:,None],se].mean(axis=(1,2));lo,hi=interval(boot)
        p,assignments=signflip(delta.mean(axis=1))
        transform.append(dict(dataset=ds,paired_contexts=15,split_clusters=5,
            mean_BA_difference_weight_eval_minus_shared=float(delta.mean()),CI_low=lo,CI_high=hi,
            sign_flip_p_raw_conditional=p,sign_assignments=assignments,minimum_two_sided_p=2/assignments,
            interpretation='same architecture and weights; input-recipe sensitivity; five split clusters'))
        control=next(s for s in strata if s.key==f'{ds}_A_cnn4_pool_contexts')
        for variant in ['shared','swin_weight_eval']:
            full=next(s for s in strata if s.key==f'{ds}_A_pool5_{variant}')
            ctrl=observed(control); obs=observed(full)
            cw=np.asarray(control.models)[ctrl['winner']];fw=np.asarray(full.models)[obs['winner']]
            rng=np.random.default_rng(keyseed(ds+':candidate_pool:'+variant))
            sp,se=draw_indices(control,rng,tag='D02 candidate pool paired sensitivity')
            difference=(cw!=fw).astype(float)
            boot=difference[sp[:,:,None],se].mean(axis=(1,2));lo,hi=interval(boot)
            pool.append(dict(dataset=ds,swin_variant=variant,contexts=15,split_clusters=5,
                selected_identity_changed_fraction=float(difference.mean()),CI_low=lo,CI_high=hi,
                observed_mean_top_two_difference_four=ctrl['margin'].mean(),observed_mean_top_two_difference_five=obs['margin'].mean(),
                interpretation='paired candidate-pool sensitivity; no cross-architecture superiority claim'))
    for row,p in zip(transform,holm([r['sign_flip_p_raw_conditional'] for r in transform])):
        row['holm_p_two_input_comparisons']=p
    write('D02_swin_input_sensitivity_v0.1.csv',transform)
    write('D02_candidate_pool_sensitivity_v0.1.csv',pool)
    return transform,pool


def combination_weights(n, size):
    indices=list(itertools.combinations(range(n),size))
    weights=np.zeros((len(indices),n))
    for i,selected in enumerate(indices):
        weights[i,list(selected)]=1/size
    return weights


def budget_analysis(s):
    a=s.arr;ns,nk,nm=a.shape
    assert ns==10
    sw={n:combination_weights(5,n) for n in range(1,6)}
    kw={n:combination_weights(nk,n) for n in range(1,nk+1)}
    partitions=[]
    all_ids=set(range(10))
    pairs=list(itertools.combinations(range(nm),2))
    for partition,disc_tuple in enumerate(itertools.combinations(range(10),5)):
        discovery=list(disc_tuple);reference=sorted(all_ids-set(discovery))
        assert not set(discovery)&set(reference)
        dd=a[discovery];rr=a[reference]
        ref_mean=rr.mean(axis=(0,1));ref_rank=ranks(ref_mean)
        ref_top=int(np.argmax(ref_mean))
        rank_mean=ranks(rr).mean(axis=(0,1));rank_top=int(np.argmin(rank_mean))
        rng=np.random.default_rng(keyseed(s.key+f':reference:{partition}'))
        si=rng.integers(0,5,(NREF,5));ki=rng.integers(0,nk,(NREF,5,nk))
        boot_ref=rr[si[:,:,None],ki].mean(axis=(1,2))
        ref_frequency=np.bincount(np.argmax(boot_ref,axis=-1),minlength=nm)/NREF
        for splits in range(1,6):
            for seeds in range(1,nk+1):
                means=np.einsum('ij,jkm,lk->ilm',sw[splits],dd,kw[seeds]).reshape(-1,nm)
                winners=np.argmax(means,axis=-1)
                ranking=ranks(means)
                ref_pair_sign=np.asarray([np.sign(ref_rank[i]-ref_rank[j]) for i,j in pairs])
                disc_pair_sign=np.stack([np.sign(ranking[:,i]-ranking[:,j]) for i,j in pairs],axis=-1)
                tau=(disc_pair_sign*ref_pair_sign).mean(axis=-1)
                counts=np.bincount(winners,minlength=nm)/len(winners)
                partitions.append({**s.info(),'partition_id':partition,
                    'discovery_split_ids':'|'.join(s.splits[i] for i in discovery),
                    'reference_split_ids':'|'.join(s.splits[i] for i in reference),
                    'budget_splits':splits,'budget_seeds':seeds,'contexts_per_model':splits*seeds,
                    'all_models_training_runs_per_budget':splits*seeds*nm,
                    'finite_subsets_scored':len(winners),'aggregate_selection_agreement':float(counts[ref_top]),
                    'complete_ranking_agreement':float(np.all(ranking==ref_rank,axis=1).mean()),
                    'mean_Kendall_tau_a_tie_resolved':float(tau.mean()),
                    'mean_rank_reference_agreement':float(counts[rank_top]),
                    'bootstrap_reference_expected_agreement':float(counts@ref_frequency),
                    'mean_BA_reference_top':s.models[ref_top],'mean_rank_reference_top':s.models[rank_top],
                    'reference_bootstrap_top_frequency_of_mean_BA_top':float(ref_frequency[ref_top]),
                    'reference_bootstrap_replicates':NREF,
                    'reference_top_frequencies_json':json.dumps(dict(zip(s.models,ref_frequency.tolist())),sort_keys=True),
                    'disjoint_split_ids':True})
    frame=pd.DataFrame(partitions)
    rows=[]
    for (splits,seeds),group in frame.groupby(['budget_splits','budget_seeds']):
        recovery=group.aggregate_selection_agreement.to_numpy()
        rows.append({**s.info(),'budget_splits':int(splits),'budget_seeds':int(seeds),
            'contexts_per_model':int(splits*seeds),'all_models_training_runs_per_budget':int(splits*seeds*nm),
            'oriented_partitions':len(group),'finite_subset_scores':int(group.finite_subsets_scored.sum()),
            'aggregate_selection_agreement':float(recovery.mean()),
            'partition_min':float(recovery.min()),'partition_max':float(recovery.max()),
            'partition_q025':float(np.quantile(recovery,.025)),'partition_q975':float(np.quantile(recovery,.975)),
            'complete_ranking_agreement':float(group.complete_ranking_agreement.mean()),
            'mean_Kendall_tau_a_tie_resolved':float(group.mean_Kendall_tau_a_tie_resolved.mean()),
            'mean_rank_reference_agreement':float(group.mean_rank_reference_agreement.mean()),
            'bootstrap_reference_expected_agreement':float(group.bootstrap_reference_expected_agreement.mean()),
            'partition_spread_type':'exact finite-grid descriptive distribution; not population CI',
            'independent_new_samples':0,'disjoint_split_ids':True,
            'reference_scope':'five held-out split IDs and all stratum seeds; image datasets may overlap between split realizations'})
    improvements=[]
    base=frame[(frame.budget_splits==1)&(frame.budget_seeds==1)].set_index('partition_id')
    for (splits,seeds),group in frame.groupby(['budget_splits','budget_seeds']):
        current=group.set_index('partition_id')
        change=current.aggregate_selection_agreement-base.aggregate_selection_agreement
        improvements.append({**s.info(),'budget_splits':int(splits),'budget_seeds':int(seeds),
            'baseline':'1x1','paired_partition_mean_change':float(change.mean()),
            'partition_change_min':float(change.min()),'partition_change_max':float(change.max()),
            'fraction_partitions_nonnegative_change':float((change>=-1e-12).mean()),
            'interpretation':'paired finite-grid budget contrast; reused partitions are not independent observations'})
    return rows,frame,improvements


def common_seed_summary(strata,stability,models,budgets):
    rows=[]
    stats={row['stratum']:row for row in stability}
    for dataset in ['organamnist','sipakmed']:
        for policy in ['A','B']:
            full=next(s for s in strata if s.dataset==dataset and s.role=='primary_original' and s.policy==policy)
            common=next(s for s in strata if s.dataset==dataset and s.role=='common_seed_sensitivity' and s.policy==policy)
            f,c=stats[full.key],stats[common.key]
            fbudget=next(r for r in budgets if r['stratum']==full.key and r['budget_splits']==5 and r['budget_seeds']==3)
            cbudget=next(r for r in budgets if r['stratum']==common.key and r['budget_splits']==5 and r['budget_seeds']==3)
            rows.append(dict(dataset=dataset,checkpoint_policy=policy,
                full_contexts=f['contexts'],common_contexts=c['contexts'],common_seed_ids='42|52|62',
                full_grid_reference_top_all5=f['reference_top_model'],full_grid_reference_top_common3=c['reference_top_model'],
                reference_selected_identity_changed=f['reference_top_model']!=c['reference_top_model'],
                LSO_agreement_all5=f['identity_correct_LSO_agreement'],LSO_agreement_common3=c['identity_correct_LSO_agreement'],
                LSO_common3_CI_low=c['LSO_CI_low'],LSO_common3_CI_high=c['LSO_CI_high'],
                full_reference_agreement_all5=f['full_grid_in_sample_agreement'],full_reference_agreement_common3=c['full_grid_in_sample_agreement'],
                normalized_entropy_all5=f['normalized_selection_entropy'],normalized_entropy_common3=c['normalized_selection_entropy'],
                disjoint_5x3_agreement_all5_seed_pool=fbudget['aggregate_selection_agreement'],
                disjoint_5x3_agreement_common3_seed_pool=cbudget['aggregate_selection_agreement'],
                inference='paired input-budget sensitivity; subset reuses original runs, not an independent replication'))
    write('D05_common_three_seed_sensitivity_v0.1.csv',rows)
    return rows


def legacy_reproduction(strata,stability):
    tables=ROOT/'research-lab/experiments/registered-workflow/stats/tables'
    ranking=pd.read_csv(source(tables/'ranking_stability_summary.csv')).set_index(['dataset','checkpoint_policy'])
    performance=pd.read_csv(source(tables/'model_level_performance_table.csv')).set_index(['dataset','checkpoint_policy','model'])
    selection=pd.read_csv(source(tables/'paired_context_bootstrap_selection_probability.csv'))
    budget=pd.read_csv(source(tables/'subsampling_resource_stability_summary.csv'))
    rows=[]
    for s in strata:
        if s.role!='primary_original': continue
        point=observed(s);old=ranking.loc[(s.dataset,s.policy)]
        assert s.models[point['full_reference']]==old.reference_top_model
        assert abs(point['full_agreement']-old.selection_frequency_reference_top)<1e-12
        for j,model in enumerate(s.models):
            assert abs(s.arr[...,j].mean()-performance.loc[(s.dataset,s.policy,model),'balanced_accuracy_mean'])<1e-12
        top_prob=selection[(selection.dataset==s.dataset)&(selection.checkpoint_policy==s.policy)&(selection.model==old.reference_top_model)].iloc[0]
        b=budget[(budget.dataset==s.dataset)&(budget.checkpoint_policy==s.policy)&(budget.resource_budget=='5_splits_x_3_seeds')].iloc[0]
        rows.append(dict(dataset=s.dataset,checkpoint_policy=s.policy,
            original_reference_top_model=old.reference_top_model,full_grid_agreement_reproduced=point['full_agreement'],
            original_flat_context_bootstrap_selection_frequency=float(top_prob.paired_context_bootstrap_top_probability),
            original_in_sample_5x3_recovery=float(b.recovered_full_reference_probability),
            new_resampling_scope='hierarchical paired split/seed; numerical difference does not mean raw metrics changed',
            new_budget_scope='disjoint reference split IDs; not same estimand as original in-sample recovery'))
    write('D02_original_estimand_reproduction_v0.1.csv',rows)
    return rows


def make_coverage():
    return [
        dict(item='Table 1',RQ='RQ1/RQ3',reviewer_ids='E.1;R2.7;R3.5',question='Within-stratum selection stability and budget/reference sensitivity',
            unit='split x seed within dataset/policy',effect='full-grid and LSO agreement; paired bootstrap selection frequency; disjoint aggregate budget agreement',
            interval='10000 hierarchical replicates for agreement; finite partition spread for budget, not CI',null='No single meaningful null for entire summary table',
            multiplicity='Pairwise BA tests separately registered; no fabricated omnibus stability p value',output='D02_ranking_stability_v0.1.csv;D04_budget_summary_v0.1.csv',
            limitation='Fixed datasets; few split clusters; overlapping source images; reference definitions differ'),
        dict(item='Figure 1',RQ='RQ1',reviewer_ids='E.1;R2.7',question='How often does each candidate rank first across observed contexts?',
            unit='paired context indicators resampled by split then seed',effect='observed selection frequency',interval='paired hierarchical percentile 95%',
            null='No prespecified scientific basis for uniform winner frequency',multiplicity='No p-value family for descriptive proportions',
            output='D02_model_performance_selection_v0.1.csv',limitation='Zero observed frequencies give degenerate nonparametric intervals, not structural impossibility'),
        dict(item='Figure 2',RQ='RQ1/RQ2',reviewer_ids='E.1;R2.7;R3.5',question='Distribution of BA difference between the two highest-ranked candidates',
            unit='paired candidate vector per context; top two reselected in each context',effect='nonnegative context order-statistic difference',
            interval='hierarchical interval for mean context difference plus distribution plot',null='Testing gap > 0 is not a model-superiority test',
            multiplicity='All six prespecified pairwise model BA differences per primary stratum have Holm; secondary global36 shown',
            output='D02_contexts_v0.1.csv;D02_paired_model_BA_differences_v0.1.csv;D01_margin_implementation_replay_v0.1.csv',
            limitation='Candidate pool dependent; association not causal; overlapping intervals not an equality test'),
        dict(item='Figure 3',RQ='RQ3',reviewer_ids='E.1;R2.3;R3.4',question='Does aggregating more split/seed evaluations recover the held-out block ranking?',
            unit='252 oriented 5/5 split partitions; aggregate candidate means per budget subset',effect='aggregate winner agreement and full-rank/tau sensitivity',
            interval='exact finite-grid partition ranges/quantiles; no population CI',null='No binomial test on reused subsets or partitions',
            multiplicity='All budget cells reported, no post hoc threshold selection',output='D04_budget_summary_v0.1.csv;D04_paired_budget_changes_v0.1.csv',
            limitation='Split IDs disjoint but source images can overlap; maximum discovery5 does not establish universal convergence'),
        dict(item='Policy supplementary table',RQ='RQ2',reviewer_ids='E.1;R2.7;R2.4',question='Checkpoint-rule sensitivity within identical split/seed/model',
            unit='paired A/B runs; split-average contrasts for conditional sign flip',effect='BA B-minus-A and selected-identity change fraction',interval='paired hierarchical percentile 95%',
            null='Symmetric sign exchangeability of split-mean BA differences under no policy difference',multiplicity='8 original dataset-model policy BA comparisons: Holm',
            output='D02_policy_B_minus_A_v0.1.csv;D02_policy_selection_sensitivity_v0.1.csv',limitation='No causal attribution of all observed ranking instability'),
        dict(item='Swin supplementary sensitivity',RQ='RQ1/RQ2 generalization boundary',reviewer_ids='R1.1;R2.2;R3.3',question='Candidate pool and input-recipe dependence at the same15 contexts',
            unit='5 splits x3 seeds, shared contexts/labels',effect='selected-identity switch fraction and paired input-recipe BA difference',interval='paired hierarchical percentile 95%',
            null='Conditional sign symmetry for input-recipe BA contrast only',multiplicity='2 dataset input-recipe contrasts: Holm; pool comparison is descriptive',
            output='D02_candidate_pool_sensitivity_v0.1.csv;D02_swin_input_sensitivity_v0.1.csv',limitation='Only5 split clusters; smallest two-sided exact p=.0625; not an architecture contest')]


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'B1_analysis_manifest_v0.1.json').exists():
        raise RuntimeError('Completed results_v0.3 exists; retain version and use a new version for another analysis.')
    source(SPEC);source(Path(__file__))
    runs=load_runs();strata=make_strata(runs)
    coverage=make_coverage()
    write('D03_figure_statistical_coverage_v0.1.csv',coverage)
    print('B1: verified 1100 raw metrics; proceeding D01',flush=True)
    d01,d01_margin=audit_d01(strata)
    print('B1: D01 different-implementation replay complete',flush=True)
    performance=[];stability=[];contexts=[];pairs=[];resampling=[]
    for s in strata:
        pp,ss,cc,dd,rr=analyze_stratum(s)
        performance.extend(pp);stability.append(ss);contexts.extend(cc);pairs.extend(dd);resampling.extend(rr)
        print('B1: D02 completed '+s.key,flush=True)
    primary_pairs=[r for r in pairs if r['role'].startswith('primary')]
    assert len(primary_pairs)==36
    for row,p in zip(primary_pairs,holm([r['sign_flip_p_raw_conditional'] for r in primary_pairs])):
        row['holm_p_all_36_primary_BA_comparisons']=p
    write('D02_model_performance_selection_v0.1.csv',performance)
    write('D02_ranking_stability_v0.1.csv',stability)
    write('D02_contexts_v0.1.csv',contexts)
    write('D02_paired_model_BA_differences_v0.1.csv',pairs)
    write('D02_resampling_design_sensitivity_v0.1.csv',resampling)
    policy,policy_selection=policy_comparisons(strata)
    transform,pool=swin_sensitivity(runs,strata)
    legacy=legacy_reproduction(strata,stability)
    budgets=[];partitions=[];changes=[]
    for s in strata:
        if s.role not in ('primary_original','primary_extension','common_seed_sensitivity'):
            continue
        rows,detail,deltas=budget_analysis(s)
        budgets.extend(rows);partitions.append(detail);changes.extend(deltas)
        print('B1: D04 aggregate budgets completed '+s.key,flush=True)
    write('D04_budget_summary_v0.1.csv',budgets)
    write('D04_partition_budget_details_v0.1.csv',pd.concat(partitions,ignore_index=True))
    write('D04_paired_budget_changes_v0.1.csv',changes)
    common=common_seed_summary(strata,stability,performance,budgets)
    write('B1_bootstrap_draw_audit_v0.1.csv',DRAW_AUDIT)
    runtime={}
    for name in ['numpy','pandas','scipy','matplotlib','Pillow','statsmodels']:
        try:
            runtime[name]=version(name)
        except PackageNotFoundError:
            runtime[name]=None
    manifest=dict(analysis_id='JIIM_B1_revision_unified_v0.1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='COMPUTED_AND_SOURCE_CHECKED; independent-person method review pending',
        input_runs=1100,original_runs=800,extension_runs=300,strata=len(strata),
        input_overlays='raw unique runs; common-seed and candidate-pool strata reuse runs; do not sum overlay counts',
        bootstrap_replicates=NBOOT,reference_bootstrap_replicates=NREF,base_seed=BASE_SEED,
        rng_seed_derivation='SHA256(stratum:purpose) first8bytes big endian XOR base_seed; D01 replays legacy20261003',
        python=platform.python_version(),python_executable=sys.executable,platform=platform.platform(),package_versions=runtime,
        source_hashes=SOURCE_HASHES,output_hashes={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))},
        original_primary_point_estimates_reproduced=True,D01_replay=d01,D01_margin_replay=d01_margin,
        limitations=['Same-agent computation cross-check is not independent-person review.',
            'Intervals describe resampling sensitivity of fixed datasets and observed evaluation grid; not clinical population validation.',
            'Repeated split IDs can reuse images and same training seed levels; nested/crossed/split-only results are reported.',
            'Sign-flip p values require sign symmetry/exchangeability; Holm does not remove dependence.',
            'Budget partitions/subsets are finite reused calculations, not independent patient samples; partition spread is not CI.',
            'No new training, manuscript rewrite, universal winner or universal evaluation budget claim.'])
    (OUT/'B1_analysis_manifest_v0.1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='completed',original_runs=800,extension_runs=300,strata=len(strata),
        performance_rows=len(performance),paired_rows=len(pairs),budget_rows=len(budgets),
        partition_budget_rows=sum(len(frame) for frame in partitions),output=str(OUT)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
