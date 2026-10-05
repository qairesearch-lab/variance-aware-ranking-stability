"""Read-only numerical/source audit using separate dataframe and direct-loop calculations."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
OUT=HERE/'results'
BASE_SEED=2026100301


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((OUT/'B1_analysis_manifest.json').read_text())
    errors=[]; checks=[]
    for path,digest in manifest['source_hashes'].items():
        if sha(ROOT/path)!=digest:
            errors.append('source hash changed: '+path)
    for path,digest in manifest['output_hashes'].items():
        if sha(OUT/path)!=digest:
            errors.append('output hash changed: '+path)
    inputs=pd.read_csv(OUT/'B1_run_inputs.csv')
    inventory=pd.read_csv(OUT/'B1_stratum_inventory.csv')
    stability=pd.read_csv(OUT/'D02_ranking_stability.csv').set_index('stratum')
    models=pd.read_csv(OUT/'D02_model_performance_selection.csv')
    details=pd.read_csv(OUT/'D04_partition_budget_details.csv')
    summary=pd.read_csv(OUT/'D04_budget_summary.csv')
    covered=set();point_checks=[];budget_checks=[];tensors={}
    for row in inventory.to_dict('records'):
        key=row['stratum']; names=row['candidate_pool'].split('|');seeds=list(map(int,row['seed_ids'].split('|')))
        data=inputs[(inputs.dataset==row['dataset'])&(inputs.checkpoint_policy==row['checkpoint_policy'])&inputs.training_seed.isin(seeds)]
        if row['role'] in ['primary_extension','candidate_pool_control']:
            data=data[(data.variant=='shared')&data.model.isin(names)]
        elif row['role']=='candidate_pool_sensitivity':
            variant='swin_weight_eval' if key.endswith('swin_weight_eval') else 'shared'
            data=data[((data.model!='swin_t')&(data.variant=='shared'))|((data.model=='swin_t')&(data.variant==variant))]
        if row['splits']==5:
            data=data[data.split_id.isin([f'split_{s:02d}' for s in range(1,6)])]
        covered.update(data.run_id)
        if len(data)!=row['runs'] or data.duplicated(['split_id','training_seed','model']).any():
            errors.append(key+': incorrect input overlay')
        wide=data.pivot(index=['split_id','training_seed'],columns='model',values='balanced_accuracy').reindex(columns=names)
        winners=wide.idxmax(axis=1)
        full_reference=data.groupby('model').balanced_accuracy.mean().reindex(names).idxmax()
        matches=[]
        for split in sorted(data.split_id.unique()):
            ref=data[data.split_id!=split].groupby('model').balanced_accuracy.mean().reindex(names).idxmax()
            matches.extend((winners.loc[split]==ref).tolist())
        saved=stability.loc[key]
        reference_matches=full_reference==saved.reference_top_model
        if not reference_matches:
            errors.append(key+': full-grid reference identity mismatch')
        agree=float((winners==full_reference).mean());lso=float(np.mean(matches))
        point_error=max(abs(agree-saved.full_grid_in_sample_agreement),abs(lso-saved.identity_correct_LSO_agreement))
        if point_error>1e-12:
            errors.append(key+': point agreement mismatch')
        for name in names:
            saved_model=models[(models.stratum==key)&(models.model==name)].iloc[0]
            if max(abs(data[data.model==name].balanced_accuracy.mean()-saved_model.mean_BA),
                   abs(float((winners==name).mean())-saved_model.observed_selection_frequency))>1e-12:
                errors.append(key+': model summary mismatch '+name)
        point_checks.append(dict(stratum=key,contexts=len(wide),full_reference_match=bool(reference_matches),
            max_point_difference=point_error,held_split_excluded=True))
        if row['splits']!=10:
            continue
        split_ids=sorted(data.split_id.unique())
        arr=np.asarray([[wide.loc[(split,seed)].to_numpy() for seed in seeds] for split in split_ids])
        tensors[key]=(arr,names,seeds)
        local=details[details.stratum==key]
        for (s,k),group in local.groupby(['budget_splits','budget_seeds']):
            if len(group)!=252 or group.partition_id.nunique()!=252:
                errors.append(key+': incorrect oriented partition count')
            if not np.all(group.finite_subsets_scored==math.comb(5,int(s))*math.comb(len(seeds),int(k))):
                errors.append(key+': finite subset denominator mismatch')
            saved_budget=summary[(summary.stratum==key)&(summary.budget_splits==s)&(summary.budget_seeds==k)].iloc[0]
            if abs(group.aggregate_selection_agreement.mean()-saved_budget.aggregate_selection_agreement)>1e-12:
                errors.append(key+': partition summary mismatch')
        for item in local.itertuples():
            dis=set(item.discovery_split_ids.split('|'));ref=set(item.reference_split_ids.split('|'))
            if len(dis)!=5 or len(ref)!=5 or dis&ref:
                errors.append(key+': reference overlap')
                break
            if abs(sum(json.loads(item.reference_top_frequencies_json).values())-1)>1e-12:
                errors.append(key+': reference bootstrap frequencies do not sum to one')
                break
        partition_list=list(itertools.combinations(range(10),5))
        for partition in [0,83,251]:
            discovery=partition_list[partition];reference=sorted(set(range(10))-set(discovery))
            reference_top=int(np.argmax(arr[reference].mean(axis=(0,1))))
            rank_values=[]
            for i in reference:
                for k in range(len(seeds)):
                    order=sorted(range(len(names)),key=lambda j:(-arr[i,k,j],names[j]))
                    rank_values.append([order.index(j)+1 for j in range(len(names))])
            rank_reference_top=int(np.argmin(np.mean(rank_values,axis=0)))
            for s,k in [(1,1),(1,3),(3,1),(3,3),(5,3)]:
                agree=[];rank_agree=[];count=0
                for si in itertools.combinations(discovery,s):
                    for ki in itertools.combinations(range(len(seeds)),k):
                        mean=np.asarray([[arr[i,j] for j in ki] for i in si]).mean(axis=(0,1))
                        selected=int(np.argmax(mean));agree.append(selected==reference_top);rank_agree.append(selected==rank_reference_top);count+=1
                record=local[(local.partition_id==partition)&(local.budget_splits==s)&(local.budget_seeds==k)].iloc[0]
                error=max(abs(np.mean(agree)-record.aggregate_selection_agreement),abs(np.mean(rank_agree)-record.mean_rank_reference_agreement))
                if error>1e-12 or count!=record.finite_subsets_scored:
                    errors.append(key+': direct-loop aggregate budget mismatch')
                budget_checks.append(dict(stratum=key,partition_id=partition,budget_splits=s,budget_seeds=k,
                    directly_enumerated_subsets=count,max_abs_difference=error))
    if covered!=set(inputs.run_id):
        errors.append('Not all1100 raw runs covered by analysis strata')
    # Reproduce original flat-context Monte Carlo stream, without importing its implementation.
    legacy_path=ROOT/'research-lab/experiments/registered-workflow/stats/tables/paired_context_bootstrap_selection_probability.csv'
    legacy=pd.read_csv(legacy_path)
    seed=int(legacy.iloc[0].bootstrap_seed);rng=np.random.default_rng(seed)
    legacy_checks=[]
    for dataset,policy in sorted(set(zip(legacy.dataset,legacy.checkpoint_policy))):
        a,names,seeds=tensors[f'{dataset}_{policy}_cnn4_all5']
        flattened=a.reshape(-1,len(names));draw=rng.integers(0,len(flattened),(10_000,len(flattened)))
        count=np.bincount(np.argmax(flattened[draw].mean(axis=1),axis=1),minlength=len(names))
        saved=legacy[(legacy.dataset==dataset)&(legacy.checkpoint_policy==policy)].set_index('model')
        for j,name in enumerate(names):
            if count[j]!=int(saved.loc[name,'bootstrap_top_count']):
                errors.append('legacy flat-context replay mismatch '+dataset+policy+name)
        legacy_checks.append(dict(dataset=dataset,policy=policy,bootstrap_replicates=10000,
            original_counts_reproduced=True,rng='single stream initialized2026051510; table per-stratum seed labels are not separate reinitializations'))
    pd.DataFrame(point_checks).to_csv(OUT/'B1_point_estimate_crosscheck.csv',index=False)
    pd.DataFrame(budget_checks).to_csv(OUT/'D04_direct_loop_crosscheck.csv',index=False)
    pd.DataFrame(legacy_checks).to_csv(OUT/'D02_legacy_flat_bootstrap_replay.csv',index=False)
    result=dict(analysis_id='JIIM_B1_numerical_source_audit_v0.1',status='PASSED' if not errors else 'FAILED',
        errors=errors,source_hashes_checked=len(manifest['source_hashes']),output_hashes_checked=len(manifest['output_hashes']),
        raw_runs_covered=len(covered),strata_point_checks=len(point_checks),direct_loop_budget_cells=len(budget_checks),
        legacy_flat_bootstrap_strata_replayed=len(legacy_checks),original_A_B_coverage=True,
        audit_script_sha256=sha(Path(__file__)),independent_person_review=False,
        interpretation='Different implementation and direct loops by same agent; numerical/source acceptance only, not independent scientific endorsement.')
    (OUT/'B1_numerical_source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:
        raise RuntimeError('B1 audit failed; no conclusion acceptance')


if __name__=='__main__':
    main()
