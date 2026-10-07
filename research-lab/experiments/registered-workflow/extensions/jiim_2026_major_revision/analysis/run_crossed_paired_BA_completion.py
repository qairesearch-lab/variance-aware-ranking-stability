"""Complete primary fixed-model paired BA intervals using adopted crossed draws.

Continues the author's sequential completion request. Existing experiments and
sign-flip/Holm results are read-only. The 36 contrasts have fixed model identities.
"""
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PREV=HERE/'crossed/agreement'
OLD=HERE/'results'
OUT=HERE/'crossed/paired_BA'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.perf_counter()
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} == {'Attempt_01_Arithmetic_Check_Stop.json'}
    else:
        OUT.mkdir()
    source=HERE/'crossed_core.py'
    spec=importlib.util.spec_from_file_location('jiim_b1_readonly_paired',source)
    b1=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b1;spec.loader.exec_module(b1)
    previous=json.loads((PREV/'Crossed_Agreement_Manifest.json').read_text())
    input_paths=[Path(__file__),source,PREV/'Crossed_Agreement_Manifest.json',
                 PREV/'Raw_Metric_Source_Hashes.json',OLD/'B1_run_inputs.csv',
                 OLD/'B1_analysis_manifest.json',OLD/'D02_paired_model_BA_differences.csv']
    old_manifest=json.loads((OLD/'B1_analysis_manifest.json').read_text())
    for p in [OLD/'B1_run_inputs.csv',OLD/'D02_paired_model_BA_differences.csv']:
        assert sha(p)==old_manifest['output_hashes'][p.name]
    assert sha(source)==old_manifest['source_hashes'][str(source.relative_to(ROOT))]
    raw_hashes=json.loads((PREV/'Raw_Metric_Source_Hashes.json').read_text())
    assert len(raw_hashes)==1040
    assert all(sha(ROOT/p)==h for p,h in raw_hashes.items())
    runs=pd.read_csv(OLD/'B1_run_inputs.csv',float_precision='round_trip')
    old_pairs=pd.read_csv(OLD/'D02_paired_model_BA_differences.csv',float_precision='round_trip')
    old_pairs=old_pairs[old_pairs.role.isin(['primary_original','primary_extension'])]
    agreement=pd.read_csv(PREV/'Crossed_Agreement_Summary.csv',float_precision='round_trip')
    rows=[];audit=[]
    for item in agreement.itertuples():
        data=runs[(runs.dataset==item.dataset)&(runs.checkpoint_policy==item.checkpoint_policy)
                  &runs.model.isin(b1.MODELS)]
        if item.role=='primary_extension':data=data[data.variant=='shared']
        s=b1.Stratum(item.stratum,item.dataset,item.checkpoint_policy,item.role,data,b1.MODELS)
        path=PREV/(s.key+'_draws.npz');input_paths.append(path)
        assert sha(path)==previous['output_hashes'][path.name]
        saved=np.load(path,allow_pickle=False)
        sp=saved['split_indices'];global_seed=saved['global_seed_indices']
        assert np.array_equal(saved['model_ids'],np.array(s.models))
        sample=s.arr[sp[:,:,None],global_seed[:,None,:]]
        means=sample.mean(axis=(1,2))
        assert np.array_equal(means.argmax(axis=1),saved['reference_indices'])
        prior=old_pairs[old_pairs.stratum==s.key]
        assert len(prior)==6
        for i,j in itertools.combinations(range(4),2):
            before=prior[(prior.model_1==s.models[i])&(prior.model_2==s.models[j])].iloc[0]
            delta=s.arr[...,i]-s.arr[...,j]
            difference=means[:,i]-means[:,j]
            direct=delta[sp[:,:,None],global_seed[:,None,:]].mean(axis=(1,2))
            max_error=float(np.max(np.abs(difference-direct)))
            tolerance=4*np.finfo(np.float64).eps*delta.size
            assert max_error<tolerance
            low,high=b1.interval(difference)
            point=float(delta.mean())
            assert point==before.mean_paired_BA_difference_model1_minus_model2
            p,assignments=b1.signflip(delta.mean(axis=1))
            assert p==before.sign_flip_p_raw_conditional and assignments==before.sign_assignments
            old_crosses=bool(before.paired_CI_low<=0<=before.paired_CI_high)
            new_crosses=bool(low<=0<=high)
            row=before.to_dict()
            row.update(paired_CI_low=float(low),paired_CI_high=float(high),
                nested_CI_low=float(before.paired_CI_low),nested_CI_high=float(before.paired_CI_high),
                nested_interval_includes_zero=old_crosses,crossed_interval_includes_zero=new_crosses,
                zero_inclusion_changed=(old_crosses!=new_crosses),
                crossed_minus_nested_width=float((high-low)-(before.paired_CI_high-before.paired_CI_low)),
                bootstrap_replicates=10000,resampling='paired crossed split-seed',
                statistical_target='mean BA difference between two fixed candidate models',
                rng_seed=item.rng_seed,draw_source=str(path.relative_to(ROOT)),
                draw_source_sha256=sha(path),pair_arithmetic_max_abs_error=max_error,
                pair_arithmetic_error_bound=tolerance,
                p_value_source='Retained split-level sign-flip enumeration; not derived from bootstrap CI')
            rows.append(row)
        np.savez_compressed(OUT/(s.key+'_mean_BA_draws.npz'),
                            mean_BA=means,model_ids=np.array(s.models))
        audit.append(dict(stratum=s.key,paired_contrasts=6,draws=10000,
                          observed_difference_check='EXACT',signflip_p_check='EXACT',
                          crossed_draw_reuse='HASH_VERIFIED',reference_replay='EXACT'))
        print(s.key+': completed 6 paired intervals',flush=True)
    frame=pd.DataFrame(rows)
    assert len(frame)==36
    assert np.array_equal(b1.holm(frame.sign_flip_p_raw_conditional),frame.holm_p_all_36_primary_BA_comparisons)
    for key,group in frame.groupby('stratum',sort=False):
        assert np.array_equal(b1.holm(group.sign_flip_p_raw_conditional),group.holm_p_within_stratum)
    frame.to_csv(OUT/'Crossed_Primary_Paired_BA_Comparisons.csv',index=False,float_format='%.17g')
    pd.DataFrame(audit).to_csv(OUT/'Crossed_Paired_BA_Computation_Audit.csv',index=False)
    changes=frame[frame.zero_inclusion_changed]
    changes.to_csv(OUT/'Paired_BA_Zero_Inclusion_Changes.csv',index=False,float_format='%.17g')
    summary=dict(contrasts=36,intervals_wider=int((frame.crossed_minus_nested_width>1e-14).sum()),
        intervals_narrower=int((frame.crossed_minus_nested_width < -1e-14).sum()),
        old_intervals_including_zero=int(frame.nested_interval_includes_zero.sum()),
        new_intervals_including_zero=int(frame.crossed_interval_includes_zero.sum()),
        zero_inclusion_changed=int(frame.zero_inclusion_changed.sum()),
        within_stratum_holm_below_005=int((frame.holm_p_within_stratum<.05).sum()),
        global36_holm_below_005=int((frame.holm_p_all_36_primary_BA_comparisons<.05).sum()),
        all_point_estimates_and_p_values_unchanged=True,
        interval_change_summary=changes[['stratum','model_1','model_2','paired_CI_low','paired_CI_high']].to_dict('records'))
    (OUT/'Crossed_Paired_BA_Result_Summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in input_paths}
    assert all(sha(ROOT/p)==h for p,h in raw_hashes.items())
    manifest=dict(analysis_id='JIIM_crossed_primary_paired_BA_completion',
        created_utc=datetime.now(timezone.utc).isoformat(),status='COMPUTED_AND_SOURCE_CHECKED',
        authorization='Author: 做好实验记录。继续下一个问题。',scope='36 primary fixed-model paired BA intervals',
        resampling='Existing adopted paired crossed indices reused; no new random draws',
        bootstrap_replicates_per_stratum=10000,raw_metric_inputs=1040,new_training_runs=0,
        input_hashes=input_hashes,raw_metric_hashes_source=str((PREV/'Raw_Metric_Source_Hashes.json').relative_to(ROOT)),
        output_hashes={p.name:sha(p) for p in sorted(OUT.iterdir())},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,executable=sys.executable,
        elapsed_seconds=time.perf_counter()-started,result_summary=summary,
        verification='Same-agent source and arithmetic checks; not independent-person review',
        application_state='New statistical materials; manuscript/Word integration pending')
    (OUT/'Crossed_Paired_BA_Manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
