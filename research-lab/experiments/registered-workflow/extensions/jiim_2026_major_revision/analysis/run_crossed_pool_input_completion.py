"""Complete existing candidate-pool and Swin input contrasts using paired crossed draws.

Raw experiments, historical analyses, model definitions, and conditional tests are read only.
"""
import hashlib,importlib.util,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]
OLD=HERE/'results';OUT=HERE/'crossed/pool_input'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    source=HERE/'crossed_core.py'
    spec=importlib.util.spec_from_file_location('b1_pool_input_readonly',source)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    m=json.loads((OLD/'B1_analysis_manifest.json').read_text())
    paths=[Path(__file__),source,OLD/'B1_analysis_manifest.json',OLD/'B1_run_inputs.csv',
        OLD/'D02_candidate_pool_sensitivity.csv',OLD/'D02_swin_input_sensitivity.csv']
    for p in paths[3:]:assert sha(p)==m['output_hashes'][p.name]
    assert sha(source)==m['source_hashes'][str(source.relative_to(ROOT))]
    runs=pd.read_csv(paths[3],float_precision='round_trip')
    runs=runs[(runs.stage=='extension') & runs.split_id.isin([f'split_{i:02d}' for i in range(1,6)])]
    assert len(runs)==runs.run_id.nunique()==180
    raw={r.metric_source_path:m['source_hashes'][r.metric_source_path] for r in runs.itertuples()}
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    oldpool=pd.read_csv(paths[4],float_precision='round_trip');oldinput=pd.read_csv(paths[5],float_precision='round_trip')
    pools=[];inputs=[];audits=[]
    for ds in ['isic2019','mura']:
        data=runs[runs.dataset==ds]
        cnn=data[(data.variant=='shared') & data.model.isin(b.MODELS)]
        control=b.Stratum(f'{ds}_A_cnn4_pool_contexts',ds,'A','candidate_pool_control',cnn,b.MODELS)
        assert (len(control.splits),len(control.seeds))==(5,3)
        co=b.observed(control);cw=np.asarray(control.models)[co['winner']]
        for variant in ['shared','swin_weight_eval']:
            swin=data[(data.model=='swin_t') & (data.variant==variant)]
            full=b.Stratum(f'{ds}_A_pool5_{variant}',ds,'A','candidate_pool_sensitivity',pd.concat([cnn,swin]),b.MODELS+['swin_t'])
            assert control.splits==full.splits and control.seeds==full.seeds
            # Check reused CNN values and model pairing before comparing pools.
            assert np.array_equal(control.arr,full.arr[:,:,[full.models.index(k) for k in control.models]])
            fo=b.observed(full);fw=np.asarray(full.models)[fo['winner']]
            changed=(cw!=fw).astype(float)
            before=oldpool[(oldpool.dataset==ds)&(oldpool.swin_variant==variant)].iloc[0]
            assert float(changed.mean())==before.selected_identity_changed_fraction
            assert float(co['margin'].mean())==before.observed_mean_top_two_difference_four
            assert float(fo['margin'].mean())==before.observed_mean_top_two_difference_five
            rngseed=b.keyseed(ds+':candidate_pool:'+variant)
            sp,sg=b.draw_indices(control,np.random.default_rng(rngseed),mode='crossed',tag='paired pool interval completion')
            se=sg[:,0,:]
            values=changed[sp[:,:,None],se[:,None,:]].mean(axis=(1,2))
            # Independently select model labels in both sampled candidate arrays.
            cs=control.arr[sp[:,:,None],se[:,None,:]];fs=full.arr[sp[:,:,None],se[:,None,:]]
            direct=(np.asarray(control.models)[cs.argmax(axis=-1)]!=np.asarray(full.models)[fs.argmax(axis=-1)]).mean(axis=(1,2))
            assert np.array_equal(values,direct)
            for i in range(32):
                scalar=np.mean([changed[int(s),int(k)] for s in sp[i] for k in se[i]])
                assert scalar==values[i]
            lo,hi=b.interval(values)
            row=before.to_dict();row.update(CI_low=float(lo),CI_high=float(hi),nested_CI_low=before.CI_low,nested_CI_high=before.CI_high,
                CI_width_change=float(hi-lo-before.CI_high+before.CI_low),
                resampling='paired crossed split-seed; same grid for four/five models',bootstrap_replicates=10000,rng_seed=str(rngseed),
                statistical_target='fraction of matched contexts with changed selected model identity',test_scope='descriptive; no added candidate-pool test')
            pools.append(row)
            np.savez_compressed(OUT/f'{ds}_pool_{variant}_draws.npz',split_indices=sp,global_seed_indices=se,changed_model_fraction=values)
            audits.append(dict(dataset=ds,comparison='pool:'+variant,paired_contexts=15,unique_runs=75,reused_CNN_values='EXACT',observed_values='EXACT',draw_identity_check='EXACT',scalar_draw_checks=32))
        shared=b.Stratum(f'{ds}_swin_shared',ds,'A','input_sensitivity',data[(data.model=='swin_t')&(data.variant=='shared')],['swin_t'])
        weight=b.Stratum(f'{ds}_swin_weight_eval',ds,'A','input_sensitivity',data[(data.model=='swin_t')&(data.variant=='swin_weight_eval')],['swin_t'])
        assert shared.splits==weight.splits==control.splits and shared.seeds==weight.seeds==control.seeds
        delta=(weight.arr-shared.arr)[:,:,0]
        before=oldinput[oldinput.dataset==ds].iloc[0]
        assert float(delta.mean())==before.mean_BA_difference_weight_eval_minus_shared
        p,n=b.signflip(delta.mean(axis=1));assert p==before.sign_flip_p_raw_conditional and n==before.sign_assignments
        rngseed=b.keyseed(ds+':swin_input')
        sp,sg=b.draw_indices(shared,np.random.default_rng(rngseed),mode='crossed',tag='paired input interval completion');se=sg[:,0,:]
        values=delta[sp[:,:,None],se[:,None,:]].mean(axis=(1,2))
        direct=(weight.arr[sp[:,:,None],se[:,None,:]]-shared.arr[sp[:,:,None],se[:,None,:]]).mean(axis=(1,2))[:,0]
        assert np.array_equal(values,direct)
        alternative=(weight.arr[sp[:,:,None],se[:,None,:]].mean(axis=(1,2))-shared.arr[sp[:,:,None],se[:,None,:]].mean(axis=(1,2)))[:,0]
        error=float(np.max(np.abs(values-alternative)));tolerance=4*np.finfo(float).eps*15;assert error<tolerance
        for i in range(32):
            scalar=np.mean([delta[int(s),int(k)] for s in sp[i] for k in se[i]])
            assert abs(scalar-values[i])<tolerance
        lo,hi=b.interval(values)
        row=before.to_dict();row.update(CI_low=float(lo),CI_high=float(hi),nested_CI_low=before.CI_low,nested_CI_high=before.CI_high,
            CI_width_change=float(hi-lo-before.CI_high+before.CI_low),interval_includes_zero=bool(lo<=0<=hi),
            nested_interval_includes_zero=bool(before.CI_low<=0<=before.CI_high),resampling='paired crossed split-seed',bootstrap_replicates=10000,
            rng_seed=str(rngseed),statistical_target='Swin mean BA difference weight_eval minus shared')
        inputs.append(row)
        np.savez_compressed(OUT/f'{ds}_input_draws.npz',split_indices=sp,global_seed_indices=se,mean_BA_weight_eval_minus_shared=values)
        audits.append(dict(dataset=ds,comparison='input:weight_eval-minus-shared',paired_contexts=15,unique_runs=30,observed_values='EXACT',draw_identity_check='EXACT',scalar_draw_checks=32,signflip_p='EXACT',sign_assignments=n,arithmetic_max_abs_error=error,arithmetic_bound=tolerance))
    pf=pd.DataFrame(pools);inf=pd.DataFrame(inputs)
    assert np.array_equal(b.holm(inf.sign_flip_p_raw_conditional),inf.holm_p_two_input_comparisons)
    pf.to_csv(OUT/'Crossed_Candidate_Pool_Sensitivity.csv',index=False,float_format='%.17g')
    inf.to_csv(OUT/'Crossed_Swin_Input_Sensitivity.csv',index=False,float_format='%.17g')
    pd.DataFrame(audits).to_csv(OUT/'Crossed_Pool_Input_Computation_Audit.csv',index=False)
    summary=dict(pool_comparisons=4,input_comparisons=2,unique_raw_runs=180,new_training_runs=0,
        pool_intervals_wider=int((pf.CI_width_change>1e-14).sum()),input_intervals_wider=int((inf.CI_width_change>1e-14).sum()),
        input_zero_inclusion_changes=int((inf.interval_includes_zero!=inf.nested_interval_includes_zero).sum()),
        point_estimates_and_original_tests_unchanged=True,
        pool_results=pf[['dataset','swin_variant','selected_identity_changed_fraction','nested_CI_low','nested_CI_high','CI_low','CI_high']].to_dict('records'),
        input_results=inf[['dataset','mean_BA_difference_weight_eval_minus_shared','CI_low','CI_high','holm_p_two_input_comparisons']].to_dict('records'))
    (OUT/'Crossed_Pool_Input_Result_Summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    result=dict(analysis_id='JIIM_crossed_pool_input_completion',created_utc=datetime.now(timezone.utc).isoformat(),status='COMPUTED_AND_SOURCE_CHECKED',
        authorization='Author: 继续（候选池/输入敏感性区间）',scope='Existing four pool and two input contrasts at the same 5 splits x 3 seeds',
        raw_unique_runs=180,reused_CNN_runs=120,Swin_shared_and_weight_eval_runs=60,new_training_runs=0,
        resampling='10000 paired crossed draws for each contrast; global seed vector shared across sampled splits',base_seed=b.BASE_SEED,
        rng_seed_derivation='Original B1 dedicated dataset:comparison key; bootstrap structure migrated to crossed',
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},raw_metric_hashes=raw,output_hashes={p.name:sha(p) for p in OUT.iterdir()},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,executable=sys.executable,elapsed_seconds=time.perf_counter()-start,
        result_summary=summary,verification='Same-agent source/arithmetic checks, not independent-person review',application_state='New statistical materials; manuscript/Word integration pending')
    (OUT/'Crossed_Pool_Input_Manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
