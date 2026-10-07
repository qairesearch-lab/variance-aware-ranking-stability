"""Complete paired A/B procedure contrasts under the adopted crossed design.

Primary original five-seed grids and existing common-three-seed sensitivity only.
Preserve raw experiments, conditional sign-flip tests, and their Holm families.
"""
import hashlib,importlib.util,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]
OLD=HERE/'results';PREV=HERE/'crossed/agreement';OUT=HERE/'crossed/AB'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    source=HERE/'crossed_core.py'
    spec=importlib.util.spec_from_file_location('b1_AB_readonly',source)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    manifest=json.loads((OLD/'B1_analysis_manifest.json').read_text())
    previous=json.loads((PREV/'Crossed_Agreement_Manifest.json').read_text())
    hashes=json.loads((PREV/'Raw_Metric_Source_Hashes.json').read_text())
    runs=pd.read_csv(OLD/'B1_run_inputs.csv',float_precision='round_trip')
    runs=runs[runs.stage=='original'];assert len(runs)==800
    raw={r.metric_source_path:hashes[r.metric_source_path] for r in runs.itertuples()}
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    paths=[Path(__file__),source,OLD/'B1_analysis_manifest.json',OLD/'B1_run_inputs.csv',
           OLD/'D02_policy_B_minus_A.csv',OLD/'D02_policy_selection_sensitivity.csv',
           OLD/'D02_resampling_design_sensitivity.csv',PREV/'Crossed_Agreement_Manifest.json',
           PREV/'Raw_Metric_Source_Hashes.json',ROOT/'research-lab/experiments/registered-workflow/configs/frozen/checkpoint_policies.yaml']
    for p in paths[3:7]:assert sha(p)==manifest['output_hashes'][p.name]
    assert sha(source)==manifest['source_hashes'][str(source.relative_to(ROOT))]
    oldBA=pd.read_csv(paths[4],float_precision='round_trip')
    oldSel=pd.read_csv(paths[5],float_precision='round_trip')
    oldSensitivity=pd.read_csv(paths[6],float_precision='round_trip')
    bas=[];sels=[];audits=[]
    for ds in ['organamnist','sipakmed']:
        for role in ['primary_original','common_seed_sensitivity']:
            data=runs[runs.dataset==ds]
            if role=='common_seed_sensitivity':data=data[data.training_seed.isin(b.COMMON)]
            nk=5 if role=='primary_original' else 3
            strata=[b.Stratum(f'{ds}_{policy}_cnn4_'+('all5' if nk==5 else 'common3'),ds,policy,role,
                             data[data.checkpoint_policy==policy],b.MODELS) for policy in ['A','B']]
            a,c=strata;assert a.splits==c.splits and a.seeds==c.seeds and a.models==c.models
            seed=b.keyseed(a.key+':crossed')
            if nk==5:
                path=PREV/(a.key+'_draws.npz');paths.append(path)
                assert sha(path)==previous['output_hashes'][path.name]
                stored=np.load(path,allow_pickle=False);sp=stored['split_indices'];se=stored['global_seed_indices']
                draw_origin='reused '+str(path.relative_to(ROOT))
            else:
                sp,seedgrid=b.draw_indices(a,np.random.default_rng(seed),mode='crossed',tag='paired A/B common3')
                se=seedgrid[:,0,:];draw_origin='B1 keyseed(stratum:crossed), common3 replay'
            aa=a.arr[sp[:,:,None],se[:,None,:]];bb=c.arr[sp[:,:,None],se[:,None,:]]
            aw=aa.argmax(axis=-1);bw=bb.argmax(axis=-1)
            delta=c.arr-a.arr;drawdelta=delta[sp[:,:,None],se[:,None,:]].mean(axis=(1,2))
            alternative=bb.mean(axis=(1,2))-aa.mean(axis=(1,2))
            err=float(np.max(np.abs(drawdelta-alternative)));tolerance=4*np.finfo(float).eps*10*nk
            assert err<tolerance
            old=oldBA[(oldBA.dataset==ds)&(oldBA.role==role)]
            for j,model in enumerate(a.models):
                before=old[old.model==model].iloc[0]
                lo,hi=b.interval(drawdelta[:,j]);point=float(delta[...,j].mean())
                assert point==before.mean_paired_BA_difference
                p,n=b.signflip(delta[...,j].mean(axis=1))
                assert p==before.sign_flip_p_raw_conditional and n==before.sign_assignments
                row=before.to_dict();row.update(CI_low=float(lo),CI_high=float(hi),
                    nested_CI_low=before.CI_low,nested_CI_high=before.CI_high,
                    interval_includes_zero=bool(lo<=0<=hi),nested_interval_includes_zero=bool(before.CI_low<=0<=before.CI_high),
                    CI_width_change=float(hi-lo-before.CI_high+before.CI_low),
                    resampling='paired crossed split-seed; same A/B draw',bootstrap_replicates=10000,rng_seed=str(seed),
                    statistical_target='fixed-model mean BA difference B minus A',draw_source=draw_origin)
                bas.append(row)
            oa,ob=b.observed(a),b.observed(c)
            oldselection=oldSel[(oldSel.dataset==ds)&(oldSel.role==role)]
            for j,model in enumerate(a.models):
                before=oldselection[oldselection.model==model].iloc[0]
                values=(bw==j).mean(axis=(1,2))-(aw==j).mean(axis=(1,2))
                point=float(ob['frequency'][j]-oa['frequency'][j]);assert point==before.frequency_difference_B_minus_A
                lo,hi=b.interval(values);row=before.to_dict()
                row.update(CI_low=float(lo),CI_high=float(hi),nested_CI_low=before.CI_low,nested_CI_high=before.CI_high,
                    resampling='paired crossed split-seed; same A/B draw',bootstrap_replicates=10000,
                    statistical_target='change in observed selection frequency B minus A',rng_seed=str(seed))
                sels.append(row)
            changed=(aw!=bw).mean(axis=(1,2));lo,hi=b.interval(changed)
            before=oldselection[oldselection.model=='ALL'].iloc[0]
            point=float((oa['winner']!=ob['winner']).mean());assert point==before.context_winner_changed_fraction
            row=before.to_dict();row.update(CI_low=float(lo),CI_high=float(hi),nested_CI_low=before.CI_low,
                nested_CI_high=before.CI_high,resampling='paired crossed split-seed; same A/B draw',
                bootstrap_replicates=10000,statistical_target='fraction of paired contexts with changed model identity',rng_seed=str(seed))
            sels.append(row)
            legacy=oldSensitivity[(oldSensitivity.stratum==a.key)&(oldSensitivity.resampling=='crossed')]
            for j,model in enumerate(a.models):
                assert abs(float((aa.mean(axis=(1,2)).argmax(axis=-1)==j).mean())-legacy[legacy.model==model].iloc[0].paired_bootstrap_selection_frequency)<1e-14
            for i in range(32):
                scalar=np.asarray([[delta[int(sp[i,s]),int(se[i,k])] for k in range(nk)] for s in range(10)])
                assert np.allclose(scalar.mean(axis=(0,1)),drawdelta[i],rtol=0,atol=tolerance)
            np.savez_compressed(OUT/(ds+'_'+role+'_AB_draws.npz'),
                split_indices=sp,global_seed_indices=se,mean_BA_B_minus_A=drawdelta,
                selection_frequency_B_minus_A=np.stack([(bw==j).mean(axis=(1,2))-(aw==j).mean(axis=(1,2)) for j in range(4)],axis=1),
                changed_model_fraction=changed,model_ids=np.asarray(a.models))
            audits.append(dict(dataset=ds,role=role,contexts=10*nk,unique_runs=10*nk*4*2,
                paired_grid_check='EXACT',observed_effects='UNCHANGED',signflip_p='EXACT',crossed_A_q_replay='EXACT_COUNTS',
                arithmetic_max_abs_error=err,arithmetic_bound=tolerance,scalar_direct_draw_checks=32))
            print(ds+' '+role+f': identity change {point:.3f} [{lo:.6f}, {hi:.6f}]',flush=True)
    baf=pd.DataFrame(bas);self=pd.DataFrame(sels)
    assert len(baf)==16 and len(self)==20
    for role,g in baf.groupby('role',sort=False):
        assert np.array_equal(b.holm(g.sign_flip_p_raw_conditional),g.holm_p_policy_family)
    baf.to_csv(OUT/'Crossed_AB_Paired_BA.csv',index=False,float_format='%.17g')
    self.to_csv(OUT/'Crossed_AB_Selection_Changes.csv',index=False,float_format='%.17g')
    pd.DataFrame(audits).to_csv(OUT/'Crossed_AB_Computation_Audit.csv',index=False)
    primary=baf[baf.role=='primary_original'];switch=self[(self.role=='primary_original')&(self.model=='ALL')]
    summary=dict(primary_BA_comparisons=8,common3_BA_comparisons=8,selection_change_summaries=20,
        primary_BA_wider=int((primary.CI_width_change>1e-14).sum()),
        primary_BA_zero_inclusion_changes=int((primary.interval_includes_zero!=primary.nested_interval_includes_zero).sum()),
        primary_Holm_below_005=int((primary.holm_p_policy_family<.05).sum()),
        point_estimates_and_original_tests_unchanged=True,
        primary_changed_identity=switch[['dataset','context_winner_changed_fraction','nested_CI_low','nested_CI_high','CI_low','CI_high']].to_dict('records'))
    (OUT/'Crossed_AB_Result_Summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    result=dict(analysis_id='JIIM_crossed_AB_completion',created_utc=datetime.now(timezone.utc).isoformat(),
        status='COMPUTED_AND_SOURCE_CHECKED',authorization='Author: 继续（A/B配对差及选择变化区间）',
        scope='8 original primary BA contrasts + 8 existing common3 sensitivity; paired frequency and identity changes',
        raw_unique_runs=800,common3_subset_unique_runs=480,common3_is_reused_subset=True,new_training_runs=0,
        resampling='10000 paired crossed draws per dataset/seed-pool; same indices for all models and A/B',
        base_seed=b.BASE_SEED,seed_derivation='B1 keyseed(A-stratum:crossed); primary saved draws reused, common3 A draws replayed',
        output_hashes={p.name:sha(p) for p in OUT.iterdir()},input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        raw_metric_hashes=raw,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,executable=sys.executable,
        elapsed_seconds=time.perf_counter()-start,result_summary=summary,
        verification='Same-agent source/arithmetic checks, not independent-person review',
        application_state='New statistical materials; manuscript/Word integration pending')
    (OUT/'Crossed_AB_Manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
