"""Audit remaining uncertainty objects and finish crossed analysis migration.

Uses existing 1,100 runs. Historical analyses and training outputs stay read only.
"""
import hashlib,importlib.util,itertools,json,math,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5];OLD=HERE/'results';OUT=HERE/'crossed/summary'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strata_for(b,runs):
    strata=[]
    for ds in ['organamnist','sipakmed']:
        for policy in ['A','B']:
            data=runs[(runs.dataset==ds)&(runs.checkpoint_policy==policy)]
            strata.append(b.Stratum(f'{ds}_{policy}_cnn4_all5',ds,policy,'primary_original',data,b.MODELS))
            strata.append(b.Stratum(f'{ds}_{policy}_cnn4_common3',ds,policy,'common_seed_sensitivity',data[data.training_seed.isin(b.COMMON)],b.MODELS))
    for ds in ['isic2019','mura']:
        data=runs[(runs.dataset==ds)&(runs.variant=='shared')&runs.model.isin(b.MODELS)]
        strata.append(b.Stratum(f'{ds}_A_cnn4_all3',ds,'A','primary_extension',data,b.MODELS))
        short=data[data.split_id.isin([f'split_{i:02d}' for i in range(1,6)])]
        strata.append(b.Stratum(f'{ds}_A_cnn4_pool_contexts',ds,'A','candidate_pool_control',short,b.MODELS))
        for variant in ['shared','swin_weight_eval']:
            sw=runs[(runs.dataset==ds)&(runs.model=='swin_t')&(runs.variant==variant)]
            strata.append(b.Stratum(f'{ds}_A_pool5_{variant}',ds,'A','candidate_pool_sensitivity',pd.concat([short,sw]),b.MODELS+['swin_t']))
    return strata

def main():
    t=time.perf_counter();OUT.mkdir(exist_ok=False)
    src=HERE/'crossed_core.py';spec=importlib.util.spec_from_file_location('b1_object_audit_readonly',src)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    manifest=json.loads((OLD/'B1_analysis_manifest.json').read_text());paths=[Path(__file__),src,OLD/'B1_analysis_manifest.json']
    assert sha(src)==manifest['source_hashes'][str(src.relative_to(ROOT))]
    names=['B1_run_inputs.csv','D02_model_performance_selection.csv','D02_ranking_stability.csv','D02_resampling_design_sensitivity.csv','D02_paired_model_BA_differences.csv','D04_partition_budget_details.csv','D04_budget_summary.csv','D01_margin_implementation_replay.csv']
    tables={}
    for name in names:
        p=OLD/name;assert sha(p)==manifest['output_hashes'][name];paths.append(p);tables[name]=pd.read_csv(p,float_precision='round_trip')
    runs=tables[names[0]];assert len(runs)==runs.run_id.nunique()==1100
    raw={r.metric_source_path:manifest['source_hashes'][r.metric_source_path] for r in runs.itertuples()}
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    strata=strata_for(b,runs);assert len(strata)==16
    perf=tables[names[1]];stability=tables[names[2]];legacy=tables[names[3]];oldpairs=tables[names[4]]
    modelrows=[];stablerows=[];pairrows=[];audits=[]
    for s in strata:
        obs=b.observed(s);rng=np.random.default_rng(b.keyseed(s.key+':crossed'))
        sp,se=b.draw_indices(s,rng,mode='crossed',tag='remaining object replay')
        sample=s.arr[sp[:,:,None],se];means=sample.mean(axis=(1,2));winner=sample.argmax(axis=-1);meanwinner=means.argmax(axis=-1)
        meanCI=b.interval(means);freqCI=b.interval(np.stack([(winner==j).mean(axis=(1,2)) for j in range(len(s.models))],axis=1))
        full=(winner==meanwinner[:,None,None]).mean(axis=(1,2));fullCI=b.interval(full)
        lsample,lsp=b.valid_lso_sample(s,sp,se,'crossed',rng,'remaining object replay')
        lso,refs=b.lso_from_sample(lsample,lsp)
        # Recompute references directly by original split identity for every draw.
        maxerr=0.
        for occurrence in range(len(s.splits)):
            mask=lsp!=lsp[:,occurrence,None]
            assert (mask.sum(axis=1)>0).all()
            direct=(lsample.mean(axis=2)*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1)[:,None]
            assert np.array_equal(direct.argmax(axis=-1),refs[:,occurrence])
        directagree=(lsample.argmax(axis=-1)==refs[:,:,None]).mean(axis=(1,2));assert np.array_equal(directagree,lso)
        for i in range(32):
            val=np.mean([lsample[i,j].argmax(axis=-1)==lsample[i,lsp[i]!=lsp[i,j]].mean(axis=(0,1)).argmax() for j in range(len(s.splits))])
            assert abs(val-lso[i])<1e-14
        old=legacy[(legacy.stratum==s.key)&(legacy.resampling=='crossed')];lCI=b.interval(lso)
        assert np.max(np.abs(old.LSO_CI_low-lCI[0]))<1e-14 and np.max(np.abs(old.LSO_CI_high-lCI[1]))<1e-14
        for j,model in enumerate(s.models):
            row=perf[(perf.stratum==s.key)&(perf.model==model)].iloc[0].to_dict();prior=old[old.model==model].iloc[0]
            q=float((meanwinner==j).mean());assert q==prior.paired_bootstrap_selection_frequency
            assert abs(meanCI[0,j]-prior.mean_BA_CI_low)<1e-14 and abs(freqCI[0,j]-prior.observed_frequency_CI_low)<1e-14
            row.update(mean_BA_CI_low=float(meanCI[0,j]),mean_BA_CI_high=float(meanCI[1,j]),selection_frequency_CI_low=float(freqCI[0,j]),selection_frequency_CI_high=float(freqCI[1,j]),
                paired_bootstrap_selection_frequency=q,selection_bootstrap_MC_SE=math.sqrt(q*(1-q)/10000),CI_method='paired crossed split-seed percentile 95%; fixed datasets',
                source_nested_q=row['paired_bootstrap_selection_frequency'],resampling='paired crossed split-seed')
            modelrows.append(row)
        row=stability[stability.stratum==s.key].iloc[0].to_dict()
        assert obs['full_agreement']==row['full_grid_in_sample_agreement'] and obs['lso_agreement']==row['identity_correct_LSO_agreement']
        margin=np.sort(sample,axis=-1)[...,-1]-np.sort(sample,axis=-1)[...,-2];mCI=b.interval(margin.mean(axis=(1,2)))
        row.update(full_grid_agreement_CI_low=float(fullCI[0]),full_grid_agreement_CI_high=float(fullCI[1]),LSO_CI_low=float(lCI[0]),LSO_CI_high=float(lCI[1]),
            mean_context_difference_CI_low=float(mCI[0]),mean_context_difference_CI_high=float(mCI[1]),resampling='paired crossed; LSO excludes all repeated original split identities')
        stablerows.append(row)
        for i,j in itertools.combinations(range(len(s.models)),2):
            row=oldpairs[(oldpairs.stratum==s.key)&(oldpairs.model_1==s.models[i])&(oldpairs.model_2==s.models[j])].iloc[0].to_dict()
            delta=s.arr[...,i]-s.arr[...,j];values=means[:,i]-means[:,j];lo,hi=b.interval(values)
            assert float(delta.mean())==row['mean_paired_BA_difference_model1_minus_model2']
            p,n=b.signflip(delta.mean(axis=1));assert p==row['sign_flip_p_raw_conditional'] and n==row['sign_assignments']
            row.update(paired_CI_low=float(lo),paired_CI_high=float(hi),CI_method='paired crossed split-seed percentile',bootstrap_replicates=10000)
            pairrows.append(row)
        np.savez_compressed(OUT/f'{s.key}_object_draws.npz',split_indices=sp,global_seed_indices=se[:,0,:],mean_BA=means,full_agreement=full,LSO_agreement=lso,LSO_split_indices=lsp)
        audits.append(dict(stratum=s.key,models=len(s.models),raw_runs=len(s.data),crossed_q_mean_frequency_replay='PASS',LSO_crossed_interval_replay='PASS',duplicate_original_split_exclusion='ALL_DRAWS_PASS',direct_first_draw_checks=32))
    mp=pd.DataFrame(modelrows);st=pd.DataFrame(stablerows);pr=pd.DataFrame(pairrows)
    assert len(mp)==68 and len(st)==16 and len(pr)==len(oldpairs)
    for _,g in pr.groupby('stratum'):
        assert np.array_equal(b.holm(g.sign_flip_p_raw_conditional),g.holm_p_within_stratum)
    # Verify primary effects/intervals against earlier completed batches.
    ap=HERE/'crossed/agreement/Crossed_Agreement_Summary.csv';pp=HERE/'crossed/paired_BA/Crossed_Primary_Paired_BA_Comparisons.csv';paths += [ap,pp]
    for r in pd.read_csv(ap,float_precision='round_trip').itertuples():
        row=st[st.stratum==r.stratum].iloc[0];assert abs(row.full_grid_agreement_CI_low-r.agreement_CI_low)<1e-14 and abs(row.full_grid_agreement_CI_high-r.agreement_CI_high)<1e-14
    for r in pd.read_csv(pp,float_precision='round_trip').itertuples():
        row=pr[(pr.stratum==r.stratum)&(pr.model_1==r.model_1)&(pr.model_2==r.model_2)].iloc[0]
        assert abs(row.paired_CI_low-r.paired_CI_low)<1e-14 and abs(row.paired_CI_high-r.paired_CI_high)<1e-14
    mp.to_csv(OUT/'Crossed_Model_Performance_Selection.csv',index=False,float_format='%.17g')
    st.to_csv(OUT/'Crossed_Ranking_Stability.csv',index=False,float_format='%.17g')
    pr.to_csv(OUT/'Crossed_All_Paired_Model_BA_Comparisons.csv',index=False,float_format='%.17g')
    print('16 strata replayed; observed values and prior crossed outputs preserved',flush=True)

    # Association uses fixed dataset intercepts and fixed observed predictor scaling.
    selected=sorted([s for s in strata if s.role.startswith('primary') and s.policy=='A'],key=lambda s:s.dataset)
    logs=np.concatenate([np.log10(b.observed(s)['margin'].ravel()+1e-6) for s in selected]);center=float(logs.mean());scale=float(logs.std(ddof=1))
    ds_order=[s.dataset for s in selected];dd=np.concatenate([[s.dataset]*s.info()['contexts'] for s in selected])
    y=np.concatenate([(b.observed(s)['winner']!=b.observed(s)['lso_reference'][:,None]).ravel() for s in selected])
    x=np.column_stack([np.ones(len(y))]+[(dd==ds).astype(float) for ds in ds_order[1:]]+[-(logs-center)/scale])
    beta,active=b.batch_logistic(x[None],y[None]);point=float(np.exp(beta[0]));assert not active.any()
    oldassoc=tables[names[7]].iloc[0];assert abs(point-oldassoc.recomputed_odds_ratio)<1e-12
    all_y=[];all_log=[];drawparams=[]
    for s in selected:
        rngseed=b.keyseed(s.key+':crossed_association');rng=np.random.default_rng(rngseed)
        sp,se=b.draw_indices(s,rng,mode='crossed',tag='association crossed',require_reference=True)
        sample=s.arr[sp[:,:,None],se];_,ref=b.lso_from_sample(sample,sp)
        all_y.append((sample.argmax(axis=-1)!=ref[:,:,None]).reshape(10000,-1))
        ss=np.sort(sample,axis=-1);all_log.append(np.log10(ss[...,-1]-ss[...,-2]+1e-6).reshape(10000,-1))
        np.savez_compressed(OUT/f'{s.key}_association_draws.npz',split_indices=sp,global_seed_indices=se[:,0,:])
        drawparams.append(dict(stratum=s.key,rng_seed=str(rngseed),excluded_all_same_identity=int(b.DRAW_AUDIT[-1]['degenerate_draws_resampled'])))
    yy=np.concatenate(all_y,axis=1);z=-(np.concatenate(all_log,axis=1)-center)/scale;draw_OR=[];failed=0
    for first in range(0,10000,256):
        zz=z[first:first+256];count=len(zz)
        xx=np.stack([np.ones_like(zz)]+[np.broadcast_to((dd==ds).astype(float),zz.shape) for ds in ds_order[1:]]+[zz],axis=-1)
        coef,bad=b.batch_logistic(xx,yy[first:first+256]);failed+=int(bad.sum());draw_OR.extend(np.exp(coef).tolist())
    assert failed==0,'Association did not converge for all draws; no failed fits silently removed'
    lo,hi=b.interval(draw_OR)
    assoc=dict(term='one SD increase in negative standardized log10(top-two BA difference + 1e-6)',odds_ratio=point,CI_low=float(lo),CI_high=float(hi),
        nested_CI_low=float(oldassoc.replay_CI_low),nested_CI_high=float(oldassoc.replay_CI_high),datasets=4,contexts=160,clusters=40,bootstrap_replicates=10000,
        failed_fits=failed,ridge=1e-8,epsilon=1e-6,observed_log_center=center,observed_log_sample_SD=scale,
        resampling='dataset-stratified paired crossed split-seed; identity-aware LSO re-estimated per draw',draw_seed_records=json.dumps(drawparams,sort_keys=True),
        interpretation='exploratory association; fixed datasets; no causal mechanism or universal threshold')
    pd.DataFrame([assoc]).to_csv(OUT/'Crossed_Exploratory_Association.csv',index=False,float_format='%.17g')
    np.savez_compressed(OUT/'association_OR_draws.npz',OR=np.asarray(draw_OR))
    print(f'Association: OR={point:.6f}, crossed CI=[{lo:.6f},{hi:.6f}], failed fits={failed}',flush=True)

    # Only resampled-reference budget sensitivity changes; finite enumeration stays fixed.
    olddetails=tables[names[5]];oldbudgets=tables[names[6]];refrows=[];budgetrows=[]
    for s in [s for s in strata if s.role.startswith('primary') or s.role=='common_seed_sensitivity']:
        ns,nk,nm=s.arr.shape;assert ns==10
        kw={k:b.combination_weights(nk,k) for k in range(1,nk+1)};sw={k:b.combination_weights(5,k) for k in range(1,6)}
        outputs={}
        for partition,discovery in enumerate(itertools.combinations(range(10),5)):
            reference=sorted(set(range(10))-set(discovery));rr=s.arr[reference];disc=s.arr[list(discovery)]
            rng=np.random.default_rng(b.keyseed(s.key+f':reference:{partition}'))
            si=rng.integers(0,5,(2000,5));ki=rng.integers(0,nk,(2000,1,nk));refmean=rr[si[:,:,None],ki].mean(axis=(1,2))
            frequency=np.bincount(refmean.argmax(axis=-1),minlength=nm)/2000;ref_top=int(rr.mean(axis=(0,1)).argmax())
            refrows.append(dict(stratum=s.key,partition_id=partition,reference_split_ids='|'.join(s.splits[i] for i in reference),
                reference_top_frequencies_json=json.dumps(dict(zip(s.models,frequency.tolist())),sort_keys=True),reference_bootstrap_top_frequency_of_mean_BA_top=float(frequency[ref_top]),
                reference_bootstrap_replicates=2000,resampling='paired crossed split-seed'))
            for splits in range(1,6):
                for seeds in range(1,nk+1):
                    means=np.einsum('ij,jkm,lk->ilm',sw[splits],disc,kw[seeds]).reshape(-1,nm)
                    counts=np.bincount(means.argmax(axis=-1),minlength=nm)/len(means)
                    outputs[(partition,splits,seeds)]=float(counts@frequency)
        for r in oldbudgets[oldbudgets.stratum==s.key].to_dict('records'):
            splits=int(r['budget_splits']);seeds=int(r['budget_seeds'])
            values=[outputs[(p,splits,seeds)] for p in range(252)];oldval=r['bootstrap_reference_expected_agreement']
            r.update(bootstrap_reference_expected_agreement=float(np.mean(values)),nested_bootstrap_reference_expected_agreement=oldval,
                resampled_reference_method='2000 paired crossed split-seed draws per held-out block',primary_budget_statistics_changed=False)
            budgetrows.append(r)
        print(s.key+': budget resampled reference completed; main curve unchanged',flush=True)
    bf=pd.DataFrame(budgetrows);assert len(bf)==len(oldbudgets)
    assert np.array_equal(bf.aggregate_selection_agreement,oldbudgets.aggregate_selection_agreement)
    assert np.array_equal(bf.partition_q025,oldbudgets.partition_q025) and np.array_equal(bf.partition_q975,oldbudgets.partition_q975)
    pd.DataFrame(refrows).to_csv(OUT/'Crossed_Budget_Reference_Frequencies.csv',index=False)
    bf.to_csv(OUT/'Crossed_Budget_Summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(audits).to_csv(OUT/'Statistical_Object_Computation_Audit.csv',index=False)
    pd.DataFrame(b.DRAW_AUDIT).to_csv(OUT/'Statistical_Object_Draw_Audit.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in raw.items())
    summary=dict(strata=16,unique_raw_runs=1100,new_training_runs=0,model_rows=len(mp),stability_rows=len(st),paired_contrasts=len(pr),
        LSO_crossed_legacy_replay=True,primary_agreement_and_pair_CI_replay=True,association=assoc,budget_rows=len(bf),budget_reference_partition_rows=len(refrows),
        main_budget_curves_and_partition_bands_unchanged=True)
    (OUT/'Statistical_Object_Result_Summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    result=dict(analysis_id='JIIM_statistical_object_audit',created_utc=datetime.now(timezone.utc).isoformat(),status='COMPUTED_AND_SOURCE_CHECKED',
        authorization='Author active goal: complete remaining revisions and integrate manuscript/attachments; preserve experiment provenance',
        scope='Remaining existing objects: 16-stratum crossed summaries/LSO, all fixed-model intervals, exploratory association, resampled budget references',
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},raw_metric_hashes=raw,output_hashes={p.name:sha(p) for p in OUT.iterdir()},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,executable=sys.executable,base_seed=b.BASE_SEED,elapsed_seconds=time.perf_counter()-t,
        summary=summary,verification='Same-agent source/arithmetic checks; no independent-person statistical certification',application_state='Not yet integrated into manuscript/Word')
    (OUT/'Statistical_Object_Manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='association'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
