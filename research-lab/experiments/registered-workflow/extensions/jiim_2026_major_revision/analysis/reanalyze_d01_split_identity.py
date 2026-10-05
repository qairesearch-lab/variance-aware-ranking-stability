#!/usr/bin/env python3
"""D01 sensitivity analysis: split-identity-aware hierarchical bootstrap.

Does not retrain models or modify the sealed extension. In each bootstrap draw,
all repeated appearances of the held original split identity are excluded from
its comparator. A draw with fewer than two distinct original split identities
is rejected and resampled; the conditioning and rejection count are reported.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
ROOT=HERE.parents[5]
INPUT=EXT/'analysis/inputs/metrics'
OLD=ROOT/'research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv'
OUT=EXT/'analysis/inputs/reference'
MODELS=sorted(['resnet18','resnet50','densenet121','efficientnet_b0'])
N_BOOT=10_000
SEED=20261003

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def top(values): return sorted(values,key=lambda m:(-values[m],m))[0]
def logistic_slope(x,y):
    n,p=x.shape; beta=np.zeros(p); ridge=np.eye(p)*1e-8; ridge[0,0]=0
    for _ in range(60):
        eta=np.clip(x@beta,-25,25); prob=1/(1+np.exp(-eta)); weight=np.clip(prob*(1-prob),1e-8,None)
        h=x.T@(weight[:,None]*x)+ridge
        step=np.linalg.solve(h,x.T@(y-prob)-ridge@beta); beta+=step
        if np.max(np.abs(step))<1e-8: break
    return float(beta[-1])
def ci(x): return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]

def load_contexts():
    old=pd.read_csv(OLD)
    old=old[(old.checkpoint_policy=='A')&(old.analysis_role=='primary_analysis')&old.model.isin(MODELS)]
    old=old[['dataset','split_id','training_seed','model','balanced_accuracy']].copy()
    ext=pd.read_csv(INPUT/'extension_run_metrics.csv')
    ext=ext[(ext.variant=='shared')&ext.model.isin(MODELS)]
    ext=ext[['dataset','split_id','training_seed','model','balanced_accuracy']].copy()
    allm=pd.concat([old,ext],ignore_index=True)
    rows=[]
    for (ds,sp,seed),g in allm.groupby(['dataset','split_id','training_seed']):
        vals=dict(zip(g.model,g.balanced_accuracy))
        if set(vals)!=set(MODELS): raise ValueError(f'incomplete/duplicate model context {ds}/{sp}/{seed}')
        order=sorted(vals,key=lambda m:(-vals[m],m))
        rows.append(dict(dataset=ds,split_id=str(sp),training_seed=int(seed),winner=order[0],margin=float(vals[order[0]]-vals[order[1]]),model_values=vals))
    ctx=pd.DataFrame(rows)
    # Identity-correct observed LSO point estimate: remove all records for held split.
    for ds,g in ctx.groupby('dataset'):
        for sp in g.split_id.unique():
            ref=allm[(allm.dataset==ds)&(allm.split_id.astype(str)!=str(sp))].groupby('model').balanced_accuracy.mean().to_dict()
            rt=top(ref)
            mask=(ctx.dataset==ds)&(ctx.split_id==str(sp))
            ctx.loc[mask,'reference_top']=rt
            ctx.loc[mask,'discordant']=(ctx.loc[mask,'winner']!=rt).astype(int)
    return allm,ctx

def bootstrap_dataset(d, rng):
    splits=sorted(d.split_id.astype(str).unique())
    seeds=sorted(int(x) for x in d.training_seed.unique())
    arr=np.empty((len(splits),len(seeds),len(MODELS)))
    wide=d.pivot(index=['split_id','training_seed'],columns='model',values='balanced_accuracy').reindex(columns=MODELS)
    for i,sp in enumerate(splits):
        for j,seed in enumerate(seeds): arr[i,j]=wide.loc[(sp,seed)].to_numpy(float)
    return splits,seeds,arr

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    allm,ctx=load_contexts(); rng=np.random.default_rng(SEED)
    stability=[]; context_tensors={}
    for ds,g in allm.groupby('dataset'):
        splits,seeds,arr=bootstrap_dataset(g,rng); context_tensors[ds]=(splits,seeds,arr)
        s_n,k_n,_=arr.shape
        point=ctx[ctx.dataset==ds]
        point_agree=float(1-point.discordant.mean())
        vals=[]; rejected=0
        for _ in range(N_BOOT):
            while True:
                chosen=rng.integers(0,s_n,size=s_n)
                if len(set(chosen.tolist()))>=2: break
                rejected+=1
            seed_idx=rng.integers(0,k_n,size=(s_n,k_n))
            matches=0
            for i in range(s_n):
                held_id=chosen[i]
                ref_pos=[j for j in range(s_n) if chosen[j]!=held_id]
                # By the guard above, at least one other original split identity exists.
                ref=np.stack([arr[chosen[j],seed_idx[j]] for j in ref_pos]).mean(axis=(0,1))
                rt=int(np.argmax(ref))
                held=arr[held_id,seed_idx[i]]
                matches+=int(np.sum(np.argmax(held,axis=1)==rt))
            vals.append(matches/(s_n*k_n))
        stability.append(dict(dataset=ds,splits=s_n,seeds_per_split=k_n,contexts=len(point),observed_LSO_agreement=point_agree,
                              identity_aware_bootstrap_CI_low=ci(vals)[0],identity_aware_bootstrap_CI_high=ci(vals)[1],
                              bootstrap_replicates=N_BOOT,bootstrap_seed=SEED,degenerate_draws_resampled=rejected,
                              conditioning='bootstrap draw contains at least two distinct original split identities'))
    # Point model using the observed identity-correct LSO comparator.
    ds_order=sorted(ctx.dataset.unique())
    logm=np.log10(ctx.margin.to_numpy(float)+1e-6); center=float(logm.mean()); scale=float(logm.std(ddof=1))
    z=(logm-center)/scale; ds=ctx.dataset.to_numpy()
    x=np.column_stack([np.ones(len(ctx))]+[(ds==name).astype(float) for name in ds_order[1:]]+[-z])
    beta=logistic_slope(x,ctx.discordant.to_numpy(float)); point_or=math.exp(beta)
    boots=[]; rejected_total=0; failed=0
    for _ in range(N_BOOT):
        yb=[]; zb=[]; db=[]
        for dataset in ds_order:
            splits,seeds,arr=context_tensors[dataset]; s_n,k_n,_=arr.shape
            while True:
                chosen=rng.integers(0,s_n,size=s_n)
                if len(set(chosen.tolist()))>=2: break
                rejected_total+=1
            seed_idx=rng.integers(0,k_n,size=(s_n,k_n))
            for i in range(s_n):
                held_id=chosen[i]
                ref_pos=[j for j in range(s_n) if chosen[j]!=held_id]
                ref=np.stack([arr[chosen[j],seed_idx[j]] for j in ref_pos]).mean(axis=(0,1))
                rt=int(np.argmax(ref)); held=arr[held_id,seed_idx[i]]
                winners=np.argmax(held,axis=1); margins=np.sort(held,axis=1)[:,-1]-np.sort(held,axis=1)[:,-2]
                yb.extend((winners!=rt).astype(int).tolist()); zb.extend(np.log10(margins+1e-6).tolist()); db.extend([dataset]*k_n)
        zboot=(np.asarray(zb)-center)/scale; dboot=np.asarray(db)
        xb=np.column_stack([np.ones(len(yb))]+[(dboot==name).astype(float) for name in ds_order[1:]]+[-zboot])
        try: boots.append(math.exp(logistic_slope(xb,np.asarray(yb,float))))
        except np.linalg.LinAlgError: failed+=1
    margin_row=dict(term='one_SD_decrease_in_log10_context_top_two_BA_difference',odds_ratio=point_or,
                    identity_aware_cluster_bootstrap_CI_low=ci(boots)[0],identity_aware_cluster_bootstrap_CI_high=ci(boots)[1],
                    datasets=len(ds_order),clusters=int(ctx.groupby('dataset').split_id.nunique().sum()),contexts=len(ctx),
                    bootstrap_replicates=len(boots),requested_replicates=N_BOOT,bootstrap_seed=SEED,
                    degenerate_draws_resampled=rejected_total,failed_logistic_fits=failed,
                    conditioning='within each dataset bootstrap draw, at least two distinct original split identities',
                    interpretation='associational; discordance is defined against the identity-correct leave-one-split-out comparator; interval is conditional on these datasets and split-generation process')
    pd.DataFrame(stability).to_csv(OUT/'d01_identity_aware_lso_intervals.csv',index=False)
    pd.DataFrame([margin_row]).to_csv(OUT/'d01_identity_aware_margin_model.csv',index=False)
    inputs=[OLD,INPUT/'extension_run_metrics.csv',HERE/'reanalyze_d01_split_identity.py']
    summary={'analysis_id':'JIIM_D01_split_identity_bootstrap_v0.2','date':'2026-10-03','status':'completed',
             'source_analysis':'extension results v0.1; original completed Rule-A runs plus shared-input CNN extension',
             'point_estimates_recomputed':True,'point_estimate_scope':'four datasets; 4 CNNs; original Rule-A 10x5 for OrganAMNIST/SIPaKMeD and extension Rule-A 10x3 for ISIC/MURA',
             'bootstrap':{'replicates':N_BOOT,'seed':SEED,'cluster':'original split identity; seeds nested within sampled split','heldout_rule':'exclude every sampled occurrence sharing held-out original split identity from its comparator','degenerate_draw':'resample if fewer than two distinct original split identities; counts recorded'},
             'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'lso_results':stability,'margin_model':margin_row,
             'limitations':['Conditioned on the four fixed benchmark datasets and observed split-generation process; no new independent patient/site sample was generated.','The margin model is associational and context margins are order statistics affected by candidate pool.','D02, D04-D09 remain outside this D01 reanalysis.']}
    (OUT/'d01_reanalysis_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    report=['# D01 重分析：按原 split 身份排除重复抽样', '', '## Analysis provenance', '', '- ID：JIIM-D01-v0.2', '- 类型：split-cluster bootstrap sensitivity reanalysis', '- 状态：ANALYZED；代码成功运行，未独立重训', '- 范围：四数据集的共同四CNN；原数据使用Rule A 10×5，扩展数据使用Rule A 10×3', '- 输入及SHA-256：见 `d01_reanalysis_manifest.json`', '- 输出：LSO区间表、分差模型区间表和manifest', '', '## 方法修正', '', '每次抽取10个split cluster并在cluster内对seed重抽样。对某个held-out原split，参考集排除本次样本中所有具有相同原split身份的抽样位置。若一次draw只含一个不同原split身份，则整次draw重抽并计数。点估计按全部原split identity排除后重新计算。', '', '## LSO结果', '', '| Dataset | contexts | identity-aware LSO agreement | 95% bootstrap interval | degenerate draws resampled |', '|---|---:|---:|---:|---:|']
    for r in stability: report.append(f"| {r['dataset']} | {r['contexts']} | {r['observed_LSO_agreement']:.3f} | {r['identity_aware_bootstrap_CI_low']:.3f}–{r['identity_aware_bootstrap_CI_high']:.3f} | {r['degenerate_draws_resampled']} |")
    report += ['', '## 分差关联模型', '', f"OR = {point_or:.3f}; identity-aware cluster-bootstrap 95% CI {margin_row['identity_aware_cluster_bootstrap_CI_low']:.3f}–{margin_row['identity_aware_cluster_bootstrap_CI_high']:.3f}; {margin_row['clusters']} dataset×split clusters and {margin_row['contexts']} contexts. This is an associational estimate conditional on these datasets; the interval is not evidence of causation or a universal threshold.", '', '## 验证边界', '', '这次修正避免同一原split因cluster bootstrap重复抽样而同时进入held-out位置与其参考集。它没有产生新的患者或机构样本，也不修复重叠holdout对总体外推的限制。只关闭D01所覆盖的LSO/分差区间算法问题；完整Rule B统一不确定性(D02)、聚合预算(D04)及其他D项需另行核验。']
    (OUT/'D01_reanalysis_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status':'completed','outputs':str(OUT),'lso':stability,'margin':margin_row},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
