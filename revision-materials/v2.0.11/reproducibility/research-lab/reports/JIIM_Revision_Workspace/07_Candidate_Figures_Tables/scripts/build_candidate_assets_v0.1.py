"""Render existing JIIM results; no training or new inferential analysis.

Run with a Python environment providing numpy, pandas and matplotlib.
All outputs are candidates, not an edited submitted manuscript.
"""
from pathlib import Path
import csv,json,hashlib,textwrap,math,platform
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[5]
WS=ROOT/'research-lab/reports/JIIM_Revision_Workspace'
OUT=WS/'07_Candidate_Figures_Tables'
DATA=ROOT/'research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3'
OLD=ROOT/'research-lab/experiments/registered-workflow/stats/tables'
FOLDERS={'正文候选':'01_Main_Candidates','补充候选':'02_Supplement_Candidates','回信或备存':'03_Response_Reserve'}
for f in list(FOLDERS.values())+['04_Data_and_Provenance']:(OUT/f).mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
ORDER=['organamnist_A_cnn4_all5','organamnist_B_cnn4_all5','sipakmed_A_cnn4_all5','sipakmed_B_cnn4_all5','isic2019_A_cnn4_all3','mura_A_cnn4_all3']
MODELS=['resnet18','resnet50','densenet121','efficientnet_b0']
ML={'resnet18':'ResNet-18','resnet50':'ResNet-50','densenet121':'DenseNet-121','efficientnet_b0':'EfficientNet-B0','swin_t':'Swin-T'}
DL={'organamnist':'OrganAMNIST','sipakmed':'SIPaKMeD','isic2019':'ISIC2019','mura':'MURA'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']
SOURCES={}; ASSETS=[]; CAPTIONS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def load(path):
    path=Path(path);SOURCES[rel(path)]=sha(path);return pd.read_csv(path)
def src(name):return load(DATA/f'{name}_v0.1.csv')
perf=src('D02_model_performance_selection');sens=src('D02_resampling_design_sensitivity');stable=src('D02_ranking_stability');ctx=src('D02_contexts');pair=src('D02_paired_model_BA_differences');budget=src('D04_budget_summary');pol=src('D02_policy_B_minus_A');psel=src('D02_policy_selection_sensitivity');pool=src('D02_candidate_pool_sensitivity');inp=src('D02_swin_input_sensitivity');common=src('D05_common_three_seed_sensitivity');inventory=src('B1_stratum_inventory')
summary=load(WS/'01_Manuscript_Revision_Checklists/Bootstrap_Revision/Bootstrap_Result_Summary_v0.2.csv')
var=load(OLD/'mixed_effects_variance_attribution_summary.csv');mstatus=load(OLD/'mixed_effects_sensitivity_model_status.csv');epoch=load(OLD/'checkpoint_policy_epoch_distribution.csv');param=load(OLD/'model_parameter_environment_summary.csv');times=load(WS/'05_B2_Evidence_Coverage/B2_D07_Per_Run_Time_Summary_v0.1.csv');assoc=src('D01_margin_implementation_replay')
def layer(st,full=False):
    r=inventory[inventory.stratum==st].iloc[0]
    s=f"{DL[r.dataset]} / {r.checkpoint_policy}"
    if not full:return s
    tag=''
    if 'common3' in st:tag=' [common3]'
    elif 'pool_contexts' in st:tag=' [CNN control]'
    elif 'pool5_swin_weight_eval' in st:tag=' [weight-eval]'
    elif 'pool5_shared' in st:tag=' [shared input]'
    return s+f" ({int(r.splits)}×{int(r.seeds)}; {int(r.models)} models)"+tag
def field(x,dec=3):return '—' if pd.isna(x) else f'{float(x):.{dec}f}'
def ci(lo,hi,scale=1,dec=3):return f'[{lo*scale:.{dec}f}, {hi*scale:.{dec}f}]'
def primary(df):return df[df.stratum.isin(ORDER)].copy()
def ordered(df):return df.assign(_order=df.stratum.map({s:i for i,s in enumerate(ORDER)})).sort_values('_order').drop(columns='_order')
def register(aid,kind,title,place,stem,paths,reviewers,mids,eids,reason,caption,algorithm):
    ASSETS.append(dict(asset_id=aid,kind=kind,title=title,recommended_placement=place,pdf_path=rel(paths['pdf']),preview_path=rel(paths['png']),data_path=rel(paths['csv']),editable_path=rel(paths.get('svg',paths.get('md',paths['csv']))),reviewer_ids=reviewers,revision_ids=mids,evidence_ids=eids,selection_reason=reason,response_use=f'回信条目{reviewers}：{reason}；引用本asset_id、证据与文件定位',algorithm_or_interval=algorithm,status='GENERATED_CANDIDATE_NOT_IN_MANUSCRIPT',file_version='v0.1'))
    CAPTIONS.append(f'## {aid} · {title}\n\n建议位置：{place}；意见：{reviewers}；修订：{mids}。\n\n**English caption / note:** {caption}\n\n**用途说明：** {reason}\n\n**算法及统计对象：** {algorithm}\n\n**文件：** [{stem}]({str(paths["pdf"].relative_to(OUT))})；[数据]({str(paths["csv"].relative_to(OUT))})。\n')
def table(aid,title,df,place,reviewers,mids,eids,reason,note,algorithm,widths=None):
    stem=f'{aid}_{title.replace(" ","_").replace("/","-")}_v0.1';folder=OUT/FOLDERS[place]
    paths={ext:folder/f'{stem}.{ext}' for ext in ['csv','md','pdf','png']}
    df.to_csv(paths['csv'],index=False,encoding='utf-8-sig')
    rows=df.astype(str).values.tolist();cols=list(df.columns)
    md=f'# {aid}: {title.replace("_"," ")}\n\nCandidate v0.1; not yet inserted in the manuscript.\n\n'
    md+='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    md+='\n'.join('| '+' | '.join(str(x).replace('|','/') for x in r)+' |' for r in rows)
    paths['md'].write_text(md+'\n\n'+note+'\n')
    with PdfPages(paths['pdf']) as pdf:
        size=12
        for page,start in enumerate(range(0,max(1,len(rows)),size)):
            part=rows[start:start+size];fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
            fig.text(.035,.955,f'{aid}  {title.replace("_"," ")}',fontsize=12,weight='bold')
            fig.text(.035,.924,f'Candidate v0.1 • page {page+1} • {len(rows)} rows • not inserted in manuscript',fontsize=8,color='#555555')
            wrappedcols=['\n'.join(textwrap.wrap(str(c),22)) for c in cols]
            wrapped=[['\n'.join(textwrap.wrap(str(v),30 if len(cols)<7 else 23)) for v in r] for r in part]
            height=min(.70,.10+len(part)*.045)
            tb=ax.table(cellText=wrapped,colLabels=wrappedcols,cellLoc='left',colLoc='left',colWidths=widths,bbox=[0,.82-height,1,height])
            tb.auto_set_font_size(False);tb.set_fontsize(8)
            for (r,c),cell in tb.get_celld().items():
                cell.set_linewidth(.4);cell.set_edgecolor('#cccccc');cell.PAD=.03;cell.set_height(.075 if r==0 else .045)
                if r==0:cell.set_facecolor('#dce8f0');cell.get_text().set_weight('bold')
                elif r%2==0:cell.set_facecolor('#f5f7f9')
            fig.text(.035,.07,'\n'.join(textwrap.wrap(note,170)),fontsize=8,va='bottom')
            fig.subplots_adjust(left=.035,right=.975,top=.90,bottom=.15)
            pdf.savefig(fig)
            if page==0:fig.savefig(paths['png'],dpi=180)
            plt.close(fig)
    register(aid,'table',title,place,stem,paths,reviewers,mids,eids,reason,note,algorithm)
def figure(aid,title,fig,df,place,reviewers,mids,eids,reason,caption,algorithm):
    stem=f'{aid}_{title.replace(" ","_")}_v0.1';folder=OUT/FOLDERS[place]
    paths={ext:folder/f'{stem}.{ext}' for ext in ['csv','pdf','png','svg']}
    df.to_csv(paths['csv'],index=False,encoding='utf-8-sig')
    for ext in ['pdf','png','svg']:fig.savefig(paths[ext],bbox_inches='tight',dpi=300)
    plt.close(fig);register(aid,'figure',title,place,stem,paths,reviewers,mids,eids,reason,caption,algorithm)
def panels():return plt.subplots(3,2,figsize=(6.9,7.8),layout='constrained')
def finish(ax):ax.grid(axis='y',color='#e5e5e5',linewidth=.5);ax.set_axisbelow(True)

# Table 1 remains exactly six columns / six primary strata.
rows=[];table1provenance=[]
for st in ORDER:
    a=stable[stable.stratum==st].iloc[0];q=summary[summary.stratum==st].iloc[0];b=budget[(budget.stratum==st)&(budget.budget_splits==5)&(budget.budget_seeds==3)].iloc[0]
    agree=a.full_grid_in_sample_agreement;lo=a.full_grid_agreement_CI_low;hi=a.full_grid_agreement_CI_high
    rows.append({'Dataset / rule':layer(st,True),'Reference top model':ML[a.reference_top_model],'Single-context agreement (95% CI)':f'{agree:.3f} {ci(lo,hi)}','Discordance (95% CI)':f'{1-agree:.3f} {ci(1-hi,1-lo)}','Paired bootstrap frequency q':f'{q.primary_frequency:.4f}','5×3 budget agreement':f'{b.aggregate_selection_agreement:.3f}'})
    table1provenance.append(dict(stratum=st,agreement=agree,agreement_low=lo,agreement_high=hi,discordance=1-agree,discordance_low=1-hi,discordance_high=1-lo,q=q.primary_frequency,budget_5x3=b.aggregate_selection_agreement,agreement_CI_source='T36 paired split-then-seed; full-grid reference re-estimated per draw',q_source='T37/T49 paired crossed; fixed reported model identity',budget_source='T40 mean-BA held-out split-ID block; 252 partitions'))
pd.DataFrame(table1provenance).to_csv(OUT/'04_Data_and_Provenance/Table1_Cell_Source_Map_v0.1.csv',index=False)
table('CT01','Model_selection_stability_Table1',pd.DataFrame(rows),'正文候选','E.1;R3.5;R3.4;R2.10;R1.1;R2.2','M23;M24;M30;M35;M40;M41','T36;T37;T40;T49;S69;S70','六列六主层；扩展并入原表；候选模型名称只定位参考，不作模型优胜结论。','The reference top model is defined by full-grid mean balanced accuracy (BA). Agreement/discordance intervals use 10,000 paired split-then-seed percentile resamples with the reference re-estimated per draw; discordance is 1 minus agreement. q is a point estimate from 10,000 paired crossed split–seed ranking resamples. The 5×3 column uses finite-grid agreement with a held-out five-split reference block, averaged over 252 oriented partitions; source images may overlap across splits. Original seed subsets come from five seeds; extension subsets from three.','agreement/discordance: existing nested T36; q: adopted crossed T37; budget: finite exact T40')

fig,axes=panels();fdata=[]
for ax,st in zip(axes.flat,ORDER):
    p=perf[perf.stratum==st].set_index('model').loc[MODELS];s=sens[(sens.stratum==st)&(sens.resampling=='crossed')].set_index('model').loc[MODELS]
    v=p.observed_selection_frequency.to_numpy();lo=s.observed_frequency_CI_low.to_numpy();hi=s.observed_frequency_CI_high.to_numpy();x=np.arange(4)
    ax.bar(x,v,color=COLORS[:4],edgecolor='#333333',linewidth=.5)
    # Draw absolute interval endpoints (never silently clamp a percentile interval).
    ax.vlines(x,lo,hi,color='black',linewidth=1);ax.hlines(lo,x-.06,x+.06,color='black');ax.hlines(hi,x-.06,x+.06,color='black')
    ax.set(ylim=(0,1.04),xticks=x,xticklabels=['R18','R50','D121','E-B0'],ylabel='Observed frequency f');ax.set_title(layer(st,True),fontsize=9);finish(ax)
    for m in MODELS:fdata.append(dict(stratum=st,model=m,f=p.loc[m,'observed_selection_frequency'],CI_low=s.loc[m,'observed_frequency_CI_low'],CI_high=s.loc[m,'observed_frequency_CI_high'],resampling='crossed',contexts=int(p.iloc[0].contexts)))
figure('CF01','Observed_selection_frequency_Figure1',fig,pd.DataFrame(fdata),'正文候选','E.1;R2.6;R2.10;R3.5','M30;M41','T31;T37','直接回应单次选择稳定性，保留Figure1统计对象。','Observed top-model selection frequencies across 50 original or 30 extension split–seed contexts per stratum. R18/R50/D121/E-B0 denote ResNet-18, ResNet-50, DenseNet-121 and EfficientNet-B0. Error bars are paired crossed split–seed percentile 95% intervals (10,000 replicates) for f, not for q. Zero observed frequencies may yield degenerate nonparametric intervals.','f point: T31; f data CI: crossed T37; no q CI')

fig,axes=panels()
for ax,st in zip(axes.flat,ORDER):
    x=ctx[ctx.stratum==st].top_two_BA_difference.to_numpy()*100
    ax.boxplot(x,positions=[1],widths=.32,showfliers=False,patch_artist=True,boxprops={'facecolor':'#cce6f2'},medianprops={'color':'#222222'})
    jitter=np.linspace(-.13,.13,len(x));ax.scatter(1+jitter,x,s=10,c=COLORS[0],alpha=.65,linewidths=0)
    ax.set(xticks=[1],xticklabels=[f'{len(x)} observed contexts'],ylabel='Top-two BA difference (pp)');ax.set_title(layer(st),fontsize=9);finish(ax)
figure('CF02','Context_top_two_BA_difference_Figure2',fig,primary(ctx),'正文候选','E.1;R1.1;R2.10','M28;M31;M41','T29;T36;T45','沿用前二性能差的原图用途，帮助解释排名变化；不检验非负差来宣称模型优胜。','Distribution of the difference between the two highest BA values within each observed context. BA differences are shown in percentage points (pp). Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all contexts. The leading pair is reselected within each context. Panel y-ranges differ to preserve readability. These distributions differ from the difference between full-grid model means.','Observed context order statistic; box/whiskers are not confidence intervals')

fig,axes=panels()
for ax,st in zip(axes.flat,ORDER):
    sub=budget[budget.stratum==st]
    for j,k in enumerate(sorted(sub.budget_seeds.unique())):
        a=sub[sub.budget_seeds==k].sort_values('budget_splits');ax.plot(a.budget_splits,a.aggregate_selection_agreement,marker=['o','s','^','D','v'][j],color=COLORS[j],label=f'{k} seed'+('s' if k>1 else ''),markersize=3,linewidth=1.1)
        if k==3:ax.fill_between(a.budget_splits,a.partition_q025,a.partition_q975,color=COLORS[j],alpha=.12)
    ax.set(ylim=(0,1.04),xticks=range(1,6),xlabel='Discovery split count',ylabel='Aggregate selection agreement');ax.set_title(layer(st),fontsize=9);ax.legend(ncol=2,fontsize=6.5,loc='lower right');finish(ax)
figure('CF03','Aggregate_evaluation_budget_Figure3',fig,primary(budget),'正文候选','E.1;R2.3;R3.4;R2.10','M27;M33;M41;M45','T40;T42;T43','用实际聚合预算回答RQ3；不把q改名为预算一致率。','Top-model agreement as a function of discovery splits and training seeds. Each point averages all eligible discovery subsets within 252 oriented five/five split-ID partitions. The reference is mean BA on the five held-out split IDs and all available seeds. Shading for the three-seed line is the 2.5th–97.5th percentile spread across the finite set of partitions, not a population confidence interval. Training-run budget is four times splits times seeds.','Finite grid exact aggregation; split IDs disjoint, source images can overlap; no population CI')

# Supplement candidates retain existing statistical evidence and source algorithms.
ba=[]
for _,r in perf.iterrows():
    s=sens[(sens.stratum==r.stratum)&(sens.model==r.model)&(sens.resampling=='crossed')].iloc[0]
    ba.append({'Stratum':layer(r.stratum,True),'Model':ML[r.model],'Mean BA':field(r.mean_BA,4),'Context SD':field(r.SD_context_BA,4),'95% crossed CI':ci(s.mean_BA_CI_low,s.mean_BA_CI_high,dec=4)})
table('ST01','Absolute_BA_uncertainty',pd.DataFrame(ba),'补充候选','R2.7;E.1','M29;M35;M38','T31;T37','直接回应绝对性能变异；同模型跨评估条件的统计描述，不用于最好架构排名。','Mean BA and context-level sample SD (ddof=1). Mean intervals use paired crossed split–seed resampling. The 16 strata include reused subsets and candidate/input sensitivities; rows must not be summed as independent training runs.','Mean/SD T31; crossed mean CI T37')
pr=ordered(primary(pair));tr=[]
for _,r in pr.iterrows():tr.append({'Stratum':layer(r.stratum),'Paired contrast':ML[r.model_1]+' − '+ML[r.model_2],'ΔBA (pp)':field(100*r.mean_paired_BA_difference_model1_minus_model2),'95% paired CI (pp)':ci(r.paired_CI_low,r.paired_CI_high,100),'Conditional p':field(r.sign_flip_p_raw_conditional,4),'Holm p (six pairs)':field(r.holm_p_within_stratum,4)})
table('ST02','Fixed_model_paired_BA_contrasts',pd.DataFrame(tr),'补充候选','E.1;R2.7','M25;M29;M31','T33','为编辑显著性要求提供明确配对问题的证据；不把Table1变成优胜模型榜。','Six alphabetically oriented fixed-model contrasts per primary stratum. Intervals retain the existing paired split-then-seed procedure. Exact sign-flip p-values use split-average contrasts under conditional joint sign symmetry; Holm adjustment is within each six-comparison stratum. These are separate from the context-wise top-two order statistic.','T33 existing nested paired CI; conditional exact sign-flip; within-stratum Holm')
polmain=pol[pol.role=='primary_original'];pt=[]
for _,r in polmain.iterrows():pt.append({'Dataset':DL[r.dataset],'Model':ML[r.model],'B − A ΔBA (pp)':field(100*r.mean_paired_BA_difference),'95% paired CI (pp)':ci(r.CI_low,r.CI_high,100),'Holm p (eight pairs)':field(r.holm_p_policy_family,4)})
table('ST03','Checkpoint_policy_BA_sensitivity',pd.DataFrame(pt),'补充候选','R2.4;E.1','M32;M44','T34','回应checkpoint复合因素及策略敏感性；不声称分离出纯checkpoint因果效应。','Original ten-split/five-seed paired B-minus-A effects. Paired nested intervals and conditional split-mean sign-flip tests retain the B1 algorithm; Holm family contains eight dataset-model comparisons. Rule A and rule B also differ in training duration/trajectory.','Existing T34, not recomputed or relabeled crossed')
pt=[]
for _,r in psel[psel.role=='primary_original'].iterrows():
    allrow=r.model=='ALL';val=r.context_winner_changed_fraction if allrow else r.frequency_difference_B_minus_A
    pt.append({'Dataset':DL[r.dataset],'Target':'Any winner change' if allrow else ML[r.model],'Statistic':'Changed-context fraction' if allrow else 'B − A observed f','Estimate':field(val),'95% paired CI':ci(r.CI_low,r.CI_high)})
table('ST04','Checkpoint_selected_identity_sensitivity',pd.DataFrame(pt),'补充候选','R2.4;R2.6','M32;M44','T35','区分性能改变与选择身份改变。','Paired A/B comparisons within identical split–seed contexts. ALL rows report a changed-winner fraction; model rows report B-minus-A observed selection-frequency differences. Both preserve the existing nested interval algorithm.','T35 paired nested, two explicitly identified targets')
pt=[]
for _,r in pool.iterrows():pt.append({'Dataset':DL[r.dataset],'Swin input':r.swin_variant,'Contexts':int(r.contexts),'Selection changed':field(r.selected_identity_changed_fraction),'95% paired CI':ci(r.CI_low,r.CI_high),'Mean top-two ΔBA: four / five (pp)':f'{100*r.observed_mean_top_two_difference_four:.3f} / {100*r.observed_mean_top_two_difference_five:.3f}'})
table('ST05','Candidate_pool_sensitivity',pd.DataFrame(pt),'补充候选','R1.3;R2.5','M36;M38','T28','回答候选池改变是否影响稳定性；同15个配对条件，不宣称Swin代表所有现代架构。','Four-CNN versus five-model candidate pools at the same five splits and three seeds. Intervals are existing paired nested sensitivity intervals. Selected-identity changes reflect candidate-pool dependence.','Existing candidate-pool sensitivity; no architecture-superiority inference')
pt=[]
for _,r in inp.iterrows():pt.append({'Dataset':DL[r.dataset],'Paired contexts':int(r.paired_contexts),'Input ΔBA (pp)':field(100*r.mean_BA_difference_weight_eval_minus_shared),'95% paired CI (pp)':ci(r.CI_low,r.CI_high,100),'Holm p (two pairs)':field(r.holm_p_two_input_comparisons,4)})
table('ST06','Swin_input_recipe_sensitivity',pd.DataFrame(pt),'补充候选','R3.3;R2.5;R1.3','M21;M37','T38','回应预处理一致性与输入方案差异；不是新模型优劣比较。','Swin weight-evaluation recipe minus shared recipe, on the same 15 contexts per dataset. Existing paired nested intervals and conditional five-split sign-flip tests; the minimum two-sided exact p is 0.0625, and Holm adjusts two dataset comparisons.','Existing T38 paired nested / exact conditional p')
pt=[]
for _,r in common.iterrows():pt.append({'Dataset / rule':f'{DL[r.dataset]} / {r.checkpoint_policy}','Reference: all5 / common3':ML[r.full_grid_reference_top_all5]+' / '+ML[r.full_grid_reference_top_common3],'LSO agreement: all5 / common3':f'{r.LSO_agreement_all5:.3f} / {r.LSO_agreement_common3:.3f}','5×3 agreement: all5 / common3 pool':f'{r.disjoint_5x3_agreement_all5_seed_pool:.3f} / {r.disjoint_5x3_agreement_common3_seed_pool:.3f}'})
table('ST07','Common_training_seed_sensitivity',pd.DataFrame(pt),'补充候选','R3.1;R3.3;R2.3','M12;M35;M39','T44','说明原五seed与扩展三seed配置差异的统计敏感性；复用原run，不作为额外独立实验。','All five original seeds compared with the shared subset 42/52/62. Common-seed results reuse original runs. Budget agreement uses the same five/five split partition definition with the indicated seed pool.','Existing T44 nested LSO point summaries / exact budget')
pt=[]
for st in ORDER:
    a=stable[stable.stratum==st].iloc[0];s=sens[(sens.stratum==st)&(sens.resampling=='crossed')].iloc[0]
    pt.append({'Stratum':layer(st),'Full-grid agreement (nested CI)':f'{a.full_grid_in_sample_agreement:.3f} {ci(a.full_grid_agreement_CI_low,a.full_grid_agreement_CI_high)}','Identity-correct LSO agreement (crossed CI)':f'{a.identity_correct_LSO_agreement:.3f} {ci(s.LSO_CI_low,s.LSO_CI_high)}'})
table('ST08','Reference_definition_sensitivity',pd.DataFrame(pt),'补充候选','R2.3;R3.4','M23;M27;M30','T36;T37','作为有限参考稳健性补充；不增加主表LSO列。','LSO excludes every occurrence of the held-out original split identity. Crossed LSO intervals use the existing identity-correct sampler. Full-grid agreement intervals retain the existing nested sampler with reference re-estimation; targets and algorithms are distinct.','Full-grid T36 nested; LSO point T36 and CI T37 crossed')
pt=[]
for _,r in primary(budget).iterrows():pt.append({'Stratum':layer(r.stratum),'Budget s×k':f'{int(r.budget_splits)}×{int(r.budget_seeds)}','Four-model runs':int(r.all_models_training_runs_per_budget),'Top agreement':field(r.aggregate_selection_agreement),'Partition 2.5–97.5% spread':ci(r.partition_q025,r.partition_q975),'Complete-rank agreement':field(r.complete_ranking_agreement),'Mean-rank ref / bootstrap ref':f'{r.mean_rank_reference_agreement:.3f} / {r.bootstrap_reference_expected_agreement:.3f}'})
table('ST09','Evaluation_budget_details',pd.DataFrame(pt),'补充候选','R2.3;R3.4;E.1','M27;M33;M45','T40;T43','保留完整预算而非挑阈值；完整CSV可作为附件，正文只择用Figure3。','All primary budget cells. Partition percentiles describe the 252 finite oriented partitions, not population CIs. Mean-rank reference is an alternative ranking rule; bootstrap-reference expected agreement uses the existing nested reference resampling.','Existing exact finite-grid budget; nested bootstrap reference remains labeled')
pt=[]
for st in ORDER[:4]:
    r=inventory[inventory.stratum==st].iloc[0];v=var[(var.dataset==r.dataset)&(var.checkpoint_policy==r.checkpoint_policy)].set_index('grp')
    pt.append({'Stratum':layer(st),'Split variance share':field(v.loc['split','variance_proportion']),'Model×split share':field(v.loc['model:split','variance_proportion']),'Residual share':field(v.loc['Residual','variance_proportion']),'Model':'Reported nonsingular fallback'})
table('ST10','Original_mixed_effects_variance_structure',pd.DataFrame(pt),'补充候选','R2.8;R2.1','M26;M34;M44','T13;T82','解释原RQ2因子结构，保留原四层fallback结果；不作为扩展混合模型新拟合。','Original four-stratum reported fallback: BA ~ model + (1|split) + (1|model:split). Variance shares describe these fitted models and are not causal attribution. This is a display of existing original outputs, not a new extension mixed-model fit.','Existing original mixed-effects outputs; no re-fit')
ms=mstatus[mstatus.model_spec.isin(['sap_full_formula_attempt','primary_reported_fallback_model'])]
pt=[{'Dataset / rule':f'{DL[r.dataset]} / {r.checkpoint_policy}','Specification':r.model_spec,'Converged':r.ok,'Singular':r.singular,'Splits / seeds':f'{int(r.splits)} / {int(r.seeds)}'} for _,r in ms.iterrows()]
table('ST11','Mixed_model_full_and_fallback_status',pd.DataFrame(pt),'补充候选','R2.8','M26;M34','T47','直接解释full模型singular与fallback；不扩展新的方差研究。','Existing original full and reported fallback fits. Full models include a seed random intercept and are singular; the reported split and model×split fallback fits are nonsingular. These are diagnostics of the existing fits.','Original stored status; no independent-person review claimed')
pt=[]
for _,r in epoch.iterrows():pt.append({'Dataset / rule':f'{DL[r.dataset]} / {r.checkpoint_policy}','Model':ML[r.model],'Runs':int(r.run_count),'Selected epoch mean / SD':f'{r.selected_epoch_mean:.2f} / {r.selected_epoch_sd:.2f}','Median [min, max]':f'{r.selected_epoch_median:.0f} [{r.selected_epoch_min:.0f}, {r.selected_epoch_max:.0f}]'})
table('ST12','Original_selected_epoch_distribution',pd.DataFrame(pt),'补充候选','R2.4;R2.9;R3.3','M20;M32;M47','T11','解释RuleA/B包含训练时长因素；原数据selected epoch不是全部训练完成epoch。','Original submitted runs only. Values describe the selected checkpoint epoch, not necessarily the total number of completed training epochs. Rule B selected epoch is fixed at 60.','Existing epoch records, not reconstructed missing metadata')
pt=[]
for _,r in times.iterrows():pt.append({'Dataset / rule':f'{DL[r.dataset]} / {r.checkpoint_policy}','Model / input':ML[r.model]+' / '+r.variant,'GPU':r.gpu_model.replace('NVIDIA GeForce RTX ','RTX '),'Runs':int(r.runs),'Seconds mean / SD':f'{r.per_run_seconds_mean:.1f} / {r.per_run_seconds_SD:.1f}','Median seconds':f'{r.per_run_seconds_median:.1f}'})
table('ST13','Recorded_per_run_wall_clock',pd.DataFrame(pt),'补充候选','R2.9','M18;M20;M47','T46','回应实际per-run时长；不同设备计时范围分别记录，不推断GPU优劣或新的预测时长。','Reported wall-clock within each recorded hardware and timing scope. Original: run-entry through export; extension: after model/data-loader initialization through export/hashing. Different scopes and parallelism prevent direct hardware-speed comparison. Exact source scope is recorded in T46.','Existing B2 operational record; no speed-ranking analysis')
pt=[]
for _,r in param.iterrows():pt.append({'Original dataset':DL[r.dataset],'Model':ML[r.model],'Class count':int(r.class_count),'Total parameters':f'{int(r.total_parameters_after_classifier_replacement):,}','Trainable parameters':f'{int(r.trainable_parameters):,}','Fine-tuning scope':r.trainable_scope})
table('ST14','Original_model_parameter_counts',pd.DataFrame(pt),'补充候选','R2.9;R3.3','M19;M20','T10','保留已记录模型参数作为复现补充；不将不同数据集classifier数量混为同一个参数值。','Original-dataset classifier configurations only. Parameter counts are architectural records; neither analysis-host package versions nor unrecorded extension parameter counts are inferred from them.','Existing original model record; training and analysis software not conflated')

# Standalone tables withdrawn from manuscript planning are generated only as reserve assets.
pt=[]
for _,r in summary.iterrows():pt.append({'Stratum':layer(r.stratum),'Adopted crossed q':field(r.crossed_frequency,4),'Nested q':field(r.nested_frequency,4),'Split-only q':field(r.split_only_frequency,4),'Original flat q':field(r.legacy_flat_frequency,4)})
table('RT01','Bootstrap_scheme_comparison',pd.DataFrame(pt),'回信或备存','R3.5','M24;M40;M42','T30;T31;T37;T49','本表只备存/回信使用；不恢复已撤销的稿件bootstrap补表。','Frequencies refer to the same reported full-grid reference model per stratum. Crossed is the adopted scheme; nested and split-only are sensitivities, flat is original replay. Original flat values are not available for the two new datasets.','Existing frequencies; q data CI not computed')
pt=[]
for _,r in summary.iterrows():pt.append({'Stratum':layer(r.stratum),'Observed f':field(r.observed_selection_frequency),'Crossed q':field(r.primary_frequency,4),'Full-grid top-two ΔBA (pp)':field(100*r.full_grid_top_two_BA_difference)})
table('RT02','Observed_and_aggregate_selection_evidence',pd.DataFrame(pt),'回信或备存','R3.5;R1.1;R2.6','M23;M31;M43','T36;T49','便于回信说明单次与聚合稳定性；指标合并会重复主表/原图，默认不另入正文。','f is observed single-context frequency; q is a resampled aggregate ranking frequency. Full-grid top-two mean difference is distinct from the within-context top-two difference in Figure2.','Existing point estimates, no extra inference')
a=assoc.iloc[0]
table('RT03','Existing_difference_instability_association',pd.DataFrame([{'Transformation':'One SD decrease in log10(context top-two BA difference + 1e-6)','Odds ratio':field(a.recomputed_odds_ratio),'95% existing interval':ci(a.replay_CI_low,a.replay_CI_high),'Analysis scope':'Four datasets, rule A, four CNNs; dataset-adjusted'}]),'回信或备存','R1.1','M28;M43','T27','已有关联结果可用于回应分差问题；区间包含1，不包装为普遍阈值或因果机制。','The reported dataset-adjusted logistic association uses standardized negative log10 of the context-wise difference plus 1e-6. The interval contains 1. This is an existing supplementary association, not a causal or threshold estimate.','Existing identity-correct D01 replay; not a new association fit')
pt=[{'Stratum':r.stratum,'Split×seed':f'{int(r.splits)}×{int(r.seeds)}','Models':int(r.models),'Contexts':int(r.contexts),'Runs in reused stratum':int(r.runs),'Role':r.role} for _,r in inventory.iterrows()]
table('RT04','Analysis_stratum_inventory',pd.DataFrame(pt),'回信或备存','R3.1;R3.3','M11;M12;M13;M42','T25','16分析层含复用子集，供回信核对设计；不把各层run相加。','Original completed primary matrix: 800 runs. Extension matrix: 300 runs. The 16 analysis strata overlap; their context/run counts are not additive independent sample sizes.','Existing B1 inventory and reuse labels')

# Additional explanatory figures are selectable, never automatically added to main text.
fig,axes=panels();badata=[]
for ax,st in zip(axes.flat,ORDER):
    p=perf[perf.stratum==st].set_index('model').loc[MODELS];s=sens[(sens.stratum==st)&(sens.resampling=='crossed')].set_index('model').loc[MODELS]
    y=np.arange(4);ax.hlines(y,s.mean_BA_CI_low,s.mean_BA_CI_high,color=COLORS[0]);ax.scatter(p.mean_BA,y,color=COLORS[0],s=20)
    ax.set(yticks=y,yticklabels=[ML[m] for m in MODELS],xlabel='Mean balanced accuracy');ax.set_title(layer(st),fontsize=9);ax.invert_yaxis();ax.grid(axis='x',color='#eeeeee')
    for m in MODELS:badata.append(dict(stratum=st,model=m,mean_BA=p.loc[m,'mean_BA'],CI_low=s.loc[m,'mean_BA_CI_low'],CI_high=s.loc[m,'mean_BA_CI_high']))
figure('SF01','Absolute_BA_intervals',fig,pd.DataFrame(badata),'补充候选','R2.7;E.1','M29','T31;T37','性能区间可回应R2.7，但与ST01重复；选图或表即可，不两者均入稿。','Model mean BA and paired crossed percentile 95% intervals (10,000 resamples). Panel x-ranges differ. Alphabetical/model-family display order does not identify a universal best architecture.','Existing crossed mean intervals T37')
fig,axes=panels()
for ax,st in zip(axes.flat,ORDER):
    p=pr[pr.stratum==st];y=np.arange(len(p));ax.hlines(y,p.paired_CI_low*100,p.paired_CI_high*100,color=COLORS[0]);ax.scatter(p.mean_paired_BA_difference_model1_minus_model2*100,y,color=COLORS[0],s=18);ax.axvline(0,color='#777777',ls=':',lw=.8)
    labs=[f'{ML[r.model_1]} − {ML[r.model_2]}' for _,r in p.iterrows()];ax.set(yticks=y,yticklabels=labs,xlabel='Paired BA difference (pp)');ax.set_title(layer(st),fontsize=9);ax.tick_params(axis='y',labelsize=6.5);ax.invert_yaxis()
figure('SF02','Paired_BA_contrast_intervals',fig,pr,'补充候选','E.1','M25;M29','T33','用于显示编辑所问统计比较；ST02有完整p值，优先选表而非把此图当模型榜。','Six fixed-model paired BA differences per primary stratum. Intervals retain the existing nested algorithm. The zero line indicates equal BA for the specified fixed contrast; significance and multiplicity are reported separately in ST02.','Existing T33 paired intervals; no new tests')
fig,ax=plt.subplots(figsize=(6.9,4),layout='constrained');y=np.arange(len(polmain));ax.hlines(y,polmain.CI_low*100,polmain.CI_high*100,color=COLORS[0]);ax.scatter(polmain.mean_paired_BA_difference*100,y,color=COLORS[0]);ax.axvline(0,color='gray',ls=':');ax.set(yticks=y,yticklabels=[f'{DL[r.dataset]} / {ML[r.model]}' for _,r in polmain.iterrows()],xlabel='B − A paired BA difference (pp)');ax.invert_yaxis()
figure('SF03','Checkpoint_policy_effect_intervals',fig,polmain,'补充候选','R2.4;E.1','M32;M44','T34','与ST03二选一；显示复合策略差异，不作因果分解。','Original paired B-minus-A BA effects with existing nested 95% intervals. Checkpoint rules also differ in training duration/optimization trajectory. Holm-adjusted tests are in ST03.','Existing T34 nested paired CI')
fig,axes=plt.subplots(1,2,figsize=(6.9,3.8),layout='constrained');ax=axes[0];y=np.arange(4);ax.hlines(y,pool.CI_low,pool.CI_high,color=COLORS[0]);ax.scatter(pool.selected_identity_changed_fraction,y,color=COLORS[0]);ax.set(yticks=y,yticklabels=[f'{DL[r.dataset]} / {r.swin_variant}' for _,r in pool.iterrows()],xlabel='Changed-selection fraction',xlim=(0,1));ax.invert_yaxis();ax.set_title('Candidate-pool sensitivity',fontsize=9)
ax=axes[1];y=np.arange(2);ax.hlines(y,inp.CI_low*100,inp.CI_high*100,color=COLORS[1]);ax.scatter(inp.mean_BA_difference_weight_eval_minus_shared*100,y,color=COLORS[1]);ax.set(yticks=y,yticklabels=[DL[x] for x in inp.dataset],xlabel='Input-recipe ΔBA (pp)');ax.axvline(0,color='gray',ls=':');ax.invert_yaxis();ax.set_title('Same-Swin input sensitivity',fontsize=9)
combined=pd.concat([pool.assign(display_object='candidate pool'),inp.assign(display_object='Swin input')],ignore_index=True)
figure('SF04','Candidate_pool_and_input_sensitivity',fig,combined,'补充候选','R1.3;R2.5;R3.3','M36;M37','T28;T38','不同对象分面展示；新架构只支持候选池/输入敏感性，不代表架构优劣。','Left: selected-identity changes when adding Swin to the four-CNN pool, at the same 15 contexts. Right: same-Swin weight-evaluation minus shared input-recipe BA difference. Intervals retain existing paired nested methods; the two panels use different targets and units.','Existing paired sensitivities, not crossed relabeling')
fig,axes=plt.subplots(1,2,figsize=(6.9,3.6),layout='constrained');y=np.arange(4);labels=[f'{DL[r.dataset]} / {r.checkpoint_policy}' for _,r in common.iterrows()]
for ax,one,two,title in [(axes[0],'LSO_agreement_all5','LSO_agreement_common3','LSO agreement'),(axes[1],'disjoint_5x3_agreement_all5_seed_pool','disjoint_5x3_agreement_common3_seed_pool','5×3 budget agreement')]:
    ax.hlines(y,common[one],common[two],color='#bbbbbb');ax.scatter(common[one],y,label='All five seeds',c=COLORS[0],marker='o');ax.scatter(common[two],y,label='Shared three seeds',c=COLORS[1],marker='s');ax.set(yticks=y,yticklabels=labels,xlim=(0,1),xlabel=title);ax.invert_yaxis();ax.legend(fontsize=7,loc='lower right')
figure('SF05','Common_seed_pool_sensitivity',fig,common,'补充候选','R3.1;R3.3;R2.3','M39','T44','与ST07二选一；说明相同run子集的敏感性，非独立复现实验。','Original all-five seed pool versus shared seeds 42/52/62. Lines link estimates from reused original runs. Left: identity-correct LSO agreement; right: five-split/three-seed agreement with a held-out split block.','Existing descriptive estimates; no new intervals')
fig,ax=plt.subplots(figsize=(6.9,3.4),layout='constrained');bottom=np.zeros(4)
for g,c,hatch in zip(['split','model:split','Residual'],COLORS[:3],['','//','xx']):
    vals=[]
    for st in ORDER[:4]:
        r=inventory[inventory.stratum==st].iloc[0];vals.append(var[(var.dataset==r.dataset)&(var.checkpoint_policy==r.checkpoint_policy)&(var.grp==g)].variance_proportion.iloc[0])
    ax.bar(range(4),vals,bottom=bottom,color=c,label=g,hatch=hatch,edgecolor='white',linewidth=.4);bottom+=vals
ax.set(xticks=range(4),xticklabels=[layer(st) for st in ORDER[:4]],ylim=(0,1),ylabel='Fitted variance proportion');ax.tick_params(axis='x',labelsize=7);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.15));finish(ax)
figure('SF06','Original_fallback_variance_structure',fig,var,'补充候选','R2.8;R2.1','M26;M34;M44','T13;T82','与ST10/11配合解释fallback，优先表；原RQ2已有因子结构展示，不新增扩展拟合。','Variance proportions from the original reported nonsingular fallback mixed model in four original strata. Split, model×split and residual components sum to one within fit. The full model with a seed random intercept was singular. These proportions describe the fitted variance structure and do not establish causal attribution.','Original stored mixed-model output, no refit')
fig,axes=panels()
for ax,st in zip(axes.flat,ORDER):
    r=budget[(budget.stratum==st)&(budget.budget_seeds==3)].sort_values('budget_splits')
    for c,m,col,label in zip(COLORS[:3],['o','s','^'],['aggregate_selection_agreement','mean_rank_reference_agreement','bootstrap_reference_expected_agreement'],['Mean BA reference','Mean rank reference','Bootstrap expected']):ax.plot(r.budget_splits,r[col],color=c,marker=m,markersize=3,label=label)
    ax.set(xticks=range(1,6),ylim=(0,1.04),xlabel='Discovery splits (3 seeds)',ylabel='Reference-specific agreement');ax.set_title(layer(st),fontsize=9);ax.legend(fontsize=6.5,loc='lower right');finish(ax)
figure('SF07','Budget_reference_sensitivity',fig,primary(budget[budget.budget_seeds==3]),'补充候选','R2.3;E.1','M27;M33','T40','回应mean-rank/bootstrap参考敏感性；不同参考定义分别标明，不替换主Figure3对象。','Three-seed budget sensitivity to reference construction. Mean-BA and mean-rank references select a top model in the held-out block; bootstrap expected agreement averages against the held-out model-selection distribution from the existing nested reference bootstrap (2,000 resamples). These are different reference-specific quantities.','Finite exact discovery enumeration; nested bootstrap reference expectation')
fig,ax=plt.subplots(figsize=(6.9,3.6),layout='constrained');y=np.arange(6)
for c,m,col,label in zip(COLORS[:3],['o','s','^'],['crossed_frequency','nested_frequency','split_only_frequency'],['Adopted crossed','Nested sensitivity','Split-only sensitivity']):ax.scatter(summary[col],y,c=c,marker=m,label=label,s=28)
ax.set(yticks=y,yticklabels=[layer(s) for s in ORDER],xlim=(0,1.04),xlabel='Bootstrap selection frequency q');ax.invert_yaxis();ax.legend(fontsize=7,loc='lower left');ax.grid(axis='x',color='#eeeeee')
figure('RF01','Bootstrap_design_comparison',fig,summary,'回信或备存','R3.5','M24;M40','T37;T49','用于回信解释频率为何变化；不恢复新增正文或bootstrap补表。','Reference-model q across three resampling schemes at the same evaluation results. Scheme choice follows the frozen crossed-factor design, not the largest numerical frequency. No data confidence interval for q is depicted.','Existing point frequencies; no new uncertainty analysis')
fig,ax=plt.subplots(figsize=(6.9,3.6),layout='constrained');y=np.arange(6);ax.hlines(y,summary.observed_selection_frequency,summary.primary_frequency,color='#cccccc');ax.scatter(summary.observed_selection_frequency,y,c=COLORS[0],marker='o',label='Observed single-context f');ax.scatter(summary.primary_frequency,y,c=COLORS[1],marker='s',label='Crossed aggregate q');ax.set(yticks=y,yticklabels=[layer(s) for s in ORDER],xlim=(0,1.04),xlabel='Selection frequency (distinct statistical objects)');ax.invert_yaxis();ax.legend(fontsize=7,loc='lower left');ax.grid(axis='x',color='#eeeeee')
figure('RF02','Single_context_and_aggregate_frequencies',fig,summary,'回信或备存','R3.5;R2.6','M23;M43','T49','帮助审稿人理解两个分母与聚合步骤；不新增核心指标，不重复主表。','Observed frequency f and crossed-bootstrap frequency q for the full-grid reference model. f uses 50 or 30 observed contexts; q uses 10,000 resampled aggregated rankings. Connecting lines aid comparison of distinct quantities and are not paired-effect estimates.','Existing descriptive f/q; no statistical test between them')
fig,ax=plt.subplots(figsize=(6.9,2.1),layout='constrained');a=assoc.iloc[0];ax.hlines([0],a.replay_CI_low,a.replay_CI_high,color=COLORS[0],lw=2);ax.scatter([a.recomputed_odds_ratio],[0],c=COLORS[0],s=40);ax.axvline(1,color='gray',ls=':');ax.set(xscale='log',yticks=[0],yticklabels=['Existing dataset-adjusted association'],xlabel='Odds ratio per 1 SD increase in −log10(top-two BA difference + 1e−6)');ax.set_ylim(-.5,.5)
figure('RF03','Existing_difference_instability_association',fig,assoc,'回信或备存','R1.1','M28;M43','T27','为margin意见提供已有模型结果；区间含1，保留备存，不扩展成新机制结论。','Existing dataset-adjusted logistic association between identity-correct LSO discordance and standardized negative log10 of the context-wise top-two BA difference plus 1e-6. OR=2.022 with interval [0.597,5.763]. The interval includes 1; this is associational evidence, not a threshold or causal estimate.','Existing D01 replay and data interval, not newly fitted')

pd.DataFrame(ASSETS).to_csv(OUT/'Candidate_Asset_Register_v0.1.csv',index=False,encoding='utf-8-sig')
(OUT/'Candidate_Captions_and_Response_Uses_v0.1.md').write_text('# Candidate captions and reviewer uses v0.1\n\n日期2026-10-03。所有材料为候选，未修改原稿；正文候选不等于最终採用。\n\n'+'\n'.join(CAPTIONS))
files=[p for f in FOLDERS.values() for p in (OUT/f).iterdir() if p.is_file()]
manifest=dict(version='v0.1',date='2026-10-03',scope='Existing-results rendering / descriptive extraction; no new training, tests, bootstrap or model fits',python=platform.python_version(),matplotlib=mpl.__version__,numpy=np.__version__,source_hashes=SOURCES,output_hashes={rel(p):sha(p) for p in files},assets=len(ASSETS),tables=sum(a['kind']=='table' for a in ASSETS),figures=sum(a['kind']=='figure' for a in ASSETS))
manifest['source_files_unchanged']=all(sha(ROOT/p)==h for p,h in SOURCES.items())
(OUT/'04_Data_and_Provenance/Asset_Source_and_Output_Manifest_v0.1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# Review thumbnails are contact sheets, not publication figures.
gallery=[]
for start in range(0,len(ASSETS),8):
    fig,axes=plt.subplots(4,2,figsize=(14,18),layout='constrained')
    for ax,a in zip(axes.flat,ASSETS[start:start+8]):
        ax.imshow(plt.imread(ROOT/a['preview_path']));ax.set_title(a['asset_id']+'  '+a['title'].replace('_',' '),fontsize=10);ax.axis('off')
    for ax in axes.flat[len(ASSETS[start:start+8]):]:ax.axis('off')
    path=OUT/f'Preview_Sheet_{start//8+1:02d}_v0.1.png';fig.savefig(path,dpi=100);plt.close(fig);gallery.append(str(path.relative_to(OUT)))
(OUT/'README_v0.1.md').write_text(f'''# JIIM候选图表与回信素材 v0.1

日期2026-10-03；{manifest['tables']}张候选/备存表、{manifest['figures']}张候选/备存图。生成不等于入稿。

## 使用顺序

1. [材料登记](Candidate_Asset_Register_v0.1.csv)：逐项审稿意见、M编号、推荐位置及选择理由。
2. [完整图表注与回信用途](Candidate_Captions_and_Response_Uses_v0.1.md)。
3. `01_Main_Candidates/`：CT01 Table1及CF01–03 Figures1–3，优先正文；Table1六列六层。
4. `02_Supplement_Candidates/`：直接回应外审的性能、检验、设计敏感性和复现材料；同一信息图/表二选一，未确定全收。
5. `03_Response_Reserve/`：重采样、f/q、已有关联及设计库存，优先回信/备存；不恢复撤销的正文新增表。
6. [来源与输出哈希](04_Data_and_Provenance/Asset_Source_and_Output_Manifest_v0.1.json)、[Table1逐格来源](04_Data_and_Provenance/Table1_Cell_Source_Map_v0.1.csv)。

图提供PDF/SVG/300dpi PNG及数据CSV；表提供CSV/Markdown/PDF与首页面PNG，长表PDF分页。所用来源算法原样标识，区间与描述范围不混用。源数据未修改；本轮未增加训练、bootstrap或推断检验。所有英文图注均可单独取用，正式图表编号后续排版确定。

## 总览预览

'''+ '\n'.join(f'- [预览{j+1}]({p})' for j,p in enumerate(gallery))+'\n')
print(json.dumps({k:manifest[k] for k in ['assets','tables','figures','source_files_unchanged']},ensure_ascii=False))
