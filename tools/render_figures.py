"""Render Figures 1 and 3 from saved numerical results; no statistical recalculation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/rendered';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results'
perf=pd.read_csv(DATA/'D02_model_performance_selection.csv')
sens=pd.read_csv(DATA/'D02_resampling_design_sensitivity.csv')
budget=pd.read_csv(DATA/'D04_budget_summary.csv')
inventory=pd.read_csv(DATA/'B1_stratum_inventory.csv')
ORDER=['organamnist_A_cnn4_all5','organamnist_B_cnn4_all5','sipakmed_A_cnn4_all5','sipakmed_B_cnn4_all5','isic2019_A_cnn4_all3','mura_A_cnn4_all3']
MODELS=['resnet18','resnet50','densenet121','efficientnet_b0']
DL={'organamnist':'OrganAMNIST','sipakmed':'SIPaKMeD','isic2019':'ISIC2019','mura':'MURA'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def layer(st,full=False):
 r=inventory[inventory.stratum==st].iloc[0];s=f"{DL[r.dataset]} / {r.checkpoint_policy}"
 return s+f" ({int(r.splits)}×{int(r.seeds)}; {int(r.models)} models)" if full else s
def panels():return plt.subplots(3,2,figsize=(6.9,7.8),layout='constrained')
def finish(ax):ax.grid(axis='y',color='#e5e5e5',linewidth=.5);ax.set_axisbelow(True)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=300)
 plt.close(fig)

fig,axes=panels();fdata=[]
for ax,st in zip(axes.flat,ORDER):
    p=perf[perf.stratum==st].set_index('model').loc[MODELS];s=sens[(sens.stratum==st)&(sens.resampling=='crossed')].set_index('model').loc[MODELS]
    v=p.observed_selection_frequency.to_numpy();lo=s.observed_frequency_CI_low.to_numpy();hi=s.observed_frequency_CI_high.to_numpy();x=np.arange(4)
    ax.bar(x,v,color=COLORS[:4],edgecolor='#333333',linewidth=.5)
    # Draw absolute interval endpoints (never silently clamp a percentile interval).
    ax.vlines(x,lo,hi,color='black',linewidth=1);ax.hlines(lo,x-.06,x+.06,color='black');ax.hlines(hi,x-.06,x+.06,color='black')
    ax.set(ylim=(0,1.04),xticks=x,xticklabels=['R18','R50','D121','E-B0'],ylabel='Observed frequency f');ax.set_title(layer(st,True),fontsize=9);finish(ax)
    for m in MODELS:fdata.append(dict(stratum=st,model=m,f=p.loc[m,'observed_selection_frequency'],CI_low=s.loc[m,'observed_frequency_CI_low'],CI_high=s.loc[m,'observed_frequency_CI_high'],resampling='crossed',contexts=int(p.iloc[0].contexts)))
save(fig,'Figure1')
fig,axes=panels()
for ax,st in zip(axes.flat,ORDER):
    sub=budget[budget.stratum==st]
    for j,k in enumerate(sorted(sub.budget_seeds.unique())):
        a=sub[sub.budget_seeds==k].sort_values('budget_splits');ax.plot(a.budget_splits,a.aggregate_selection_agreement,marker=['o','s','^','D','v'][j],color=COLORS[j],label=f'{k} seed'+('s' if k>1 else ''),markersize=3,linewidth=1.1)
        if k==3:ax.fill_between(a.budget_splits,a.partition_q025,a.partition_q975,color=COLORS[j],alpha=.12)
    ax.set(ylim=(0,1.04),xticks=range(1,6),xlabel='Discovery split count',ylabel='Aggregate selection agreement');ax.set_title(layer(st),fontsize=9);ax.legend(ncol=2,fontsize=6.5,loc='lower right');finish(ax)
save(fig,'Figure3')
