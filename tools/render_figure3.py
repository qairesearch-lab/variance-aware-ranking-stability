from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/rendered';OUT.mkdir(parents=True,exist_ok=True)
b=pd.read_csv(ROOT/'figures/data/Figure3.csv',float_precision='round_trip');b=b[b.role.str.startswith('primary')]
figures=[]
def sha(p):return ''
def rel(p):return str(p.relative_to(ROOT))
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#666666'];markers=['o','s','^','D','v'];labels=['1 seed','2 seeds','3 seeds','4 seeds','5 seeds'];panels=[('organamnist','A','OrganAMNIST / A'),('organamnist','B','OrganAMNIST / B'),('sipakmed','A','SIPaKMeD / A'),('sipakmed','B','SIPaKMeD / B'),('isic2019','A','ISIC 2019 / A (1–3 seeds)'),('mura','A','MURA / A (1–3 seeds)')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'black','text.color':'black','axes.edgecolor':'black','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(3,2,figsize=(9.6,10.2));fig.subplots_adjust(left=.10,right=.98,bottom=.07,top=.905,hspace=.43,wspace=.28);curves=[]
for ax,(ds,policy,title) in zip(axes.ravel(),panels):
    q=b[(b.dataset==ds)&(b.checkpoint_policy==policy)]
    shade=q[q.budget_seeds==3].sort_values('budget_splits');ax.fill_between(shade.budget_splits,shade.partition_q025,shade.partition_q975,color=colors[2],alpha=.12,zorder=1)
    for seed in sorted(q.budget_seeds.unique()):
        rows=q[q.budget_seeds==seed].sort_values('budget_splits');x=rows.budget_splits.to_numpy();y=rows.aggregate_selection_agreement.to_numpy();line=ax.plot(x,y,color=colors[seed-1],marker=markers[seed-1],lw=1.8,ms=4.4,zorder=3)[0]
        assert np.array_equal(line.get_xdata(),x) and np.array_equal(line.get_ydata(),y)
        curves.append(dict(dataset=ds,policy=policy,seeds=int(seed),points=len(x),numeric_values_unchanged=True))
    ax.set_title(title,fontsize=11.5,pad=7);ax.set_xlabel('Discovery split count',fontsize=10);ax.set_ylabel('Aggregate selection agreement',fontsize=10);ax.set_xlim(.8,5.2);ax.set_ylim(0,1.04);ax.set_xticks(range(1,6));ax.set_yticks(np.linspace(0,1,6));ax.grid(axis='y',color='#D9D9D9',lw=.65);ax.set_axisbelow(True)
legend=fig.legend([Line2D([],[],color=c,marker=m,lw=1.8,ms=4.4) for c,m in zip(colors,markers)],labels,loc='upper center',bbox_to_anchor=(.54,.978),ncol=5,frameon=False,fontsize=10,columnspacing=1.6)
fig.canvas.draw();renderer=fig.canvas.get_renderer();lb=legend.get_window_extent(renderer);assert all(not lb.overlaps(ax.get_window_extent(renderer)) for ax in axes.ravel())
figures=[]
for ext in ['png','svg','pdf']:
    out=OUT/('Figure3.'+ext);fig.savefig(out,dpi=300,facecolor='white');figures.append(dict(output=rel(out),sha256=sha(out)))
plt.close(fig)
