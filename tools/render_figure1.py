from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures/rendered';OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'figures/data/Figure1.csv');order=['organamnist_A_cnn4_all5','organamnist_B_cnn4_all5','sipakmed_A_cnn4_all5','sipakmed_B_cnn4_all5','isic2019_A_cnn4_all3','mura_A_cnn4_all3'];models=['resnet18','resnet50','densenet121','efficientnet_b0']
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans'})
fig,axes=plt.subplots(3,2,figsize=(7,8),layout='constrained')
for ax,st in zip(axes.ravel(),order):
 q=d[d.stratum==st].set_index('model').loc[models];y=q.observed_selection_frequency;lo=q.selection_frequency_CI_low;hi=q.selection_frequency_CI_high
 ax.bar(range(4),y,color=['#0072B2','#D55E00','#009E73','#CC79A7']);ax.errorbar(range(4),y,yerr=[y-lo,hi-y],fmt='none',c='black',capsize=3);ax.set_xticks(range(4),['R18','R50','D121','E-B0']);ax.set_ylim(0,1.05);ax.set_title(st.split('_cnn')[0]);ax.set_ylabel('Observed selection frequency')
for ext in ['png','svg','pdf']:fig.savefig(OUT/('Figure1.'+ext),dpi=300)
plt.close(fig)
