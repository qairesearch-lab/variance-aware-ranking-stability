from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'figures';OUT=A/'rendered';OUT.mkdir(parents=True,exist_ok=True);stem='Figure2'
srca=A/'data/Figure2a.csv';srcb=A/'data/Figure2b.csv'
ra=list(csv.DictReader(srca.open(encoding='utf-8-sig')));rb=list(csv.DictReader(srcb.open(encoding='utf-8-sig')))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(11.4,9.4));gs=fig.add_gridspec(3,6,height_ratios=[1,1,1.15],left=.09,right=.98,top=.93,bottom=.13,hspace=.78,wspace=.9)
order=['organamnist_A_cnn4_all5','organamnist_B_cnn4_all5','sipakmed_A_cnn4_all5','sipakmed_B_cnn4_all5','isic2019_A_cnn4_all3','mura_A_cnn4_all3'];labels=['OrganAMNIST / A','OrganAMNIST / B','SIPaKMeD / A','SIPaKMeD / B','ISIC2019 / A','MURA / A']
fig.text(.035,.975,'a  Context-wise differences between the two highest BA values',fontsize=12,weight='bold')
for i,(st,label) in enumerate(zip(order,labels)):
 ax=fig.add_subplot(gs[i//3,(i%3)*2:(i%3)*2+2]);x=np.array([float(r['top_two_BA_difference']) for r in ra if r['stratum']==st])*100
 ax.boxplot(x,positions=[1],widths=.32,showfliers=False,patch_artist=True,boxprops={'facecolor':'#cce6f2'},medianprops={'color':'#222222'})
 ax.scatter(1+np.linspace(-.13,.13,len(x)),x,s=9,c='#0072B2',alpha=.65,linewidths=0)
 ax.set(xticks=[1],xticklabels=[f'{len(x)} observed contexts'],ylabel='Top-two BA difference (pp)');ax.set_title(label,fontsize=10)
 ax.grid(axis='y',color='#e5e5e5',linewidth=.5);ax.set_axisbelow(True)
 for edge in ['top','right']:ax.spines[edge].set_visible(False)
fig.text(.035,.363,'b  Changes in selected model identity under alternative evaluation conditions',fontsize=12,weight='bold')
for panel,ax,color,title in [('A/B',fig.add_subplot(gs[2,:3]),'#0072B2','A/B evaluation procedures'),('Candidate pool',fig.add_subplot(gs[2,3:]),'#D55E00','Candidate pool: four CNNs versus four CNNs + Swin-T')]:
 rows=[r for r in rb if r['panel']==panel];y=np.array([1,0]);ax.set_xlim(0,103);ax.set_ylim(-.4,1.4)
 for yy,r in zip(y,rows):
  estimate=float(r['estimate'])*100;lo=float(r['CI_low'])*100;hi=float(r['CI_high'])*100
  ax.errorbar(estimate,yy,xerr=[[estimate-lo],[hi-estimate]],fmt='o',color=color,markersize=6,capsize=4,elinewidth=1.4)
  ax.text(estimate,yy+.16,f'{estimate:.0f}%',ha='center',weight='bold',color=color,fontsize=9)
 ax.set_yticks(y,[f'{r["dataset"]}\n(n={r["contexts"]})' for r in rows],fontsize=8.5);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Contexts with a different selected model (%)',fontsize=9);ax.set_title(title,fontsize=9,pad=11)
 ax.xaxis.grid(True,color='#dfe3e6',linewidth=.7);ax.set_axisbelow(True)
 for edge in ['top','right']:ax.spines[edge].set_visible(False)
fig.text(.09,.025,'Panel a: boxes and points describe observed distributions; y-ranges differ.\nPanel b: points are changed-context fractions; bars show paired crossed 95% intervals.',fontsize=8.5)
for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=300,facecolor='white')
plt.close(fig)
