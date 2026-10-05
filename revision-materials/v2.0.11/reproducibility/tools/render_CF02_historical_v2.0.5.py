from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/Users/qijiansheng/PycharmProjects/CevicalCancerV2');A=ROOT/'research-lab/reports/JIIM_Revision_Workspace/07_Candidate_Figures_Tables';OUT=A/'01_Main_Candidates';V='v2.0.5';stem='CF02_Expanded_Figure2_'+V
srca=OUT/'CF02_Context_top_two_BA_difference_Figure2_v0.1.csv';srcb=OUT/'CF04_Selected_identity_sensitivity_Figure4_v2.0.4.csv'
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
for panel,ax,color,title in [('A',fig.add_subplot(gs[2,:3]),'#0072B2','Checkpoint selection rule: A versus B'),('B',fig.add_subplot(gs[2,3:]),'#D55E00','Candidate pool: four CNNs versus four CNNs + Swin-T')]:
 rows=[r for r in rb if r['panel']==panel];y=np.array([1,0]);ax.set_xlim(0,103);ax.set_ylim(-.4,1.4)
 for yy,r in zip(y,rows):
  estimate=float(r['changed_fraction'])*100;lo=float(r['CI_low'])*100;hi=float(r['CI_high'])*100
  ax.errorbar(estimate,yy,xerr=[[estimate-lo],[hi-estimate]],fmt='o',color=color,markersize=6,capsize=4,elinewidth=1.4)
  ax.text(estimate,yy+.16,f'{estimate:.0f}%',ha='center',weight='bold',color=color,fontsize=9)
 ax.set_yticks(y,[f'{r["dataset"]}\n(n={r["contexts"]})' for r in rows],fontsize=8.5);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Contexts with a different selected model (%)',fontsize=9);ax.set_title(title,fontsize=9,pad=11)
 ax.xaxis.grid(True,color='#dfe3e6',linewidth=.7);ax.set_axisbelow(True)
 for edge in ['top','right']:ax.spines[edge].set_visible(False)
fig.text(.09,.025,'Panel a: boxes and points describe observed distributions; y-ranges differ.\nPanel b: points are changed-context fractions; bars retain existing paired 95% intervals.',fontsize=8.5)
for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=300,facecolor='white')
plt.close(fig)
# Exact source CSV copies retain all original values, column meanings and source versions.
pa=OUT/(stem+'_panel_a.csv');pb=OUT/(stem+'_panel_b.csv');pa.write_bytes(srca.read_bytes());pb.write_bytes(srcb.read_bytes())
caption='''# Figure2扩展候选：2a与2b v2.0.5

作者确认Figure2合并展示；正文仍1表3图，不另列Figure4。原前二BA差为2a，新增规则/候选池敏感性为2b。当前是候选排图，正文v2.0未整合。

**English caption:** Performance separation and sensitivity of model selection to evaluation conditions. (a) Distribution of the difference between the two highest balanced-accuracy (BA) values within each observed split–seed context, in percentage points (pp). The leading pair is reselected within each context. Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all observed contexts. Subplot y-ranges differ. (b) Fractions of matched split–seed contexts selecting a different model under alternative evaluation conditions. The checkpoint selection rule comparison uses 10 splits and five seeds per original dataset. The candidate-pool comparison adds shared-input Swin-T to four CNNs using five splits and three seeds per extension dataset. Horizontal bars retain existing paired nested percentile 95% intervals. The two condition comparisons are interpreted separately; their fractions describe changes in selected model identity, not changes in discordance rates.

来源：2a逐行复制原CF02数据/T52；2b逐行复制原CF04数据/T83（T57/T58重排）。不增加实验、重采样或统计检验。所有字段、原始数值和区间保持来源版本。2a的数据位置分布与2b的条件对照比例分别定义；固定模型显著性分析在ST02，A/B BA差在ST03。
'''
(OUT/(stem+'.md')).write_text(caption)
manifest={'version':V,'operation':'existing statistics render only','sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [srca,srcb]},'outputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pa,pb,*[OUT/(stem+'.'+x) for x in ['png','pdf','svg','md']]]}}
(A/'04_Data_and_Provenance'/('CF02_Expanded_Figure2_Manifest_'+V+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'preview':str(OUT/(stem+'.png')),'panel_a_rows':len(ra),'panel_b_rows':len(rb)}))
