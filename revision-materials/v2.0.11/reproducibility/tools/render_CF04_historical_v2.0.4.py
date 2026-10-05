from pathlib import Path
import csv,re,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'research-lab/reports/JIIM_Revision_Workspace/07_Candidate_Figures_Tables'
OUT=ASSET/'01_Main_Candidates';V='v2.0.4';stem='CF04_Selected_identity_sensitivity_Figure4_'+V
policy=ASSET/'02_Supplement_Candidates/ST04_Checkpoint_selected_identity_sensitivity_v0.1.csv'
pool=ASSET/'02_Supplement_Candidates/ST05_Candidate_pool_sensitivity_v0.1.csv'
rows=[]
for source,panel in [(policy,'A'),(pool,'B')]:
    for r in csv.DictReader(source.open(encoding='utf-8-sig')):
        if panel=='A' and r['Target']!='Any winner change':continue
        if panel=='B' and r['Swin input']!='shared':continue
        dataset=r['Dataset'];estimate=float(r['Estimate'] if panel=='A' else r['Selection changed']);ci=[float(v) for v in re.findall(r'[-+]?\d*\.?\d+',r['95% paired CI'])]
        rows.append(dict(panel=panel,dataset=dataset,condition='Rule A versus Rule B' if panel=='A' else 'Four CNNs versus four CNNs plus shared-input Swin-T',split_count=10 if panel=='A' else 5,training_seed_count=5 if panel=='A' else 3,contexts=50 if panel=='A' else int(r['Contexts']),changed_fraction=estimate,CI_low=ci[0],CI_high=ci[1],source_path=str(source.relative_to(ROOT)),source_evidence='T57' if panel=='A' else 'T58',interval_algorithm='existing paired nested percentile interval; unchanged'))
with (OUT/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),sharex=True)
for ax,panel,color,title,sub in zip(axes,['A','B'],['#0072B2','#D55E00'],['A  Checkpoint selection rule','B  Candidate model pool'],['Rule A versus Rule B\n10 splits × 5 training seeds','Four CNNs versus four CNNs + Swin-T\nShared input; 5 splits × 3 training seeds']):
    rr=[r for r in rows if r['panel']==panel]
    for y,r in zip([1,0],rr):
        e=r['changed_fraction']*100;low=r['CI_low']*100;high=r['CI_high']*100
        ax.errorbar(e,y,xerr=[[e-low],[high-e]],fmt='o',color=color,markersize=7,capsize=5,elinewidth=1.6)
        ax.text(e,y+0.19,f'{e:.0f}%',ha='center',color=color,weight='bold')
    ax.set_yticks([1,0],[r['dataset']+f' (n={r["contexts"]})' for r in rr]);ax.set_xlim(0,103);ax.set_ylim(-0.5,1.55)
    ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Contexts with a different selected model (%)',fontsize=9)
    ax.set_title(title,loc='left',fontsize=11,weight='bold',pad=25)
    ax.text(0,1.025,sub,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5)
    ax.xaxis.grid(True,color='#dfe3e6',linewidth=.7);ax.set_axisbelow(True)
    for edge in ['top','right']:ax.spines[edge].set_visible(False)
fig.subplots_adjust(left=.19,right=.98,bottom=.31,top=.73,wspace=.78)
fig.text(.19,.045,'Points: observed changed-context fractions. Bars: existing paired 95% intervals.\nEach panel uses matched split–seed contexts within its own comparison.',fontsize=8.5)
for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=300,facecolor='white')
plt.close(fig)
(OUT/(stem+'.md')).write_text('''# CF04：选择规则与候选池变化下的首位模型身份变化（候选）

Historical standalone rendering v2.0.4 from ST04/ST05; these existing results supply Figure 2b. No new inferential analysis.

**English caption:** Changes in selected model identity under alternative evaluation conditions. (A) Fraction of matched split–seed contexts selecting different models under checkpoint selection Rules A and B, using 10 splits and five training seeds per original dataset. (B) Fraction selecting a different model after adding shared-input Swin-T to the four-CNN candidate pool, using five splits and three training seeds per extension dataset. Points show observed changed-context fractions; horizontal bars retain the existing paired nested percentile 95% intervals. Comparisons are paired within each panel; the panels represent different evaluation changes and are interpreted separately.

**用途：** 原RQ2的选择条件敏感性，R2.4的选择规则解释，R1.3/R2.5的候选池异质性回应。原Figure2继续提供BA差分布背景；本图显示选择条件改变时的模型身份变化。完整BA效应、输入方案对照与规格放补充材料。

**来源：** Panel A=ST04/T57（OrganAMNIST 0.460 [0.280,0.640]；SIPaKMeD 0.560 [0.360,0.740]）；Panel B=ST05/T58共享输入（ISIC2019 0.400 [0.067,0.733]；MURA 0.800 [0.533,1.000]）。本图按已生成表的显示精度作图；正式排图可直接使用同源未舍入CSV，不改变既定算法。
''')
manifest=dict(version=V,operation='render existing statistics only',source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [policy,pool]},outputs={str((OUT/(stem+'.'+ext)).relative_to(ROOT)):hashlib.sha256((OUT/(stem+'.'+ext)).read_bytes()).hexdigest() for ext in ['csv','png','pdf','svg','md']})
(ASSET/'04_Data_and_Provenance'/('CF04_Source_and_Output_Manifest_'+V+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(asset='CF04',rows=len(rows),preview=str(OUT/(stem+'.png')))))
