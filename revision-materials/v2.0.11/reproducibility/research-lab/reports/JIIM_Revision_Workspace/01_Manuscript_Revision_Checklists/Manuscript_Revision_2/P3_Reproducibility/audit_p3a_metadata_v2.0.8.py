"""Prepare reporting metadata from existing runs, without loading tensor data.

Checkpoint pickle decoding is restricted to symbolic tensor shapes, storage
labels and OrderedDict. No torch import, weight download, model forward pass,
training, resampling, fitting or significance testing is performed.
"""
from pathlib import Path
import csv,json,hashlib,zipfile,pickle,io,math,statistics,collections
from datetime import datetime
ROOT=Path(__file__).resolve().parents[6]
WS=ROOT/'research-lab/reports/JIIM_Revision_Workspace'; OUT=Path(__file__).parent
EXT=ROOT/'research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision'
V='v2.0.8'; sources={}; errors=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def source(p):sources[rel(p)]=sha(p);return p
def readjson(p):return json.loads(source(p).read_text())
def readcsv(p):return list(csv.DictReader(source(p).open(encoding='utf-8-sig')))
def writecsv(name,rows):
 p=OUT/f'{name}_{V}.csv'
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 return p
def ck(ok,msg):
 if not ok:errors.append(msg)
class FloatStorage:pass
class LongStorage:pass
def tensor_shape(storage,offset,size,stride,requires_grad,hooks,*args):
 assert all(isinstance(n,int) and n>=0 for n in size)
 return dict(shape=tuple(size),storage=storage,stride=tuple(stride),requires_grad=bool(requires_grad))
class ShapeReader(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('collections','OrderedDict'):collections.OrderedDict,('torch','FloatStorage'):FloatStorage,('torch','LongStorage'):LongStorage,('torch._utils','_rebuild_tensor_v2'):tensor_shape}
  if (module,name) not in allowed:raise ValueError(f'Unexpected checkpoint global: {module}.{name}')
  return allowed[(module,name)]
 def persistent_load(self,pid):
  if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or pid[1] not in [FloatStorage,LongStorage]:raise ValueError('Unexpected storage metadata')
  return dict(kind=pid[1].__name__,key=pid[2],device=pid[3],elements=pid[4])
def checkpoint_meta(p):
 with zipfile.ZipFile(p) as z:
  name=next(n for n in z.namelist() if n.endswith('/data.pkl'));raw=z.read(name)
  if len(raw)>2_000_000:raise ValueError('Unexpected large checkpoint metadata')
  obj=ShapeReader(io.BytesIO(raw)).load()
 return obj,hashlib.sha256(raw).hexdigest()
prefixes={'resnet18':('layer4.','fc.'),'resnet50':('layer4.','fc.'),'densenet121':('features.denseblock4.','features.norm5.','classifier.'),'efficientnet_b0':('features.8.','classifier.'),'swin_t':('features.7.','norm.','head.')}
head={'resnet18':'fc.weight','resnet50':'fc.weight','densenet121':'classifier.weight','efficientnet_b0':'classifier.1.weight','swin_t':'head.weight'}
pilot=readjson(EXT/'pilot/results/preprocessing_dual_variant_smoke_20260926.json');pilotcounts={(r['dataset'],r['model'],r['variant']):r['trainable_parameters'] for r in pilot['cases']}
script_hash=sha(source(EXT/'train_extension_run.py'));prep_hash=sha(source(EXT/'preprocessing.py'));ck(pilot['preprocessing_script_sha256']==prep_hash,'pilot/formal preprocessing source differs')
epoch_saved=readcsv(EXT/'analysis/input_audit_v0.1/D07_epoch_timing_config_by_run_v0.1.csv'); saved={r['run_id']:r for r in epoch_saved}
epoch_summary=readcsv(EXT/'analysis/input_audit_v0.1/D07_epoch_summary_v0.1.csv')
time_summary=readcsv(WS/'05_B2_Evidence_Coverage/B2_D07_Per_Run_Time_Summary_v0.1.csv')
allruns=[];paramruns=[];templates={};shape_templates={};envcounts=collections.Counter();representative={}
for stage,manifest in [('original',ROOT/'research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv'),('extension',EXT/'run_manifests/jiim_extension_300_v0.6.csv')]:
 for r in readcsv(manifest):
  run=r['run_id'];p=ROOT/r['output_dir'];conf=readjson(p/'run_config.json');status=readjson(p/'run_status.json');history=readcsv(p/'history.csv');checkpoint=readjson(p/'checkpoint_manifest.json');metric=readjson(p/'metrics.json');old=saved[run]
  ck(status['status'] in ['completed','rerun_completed'],'incomplete '+run)
  ck([int(h['epoch']) for h in history]==list(range(1,len(history)+1)),'non-contiguous history '+run)
  selected=int(checkpoint['selected_epoch'] if stage=='original' else checkpoint['best_epoch'])
  trained=len(history); ck(trained==int(old['trained_epochs']) and selected==int(old['selected_epoch']),'saved epoch mismatch '+run)
  if stage=='original':
   last=[int(c['epoch']) for c in checkpoint['checkpoints'] if c['role']=='last_epoch'];ck(last==[trained],'last epoch mismatch '+run)
   policy=r['checkpoint_policy'];variant='submitted';env=conf['package_versions'];gpu='; '.join(conf['device_info']['gpu_models']);cuda=conf['device_info']['cuda_version'];elapsed=(datetime.fromisoformat(status['ended_at_utc'])-datetime.fromisoformat(status['started_at_utc'])).total_seconds()
   if policy=='B':ck(trained==selected==60,'original B not 60 '+run)
   frozen=conf['resolved_frozen_config']; common=frozen['training_common']; opt=common['optimization']; override=conf.get('validation_overrides',{}).get('num_workers_override'); workers=override if override is not None else common['reproducibility']['num_workers']; max_epochs=frozen['checkpoint_policy']['max_epochs']
  else:
   timing=readjson(p/'timing.json');ck(int(timing['epochs_executed'])==trained,'timing epochs mismatch '+run);ck(int(metric['selected_epoch'])==selected,'selected metric mismatch '+run)
   policy=checkpoint['rule'];variant=r['variant'];env=conf['environment'];gpu=env['gpu'];cuda=env['cuda'];elapsed=float(timing['elapsed_seconds']);ck(policy=='A','extension policy mismatch '+run)
   ck(conf['script_sha256']==script_hash and conf['preprocessing_sha256']==prep_hash,'execution source hash mismatch '+run)
   cp=p/'checkpoints/best_validation_loss.pt';obj,msha=checkpoint_meta(cp);ck(obj['run_id']==run and obj['variant']==variant and int(obj['epoch'])==selected,'checkpoint identity mismatch '+run)
   params={};buffers={}
   for name,t in obj['model_state_dict'].items():
    if name.endswith(('.running_mean','.running_var','.num_batches_tracked','.relative_position_index')):buffers[name]=t
    elif name.endswith(('.weight','.bias','.relative_position_bias_table')):params[name]=t
    else:raise ValueError('Unclassified tensor '+name)
   total=sum(math.prod(t['shape']) for t in params.values());trainable=sum(math.prod(t['shape']) for name,t in params.items() if name.startswith(prefixes[r['model']]))
   ck(trainable==pilotcounts[(r['dataset'],r['model'],variant)],'pilot trainable count differs '+run)
   classes=8 if r['dataset']=='isic2019' else 2;ck(params[head[r['model']]]['shape'][0]==classes,'classifier class count '+run)
   shapes={n:list(t['shape']) for n,t in params.items()};shapehash=hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest();key=(r['dataset'],r['model'])
   ck(key not in templates or templates[key]==(total,trainable,shapehash),'parameter shape/count varies within model '+run);templates[key]=(total,trainable,shapehash);shape_templates[key]=shapes
   rawrow=dict(run_id=run,dataset=r['dataset'],model=r['model'],variant=variant,total_parameters=total,trainable_parameters=trainable,class_count=classes,checkpoint_path=rel(cp),recorded_checkpoint_sha256=checkpoint['selected_checkpoint_sha256'],data_pickle_sha256=msha,parameter_shape_sha256=shapehash,source_freeze_modules=';'.join(prefixes[r['model']]),checkpoint_identity_verified=True,pilot_trainable_match=True)
   groupkey=(r['dataset'],r['model'],variant)
   if groupkey not in representative:
    actualsha=sha(cp);ck(actualsha==checkpoint['selected_checkpoint_sha256'],'representative checkpoint full hash '+run);sources[rel(cp)]=actualsha;representative[groupkey]=run
   paramruns.append(rawrow)
   opt=conf['hyperparameters'];workers=opt['num_workers'];max_epochs=opt['max_epochs']
  for field in ['learning_rate','weight_decay','batch_size']:ck(float(opt[field])==float(old[field]),'optimization field differs '+run+'/'+field)
  ck(str(opt['optimizer']).lower()==old['optimizer'].lower(),'optimizer differs '+run)
  ck(int(workers)==int(old['num_workers']) and int(max_epochs)==int(old['max_epochs']),'worker/max-epoch mismatch '+run)
  ck(abs(elapsed-float(old['elapsed_seconds']))<1e-8,'saved timing differs '+run)
  ck(1<=selected<=trained<=int(old['max_epochs']),'epoch ordering '+run)
  py=env['python'];torch=env['torch'];vision=env['torchvision'];ck((py,torch,vision,gpu)==(old['python'],old['torch'],old['torchvision'],old['gpu_model']),'saved runtime differs '+run)
  envcounts[(stage,r['dataset'],py,torch,vision,cuda,gpu)]+=1
  allruns.append(dict(run_id=run,stage=stage,dataset=r['dataset'],model=r['model'],checkpoint_rule=policy,variant=variant,trained_epochs=trained,selected_epoch=selected,max_epochs=max_epochs,batch_size=opt['batch_size'],num_workers=workers,optimizer=opt['optimizer'],learning_rate=opt['learning_rate'],weight_decay=opt['weight_decay'],elapsed_seconds=elapsed,timing_scope=old['timing_scope'],python=py,torch=torch,torchvision=vision,cuda=cuda,gpu=gpu,source_dir=rel(p),epoch_record_match=True,timing_record_match=True))
ck(len(allruns)==1100 and len(paramruns)==300,'formal counts');ck(len(envcounts)==4,'runtime variants within dataset')
# Verify every stored summary against the raw values; these are descriptive metadata.
grouped=collections.defaultdict(list)
for r in allruns:grouped[(r['stage'],r['dataset'],r['model'],r['checkpoint_rule'],r['variant'])].append(r)
for r in epoch_summary:
 rr=grouped[(r['stage'],r['dataset'],r['model'],r['policy'],r['variant'])];ck(len(rr)==int(r['runs']),'summary count')
 for f in ['trained_epochs','selected_epoch']:
  vals=[x[f] for x in rr]
  for suffix,v in [('mean',statistics.mean(vals)),('SD',statistics.stdev(vals)),('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]:ck(abs(v-float(r[f+'_'+suffix]))<1e-10,'epoch summary '+r['dataset']+'/'+r['model']+'/'+f+'/'+suffix)
for r in time_summary:
 rr=grouped[(r['stage'],r['dataset'],r['model'],r['checkpoint_policy'],r['variant'])]; vals=[x['elapsed_seconds'] for x in rr];ck(len(rr)==int(r['runs']),'timing summary N')
 for suffix,v in [('mean',statistics.mean(vals)),('SD',statistics.stdev(vals)),('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]:ck(abs(v-float(r['per_run_seconds_'+suffix]))<1e-8,'timing summary '+r['dataset']+'/'+r['model']+'/'+suffix)
original=readcsv(ROOT/'research-lab/experiments/registered-workflow/stats/tables/model_parameter_environment_summary.csv')
originalkeys=list(original[0]); model_rows=[]
for r in original:
 model_rows.append(dict(stage='original',dataset=r['dataset'],model=r['model'],classes=r['class_count'],total_parameters=r['total_parameters_after_classifier_replacement'],trainable_parameters=r['trainable_parameters'],source_kind='retained original parameter-count table',model_source='research-lab/experiments/scripts/train_one_run.py',freeze_modules=';'.join(prefixes[r['model']]),formal_runs=400//4,variants='original A/B',parameter_shape_sha256='',provenance_note='original saved count; not newly instantiated'))
for (ds,m),(total,trainable,h) in sorted(templates.items()):
 rr=[r for r in paramruns if r['dataset']==ds and r['model']==m]
 model_rows.append(dict(stage='extension',dataset=ds,model=m,classes=8 if ds=='isic2019' else 2,total_parameters=total,trainable_parameters=trainable,source_kind='formal checkpoint shape count + source freeze rules; matched historical pilot count',model_source=rel(EXT/'preprocessing.py'),freeze_modules=';'.join(prefixes[m]),formal_runs=len(rr),variants=';'.join(sorted(set(r['variant'] for r in rr))),parameter_shape_sha256=h,provenance_note='all formal checkpoints in this model/class group share parameter shapes; no model fitting'))
environment=[dict(stage=k[0],dataset=k[1],runs=n,python=k[2],torch=k[3],torchvision=k[4],cuda=k[5],gpu=k[6],source_kind='formal run runtime metadata') for k,n in sorted(envcounts.items())]
b1=readjson(EXT/'analysis/results_v0.3/B1_analysis_manifest_v0.1.json');runtime=dict(analysis_id=b1['analysis_id'],created_utc=b1['created_utc'],python=b1['python'],package_versions=b1['package_versions'],source_manifest=rel(EXT/'analysis/results_v0.3/B1_analysis_manifest_v0.1.json'),historical_training_environment=False)
paths=[writecsv('P3a_Model_Parameter_Counts',model_rows),writecsv('P3a_Parameter_Checkpoint_By_Run',paramruns),writecsv('P3a_Epoch_Summary',epoch_summary),writecsv('P3a_Recorded_Timing_Summary',time_summary),writecsv('P3a_Training_Environment',environment),writecsv('P3a_Run_Metadata_Check',allruns)]
rp=OUT/f'P3a_Revised_Analysis_Runtime_{V}.json';rp.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n');paths.append(rp)
for key,shapes in shape_templates.items():
 p=OUT/f'Parameter_Shapes_{key[0]}_{key[1]}_{V}.json';p.write_text(json.dumps(shapes,indent=2)+'\n');paths.append(p)
source(Path(__file__))
report=dict(version=V,errors=errors,error_count=len(errors),formal_runs=1100,checkpoint_metadata_runs=300,representative_full_checkpoint_hashes=len(representative),parameter_rows=len(model_rows),epoch_summary_rows=len(epoch_summary),timing_summary_rows=len(time_summary),runtime_rows=len(environment),original_B_runs_at_60=sum(r['stage']=='original' and r['checkpoint_rule']=='B' for r in allruns),extension_checkpoint_rules=sorted(set(r['checkpoint_rule'] for r in allruns if r['stage']=='extension')),parameter_count_method='restricted symbolic checkpoint metadata, exclude recorded buffers, count parameter shapes and apply exact source module-prefix freeze rules; cross-check historical pilot trainable counts',weights_loaded=False,new_training=False,new_inferential_statistics=False,original_manuscript_modified=False,sources=sources,outputs={rel(p):sha(p) for p in paths},representative_runs={':'.join(k):v for k,v in representative.items()})
(OUT/f'P3a_Metadata_Audit_{V}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['sources','outputs','representative_runs']},ensure_ascii=False));raise SystemExit(1 if errors else 0)
