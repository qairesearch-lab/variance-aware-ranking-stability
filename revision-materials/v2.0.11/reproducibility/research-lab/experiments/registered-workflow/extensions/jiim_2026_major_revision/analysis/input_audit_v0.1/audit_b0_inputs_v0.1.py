"""Read-only B0 source audit; writes only versioned audit artifacts beside this script.

No training, performance comparison, bootstrap, or modifications to source data.
"""
import csv
import hashlib
import json
import math
import platform
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
BASE = ROOT / 'research-lab/experiments/registered-workflow'
EXT = BASE / 'extensions/jiim_2026_major_revision'
SOURCES = {}
ISSUES = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(p):
    SOURCES[str(p.relative_to(ROOT))] = sha(p)


def read_json(p):
    source(p)
    return json.loads(p.read_text())


def read_csv(p, register=True):
    if register:
        source(p)
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def issue(item, severity, finding, path=''):
    ISSUES.append(dict(item=item, severity=severity, finding=finding, source=str(path)))


def split_audit():
    records, cache = [], {}
    for ds in ['sipakmed', 'organamnist', 'isic2019', 'mura']:
        folder = BASE/'splits'/ds if ds in ['sipakmed', 'organamnist'] else EXT/'formal_splits_v0.6'/ds
        for p in sorted(folder.glob('split_*.csv')):
            rows = read_csv(p)
            by_subset = Counter(r['subset'] for r in rows)
            sample_partition, group_partition = defaultdict(set), defaultdict(set)
            for r in rows:
                sample_partition[r['sample_id']].add(r['subset'])
                if r.get('group_id'):
                    group_partition[r['group_id']].add(r['subset'])
            dup = len(rows)-len(sample_partition)
            overlap = sum(len(x)>1 for x in group_partition.values())
            if dup or overlap:
                issue('D05', 'INPUT_ERROR', f'{ds}: duplicate sample rows={dup}; group partition overlaps={overlap}', p.relative_to(ROOT))
            test = [r for r in rows if r['subset']=='test']
            units = len({r['study_id'] for r in test}) if ds=='mura' else len(test)
            rec = dict(dataset=ds, split_id=p.stem, rows=len(rows), train_rows=by_subset['train'],
                       validation_rows=by_subset.get('validation', by_subset.get('val',0)),
                       test_images=len(test), primary_test_units=units,
                       primary_prediction_unit='study' if ds=='mura' else 'image',
                       grouping_unit={'sipakmed':'unavailable; image-level', 'organamnist':'unavailable; image-level',
                                      'isic2019':'lesion group_id', 'mura':'patient group_id'}[ds],
                       duplicate_sample_rows=dup, group_partition_overlaps=overlap,
                       sha256=sha(p), source_path=str(p.relative_to(ROOT)))
            records.append(rec); cache[(ds,p.stem)] = rec
    write_csv('B0_split_input_audit_v0.1.csv', records)
    return cache


def audit_runs(split_cache):
    records, common, strata = [], [], Counter()
    envs = defaultdict(list)
    historical = {}
    for p in [BASE/'outputs/primary/analysis_io/run_metrics_table.csv', EXT/'analysis/results_v0.1/extension_run_metrics.csv']:
        for r in read_csv(p):
            historical[r['run_id']] = r
    specs = [('original',BASE/'run-manifests/primary_run_manifest.csv'),
             ('extension',EXT/'run_manifests/jiim_extension_300_v0.6.csv')]
    for stage, manifest in specs:
        manifest_rows=read_csv(manifest)
        if len({r['run_id'] for r in manifest_rows}) != len(manifest_rows):
            issue('D05','INPUT_ERROR','duplicate manifest run IDs',manifest.relative_to(ROOT))
        for r in manifest_rows:
            p = ROOT / r['output_dir']; local=[]
            files = ['run_config.json','run_status.json','metrics.json','checkpoint_manifest.json','history.csv','predictions.csv']
            if stage=='extension': files.append('timing.json')
            missing=[n for n in files if not (p/n).is_file()]
            if missing:
                issue('D05','INPUT_ERROR',f"{r['run_id']}: missing {missing}",p.relative_to(ROOT));continue
            conf=read_json(p/'run_config.json');status=read_json(p/'run_status.json');metric=read_json(p/'metrics.json')
            checkpoint=read_json(p/'checkpoint_manifest.json');history=read_csv(p/'history.csv')
            with (p/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
                prediction_first=next(csv.DictReader(f),{})
            epochs=[int(x['epoch']) for x in history]
            if epochs != list(range(1,len(history)+1)):local.append('history epochs not contiguous from 1')
            if stage=='original':
                identity=conf['manifest_row'];policy=r['checkpoint_policy'];variant='submitted';software=conf['package_versions']
                env={k:software.get(k,'') for k in ['python','torch','torchvision','numpy','pandas','Pillow','scikit-learn','PyYAML']}
                device=conf.get('device_info',{});gpu='; '.join(device.get('gpu_models',[]));cuda=device.get('cuda_version','')
                train=conf['resolved_frozen_config']['training_common'];opt=train['optimization'];overrides=conf.get('validation_overrides',{})
                params={'optimizer':opt['optimizer'], 'learning_rate':opt['learning_rate'], 'weight_decay':opt['weight_decay'],
                        'batch_size':opt['batch_size'], 'num_workers':overrides.get('num_workers_override') if overrides.get('num_workers_override') is not None else train['reproducibility']['num_workers'],
                        'mixed_precision':train.get('mixed_precision', train.get('precision','not recorded in frozen config'))}
                selected=checkpoint['selected_epoch'];max_epoch=conf['resolved_frozen_config']['checkpoint_policy']['max_epochs']
                last=[c.get('epoch') for c in checkpoint.get('checkpoints',[]) if c.get('role')=='last_epoch']
                if last != [len(history)]:local.append('last checkpoint epoch differs from history')
                for key in ['run_id','dataset','split_id','training_seed','model','checkpoint_policy','config_hash','split_hash']:
                    if str(identity.get(key))!=str(r.get(key)) or str(metric.get(key))!=str(r.get(key)):
                        local.append(f'{key}: manifest/config/metric mismatch')
                if metric.get('metric_source')!='primary_train':local.append('metric not primary_train')
                restrictive=[k for k in ['max_epochs','sample_limit_per_subset','max_train_batches'] if overrides.get(k) is not None]
                if restrictive or overrides.get('disable_train_augmentation'):local.append('training/sample override: '+str(restrictive))
                elapsed=(datetime.fromisoformat(status['ended_at_utc'])-datetime.fromisoformat(status['started_at_utc'])).total_seconds()
                timing_scope='run entry through output export; wall-clock UTC difference'
                split_hash=r['split_hash']
            else:
                identity=conf;policy=checkpoint['rule'];variant=r['variant'];params=conf['hyperparameters'];env_raw=conf['environment']
                aliases={'Pillow':'pillow','scikit-learn':'scikit_learn','PyYAML':'pyyaml'}
                env={k:env_raw.get(aliases.get(k,k),'') for k in ['python','torch','torchvision','numpy','pandas','Pillow','scikit-learn','PyYAML']}
                gpu=env_raw['gpu'];cuda=env_raw['cuda'];selected=checkpoint['best_epoch'];max_epoch=params['max_epochs']
                timing=read_json(p/'timing.json');elapsed=timing['elapsed_seconds']
                timing_scope='after model/data-loader initialization through checkpoint hashing and metric export; perf_counter'
                if int(timing['epochs_executed'])!=len(history):local.append('timing epoch count differs from history')
                for key in ['run_id','dataset','model','variant','training_seed']:
                    if str(identity.get(key))!=str(r.get(key)):local.append(key+': config/manifest mismatch')
                if metric.get('selected_epoch')!=selected:local.append('metric/checkpoint selected epoch mismatch')
                if conf.get('technical_smoke') or metric.get('technical_smoke_not_research_result'):local.append('technical smoke included')
                split_hash=r['split_sha256']
                if conf['split_sha256']!=split_hash:local.append('config split hash mismatch')
                if conf['run_manifest_sha256']!=sha(manifest):local.append('sealed manifest hash mismatch')
                index=ROOT/r['dataset_index_file']
                if str(index.relative_to(ROOT)) not in SOURCES: source(index)
                if SOURCES[str(index.relative_to(ROOT))]!=r['dataset_index_sha256'] or conf['dataset_index_sha256']!=r['dataset_index_sha256']:local.append('dataset index hash mismatch')
            split=split_cache[(r['dataset'],r['split_id'])]
            if split_hash!=split['sha256']:local.append('split content hash mismatch')
            if status.get('run_id')!=r['run_id'] or status['status'] not in ['completed','rerun_completed']:local.append('run status mismatch/incomplete')
            if metric.get('run_id')!=r['run_id'] or metric.get('dataset')!=r['dataset']:local.append('metric identity mismatch')
            if prediction_first.get('run_id')!=r['run_id']:local.append('prediction header/first record identity mismatch')
            if not (1<=int(selected)<=len(history)<=int(max_epoch)):local.append('selected/trained/max epoch inconsistent')
            ba=float(metric['balanced_accuracy'])
            if not math.isfinite(ba) or not 0<=ba<=1:local.append('BA not finite/in range')
            n=sum(sum(row) for row in metric['confusion_matrix'])
            if n!=split['primary_test_units']:local.append('confusion matrix N differs from split primary test units')
            old=historical.get(r['run_id'])
            if old is None or abs(float(old['balanced_accuracy'])-ba)>1e-12:local.append('previous analysis metric does not match raw run JSON')
            for msg in local:issue('D05','INPUT_ERROR',r['run_id']+': '+msg,p.relative_to(ROOT))
            rec=dict(run_id=r['run_id'],stage=stage,dataset=r['dataset'],model=r['model'],split_id=r['split_id'],
                     training_seed=int(r['training_seed']),checkpoint_policy=policy,variant=variant,run_state=status['status'],
                     required_files_present=True,input_audit='PASS' if not local else 'NEEDS_REVIEW',issue_count=len(local),
                     prediction_unit=split['primary_prediction_unit'],test_units=n,balanced_accuracy=ba,
                     trained_epochs=len(history),selected_epoch=int(selected),max_epochs=int(max_epoch),
                     elapsed_seconds=elapsed,timing_scope=timing_scope,gpu_model=gpu,cuda=cuda,
                     **env, **{k:v for k,v in params.items() if k!='max_epochs'},source_dir=str(p.relative_to(ROOT)),
                     metrics_sha256=sha(p/'metrics.json'),run_config_sha256=sha(p/'run_config.json'))
            records.append(rec)
            strata[(stage,r['dataset'],policy,variant,r['model'])]+=1
            envs[(stage,r['dataset'],json.dumps(env,sort_keys=True),gpu,cuda)].append(r['run_id'])
            if stage=='original' and int(r['training_seed']) in [42,52,62]:common.append({k:rec[k] for k in ['run_id','dataset','model','split_id','training_seed','checkpoint_policy','input_audit','source_dir']})
    write_csv('B0_run_input_audit_v0.1.csv',records)
    write_csv('D05_original_common_three_seeds_input_v0.1.csv',common)
    write_csv('B0_stratum_inventory_v0.1.csv',[dict(stage=k[0],dataset=k[1],policy=k[2],variant=k[3],model=k[4],runs=v) for k,v in sorted(strata.items())])
    envrows=[dict(stage=k[0],dataset=k[1],runs=len(v),gpu_model=k[3],cuda=k[4],**json.loads(k[2]),first_run=v[0],last_run=v[-1]) for k,v in sorted(envs.items())]
    write_csv('D07_environment_by_dataset_v0.1.csv',envrows)
    write_csv('D07_epoch_timing_config_by_run_v0.1.csv',[{k:r[k] for k in ['run_id','stage','dataset','model','checkpoint_policy','variant','trained_epochs','selected_epoch','max_epochs','elapsed_seconds','timing_scope','gpu_model','python','torch','torchvision','numpy','Pillow','batch_size','num_workers','optimizer','learning_rate','weight_decay','mixed_precision','source_dir']} for r in records])
    return records,common,envrows


def audit_images():
    p=BASE/'configs/frozen/checksums/organamnist_primary_images_sha256.csv'
    expected=read_csv(p);checks=[];counts=Counter();matched=0
    for r in expected:
        img=ROOT/r['relative_path'];present=img.is_file();actual=sha(img) if present else '';ok=actual==r['sha256']
        width=height=None;mode='missing';verified=False
        if present:
            try:
                with Image.open(img) as im:
                    width,height=im.size;mode=im.mode;im.verify();verified=True
            except Exception as exc:issue('D06','INPUT_ERROR',str(exc),r['relative_path'])
        counts[(r['relative_path'].split('/')[3],width,height,mode,verified)]+=1
        matched+=ok
        if not ok:issue('D06','INPUT_ERROR','current image missing or hash differs from frozen content manifest',r['relative_path'])
        checks.append(dict(relative_path=r['relative_path'],frozen_sha256=r['sha256'],actual_sha256=actual,
                           hash_matches=ok,width=width,height=height,mode=mode,png_verified=verified))
    write_csv('D06_organamnist_image_audit_v0.1.csv',checks)
    summary=[dict(subset=k[0],width=k[1],height=k[2],mode=k[3],png_verified=k[4],images=v) for k,v in sorted(counts.items())]
    write_csv('D06_organamnist_shape_mode_summary_v0.1.csv',summary)
    csvfacts=[]
    for subset in ['train','val','test']:
        p=ROOT/'research-lab/data/organamnist'/subset/'organamnist.csv';source(p)
        with p.open(newline='') as f:rows=list(csv.reader(f))
        csvfacts.append(dict(subset=subset,rows=len(rows),sha256=sha(p),source_path=str(p.relative_to(ROOT))))
    write_csv('D06_organamnist_csv_counts_v0.1.csv',csvfacts)
    return dict(images=len(expected),frozen_hash_matches=matched,shape_mode_counts=summary,csv_counts=csvfacts,
                npz_present_in_local_dataset=bool(list((ROOT/'research-lab/data/organamnist').rglob('*.npz'))),
                evidence_boundary='Current PNG content verified against frozen manifest; original NPZ arrays not reverified; exporter file is not proof of its execution in April.')


def main():
    cache=split_audit();records,common,envrows=audit_runs(cache);images=audit_images()
    groups=defaultdict(list)
    for r in records:
        groups[(r['stage'],r['dataset'],r['model'],r['checkpoint_policy'],r['variant'])].append(r)
    epoch_summary=[]
    for key, group in sorted(groups.items()):
        item=dict(stage=key[0],dataset=key[1],model=key[2],policy=key[3],variant=key[4],runs=len(group))
        for field in ['trained_epochs','selected_epoch']:
            values=[r[field] for r in group]
            item.update({field+'_mean':statistics.mean(values),field+'_SD':statistics.stdev(values) if len(values)>1 else None,
                         field+'_median':statistics.median(values),field+'_min':min(values),field+'_max':max(values)})
        epoch_summary.append(item)
    write_csv('D07_epoch_summary_v0.1.csv',epoch_summary)
    for p in [ROOT/'research-lab/experiments/scripts/export_organamnist_npz.py',ROOT/'research-lab/experiments/scripts/train_one_run.py',
              BASE/'configs/frozen/training_common.yaml',BASE/'configs/frozen/dataset_manifest.md',
              EXT/'train_extension_run.py',EXT/'preprocessing.py',EXT/'configs/environment_target_v1.yaml',
              ROOT/'research-lab/experiments/experiment-logs/sessions/2026-04-25_OrganAMNIST下载校验与数据记录_05.md',
              ROOT/'research-lab/experiments/experiment-logs/sessions/2026-04-26_数据完整性与上传前冻结配置复核_07.md']:
        source(p)
    issue('D07','DISCLOSE','Training software differs by dataset: OrganAMNIST torch2.10/python3.12 vs SIPaKMeD and extension torch2.4/python3.10; fixed within each dataset; no claim that 1100 runs share one environment.','D07_environment_by_dataset_v0.1.csv')
    issue('D07','DISCLOSE','num_workers execution overrides and timing origins differ; retain factual metadata only, no speed/hardware comparison.','D07_epoch_timing_config_by_run_v0.1.csv')
    issue('D06','LIMIT','Original NPZ not present in dataset folder. Historical frozen hashes identify all actual training PNGs; do not assert NPZ regeneration or array-to-PNG pixel equivalence was verified.','D06_organamnist_image_audit_v0.1.csv')
    write_csv('B0_findings_v0.1.csv',ISSUES)
    out=dict(audit_id='JIIM-B0-input-audit-v0.1',date='2026-10-03',status='INPUT_AUDITED_NOT_RETRAINED',
             runtime=dict(python=sys.version.split()[0],platform=platform.platform()),
             declared_scope='D05-D07 input verification only; no inferential analysis; no speed comparison',
             run_rows=len(records),run_passed=sum(r['input_audit']=='PASS' for r in records),
             split_rows=len(cache),common_three_seed_original_runs=len(common),
             stage_counts=dict(Counter(r['stage'] for r in records)),environments=envrows,organamnist=images,
             input_errors=[r for r in ISSUES if r['severity']=='INPUT_ERROR'],disclosures=[r for r in ISSUES if r['severity']!='INPUT_ERROR'],
             source_hashes=SOURCES,output_hashes={p.name:sha(p) for p in sorted(HERE.glob('*.csv'))},
             limitations=['Predictions checked for presence/schema/first-record identity, not fully recomputed; BA was checked against existing analysis table and confusion-matrix sample counts.',
                          'Checkpoint weight files not rehashed; they are not inputs to the planned tabular statistical analyses.',
                          'Image.verify checks file integrity; no re-export, retraining, or pixel equivalence test against unavailable NPZ.',
                          'Within-split group disjointness was audited; repeated splits can share images, so distinct split IDs do not imply independent observations.'])
    (HERE/'B0_input_audit_manifest_v0.1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['audit_id','run_rows','run_passed','split_rows','common_three_seed_original_runs','stage_counts','input_errors']},ensure_ascii=False))
    print(json.dumps(images,ensure_ascii=False))


if __name__=='__main__':main()
