import csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,brier_score_loss

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p,obj):
    temp=p.with_suffix(p.suffix+'.part');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(p)

def load_data(anatomy,fmri,out):
    for base in (anatomy,fmri):
        if json.loads((base/'results/results.json').read_text()).get('status')!='complete': raise ValueError('Source study incomplete: '+str(base))
    old=json.loads((anatomy/'results/contract.json').read_text())
    if sha(anatomy/'data/cohort.json')!=old['cohort_sha256'] or sha(anatomy/'data/anatomy.csv')!=old['anatomy_sha256']: raise ValueError('Anatomical source changed since completed run')
    original=json.loads((anatomy/'data/cohort.json').read_text());frows=json.loads((fmri/'prepared/eligible_cohort.json').read_text())
    if sha(fmri/'prepared/eligible_cohort.json')!=json.loads((fmri/'results/contract.json').read_text())['cohort_sha256']: raise ValueError('fMRI eligible roster changed since completed run')
    for name in ('eligibility.csv','summary.json'):
        p=fmri/'prepared'/name
        if p.is_file(): (out/('fmri_source_'+name)).write_bytes(p.read_bytes())
    if len({r['subject_id'] for r in original})!=len(original) or len({r['subject_id'] for r in frows})!=len(frows): raise ValueError('Duplicate source IDs')
    fmap={r['subject_id']:r for r in frows}
    frame=pd.read_csv(anatomy/'data/anatomy.csv').set_index('subject_id')
    if not frame.index.is_unique or set(frame.index)!={r['subject_id'] for r in original} or frame.shape[1]!=185: raise ValueError('Anatomy table and roster differ')
    selected=[];excluded=[];fc=[];cache_hashes={}
    for a in original:
        sid=a['subject_id']
        if sid not in fmap:
            excluded.append(dict(subject_id=sid,partition=a['historical_partition'],reason='Absent from completed fMRI eligible roster'))
            continue
        f=fmap[sid]
        if not sid.startswith('ABIDE_I::') or a['historical_partition']!=f['partition'] or int(a['group']=='ASD')!=f['asd_true']: raise ValueError('Dataset, label or split mismatch: '+sid)
        if f['partition'] not in ('train','val'): raise ValueError('Non-development participant')
        path=fmri/'prepared/participants'/(hashlib.sha256(sid.encode()).hexdigest()+'.npz')
        if sha(path)!=f['cache_sha256']: raise ValueError('FC cache changed: '+sid)
        cache_hashes[sid]=f['cache_sha256']
        with np.load(path,allow_pickle=False) as z:
            if str(z['subject_id'].item())!=sid or int(z['asd_true'])!=f['asd_true'] or str(z['source_sha256'].item())!=f['source_sha256']: raise ValueError('FC cache identity differs: '+sid)
            fc.append(z['fc'].copy())
        selected.append(dict(f,anatomy_source='FreeSurfer185'))
    if not selected: raise ValueError('No paired participants')
    frame=frame.loc[[r['subject_id'] for r in selected]]
    feature_names=list(frame.columns)
    for c in frame.columns:
        if not c.endswith('_thickness'): frame[c]=np.log1p(frame[c])
    x=frame.to_numpy(float);f=np.asarray(fc,float)
    if f.shape[1]!=6670 or not np.isfinite(f).all() or np.isinf(x).any(): raise ValueError('Invalid feature shapes/values')
    save(out/'anatomy_feature_names.json',feature_names);save(out/'fc_cache_hashes.json',cache_hashes);save(out/'matching_exclusions.json',excluded)
    with (out/'matched_cohort.csv').open('w',newline='') as stream:
        names=['subject_id','dataset','site_id','partition','asd_true','mean_fd','usable_frames','tensor_tr'];w=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore');w.writeheader();w.writerows(selected)
    return selected,x,f

def metrics(y,p):
    if not np.isfinite(p).all(): raise ValueError('Nonfinite fusion probabilities')
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),accuracy=float(np.mean(y==(p>=.5))),balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),f1=float(f1_score(y,p>=.5,zero_division=0)),brier=float(brier_score_loss(y,p)))
