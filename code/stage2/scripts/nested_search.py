#!/usr/bin/env python3
"""Site-nested feature-budget / regularization / fusion search on ABIDE-I.

One primary adaptive pipeline is predeclared. Every selected configuration comes
from training-site-only LOSO inner validation. Outer held-site predictions are
used ONLY for final evaluation, never to choose the configuration, rank feature
selectors, or stop an experiment at a favorable AUC.
"""
from __future__ import annotations
import argparse,hashlib,inspect,json,os,sys,traceback
from dataclasses import asdict,dataclass
from pathlib import Path
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v,'4')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

@dataclass(frozen=True)
class Candidate:
    name:str
    k_anat:int
    k_fc:int
    C_anat:float
    C_fc:float
    weight_anat:float
    fusion:str='prob'

# One pre-specified, constrained list. The user cannot modify it based on
# individual outer-site test results without creating a new exploratory study.
CANDIDATES=(
 Candidate('baseline_40_256',40,256,.025,.075,.50),
 Candidate('baseline_80_512',80,512,.025,.075,.50),
 Candidate('wider_120_768',120,768,.025,.075,.50),
 Candidate('wide_160_1024',160,1024,.025,.075,.50),
 Candidate('weighted_80_512',80,512,.025,.075,.60),
 Candidate('weighted_120_768',120,768,.025,.075,.60),
 Candidate('shrink_more_80_512',80,512,.0125,.040,.50),
 Candidate('shrink_less_80_512',80,512,.050,.150,.50),
 Candidate('logit_blend_80_512',80,512,.025,.075,.50,'logit'),
)
EXPECTED_SITES=17
EXPECTED_SUBJECTS=720


def fuse(pa,pf,w,mode):
    pa,pf=np.asarray(pa,dtype=np.float64),np.asarray(pf,dtype=np.float64)
    if pa.shape!=pf.shape: raise ValueError('Different modality prediction shapes')
    if mode=='prob':return w*pa+(1-w)*pf
    if mode=='logit':
        from scipy.special import expit,logit
        return expit(w*logit(np.clip(pa,1e-6,1-1e-6))+(1-w)*logit(np.clip(pf,1e-6,1-1e-6)))
    raise ValueError(f'Unknown fusion mode {mode}')


def preflight_import(lab,root_results):
    lab=Path(lab).resolve();root_results=Path(root_results).resolve()
    if not (lab/'pipeline/selection.py').exists() or not (lab/'pipeline/data.py').exists():
        raise RuntimeError('Original ASD_FEATURE_ATTENTION_LAB missing; please restore the original training ZIP at --original-lab')
    if not (root_results/'RUN_CONTRACT.json').exists():
        raise RuntimeError('Missing original RUN_CONTRACT.json; must verify prior frozen cohort and source')
    contract=json.loads((root_results/'RUN_CONTRACT.json').read_text())
    expected=contract.get('code_sha256')
    h=hashlib.sha256()
    for file in sorted(lab.rglob('*.py')):
        if 'tests' in file.parts:continue
        h.update(str(file.relative_to(lab)).encode());h.update(file.read_bytes())
    digest=h.hexdigest()
    if digest!=expected:
        raise RuntimeError('Original LAB source changed since completed experiment. Expected '+str(expected)+' vs actual '+digest+'. Restore original source. DO NOT override this check.')
    if contract.get('cohort_sha256')!='af538bf3da2091f3dfb78564c1127bf615d10aed586899e697f670127e283953':
        raise RuntimeError('Prior cohort checksum differs from reviewed 720-subject cohort')
    if str(lab) not in sys.path:sys.path.insert(0,str(lab))
    from pipeline.selection import FoldFeatureSelector
    from pipeline.data import load_cohort, feature_fingerprint
    from pipeline.features import inverse_sqrt_site_weights
    from pipeline.splits import outer_loso,inner_site_splits
    if len(inspect.signature(FoldFeatureSelector.fit).parameters)!=6: # self + anat fc sites y meta
        raise RuntimeError('Original FoldFeatureSelector.fit signature differs from expected')
    return contract,(FoldFeatureSelector,load_cohort,feature_fingerprint,
                     inverse_sqrt_site_weights,outer_loso,inner_site_splits)


def atomic_json(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False));tmp.replace(p)


def site_macro_auc(y,p,sites):
    auc=[]
    for site in np.unique(sites):
        sel=sites==site
        if len(set(y[sel]))!=2:
            raise ValueError('Inner held-out site lacks both diagnostic groups: '+str(site))
        auc.append(roc_auc_score(y[sel],p[sel]))
    return float(np.mean(auc))


def train_pair(c,itr,k_a,k_fc,seed,n_boot,selector_class,weights_fn):
    """Fit selector once for a feature budget; re-use only within same inner fold."""
    selector=selector_class(k_a=k_a,k_f=k_fc,fc_mode='edges',seed=seed,n_boot=n_boot)
    selector.fit(c.anat[itr],c.fc[itr],c.sites[itr],c.y[itr],c.meta[itr])
    return selector


def eval_bundle(c,itr,iv,configs,selector,weights_fn):
    ax,fx=selector.transform(c.anat[itr],c.fc[itr],c.meta[itr])
    av,fv=selector.transform(c.anat[iv],c.fc[iv],c.meta[iv])
    w=weights_fn(c.sites[itr],c.y[itr]);ys=c.y[itr]
    memo={}
    for cand in configs:
        ca,cf=cand.C_anat,cand.C_fc
        for mod,reg,features_train,features_val in (
            ('anat',ca,ax,av),('fc',cf,fx,fv)):
            key=(mod,reg)
            if key not in memo:
                model=LogisticRegression(C=reg,solver='liblinear',max_iter=2500,random_state=17)
                model.fit(features_train,ys,sample_weight=w)
                memo[key]=model.predict_proba(features_val)[:,1]
    return {cand.name:fuse(memo[('anat',cand.C_anat)],memo[('fc',cand.C_fc)],
                      cand.weight_anat,cand.fusion) for cand in configs}


def nested_choose(c,tr,seed,n_boot,n_inner,adapter):
    Selector,_,_,weights_fn,_,inner_splits=adapter
    val_results={x.name:[] for x in CANDIDATES}
    detail=[]
    groups={}
    for cfg in CANDIDATES:
        groups.setdefault((cfg.k_anat,cfg.k_fc),[]).append(cfg)
    splits=list(inner_splits(c.y,c.sites,tr,seed=seed,count=n_inner))
    if len(splits)<2:raise ValueError('At least 2 inner held-site folds are required')
    for k,(itr,iv,sites) in enumerate(splits):
        if set(c.sites[itr]).intersection(set(c.sites[iv])):
            raise RuntimeError('TRAIN/VALIDATION SITE LEAKAGE')
        if set(c.subjects[itr]).intersection(set(c.subjects[iv])):
            raise RuntimeError('PARTICIPANT LEAKAGE')
        iv_labels=c.y[iv];iv_sites=c.sites[iv]
        scores={}
        for j,((k_a,k_fc),configs) in enumerate(groups.items()):
            selector=train_pair(c,itr,k_a,k_fc,seed+101*k+j,n_boot,Selector,weights_fn)
            options=eval_bundle(c,itr,iv,configs,selector,weights_fn)
            for name,pred in options.items():
                auc=site_macro_auc(iv_labels,pred,iv_sites)
                scores[name]=auc
                val_results[name].append(auc)
        detail.append({'inner_fold':k,'held_sites':[str(x) for x in sites],
            'n_train':len(itr),'n_validation':len(iv), 'candidate_auc':scores})
    ranking=sorted(CANDIDATES,key=lambda x:np.mean(val_results[x.name]),reverse=True)
    top=ranking[0]
    # Fixed, conservative tie preference: near-optimal 80+512 equal over
    # unnecessarily elaborate tuning; tolerance chosen in advance.
    simple=next(x for x in CANDIDATES if x.name=='baseline_80_512')
    top_score=float(np.mean(val_results[top.name]))
    score_simple=float(np.mean(val_results[simple.name]))
    chosen=(simple if score_simple>=top_score-.005 else top)
    choice={'selected':asdict(chosen),'best_raw':asdict(top),
      'tie_tolerance':.005,'inner_score_selected':float(np.mean(val_results[chosen.name])),
      'inner_score_best':top_score,
      'candidate_means':{n:float(np.mean(v)) for n,v in val_results.items()},
      'inner_folds':detail}
    return chosen,choice


def fit_final_predict(c,tr,held,cfg,seed,n_boot,adapter):
    Selector,_,_,weights_fn,_,_=adapter
    selector=train_pair(c,tr,cfg.k_anat,cfg.k_fc,seed+10101,n_boot,Selector,weights_fn)
    preds=eval_bundle(c,tr,held,[cfg],selector,weights_fn)
    p=preds[cfg.name]
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid final probabilities')
    details={'structural_feature_indices':selector.evidence.anat_indices.astype(int).tolist(),
        'functional_edge_indices':selector.evidence.fc_indices.astype(int).tolist()}
    return p,details


def summarize(out,expected_sites):
    from sklearn.metrics import roc_auc_score,balanced_accuracy_score
    out=Path(out);rows=[]
    for f in sorted((out/'private_predictions').glob('*.csv')):
        d=pd.read_csv(f)
        if not d.subject_id.is_unique or d.site.nunique()!=1 or d.model.nunique()!=1 or d.seed.nunique()!=1:raise ValueError('Corrupt prediction file '+str(f))
        rows.append({'model':d.model.iloc[0],'seed':int(d.seed.iloc[0]),
            'site':d.site.iloc[0],'n':len(d),'auc':float(roc_auc_score(d.label,d.prob)),
            'balanced_accuracy':float(balanced_accuracy_score(d.label,d.prob>=.5))})
    if not rows:return pd.DataFrame(),pd.DataFrame()
    site=pd.DataFrame(rows)
    site.to_csv(out/'aggregates'/'PER_SITE_AUC.csv',index=False)
    summary=[]
    for (model,seed),g in site.groupby(['model','seed']):
        summary.append({'model':model,'seed':int(seed),'completed_sites':len(g),
          'expected_sites':expected_sites,'n_subjects':int(g.n.sum()),
          'complete_17_site':bool(len(g)==17 and g.n.sum()==720),
          'macro_auc':float(g.auc.mean()),
          'macro_balanced_accuracy':float(g.balanced_accuracy.mean())})
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'aggregates'/'MODEL_SEED_RESULTS.csv',index=False)
    return site,summary


def aggregate_zip(out,output):
    import zipfile
    out,output=Path(out),Path(output)
    output.parent.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['aggregates','public_fold_metadata']:
            for p in (out/folder).rglob('*'):
                if p.is_file():z.write(p,p.relative_to(out))
        z.write(out/'RUN_CONTRACT.json','RUN_CONTRACT.json')


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--original-lab',default='/home/jovyan/ASD_FEATURE_ATTENTION_LAB')
    p.add_argument('--original-out',default='/home/jovyan/ASD_FEATURE_ATTENTION_RESULTS')
    p.add_argument('--anatomy',default='/home/jovyan/anatomy_mamba_comparison')
    p.add_argument('--fmri',default='/home/jovyan/fmri_gat_controlled')
    p.add_argument('--out',default='/home/jovyan/ASD_TO_070_RESULTS/nested')
    p.add_argument('--aggregate-zip',default='/home/jovyan/ASD_TO_070_NESTED_AGGREGATES.zip')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,41])
    p.add_argument('--n-boot',type=int,default=8)
    p.add_argument('--inner-splits',type=int,default=3)
    p.add_argument('--pilot',action='store_true')
    return p.parse_args()


def main():
    args=parse();out=Path(args.out)
    adapter_contract,adapter=preflight_import(args.original_lab,args.original_out)
    Selector,load_cohort,feature_fingerprint,weights_fn,outer_loso,_=adapter
    for p in ('private_predictions','private_selection','aggregates','public_fold_metadata'):
        (out/p).mkdir(exist_ok=True,parents=True)
    c=load_cohort(args.anatomy,args.fmri,out/'private_selection')
    if len(c.y)!=EXPECTED_SUBJECTS or c.sites.size!=EXPECTED_SUBJECTS or len(set(c.sites))!=EXPECTED_SITES:
        raise RuntimeError('Unexpected cohort! 720 participants/17 families required.')
    cohort_hash=feature_fingerprint(c)
    if cohort_hash!=adapter_contract['cohort_sha256']:
        raise RuntimeError('Frozen MRI cohort checksum changed: refuse to proceed')
    folds=list(outer_loso(c))
    if args.pilot:folds=folds[:2]
    own_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract={'prior_cohort_sha':cohort_hash,'prior_source_sha':adapter_contract['code_sha256'],
      'this_runner_sha':own_sha,'candidate_configs':[asdict(x) for x in CANDIDATES],
      'seeds':args.seeds,'n_boot':args.n_boot,'inner_splits':args.inner_splits,
      'site_names':[str(site) for site,*_ in folds], 'pilot':args.pilot,
      'scientific_status':'Exploratory ABIDE-I model development, not independent external validation'}
    path=out/'RUN_CONTRACT.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise RuntimeError('Out-dir contains a different run contract. Start a new --out folder.')
    if not path.exists():atomic_json(path,contract)
    completed=0;total=len(folds)*len(args.seeds)
    for seed in args.seeds:
        for site,tr,held in folds:
            short=str(site).split('::')[-1]
            filename=f'nested_primary__s{seed}__{short}'
            predictions=out/'private_predictions'/f'{filename}.csv'
            public=out/'public_fold_metadata'/f'{filename}.json'
            if predictions.exists() and public.exists():
                print(f'[SKIP] {filename}',flush=True);completed+=1;continue
            if predictions.exists()!=public.exists():
                raise RuntimeError(f'Partial fold {filename}; inspect files, do not silently overwrite')
            if len(set(c.sites[tr]) & set(c.sites[held])):raise RuntimeError('Outer site contamination')
            print(f'[START] {filename} n_train={len(tr)} n_test={len(held)}',flush=True)
            cfg,choice=nested_choose(c,tr,seed,args.n_boot,args.inner_splits,adapter)
            prob,indices=fit_final_predict(c,tr,held,cfg,seed,args.n_boot,adapter)
            private=pd.DataFrame({'subject_id':c.subjects[held], 'site':c.sites[held],
                 'label':c.y[held], 'model':'nested_primary','seed':seed,'prob':prob})
            tmp=predictions.with_suffix('.csv.partial');private.to_csv(tmp,index=False);tmp.replace(predictions)
            atomic_json(out/'private_selection'/f'{filename}.json',indices)
            site_auc=float(roc_auc_score(c.y[held],prob))
            choice_meta={'model':'nested_primary','seed':seed,'held_site':str(site),
              'n_train':len(tr),'n_held':len(held),'site_auc':site_auc,
              'selected_config':asdict(cfg),'nested_macro_auc':choice['inner_score_selected'],
              'scores_by_candidate':choice['candidate_means']}
            atomic_json(public,choice_meta)
            print(f'[DONE] {filename} held-site AUC={site_auc:.4f} | train-only choice={cfg.name}',flush=True)
            completed+=1
            _,status=summarize(out,len(folds));aggregate_zip(out,args.aggregate_zip)
            print(f'[PROGRESS] {completed}/{total}, all-site results only valid after 17/17 per seed',flush=True)
    _,status=summarize(out,len(folds))
    print('[FINISHED] See',args.aggregate_zip,'\n',status.to_string(index=False))

if __name__=='__main__':
    main()
