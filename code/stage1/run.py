#!/usr/bin/env python3
"""Sequential, resuming ABIDE-I LOSO feature selection + fusion study.

No person-level data or weights are copied into aggregate result archives.
Every feature-selection/tuning/early-stop fit excludes the outer held-out site.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(k,'4')
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from pipeline.data import load_cohort, feature_fingerprint
from pipeline.splits import outer_loso
from pipeline.tuning import tune_train_only
from pipeline.experts import SelectedExperts
from experiments.smri_stability.model import StableStructuralModel
from experiments.fmri_edges.model import StableFunctionalEdges
from experiments.fmri_roi.model import StableFunctionalROI
from experiments.fusion_fixed.model import FixedSelectedFusion
from pipeline.reporting import recompute, export_zip
from experiments.registry import EXPERIMENTS,DL
from experiments.reference_v4.model import V4Reference
from experiments.fusion_gated.model import AdaptiveGate
from experiments.fusion_attention.model import AttentionExperiment

ROOT=Path(__file__).resolve().parent


def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False));tmp.replace(path)


def source_digest():
    h=hashlib.sha256()
    for p in sorted(ROOT.rglob('*.py')):
        if 'tests' in p.parts:continue
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def get_args():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/home/jovyan/ASD_FEATURE_ATTENTION_RESULTS')
    p.add_argument('--anatomy',default='/home/jovyan/anatomy_mamba_comparison')
    p.add_argument('--fmri',default='/home/jovyan/fmri_gat_controlled')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--only-sites',nargs='*')
    p.add_argument('--models',nargs='*',choices=EXPERIMENTS)
    p.add_argument('--bootstraps',type=int,default=8)
    p.add_argument('--epochs',type=int,default=65)
    p.add_argument('--inner-splits',type=int,default=2)
    p.add_argument('--device',default=None)
    p.add_argument('--aggregate-zip',default='/home/jovyan/ASD_FEATURE_ATTENTION_AGGREGATES.zip')
    return p.parse_args()


def safe_basename(model,seed,site):
    return f'{model}__s{seed}__{site.split("::")[-1]}'


def selection_for(model,chosen):
    key=('smri_stable' if model=='smri_stable'
         else 'fmri_edges' if model=='fmri_edges'
         else 'fmri_roi' if model=='fmri_roi' else 'fusion')
    return chosen[key]


def selected_evidence(fold_model,model):
    obj=fold_model.selector if hasattr(fold_model,'selector') else None
    if obj is None and hasattr(fold_model,'base') and hasattr(fold_model.base,'selector'):
        obj=fold_model.base.selector
    if obj is None and hasattr(fold_model,'prep') and hasattr(fold_model.prep,'evidence'):
        obj=fold_model.prep
    if obj is None:return None
    return {'anat':obj.evidence.anat_indices.tolist(),
            'fc':obj.evidence.fc_indices.tolist(),
            'roi':obj.evidence.roi_indices.tolist()}


def main():
    args=get_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for d in ('private_predictions','private_selection','private_models','fold_summaries','aggregates','audit_private'):
        (out/d).mkdir(exist_ok=True)
    print('[DATA] Reading original verified V4 cache; no copies of MRI scans are made.',flush=True)
    c=load_cohort(args.anatomy,args.fmri,out/'audit_private')
    # V4 comparator requires original acquisition IDs (LEUVEN_1 vs _2 etc).
    assert len(c.raw_sites)==len(c.subjects)
    models=args.models or list(EXPERIMENTS)
    sites=['CALTECH','NYU'] if args.pilot and not args.only_sites else args.only_sites
    outer=list(outer_loso(c,sites))
    config={'cohort_sha256':feature_fingerprint(c),
        'code_sha256':source_digest(), 'pilot':bool(args.pilot),
        'models':models,'sites':[s for s,_,_ in outer],
        'bootstraps':args.bootstraps,'epochs':args.epochs,
        'inner_splits':args.inner_splits,
        'v4_source_sha256':{p:file_sha(ROOT/'vendor'/'v4'/p)
          for p in ('controls.py','feature_suite.py','advanced_fusion.py','matched_data.py','metadata.py','cv_utils.py')},
        'caution':'Previously analyzed ABIDE-I; exploratory model development only.'}
    contract=out/'RUN_CONTRACT.json'
    if contract.exists():
        if json.loads(contract.read_text())!=config:
            raise RuntimeError('Run configuration, source, or cohort changed! Use NEW --out; do not mix runs.')
    else:atomic_json(contract,config)
    planned=sum((2 if m in DL else 1)*len(outer) for m in models)
    print(f'[PLAN] {len(models)} models, {len(outer)} held-out sites, '
          f'{planned} complete fold evaluations; accelerator={args.device or ("cuda" if torch.cuda.is_available() else "cpu")}',flush=True)
    done=0
    for model in models:
        seed_list=(17,41) if model in DL else (17,)
        for seed in seed_list:
            for site,tr,held in outer:
                name=safe_basename(model,seed,site)
                pth=out/'private_predictions'/f'{name}.csv'
                summary_file=out/'fold_summaries'/f'{name}.json'
                if pth.exists() and summary_file.exists():
                    print(f'[SKIP] {name} (already completed)',flush=True);done+=1;continue
                if pth.exists() != summary_file.exists():
                    raise RuntimeError(f'Incomplete output for {name}; inspect before resume')
                print(f'[START] {name} : train={len(tr)}, held={len(held)}',flush=True)
                details={'model':model,'seed':seed,'held_site':site,'n_train':int(len(tr)),'n_held':int(len(held))}
                try:
                    if model.startswith('v4_'):
                        estimator=V4Reference(model).fit(c,tr)
                        probabilities=estimator.predict(c,held)
                        details['feature_selection']='exact V4 unmodified archived recipe'
                    else:
                        tune_cache=out/'audit_private'/f'nested_tuning__seed17__{site.split("::")[-1]}.json'
                        if not tune_cache.exists():
                            nested=tune_train_only(c,tr,seed=17,n_boot=args.bootstraps,
                                                   inner_splits=args.inner_splits)
                            atomic_json(tune_cache,nested)
                        nested=json.loads(tune_cache.read_text())
                        psel=selection_for(model,nested)
                        ka,kf=int(psel['k_anat']),int(psel['k_fc'])
                        details.update({'selected_anat_count':ka,'selected_fc_count':kf,
                                        'inner_selection_macro_auc':psel['mean_inner_macro_auc'],
                                        'selected_fc_mode':psel['mode']})
                        if model=='selected_gate':
                            estimator=AdaptiveGate(ka,kf,seed=seed,nb=args.bootstraps).fit(c,tr)
                            probabilities,weights=estimator.predict(c,held)
                            details['mean_gate_anatomy_weight']=float(np.mean(weights))
                        elif model in DL:
                            estimator=AttentionExperiment(full=model=='full_attention',ka=ka,kf=kf,
                                seed=seed,n_boot=args.bootstraps,inner_splits=args.inner_splits,
                                max_epochs=args.epochs,device=args.device).fit(c,tr)
                            probabilities,attention=estimator.predict(c,held)
                            details.update({'chosen_epochs':estimator.final_epochs,
                                'inner_val_aucs':[round(s,6) for s in estimator.inner_scores],
                                'mean_attention_anatomy_weight':float(np.mean(attention[:,0]))})
                        else:
                            if model=='smri_stable': estimator=StableStructuralModel(ka,kf,seed,nb=args.bootstraps)
                            elif model=='fmri_edges': estimator=StableFunctionalEdges(ka,kf,seed,nb=args.bootstraps)
                            elif model=='fmri_roi': estimator=StableFunctionalROI(ka,kf,seed,nb=args.bootstraps)
                            elif model=='selected_equal': estimator=FixedSelectedFusion(ka,kf,'equal',seed,nb=args.bootstraps)
                            elif model=='selected_anatomy75': estimator=FixedSelectedFusion(ka,kf,'anatomy75',seed,nb=args.bootstraps)
                            else: raise ValueError('Unknown experiment '+model)
                            estimator.fit(c,tr)
                            probabilities=estimator.predict(c,held)
                    if len(probabilities)!=len(held) or not np.isfinite(probabilities).all():
                        raise ValueError('Bad held-site predictions')
                    from sklearn.metrics import roc_auc_score
                    auc=float(roc_auc_score(c.y[held],probabilities))
                    import io
                    pred_df=pd.DataFrame(dict(subject_id=c.subjects[held],label=c.y[held],
                        site=c.sites[held],model=model,seed=seed,prob=probabilities))
                    tmp=pth.with_suffix('.csv.tmp')
                    pred_df.to_csv(tmp,index=False);tmp.replace(pth)
                    details['auc']=auc;details['status']='complete'
                    atomic_json(summary_file,details)
                    if model not in ('v4_anatomy75','v4_pca128'):
                        evidence=selected_evidence(estimator,model)
                        if evidence:
                            atomic_json(out/'private_selection'/f'{name}.json',evidence)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        joblib.dump(estimator,out/'private_models'/f'{name}.joblib',compress=3)
                    print(f'[DONE] {name}: AUC {auc:.4f}',flush=True)
                except Exception:
                    print('[ERROR]',name,traceback.format_exc(),flush=True)
                    raise
                done+=1
                record=recompute(out,expected_sites=len(outer))
                export_zip(out,args.aggregate_zip)
                print(f'[PROGRESS] {done}/{planned} completed; aggregate ZIP refreshed',flush=True)
    record=recompute(out,expected_sites=len(outer))
    # Pilot intentionally has 2 sites; report is not full-17 evaluation.
    record['status']='COMPLETED_PILOT' if args.pilot else 'COMPLETED_FULL_SITE_SUITE'
    atomic_json(out/'aggregates'/'SUMMARY.json',record)
    export_zip(out,args.aggregate_zip)
    print('[COMPLETE] Aggregate ZIP: '+args.aggregate_zip,flush=True)

if __name__=='__main__':main()
