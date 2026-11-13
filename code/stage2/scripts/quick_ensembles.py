#!/usr/bin/env python3
"""No-refit, locked-weight analysis of archived ABIDE-I outer-LOSO probabilities.

Every mixture is specified in source BEFORE inspecting its full-site performance.
These are additional retrospective, exploratory comparisons. Absolutely no fit,
weight search, or cherry picking using held-out site labels.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Ordered by pre-specified hypothesis, not observed outcome.
ENSEMBLES = {
    'selected_equal_prior': {'selected_equal': 1.0},
    'selected_gate_prior': {'selected_gate': 1.0},
    'pca128_prior': {'v4_pca128': 1.0},
    'blend_selected_gate_half': {'selected_equal': .5, 'selected_gate': .5},
    'blend_selected_pca_half': {'selected_equal': .5, 'v4_pca128': .5},
    'blend_selected_pca_75_25': {'selected_equal': .75, 'v4_pca128': .25},
    'blend_selected_gate_pca_50_25_25': {
        'selected_equal': .5, 'selected_gate': .25, 'v4_pca128': .25},
    'blend_gate_pca_half': {'selected_gate': .5, 'v4_pca128': .5},
}
EXPECTED_N = 720
EXPECTED_SITES = 17


def hashlib_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_one_model(root, name, expected_sites=EXPECTED_SITES, expected_n=EXPECTED_N):
    folder=Path(root)/'private_predictions'
    paths=sorted(folder.glob(f'{name}__s17__*.csv'))
    if len(paths) != expected_sites:
        raise RuntimeError(f'{name}: expected {expected_sites} site files, found {len(paths)} under {folder}')
    frames=[]
    for path in paths:
        df=pd.read_csv(path,dtype={'subject_id':str,'site':str})
        needed={'subject_id','site','label','prob','model','seed'}
        if not needed.issubset(df.columns):
            raise ValueError(f'Missing columns in {path}: {needed-set(df.columns)}')
        if df['subject_id'].isna().any() or not df.subject_id.is_unique:
            raise ValueError(f'Null or repeated IDs: {path}')
        if df.site.nunique()!=1 or df.model.nunique()!=1 or str(df.model.iloc[0])!=name or df.seed.nunique()!=1 or int(df.seed.iloc[0])!=17:
            raise ValueError(f'Wrong site, model or seed: {path}')
        if not df.prob.between(0,1).all() or df.prob.isna().any():
            raise ValueError(f'Invalid probability data: {path}')
        frames.append(df[['subject_id','site','label','prob']])
    data=pd.concat(frames,ignore_index=True)
    if len(data)!=expected_n or data.subject_id.duplicated().any() or data.site.nunique()!=expected_sites:
        raise ValueError(f'{name}: subject count or site families inconsistent')
    if not set(data.label.unique())=={0,1}:
        raise ValueError(f'{name}: unexpected diagnoses')
    if (data.groupby('site').label.nunique()!=2).any():
        raise ValueError(f'{name}: at least one site does not have both classes')
    return data.sort_values('subject_id',kind='stable').reset_index(drop=True)


def score_site_frame(df,prob_col):
    rows=[]
    for site,grp in df.groupby('site'):
        rows.append({'site':site,'n':len(grp), 'auc':roc_auc_score(grp.label,grp[prob_col]),
              'balanced_accuracy': balanced_accuracy_score(grp.label,grp[prob_col]>=.5)})
    return pd.DataFrame(rows)


def evaluate(root,out,expected_sites=EXPECTED_SITES,expected_n=EXPECTED_N):
    root=Path(root);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    models=set(m for w in ENSEMBLES.values() for m in w)
    observed={m:load_one_model(root,m,expected_sites,expected_n) for m in sorted(models)}
    ref=observed['selected_equal']
    base=ref[['subject_id','site','label']].copy()
    for key,data in observed.items():
        if not np.array_equal(data.subject_id.to_numpy(),ref.subject_id.to_numpy()) or not np.array_equal(data.site.to_numpy(),ref.site.to_numpy()) or not np.array_equal(data.label.to_numpy(),ref.label.to_numpy()):
            raise RuntimeError(f'MISMATCH between {key} and selected_equal in subject/diagnosis/site ordering')
    site_rows=[];overall_rows=[]
    for name,w in ENSEMBLES.items():
        assert abs(sum(w.values())-1.)<1e-12 and all(0<=v<=1 for v in w.values())
        base['score']=sum(k_weight*observed[k]['prob'].to_numpy() for k,k_weight in w.items())
        agg=score_site_frame(base,'score')
        agg.insert(0,'model',name)
        site_rows.append(agg)
        overall_rows.append({'model':name,'n_site_families':agg.site.nunique(),
           'n_participants':int(agg.n.sum()),'site_macro_auc':float(agg.auc.mean()),
           'site_macro_balanced_accuracy':float(agg.balanced_accuracy.mean()),
           'weights':json.dumps(w,sort_keys=True)})
    site=pd.concat(site_rows,ignore_index=True)
    summary=pd.DataFrame(overall_rows)
    primary='blend_selected_gate_half'
    baseline=float(summary.loc[summary.model=='pca128_prior','site_macro_auc'].iloc[0])
    sel=float(summary.loc[summary.model=='selected_equal_prior','site_macro_auc'].iloc[0])
    summary['delta_vs_pca128']=summary.site_macro_auc-baseline
    summary['delta_vs_selected_equal']=summary.site_macro_auc-sel
    summary.sort_values('site_macro_auc',ascending=False).to_csv(out/'EXPLORATORY_FIXED_ENSEMBLES.csv',index=False)
    site.to_csv(out/'EXPLORATORY_ENSEMBLE_SITE_SCORES.csv',index=False)
    metadata={'n':expected_n,'site_families':expected_sites, 'primary_ensemble':primary,
      'predeclared_ensemble_weights':ENSEMBLES,
      'raw_predictions_location':str(root/'private_predictions'),
      'cohort_hash':json.loads((root/'RUN_CONTRACT.json').read_text()).get('cohort_sha256') if (root/'RUN_CONTRACT.json').exists() else None,
      'data_policy':'No participant-level predictions in the export.',
      'scientific_warning':'Exploratory fixed-weight ensembles evaluated on previously explored ABIDE-I. No ensemble is an independently confirmed gain.'}
    (out/'QUICK_ENSEMBLE_MANIFEST.json').write_text(json.dumps(metadata,indent=2))
    print(summary.sort_values('site_macro_auc',ascending=False)[['model','site_macro_auc','delta_vs_pca128','site_macro_balanced_accuracy']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'Pre-specified primary 50/50 selected_equal+gate: {float(summary.loc[summary.model==primary,"site_macro_auc"].iloc[0]):.4f}')
    return summary,site


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-out',default='/home/jovyan/ASD_FEATURE_ATTENTION_RESULTS')
    p.add_argument('--out',default='/home/jovyan/ASD_TO_070_RESULTS/quick')
    args=p.parse_args()
    evaluate(args.original_out,args.out)

if __name__=='__main__':main()
