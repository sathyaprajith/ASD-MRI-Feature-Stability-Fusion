#!/usr/bin/env python3
"""Fixed-prediction paired comparisons of a NEW method with the prior reference.

Participant bootstrap samples within site x diagnosis while retaining the
same resampled rows for each method. This is *conditional* uncertainty only,
not post-selection/parameter-refit uncertainty or independent validation.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from quick_ensembles import load_one_model


def site_means(d,score_col):
    return np.array([roc_auc_score(grp.label,grp[score_col]) for _,grp in d.groupby('site')])


def paired(a,b,draws=4000,seed=20261110):
    """a,b rows sorted identically, with columns subject_id,site,label,prob."""
    a=a.sort_values('subject_id').reset_index(drop=True)
    b=b.sort_values('subject_id').reset_index(drop=True)
    if not np.array_equal(a[['subject_id','site','label']].to_numpy(),b[['subject_id','site','label']].to_numpy()):
        raise ValueError('Cannot compare different subject labels or held sites')
    frame=a[['subject_id','site','label']].copy()
    frame['new']=a.prob
    frame['baseline']=b.prob
    delta=site_means(frame,'new')-site_means(frame,'baseline')
    gen=np.random.default_rng(seed)
    # Stratified by each outer held-out family x ASD/TD, paired across models.
    groups=[]
    for site,grp in frame.groupby('site'):
        grp_bins=[grp.index[grp.label==y].to_numpy() for y in [0,1]]
        if any(len(x)==0 for x in grp_bins):raise ValueError('Site lacks both classes')
        groups.append(grp_bins)
    boot=np.empty(draws,dtype=float)
    y=frame.label.to_numpy();new=frame.new.to_numpy();old=frame.baseline.to_numpy()
    for d in range(draws):
        diffs=[]
        for grp in groups:
            idx=np.concatenate([gen.choice(x,len(x),replace=True) for x in grp])
            diffs.append(roc_auc_score(y[idx],new[idx])-roc_auc_score(y[idx],old[idx]))
        boot[d]=float(np.mean(diffs))
    gen=np.random.default_rng(seed+1)
    site_boot=delta[gen.integers(0,len(delta),size=(draws,len(delta)))].mean(axis=1)
    result={'n_sites':len(groups),'n_subjects':len(frame),'new_site_macro_auc':float(site_means(frame,'new').mean()),
      'reference_site_macro_auc':float(site_means(frame,'baseline').mean()),
      'paired_macro_auc_difference':float(delta.mean()),
      'participant_stratified_pair_bootstrap_CI_95':np.quantile(boot,[.025,.975]).tolist(),
      'site_resampling_pair_bootstrap_CI_95':np.quantile(site_boot,[.025,.975]).tolist(),
      'site_deltas_positive':int(np.sum(delta>1e-10)),'site_deltas_negative':int(np.sum(delta < -1e-10)),
      'site_deltas_tied':int(np.sum(abs(delta)<=1e-10)),
      'caveat':'Conditional repeated-sample inference for frozen predictions only; not multiplicity adjusted, not independent-cohort validation.'}
    return result


def run(original,nested,out,seeds):
    original=Path(original);nested=Path(nested);out=Path(out)
    out.mkdir(exist_ok=True,parents=True)
    baselines={name:load_one_model(original,name) for name in ('v4_pca128','selected_equal')}
    summaries=[]
    for seed in seeds:
        pred_files=sorted((nested/'private_predictions').glob(f'nested_primary__s{seed}__*.csv'))
        if len(pred_files)!=17:
            print(f'[SKIP] Seed {seed} not fully complete ({len(pred_files)}/17)');continue
        new=pd.concat((pd.read_csv(x,dtype={'subject_id':str,'site':str}) for x in pred_files),ignore_index=True)
        if len(new)!=720 or len(set(new.site))!=17 or new.subject_id.duplicated().any():raise ValueError('Incomplete/duplicate new predictions')
        for refname,ref in baselines.items():
            r=paired(new,ref)
            r.update({'seed':seed,'against_model':refname})
            summaries.append(r)
    (out/'PAIRED_INFERENCE.json').write_text(json.dumps(summaries,indent=2))
    if summaries:
        pd.json_normalize(summaries).to_csv(out/'PAIRED_INFERENCE.csv',index=False)
        for r in summaries:
            print(f"seed {r['seed']} vs {r['against_model']}: delta={r['paired_macro_auc_difference']:+.4f} "
                  f"conditional 95% CI {r['participant_stratified_pair_bootstrap_CI_95']}")
    else:print('No full 17-site new model run yet; results not calculated')
    return summaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-out',default='/home/jovyan/ASD_FEATURE_ATTENTION_RESULTS')
    p.add_argument('--nested-out',default='/home/jovyan/ASD_TO_070_RESULTS/nested')
    p.add_argument('--out',default='/home/jovyan/ASD_TO_070_RESULTS/paired')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,41])
    a=p.parse_args();run(a.original_out,a.nested_out,a.out,a.seeds)

if __name__=='__main__':main()
