"""Nested candidate selection entirely within outer training families."""
import numpy as np
from sklearn.metrics import roc_auc_score
from pipeline.splits import inner_site_splits
from pipeline.experts import SelectedExperts
from experiments.registry import CANDIDATES


def macro_score(y,p,sites):
    results=[]
    for s in np.unique(sites):
        m=sites==s
        if len(np.unique(y[m])) == 2:
            results.append(float(roc_auc_score(y[m],p[m])))
    if not results:raise ValueError('Validation has no site containing both classes')
    return float(np.mean(results))


def tune_train_only(cohort,train,seed=17,n_boot=8,inner_splits=2):
    """Choose counts on training-site-held inner validation; never outer held site."""
    vals={'smri_stable':{},'fmri_edges':{},'fmri_roi':{},'fusion':{}}
    splits=list(inner_site_splits(cohort.y,cohort.sites,train,seed=seed,count=inner_splits))
    for k_a,k_f in CANDIDATES:
        key=f'{k_a}_{k_f}'
        for typ in ('edges','roi'):
            choices=[]
            for i,(it,iv,sites) in enumerate(splits):
                e=SelectedExperts(k_a,k_f,typ,seed=seed+i,nb=n_boot).fit(cohort,it)
                pa,pf=e.predict(cohort,iv)
                yt=cohort.y[iv];st=cohort.sites[iv]
                choices.append((macro_score(yt,pa,st),macro_score(yt,pf,st),
                                macro_score(yt,.75*pa+.25*pf,st)))
            v=np.mean(choices,axis=0)
            if typ=='edges':
                vals['smri_stable'][key]=float(v[0]);vals['fmri_edges'][key]=float(v[1]);
                vals['fusion'][key]=float(v[2])
            else:
                vals['fmri_roi'][key]=float(v[1])
    selected={}
    for name,results in vals.items():
        # fixed tie rule: smaller feature count comes first
        best=max(sorted(results,key=lambda z: tuple(map(int,z.split('_'))),reverse=True),key=lambda z:results[z])
        a,f=map(int,best.split('_'))
        selected[name]={'k_anat':a,'k_fc':f,'mode':'roi' if name=='fmri_roi' else 'edges',
                        'mean_inner_macro_auc':float(results[best]),'all_candidates':results,
                        'n_inner_validations':len(splits)}
    return selected
