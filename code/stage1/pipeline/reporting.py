"""Reproducible aggregate-only export. Private participant scores never enter ZIP."""
from pathlib import Path
import csv,json,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score


def recompute(out,expected_sites=17):
    out=Path(out)
    pred_files=sorted((out/'private_predictions').glob('*.csv'))
    if not pred_files:
        return dict(completed=0,all_complete=False)
    site_rows=[]
    for path in pred_files:
        p=pd.read_csv(path)
        assert p.subject_id.is_unique and p.label.nunique()==2
        assert p.prob.between(0,1).all()
        site_rows.append({'model':p.model.iloc[0],'seed':int(p.seed.iloc[0]),
            'site':p.site.iloc[0], 'n':len(p),'n_asd':int(p.label.sum()),
            'auc':float(roc_auc_score(p.label,p.prob)),
            'balanced_accuracy':float(balanced_accuracy_score(p.label,p.prob>=.5)),
            'average_precision':float(average_precision_score(p.label,p.prob))})
    per_site=pd.DataFrame(site_rows).sort_values(['model','seed','site'])
    if per_site.duplicated(['model','seed','site']).any():raise ValueError('Duplicate held site')
    agg_dir=out/'aggregates';agg_dir.mkdir(parents=True,exist_ok=True)
    per_site.to_csv(agg_dir/'PER_SITE_AUC.csv',index=False)
    res=[]
    for (model,seed),g in per_site.groupby(['model','seed']):
        res.append({'model':model,'seed':int(seed),'n_held_sites':int(g.site.nunique()),
            'n_subjects':int(g.n.sum()),'complete':bool(len(g)==expected_sites and g.n.sum()==720),
            'site_macro_auc':float(g.auc.mean()),
            'weighted_site_auc':float(np.average(g.auc,weights=g.n)),
            'macro_balanced_accuracy':float(g.balanced_accuracy.mean()),
            'macro_average_precision':float(g.average_precision.mean())})
    agg=pd.DataFrame(res).sort_values(['model','seed'])
    agg.to_csv(agg_dir/'MODEL_SEED_SUMMARY.csv',index=False)
    complete=agg[agg.complete].copy()
    if len(complete):
        rank=complete.groupby('model',as_index=False).agg(
            mean_site_macro_auc=('site_macro_auc','mean'),
            sd_across_seeds=('site_macro_auc','std'),
            n_complete_seeds=('seed','size'))
        rank.sort_values('mean_site_macro_auc',ascending=False).to_csv(
            agg_dir/'FULL_SITE_RANKING_ONLY.csv',index=False)
    selection_dir=out/'private_selection'
    # Frequencies calculated WITHIN each model, preventing models that share
    # feature subsets from being counted as independent site replications.
    names_path=out/'audit_private'/'anatomy_feature_names.json'
    if names_path.exists():
        anat_names=[n for n in json.loads(names_path.read_text())
                    if n != 'EstimatedTotalIntraCranialVol']
    else:
        anat_names=[]
    triangle=np.triu_indices(116,1)
    counts={};model_folds={}
    for path in sorted(selection_dir.glob('*.json')):
        try:d=json.loads(path.read_text())
        except Exception:continue
        model=path.name.split('__s')[0]
        model_folds[model]=model_folds.get(model,0)+1
        for kind in ('anat','fc'):
            for idx in set(d.get(kind,[])):
                key=(model,kind,int(idx))
                counts[key]=counts.get(key,0)+1
    if counts:
        rows=[]
        for (model,kind,idx),count in counts.items():
            if kind=='anat':
                description=anat_names[idx] if idx < len(anat_names) else f'AnatomyFeature_{idx}'
            else:
                description=f'AAL_ROI_{triangle[0][idx]+1:03d}__AAL_ROI_{triangle[1][idx]+1:03d}'
            rows.append({'model':model,'feature_type':kind,'feature_index_0based':idx,
                'feature_name_or_edge':description,
                'n_outer_model_folds_selected':count,
                'n_available_outer_model_folds':model_folds[model],
                'selection_rate':float(count/model_folds[model])})
        pd.DataFrame(rows).sort_values(['model','selection_rate','feature_name_or_edge'],
                        ascending=[True,False,True]).to_csv(
            agg_dir/'FEATURE_SELECTION_STABILITY_BY_MODEL.csv',index=False)
    record={'completed_folds':len(site_rows),'complete_model_seed_runs':int(len(complete)),
            'status':'PARTIAL',
            'warning':'Previously explored ABIDE-I cohort; exploratory model comparisons, not prospective SOTA.'}
    (agg_dir/'SUMMARY.json').write_text(json.dumps(record,indent=2))
    return record


def export_zip(out,archive):
    out,archive=Path(out),Path(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for d in ('aggregates','fold_summaries','docs'):
            for p in sorted((out/d).rglob('*')) if (out/d).exists() else []:
                if p.is_file():z.write(p,arcname=str(p.relative_to(out)))
        if (out/'RUN_CONTRACT.json').exists():z.write(out/'RUN_CONTRACT.json','RUN_CONTRACT.json')
    return archive
