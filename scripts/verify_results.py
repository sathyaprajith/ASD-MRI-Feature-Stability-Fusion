#!/usr/bin/env python3
"""Recompute paper summaries from supplied aggregate tables; no model training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/aggregate'
def verify():
 site=pd.read_csv(DATA/'Additional_file_1_Site_and_Model_Results.csv')
 summary=pd.read_csv(DATA/'Additional_file_3_Model_and_Ensemble_Summary.csv',dtype={'seed':str})
 for row in summary.itertuples():
  seed='deterministic_derived' if str(row.seed)=='derived' else str(row.seed)
  sel=site[(site.stage==row.stage)&(site.recipe==row.recipe)&(site.seed.astype(str)==seed)]
  assert len(sel)==17,(row.stage,row.recipe,seed,len(sel))
  assert int(sel.n_subjects.sum())==720
  assert int(sel.n_asd.sum())==336
  assert abs(sel.auc.mean()-row.site_macro_auc)<2e-9
  assert abs(sel.balanced_accuracy.mean()-row.site_macro_balanced_accuracy)<2e-9
  assert abs(np.average(sel.auc,weights=sel.n_subjects)-row.weighted_site_auc)<2e-9
 expected={'v4_pca128':.676956201,'selected_equal':.692837956,'selected_gate':.692035548,'blend_gate_pca_half':.702276789}
 for recipe,value in expected.items():
  observed=summary.loc[summary.recipe==recipe,'site_macro_auc']
  assert len(observed)==1 and abs(float(observed.iloc[0])-value)<1e-9
 stable=pd.read_csv(DATA/'Additional_file_2_Feature_Selection_Stability.csv')
 stable=stable[stable.model=='selected_equal']
 a=stable[stable.feature_type=='anat'];f=stable[stable.feature_type=='fc']
 checks={'structural_ever_selected':len(a),'fc_ever_selected':len(f),'structural_all_17':int((a.n_outer_model_folds_selected==17).sum()),'fc_all_17':int((f.n_outer_model_folds_selected==17).sum()),'structural_at_least_75_percent':int((a.n_outer_model_folds_selected>=13).sum()),'fc_at_least_75_percent':int((f.n_outer_model_folds_selected>=13).sum())}
 assert checks=={'structural_ever_selected':137,'fc_ever_selected':1186,'structural_all_17':24,'fc_all_17':112,'structural_at_least_75_percent':49,'fc_at_least_75_percent':272},checks
 winner=site[(site.stage=='retrospective_ensemble_comparisons')&(site.recipe=='blend_gate_pca_half')].set_index('site_family')
 reference=site[(site.stage=='original_10_arm_suite')&(site.recipe=='v4_pca128')].set_index('site_family')
 delta=winner.auc-reference.auc
 assert (int((delta>1e-8).sum()),int((abs(delta)<=1e-8).sum()),int((delta < -1e-8).sum()))==(11,2,4)
 return {'status':'passed','model_seed_summaries':len(summary),'sites_per_summary':17,'participants_per_summary':720,**checks,'warning':'Aggregate arithmetic checks do not validate model training or independent generalization.'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
