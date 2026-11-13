#!/usr/bin/env python3
"""Regenerate main Figures 1-5 from architecture definitions and archived aggregates.
Figures 1-4 are newly written plotting implementations; not recovered originals.
No participant-level input or invented experimental values are used.
"""
from pathlib import Path
import subprocess,sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/aggregate'
OUT=ROOT/'figures/regenerated';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
TEAL='#187f72';BLUE='#446884';GRAY='#87949f'

def export(fig,stem):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.2)
 plt.close(fig)

def workflow():
 fig,ax=plt.subplots(figsize=(10,7.0));ax.set_xlim(0,10);ax.set_ylim(0,7.1);ax.axis('off')
 def box(x,y,w,h,text,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',facecolor=color,edgecolor='#72808b',linewidth=1))
  ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
 def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#647783','lw':1.3})
 ax.text(5,6.92,'Multimodal MRI fusion under site-held-out evaluation',ha='center',fontsize=15,fontweight='bold')
 ax.text(5,6.55,'17 outer folds  |  Fit on 16 site families; score the held-out family',ha='center',fontsize=10,color='#45556c')
 box(1.5,5.45,7,.7,'Matched ABIDE-I representations\n184 structural measures + 6,670 functional connections','#edf1f5')
 box(.15,3.95,2.8,.8,'Structural selection\nTraining-fold-only ranking','#eaf2f8')
 box(3.55,3.95,2.8,.8,'Connectivity selection\nTraining-fold-only edge ranking','#eaf2f8')
 box(.15,2.65,2.8,.7,'Structural expert\nRegularized logistic model','#edf5f5')
 box(3.55,2.65,2.8,.7,'Functional expert\nRegularized logistic model','#edf5f5')
 box(7.0,3.05,2.8,1.2,'PCA-128 reference\nSeparate historical\nsMRI + FC fusion','#faf1e3')
 box(1.1,1.1,5.2,.8,'Adaptive gated prediction\nGate trained on site-aware out-of-fold scores','#dcefe9')
 box(7.0,1.1,2.8,.8,'Exploratory 50:50 blend\nGate + PCA-128','#dcefe9')
 for a,b in [((3,5.45),(1.55,4.75)),((4.9,5.45),(4.95,4.75)),((7,5.45),(8.4,4.25)),((1.55,3.95),(1.55,3.35)),((4.95,3.95),(4.95,3.35)),((1.55,2.65),(2.2,1.9)),((4.95,2.65),(4.95,1.9)),((6.3,1.5),(7,1.5)),((8.4,3.05),(8.4,1.9))]:arrow(a,b)
 ax.text(5,.3,'The ensemble pairing was selected retrospectively. Other comparator arms are omitted.',ha='center',fontsize=9,color='#45556c')
 export(fig,'Figure_1_Study_Workflow')

def leaderboard(summary):
 base=summary[summary.stage=='original_10_arm_suite'].groupby('recipe',sort=False).site_macro_auc.mean()
 base.loc['blend_gate_pca_half']=float(summary.loc[summary.recipe=='blend_gate_pca_half','site_macro_auc'].iloc[0]);base=base.sort_values()
 names={'v4_pca128':'PCA-128 reference','v4_anatomy75':'V4 anatomy-75 fusion','smri_stable':'Selected sMRI logistic','fmri_edges':'Selected FC-edge logistic','fmri_roi':'ROI-prioritized FC logistic','selected_equal':'Selected equal fusion','selected_anatomy75':'Selected anatomy-75 fusion','selected_gate':'Adaptive gated fusion','full_attention':'Full-input attention (2-seed mean)','selected_attention':'Selected attention (2-seed mean)','blend_gate_pca_half':'Gate + PCA-128 (retrospective)'}
 fig,ax=plt.subplots(figsize=(10,6));ys=np.arange(len(base))
 colors=[TEAL if r=='blend_gate_pca_half' else BLUE if r=='selected_equal' else '#a9bbc9' for r in base.index]
 ax.barh(ys,base.values,color=colors,height=.58);ax.set_yticks(ys,[names[x] for x in base.index])
 for y,v in zip(ys,base.values):ax.text(v+.002,y,f'{v:.4f}',va='center',fontsize=9)
 ax.axvline(base['v4_pca128'],ls='--',lw=1,color='#566875');ax.set_xlim(.56,.735)
 ax.set_xlabel('Unweighted mean ROC-AUC across 17 held-out site families')
 ax.set_title('Full-site model comparison (exploratory)',fontweight='bold',pad=15)
 fig.text(.02,.015,'Descriptive values. The highest blend is not an independently validated improvement.',fontsize=9,color='#45556c')
 fig.tight_layout(rect=[0,.05,1,1]);export(fig,'Figure_2_Model_Leaderboard')

def differences(site):
 a=site[(site.stage=='retrospective_ensemble_comparisons')&(site.recipe=='blend_gate_pca_half')].set_index('site_family')
 b=site[(site.stage=='original_10_arm_suite')&(site.recipe=='v4_pca128')].set_index('site_family')
 delta=(a.auc-b.auc).sort_values();names=[x.split('::')[-1] for x in delta.index]
 fig,ax=plt.subplots(figsize=(9,6));ax.barh(names,delta.values,color=[TEAL if v>=0 else '#aa6f6f' for v in delta]);ax.axvline(0,color='#5b6670',lw=.9)
 for i,n in enumerate(delta.index):ax.text(.123,i,f'n={int(a.loc[n,"n_subjects"])}',fontsize=8,va='center',ha='right')
 ax.set_xlim(-.055,.13);ax.set_xlabel('Site AUC difference: gate + PCA-128 minus PCA-128')
 ax.set_title('Performance differences vary across held-out sites',fontweight='bold',pad=14)
 fig.tight_layout();export(fig,'Figure_3_Site_Differences')

def anatomical(stability):
 a=stability[(stability.model=='selected_equal')&(stability.feature_type=='anat')].copy()
 counts=a.n_outer_model_folds_selected.value_counts().reindex(range(1,18),fill_value=0)
 examples=a[a.n_outer_model_folds_selected==17].sort_values('feature_index_0based').head(7)
 fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1,1.4]})
 axes[0].bar(counts.index,counts.values,color=[TEAL if i==17 else '#8caabd' for i in counts.index]);axes[0].set_xticks(range(1,18,2));axes[0].set_xlabel('Number of outer training folds (of 17)');axes[0].set_ylabel('Number of structural measurements');axes[0].set_title(f'All {len(a)} ever-selected features')
 axes[1].barh(examples.feature_name_or_edge.iloc[::-1],examples.n_outer_model_folds_selected.iloc[::-1],color=TEAL);axes[1].set_xlim(0,18.8);axes[1].set_xticks([0,5,10,15,17]);axes[1].set_xlabel('Folds selecting feature');axes[1].set_title('Seven examples recurring in all 17 folds')
 for i in range(len(examples)):axes[1].text(17.15,i,'17/17',va='center',fontsize=8)
 fig.suptitle('Structural feature-selection recurrence',fontweight='bold',fontsize=14)
 fig.text(.02,.015,'Selected-equal branch. Overlapping training sets are not independent biological replications.',fontsize=9,color='#45556c')
 fig.tight_layout(rect=[0,.04,1,.95]);export(fig,'Figure_4_Anatomical_Stability')

def main():
 summary=pd.read_csv(DATA/'Additional_file_3_Model_and_Ensemble_Summary.csv')
 site=pd.read_csv(DATA/'Additional_file_1_Site_and_Model_Results.csv')
 stability=pd.read_csv(DATA/'Additional_file_2_Feature_Selection_Stability.csv')
 workflow();leaderboard(summary);differences(site);anatomical(stability)
 subprocess.run([sys.executable,str(ROOT/'scripts/plot_fc_stability.py'),'--input',str(DATA/'Additional_file_2_Feature_Selection_Stability.csv'),'--out',str(OUT)],check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/plot_supplementary.py')],check=True)
 print('Regenerated all main and supplementary figures from bundled aggregates.')
if __name__=='__main__':main()
