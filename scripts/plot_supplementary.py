"""Rebuild supplement Figures S1-S4 from the unchanged Additional files 1-3.

Run from this repository:
    python scripts/plot_supplementary.py
Requires matplotlib and numpy. No synthetic values, predictions or external data.
"""
from pathlib import Path
import csv
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'results/aggregate'
OUT=ROOT/'figures/regenerated/supplementary'
OUT.mkdir(parents=True,exist_ok=True)

def rows(fn):
    with (IN/fn).open(encoding='utf-8-sig', newline='') as f:return list(csv.DictReader(f))
site=rows('Additional_file_1_Site_and_Model_Results.csv')
stability=rows('Additional_file_2_Feature_Selection_Stability.csv')
summary=rows('Additional_file_3_Model_and_Ensemble_Summary.csv')
BLUE='#24569A';TEAL='#0C847E';SLATE='#45556C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def export(fig,name):
    for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.17)
    plt.close(fig)
labels={'fmri_edges':'Selected FC-edge LR','fmri_roi':'ROI-prioritized FC LR','full_attention':'Full-input attention','selected_anatomy75':'Selected anatomy-75 fusion','selected_attention':'Selected-feature attention','selected_equal':'Selected equal fusion','selected_gate':'Adaptive gate','smri_stable':'Selected sMRI logistic','v4_anatomy75':'V4 anatomy-75 fusion','v4_pca128':'PCA-128 reference'}
recipes=['v4_pca128','v4_anatomy75','smri_stable','fmri_edges','fmri_roi','selected_equal','selected_anatomy75','selected_gate','full_attention','selected_attention']
original=[r for r in site if r['stage']=='original_10_arm_suite']
sites=sorted(set(r['site_family'] for r in original))
mat=[[np.mean([float(r['auc']) for r in original if r['recipe']==recipe and r['site_family']==site_id]) for site_id in sites] for recipe in recipes]
fig,ax=plt.subplots(figsize=(14,6.1))
im=ax.imshow(mat,cmap='YlGnBu',vmin=.25,vmax=.95,aspect='auto')
ax.set_xticks(range(len(sites)),[s.replace('ABIDE_I::','').replace('ABIDE_II::','II-') for s in sites],rotation=55,ha='right',fontsize=8)
ax.set_yticks(range(len(recipes)),[labels[s] for s in recipes],fontsize=9)
ax.set_title('Held-site ROC-AUC by model family | original 720-person ABIDE-I suite',fontsize=13,fontweight='bold',pad=14)
ax.set_xlabel('Held-out site family  •  Attention rows average two available seeds')
fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label='Site ROC-AUC')
fig.tight_layout();export(fig,'Figure_S1_17_Site_Model_AUC_Heatmap')

orig=[r for r in summary if r['stage']=='original_10_arm_suite']
combo=[r for r in summary if r['stage']=='retrospective_ensemble_comparisons']
fig,ax=plt.subplots(figsize=(9.6,6.8))
for rs,label,color,marker in [(orig,'Original fitted model-seed summaries',BLUE,'o'),(combo,'Retrospective prediction mixtures',TEAL,'D')]:
    ax.scatter([float(r['site_macro_auc']) for r in rs],[float(r['site_macro_balanced_accuracy']) for r in rs],s=75,alpha=.83,color=color,marker=marker,edgecolor='white',linewidth=.6,label=label)
for rec,name,dx,dy in [('v4_pca128','PCA-128 reference',-20,12),('selected_equal','Selected equal',-50,-19),('selected_gate','Adaptive gate',8,-12),('blend_gate_pca_half','Post-hoc gate+PCA',-135,9),('blend_selected_gate_half','First-choice mixture',0,14)]:
    for r in orig+combo:
        if r['recipe']==rec:
            ax.annotate(name,(float(r['site_macro_auc']),float(r['site_macro_balanced_accuracy'])),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=SLATE,arrowprops={'arrowstyle':'-','color':'#AAB2BD','lw':.6} if abs(dx)>30 else None)
ax.set_xlabel('Site-macro ROC-AUC');ax.set_ylabel('Site-macro balanced accuracy at threshold 0.5')
ax.set_title('AUC and balanced-accuracy tradeoff | exploratory comparison',fontweight='bold',pad=13)
ax.grid(alpha=.14);ax.legend(loc='lower left',fontsize=8);ax.set_xlim(.585,.735);ax.set_ylim(.545,.665)
fig.tight_layout();export(fig,'Figure_S2_Ensemble_AUC_vs_Balanced_Accuracy')

for ft,name,title,color in [('anat','Figure_S3_Structural_Selection_Frequency','Structural feature recurrence | selected-equal model',BLUE),('fc','Figure_S4_FC_Edge_Selection_Frequency','AAL116 FC edge-selection recurrence | selected-equal model',TEAL)]:
    sel=[r for r in stability if r['model']=='selected_equal' and r['feature_type']==ft]
    freq=Counter(int(r['n_outer_model_folds_selected']) for r in sel)
    fig,ax=plt.subplots(figsize=(9.2,5.0))
    ax.bar(range(1,18),[freq.get(i,0) for i in range(1,18)],color=color,width=.8)
    ax.set_xticks(range(1,18));ax.set_ylabel('Number of distinct ever-selected features' if ft=='anat' else 'Number of distinct ever-selected FC edges')
    ax.set_xlabel('Number of outer training folds selecting the feature (out of 17)')
    ax.set_title(title,fontweight='bold',pad=12);ax.grid(axis='y',alpha=.17)
    if ft=='fc':
        assert len(sel)==1186 and freq[17]==112
        ax.text(.99,.96,'1,186 of 6,670 candidate edges selected at least once\n112 selected in all 17 folds',transform=ax.transAxes,va='top',ha='right',fontsize=9,color=SLATE,bbox=dict(facecolor='#E8F1F8',edgecolor='none',boxstyle='round,pad=.5'))
    fig.tight_layout();export(fig,name)
print('Recreated Figures S1-S4 directly from existing aggregate CSVs.')
