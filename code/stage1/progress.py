#!/usr/bin/env python3
"""Read-only compact status of all models, no log scrolling."""
from pathlib import Path
import pandas as pd
from experiments.registry import EXPERIMENTS,DL
root=Path('/home/jovyan/ASD_FEATURE_ATTENTION_RESULTS')
files=list((root/'private_predictions').glob('*.csv'))
print('\n   FEATURE SELECTION + ATTENTION STUDY PROGRESS')
print('   '+'-'*67)
print(f"   {'MODEL':<24} {'FOLDS':>8} {'MACRO AUC':>13}  STATUS")
print('   '+'-'*67)
if not files:
    print('   No completed folds yet. Check training log for preflight errors.')
else:
    s=root/'aggregates'/'MODEL_SEED_SUMMARY.csv'
    df=pd.read_csv(s) if s.exists() else pd.DataFrame()
    for model in EXPERIMENTS:
        n=len(list((root/'private_predictions').glob(f'{model}__s*__*.csv')))
        expected=34 if model in DL else 17
        rows=df[(df.model==model)&(df.complete)] if len(df) else pd.DataFrame()
        score=f'{rows.site_macro_auc.mean():.4f}' if len(rows) else '--'
        status='COMPLETE' if n==expected else 'PARTIAL' if n else 'PENDING'
        print(f'   {model:<24} {f"{n}/{expected}":>8} {score:>13}  {status}')
    total=sum(len(list((root/'private_predictions').glob(f'{model}__s*__*.csv'))) for model in EXPERIMENTS)
    print('   '+'-'*67)
    print(f'   Overall: {total}/204 folds completed ({100*total/204:.1f}%).')
    print('   AUC appears only after all 17 held-site evaluations of a seed.')
    print('   Previous archived reference: 0.6770 macro AUC (V4 PCA128).')
print()
