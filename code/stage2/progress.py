#!/usr/bin/env python3
"""Read-only progress; no training or mutation."""
from pathlib import Path
import pandas as pd
base=Path('/home/jovyan/ASD_TO_070_RESULTS')
print('\nASD AUC >= 0.700 RESEARCH CHECK — exploratory only')
quick=base/'quick/EXPLORATORY_FIXED_ENSEMBLES.csv'
if quick.exists():
    df=pd.read_csv(quick)
    print('\n[Stage 1: fixed-weight ensemble on previously held-out site scores]')
    for _,row in df.iterrows():
        print(f"  {row['model']:<43} macro-AUC={row['site_macro_auc']:.4f}")
else:print('\n[Stage 1] Pending')
summary=base/'nested/aggregates/MODEL_SEED_RESULTS.csv'
if summary.exists():
    d=pd.read_csv(summary)
    print('\n[Stage 2: train-site nested feature/fusion search]')
    for _,row in d.iterrows():
        score=f"{row['macro_auc']:.4f}" if row['complete_17_site'] else 'not a full-site score'
        print(f"  seed={int(row['seed'])} sites={int(row['completed_sites'])}/17 full macro-AUC={score}")
else:print('\n[Stage 2] Pending')
paired=base/'paired/PAIRED_INFERENCE.json'
print('\n[Stage 3]', 'Paired evaluation present' if paired.exists() else 'Pending')
print('Share: /home/jovyan/ASD_TO_070_FINAL_AGGREGATES.zip')
print('Never judge success from a partial 17-site run or from the best individual fold.')
