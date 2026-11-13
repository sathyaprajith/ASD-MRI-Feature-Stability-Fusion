#!/usr/bin/env python3
"""Plot REAL AAL-116 functional-edge selection frequencies, not synthetic connectivity.

Reads Additional_file_2_Feature_Selection_Stability.csv or
aggregates/FEATURE_SELECTION_STABILITY_BY_MODEL.csv from an archive/CSV.
Default model: selected_equal (17 outer LOSO folds).
"""
import argparse
import hashlib
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f'Missing plotting dependency: {exc}. Install numpy and matplotlib in the plotting environment.')

TARGETS = ('Additional_file_2_Feature_Selection_Stability.csv',
           'FEATURE_SELECTION_STABILITY_BY_MODEL.csv')

def locate_input(specified):
    if specified:
        p = Path(specified).expanduser().resolve()
        if not p.is_file(): sys.exit(f'Source does not exist: {p}')
        return p
    return Path(__file__).resolve().parents[1]/'results/aggregate/Additional_file_2_Feature_Selection_Stability.csv'

def read_table(p):
    if p.suffix.lower() == '.csv':
        return list(csv.DictReader(p.open(encoding='utf-8-sig', newline=''))), str(p)
    if p.suffix.lower() != '.zip': sys.exit(f'Expected .csv or .zip, got {p}')
    with zipfile.ZipFile(p) as z:
        matching = [n for n in z.namelist() if not n.endswith('/') and
                    n.split('/')[-1] in TARGETS]
        if not matching: sys.exit(f'No feature-selection CSV in {p}')
        n = next((n for n in matching if n.endswith('Additional_file_2_Feature_Selection_Stability.csv')), matching[0])
        with z.open(n) as data:
            rows = list(csv.DictReader(io.TextIOWrapper(data, encoding='utf-8-sig', newline='')))
    return rows, str(p) + '::' + n

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', help='Input CSV or containing ZIP. Auto-detect if omitted.')
    ap.add_argument('--model', default='selected_equal',
                    help='Model in stability CSV (default: selected_equal)')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1]/'figures/regenerated'))
    args = ap.parse_args()
    rows, source = read_table(locate_input(args.input))
    chosen = [r for r in rows if r['model'] == args.model and r['feature_type'] == 'fc']
    if not chosen: sys.exit(f'No FC rows for model {args.model!r}.')
    upper_i, upper_j = np.triu_indices(116, k=1)
    assert len(upper_i) == 6670
    matrix = np.zeros((116, 116), dtype=float)
    seen = set()
    fold_counts = set()
    for r in chosen:
        idx = int(r['feature_index_0based'])
        freq = int(r['n_outer_model_folds_selected'])
        folds = int(r['n_available_outer_model_folds'])
        if idx in seen: raise ValueError(f'Duplicate FC index: {idx}')
        if not (0 <= idx < 6670 and 0 <= freq <= folds):
            raise ValueError(f'Invalid FC index/frequency/folds: {idx}, {freq}, {folds}')
        i,j = int(upper_i[idx]),int(upper_j[idx])
        name = r.get('feature_name_or_edge','')
        match = re.fullmatch(r'AAL_ROI_(\d+)__AAL_ROI_(\d+)', name)
        if match and (int(match.group(1)),int(match.group(2))) != (i+1,j+1):
            raise ValueError(f'ROI name disagrees with upper triangle index {idx}: {name} vs ({i+1},{j+1})')
        matrix[i,j] = matrix[j,i] = freq
        seen.add(idx)
        fold_counts.add(folds)
    if fold_counts != {17}:
        raise ValueError(f'Expected 17 outer folds for selected model; found {fold_counts}')
    stable = sum(int(r['n_outer_model_folds_selected']) == 17 for r in chosen)
    all_selections = sum(int(r['n_outer_model_folds_selected']) for r in chosen)
    if args.model == 'selected_equal' and stable != 112:
        raise ValueError(f'Unexpected fully stable edge count: {stable} (expected 112)')

    np.fill_diagonal(matrix, np.nan)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e5e7eb')
    fig, ax = plt.subplots(figsize=(8.7, 7.6), dpi=160)
    heat = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=17, cmap=cmap,
                     interpolation='nearest', origin='upper')
    ticks = np.array([1,10,20,30,40,50,60,70,80,90,100,110,116])
    ax.set_xticks(ticks-1, labels=[str(v) for v in ticks])
    ax.set_yticks(ticks-1, labels=[str(v) for v in ticks])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('AAL region index (1–116)', fontsize=11)
    ax.set_ylabel('AAL region index (1–116)', fontsize=11)
    ax.set_title('Functional connectivity feature-selection stability', fontsize=14, fontweight='bold', pad=12)
    cb = fig.colorbar(heat, ax=ax, fraction=0.048, pad=0.024, ticks=[0,4,8,12,17])
    cb.set_label('Number of LOSO folds selecting each edge', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.text(0.49, 0.022,
             f'{len(seen):,} edges selected at least once; {stable} in all 17 folds. '
             'Light-gray diagonal marks excluded self-connections.', ha='center', fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.11)
    stem = f'Figure_5_FC_Selection_Stability_{args.model}'
    fig.savefig(out/f'{stem}.png', dpi=400, facecolor='white', bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    summary = {'source': Path(source).name, 'source_sha256': hashlib.sha256(locate_input(args.input).read_bytes()).hexdigest(), 'model': args.model, 'outer_folds': 17,
               'region_count': 116, 'candidate_edges': 6670,
               'edges_selected_at_least_once': len(seen),
               'edges_selected_in_all_17_folds': stable,
               'aggregate_feature_selection_occurrences': all_selections,
               'mean_selected_edges_per_fold': all_selections/17,
               'data_type': 'real archived experiment selection counts; no synthetic values'}
    (out/'PROVENANCE_AND_COUNTS.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print('\nPNG:', out/f'{stem}.png')
    print('PDF:', out/f'{stem}.pdf')

if __name__ == '__main__':
    main()
