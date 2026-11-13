"""Acquisition-site-family grouped split checks for the archived 720 ABIDE-I pairs.

All 720 were previously used in development; these are exploratory estimates.
No family may be split across a training/assessment boundary.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 17
FAMILY_MERGES = {
    'ABIDE_I::LEUVEN_1': 'ABIDE_I::LEUVEN',
    'ABIDE_I::LEUVEN_2': 'ABIDE_I::LEUVEN',
    'ABIDE_I::UCLA_1': 'ABIDE_I::UCLA',
    'ABIDE_I::UCLA_2': 'ABIDE_I::UCLA',
    'ABIDE_I::UM_1': 'ABIDE_I::UM',
    'ABIDE_I::UM_2': 'ABIDE_I::UM',
}


def site_families(site_ids):
    return np.asarray([FAMILY_MERGES.get(str(s), str(s)) for s in site_ids], dtype=str)


def checked_splits(labels, families, n_folds, seed=SEED, min_td=20):
    y = np.asarray(labels, dtype=int)
    f = np.asarray(families, dtype=str)
    if len(y) != len(f) or len(y) == 0:
        raise ValueError('Invalid labels/families lengths')
    if len(np.unique(f)) < n_folds:
        raise ValueError('Insufficient distinct site families for n_folds')
    parts = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                  random_state=seed).split(np.zeros((len(y), 1)), y, groups=f))
    held_all = []
    for i, (train, held) in enumerate(parts, 1):
        if not len(train) or not len(held):
            raise ValueError(f'Empty fold {i}')
        if set(f[train]).intersection(f[held]):
            raise ValueError(f'Site family leakage in fold {i}')
        if set(y[train]) != {0, 1} or set(y[held]) != {0, 1}:
            raise ValueError(f'Missing class in fold {i}')
        if (y[train] == 0).sum() < min_td:
            raise ValueError(f'Not enough training TD anchors in fold {i}')
        held_all.extend(held.tolist())
    if sorted(held_all) != list(range(len(y))):
        raise ValueError('Participants missing from/multiplied in outer held folds')
    return parts


def assert_nested(labels, families, n_outer):
    outer = checked_splits(labels, families, n_outer)
    y = np.asarray(labels, dtype=int)
    fam = np.asarray(families, dtype=str)
    for i, (train, _) in enumerate(outer, 1):
        for j, (local_train, _) in enumerate(checked_splits(y[train],fam[train],3), 1):
            try:
                checked_splits(y[train][local_train],fam[train][local_train],3)
            except ValueError as e:
                raise ValueError(f'outer{i} headOOF{j}: {e}') from e
    return outer


def validate_roster(frame):
    required = {'subject_id','site_id','partition','asd_true'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing roster columns')
    if len(frame) != 720 or frame.subject_id.duplicated().any():
        raise ValueError('Must have exactly 720 distinct matched participants')
    if set(frame.partition) != {'train','val'}:
        raise ValueError('Unexpected historical partition')
    if frame.partition.value_counts().to_dict() != {'train':583,'val':137}:
        raise ValueError('Historical 583/137 composition changed')
    if frame.site_id.nunique() != 20 or site_families(frame.site_id).size != 720:
        raise ValueError('Site inventory changed')
    if len(set(site_families(frame.site_id))) != 17:
        raise ValueError('Site family inventory changed')
    if not set(frame.asd_true).issubset({0,1}) or frame.asd_true.sum() != 336:
        raise ValueError('Unexpected ASD/TD composition')
    if frame.groupby(site_families(frame.site_id)).asd_true.nunique().min() != 2:
        raise ValueError('At least one family has only one diagnosis class')
    return frame


def preflight(roster, fold_counts=(5,10)):
    r = validate_roster(roster)
    labels = r.asd_true.to_numpy(dtype=int)
    families = site_families(r.site_id)
    out = {'n':len(r),'asd':int(labels.sum()),'td':int((labels==0).sum()),
           'site_ids':int(r.site_id.nunique()),'families':int(len(set(families))),
           'historical_train':583,'historical_val_reused':137,
           'family_counts':[]}
    for f, group in r.assign(family=families).groupby('family'):
        out['family_counts'].append({'family':f, 'n':len(group),
                                     'asd':int(group.asd_true.sum()),
                                     'td':int((group.asd_true==0).sum())})
    for n in fold_counts:
        p=assert_nested(labels,families,n)
        out[f'cv{n}'] = [{'fold':i,'train':len(train),'held':len(held),
                           'asd':int(labels[held].sum()),'td':int((labels[held]==0).sum()),
                           'held_families':sorted(set(families[held]))}
                          for i,(train,held) in enumerate(p,1)]
    return out


def print_preflight(data):
    print(f"EXPLORED ABIDE-I PAIRED: {data['n']} | ASD {data['asd']} | TD {data['td']} | "
          f"{data['site_ids']} site IDs / {data['families']} families | "
          f"previously used validation {data['historical_val_reused']}")
    for n in (5,10):
        if f'cv{n}' in data:
            print(f'\n{n}-fold site-family grouped CV:')
            for r in data[f'cv{n}']:
                print(f"  fold {r['fold']:2d}: fit {r['train']:3d}, assess {r['held']:3d} "
                      f"(ASD {r['asd']:2d}, TD {r['td']:2d}) | {', '.join(r['held_families'])}")
            if any(len(r['held_families'])==1 for r in data[f'cv{n}']):
                print(' WARNING: single-site-family assessment fold(s); report by-family metrics.')
