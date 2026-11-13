"""Site-held-out outer splits and inner site-held-out model-selection splits."""
import numpy as np


def outer_loso(cohort, only_sites=None):
    sites = sorted(set(cohort.sites))
    if only_sites:
        allowed = {s if s.startswith('ABIDE_I::') else f'ABIDE_I::{s}' for s in only_sites}
        sites = [s for s in sites if s in allowed]
        if len(sites) != len(allowed):
            raise ValueError(f'Unknown held-out sites: {allowed - set(sites)}')
    for site in sites:
        held = np.flatnonzero(cohort.sites == site)
        train = np.flatnonzero(cohort.sites != site)
        assert len(set(cohort.sites[train]).intersection(set(cohort.sites[held]))) == 0
        assert len(held) and set(cohort.y[held]) == {0,1}
        yield site, train, held


def inner_site_splits(y, sites, train_idx, seed=17, count=2, sites_per_split=2):
    """Deterministic disjoint site-pair validation groups drawn ONLY from outer train.

    These splits are not selected by outer-site scores. Pairing is approximately
    balanced by site size and label distribution; no held-out-site information.
    """
    rng = np.random.default_rng(seed)
    uniq = sorted(set(sites[train_idx]))
    if len(uniq) < 6: raise ValueError('Need >=6 inner training families')
    summaries=[]
    for site in uniq:
        ix = train_idx[sites[train_idx] == site]
        summaries.append((site, len(ix), float(y[ix].mean())))
    # Partition shuffled sites into reproducible size-aware rank blocks.
    summaries.sort(key=lambda item: (item[1], item[2], item[0]))
    groups = [[], [], [], []]
    for j, block_start in enumerate(range(0, len(summaries), 4)):
        batch = [x[0] for x in summaries[block_start:block_start+4]]
        rng.shuffle(batch)
        for offset, site in enumerate(batch):
            groups[(offset+j)%4].append(site)
    val_groups=[]
    for k in range(count):
        chosen=[]
        for pool in (groups[k%4], groups[(k+2)%4], uniq):
            for site in pool:
                if site not in chosen:
                    chosen.append(site)
                    break
            if len(chosen) >= sites_per_split: break
        val_groups.append(chosen[:sites_per_split])
    for val_sites in val_groups:
        val = train_idx[np.isin(sites[train_idx], val_sites)]
        inner_train = train_idx[~np.isin(sites[train_idx], val_sites)]
        if len(inner_train) < 50 or len(val) < 6 or set(y[val]) != {0,1}:
            raise RuntimeError(f'Invalid inner site split: {val_sites}')
        if set(sites[inner_train]) & set(sites[val]): raise RuntimeError('Inner site overlap')
        yield inner_train, val, val_sites
