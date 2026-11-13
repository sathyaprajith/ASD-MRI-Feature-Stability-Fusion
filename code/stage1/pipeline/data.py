"""Verified V4 ABIDE-I cache adapter. No subject data are shipped with this bundle."""
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'v4'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from matched_data import load_data, sha  # noqa: E402
from metadata import load_metadata  # noqa: E402
from cv_utils import site_families  # noqa: E402

EXPECTED_ROSTER_SHA = '124f9dd311d3038417687ccddf14047fc625e133337b831acccca2384972384e'

@dataclass
class Cohort:
    subjects: np.ndarray
    sites: np.ndarray
    raw_sites: np.ndarray
    y: np.ndarray
    anat: np.ndarray
    fc: np.ndarray
    meta: np.ndarray
    names: list


def load_cohort(anatomy_root, fmri_root, audit_root):
    anatomy_root, fmri_root = Path(anatomy_root), Path(fmri_root)
    audit_root = Path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    rows, anat185, fc = load_data(anatomy_root, fmri_root, audit_root)
    if sha(audit_root / 'matched_cohort.csv') != EXPECTED_ROSTER_SHA:
        raise RuntimeError('The participant roster differs from verified V4. Stop; never silently redefine the cohort.')
    meta = load_metadata(rows, anatomy_root, audit_root)
    names = json.loads((audit_root / 'anatomy_feature_names.json').read_text())
    keep = [i for i,n in enumerate(names) if n != 'EstimatedTotalIntraCranialVol']
    subjects = np.asarray([r['subject_id'] for r in rows])
    raw_sites = np.asarray([r['site_id'] for r in rows])
    sites = site_families(raw_sites)
    y = np.asarray([int(r['asd_true']) for r in rows], dtype=np.int64)
    anat = np.asarray(anat185[:, keep], np.float32)
    fc = np.asarray(fc, np.float32)
    meta = np.asarray(meta, np.float32)
    if not (len(rows) == 720 and len(set(subjects)) == 720 and len(set(sites)) == 17
            and y.sum() == 336 and anat.shape == (720, 184)
            and fc.shape == (720, 6670) and meta.shape == (720, 6)):
        raise RuntimeError('Unexpected ABIDE-I dataset shape, subject labels, or family count')
    if np.isinf(anat).any() or not np.isfinite(fc).all() or not np.isfinite(meta).all():
        raise RuntimeError('Nonfinite inherited cache values')
    return Cohort(subjects, sites, raw_sites, y, anat, fc, meta, [names[i] for i in keep])


def feature_fingerprint(cohort):
    h = hashlib.sha256()
    for arr in (cohort.subjects, cohort.sites):
        h.update('\n'.join(map(str, arr.tolist())).encode())
    for arr in (cohort.y, cohort.anat, cohort.fc, cohort.meta):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
