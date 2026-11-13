"""End-to-end 10-model synthetic run, ZIP privacy, and restart check."""
import json,zipfile,sys
from pathlib import Path


def test_all_10_models_runner(monkeypatch,synthetic,tmp_path):
    import run
    monkeypatch.setattr(run,'load_cohort',lambda *a,**k:synthetic)
    args=['run.py','--pilot','--only-sites','S0',
          '--models','v4_anatomy75','v4_pca128','smri_stable',
          'fmri_edges','fmri_roi','selected_equal','selected_anatomy75',
          'selected_gate','selected_attention','full_attention',
          '--bootstraps','2','--epochs','8','--inner-splits','1',
          '--out',str(tmp_path/'results'),'--aggregate-zip',str(tmp_path/'aggregates.zip')]
    monkeypatch.setattr(sys,'argv',args)
    run.main()
    with zipfile.ZipFile(tmp_path/'aggregates.zip') as z:
        names=z.namelist()
        assert 'aggregates/PER_SITE_AUC.csv' in names
        assert 'RUN_CONTRACT.json' in names
        assert not any('private' in s or s.endswith('.joblib') for s in names)
    assert len(list((tmp_path/'results'/'private_predictions').glob('*.csv')))==12
    run.main()  # exact-repeat resume
    assert len(list((tmp_path/'results'/'private_predictions').glob('*.csv')))==12
