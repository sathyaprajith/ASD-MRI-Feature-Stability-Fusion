"""All tests use synthetic participants ONLY (not ASD MRI results)."""
import importlib,sys,subprocess,zipfile,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from quick_ensembles import evaluate,load_one_model
from nested_search import Candidate,CANDIDATES,nested_choose,fit_final_predict,fuse
from compare_paired import paired


def dummy_data(nsites=7,n_per=12,seed=10):
    rng=np.random.default_rng(seed)
    n=nsites*n_per
    y=np.tile(np.tile([0,1],n_per//2),nsites)
    sites=np.repeat([f'ABIDE_I::SITE_{j}' for j in range(nsites)],n_per)
    idx=np.arange(n)
    # Train/held-out different site effects
    x_a=rng.normal(size=(n,184)).astype('float32')
    x_f=rng.normal(size=(n,6670)).astype('float32')
    x_a[:,3]+=.6*y
    x_f[:,8]+=.8*y
    meta=rng.normal(size=(n,6)).astype('float32')
    from types import SimpleNamespace
    return SimpleNamespace(anat=x_a,fc=x_f,meta=meta,y=y,sites=sites,
        subjects=np.array([f'COHORT_{i}' for i in range(n)]))


def create_fake_original(base,nsites=7,n_per=12):
    base=Path(base)
    root=base/'private_predictions';root.mkdir(parents=True)
    rng=np.random.default_rng(42)
    for model in ('selected_equal','selected_gate','v4_pca128'):
        for site_idx in range(nsites):
            s=f'SITE_{site_idx}'
            ids=[f'ID_{site_idx}_{j}' for j in range(n_per)]
            labels=[j%2 for j in range(n_per)]
            preds=np.clip(np.array(labels)*.3+.35+rng.normal(0,.15,n_per),.005,.995)
            df=pd.DataFrame({'subject_id':ids,'site':[f'ABIDE_I::{s}']*n_per,
               'label':labels,'prob':preds,'model':model,'seed':[17]*n_per})
            df.to_csv(root/f'{model}__s17__{s}.csv',index=False)
    return base


def test_quick_ensembles_synthetic(tmp_path):
    root=create_fake_original(tmp_path/'old')
    summary,site=evaluate(root,tmp_path/'public',7,84)
    assert len(summary)==8
    assert set(summary['n_site_families'])=={7}
    assert len(site)==56
    assert summary.site_macro_auc.between(0,1).all()


def test_quick_detects_misaligned_subject_labels(tmp_path):
    root=create_fake_original(tmp_path/'old')
    path=root/'private_predictions'/'selected_gate__s17__SITE_2.csv'
    d=pd.read_csv(path);d.loc[0,'label']=1-d.loc[0,'label'];d.to_csv(path,index=False)
    with pytest.raises(RuntimeError,match='MISMATCH'):
        evaluate(root,tmp_path/'pub',7,84)


def test_fixed_fusions_are_specified_and_deterministic():
    a=np.array([.1,.5,.9]);f=np.array([.7,.5,.3])
    assert np.allclose(fuse(a,f,.5,'prob'),[.4,.5,.6])
    assert np.all((fuse(a,f,.5,'logit')>0) &(fuse(a,f,.5,'logit')<1))
    assert len(CANDIDATES)==9


class MockSelector:
    def __init__(self,k_a,k_f,fc_mode,seed,n_boot):self.ka=k_a;self.kf=k_f
    def fit(self,a,f,sites,y,meta):
        from types import SimpleNamespace
        self.evidence=SimpleNamespace(anat_indices=np.arange(self.ka),fc_indices=np.arange(self.kf))
        self.mean_a=np.mean(a,axis=0);self.mean_f=np.mean(f,axis=0)
        return self
    def transform(self,a,f,meta):
        return (np.asarray(a[:,:self.ka],dtype=np.float32),
                np.asarray(f[:,:self.kf],dtype=np.float32))


def fake_inner_splits(y,sites,tr,seed,count):
    tr=np.asarray(tr)
    uniq=np.unique(sites[tr])
    for s in uniq[:count]:
        iv=tr[sites[tr]==s]
        it=tr[sites[tr]!=s]
        yield it,iv,[s]


def test_nested_train_site_isolation_synthetic():
    c=dummy_data()
    tr=np.flatnonzero(c.sites!='ABIDE_I::SITE_0')
    held=np.flatnonzero(c.sites=='ABIDE_I::SITE_0')
    adapter=(MockSelector,None,None,lambda sites,y:np.ones(len(y)),None,fake_inner_splits)
    choice,report=nested_choose(c,tr,seed=17,n_boot=2,n_inner=2,adapter=adapter)
    assert choice.name in [x.name for x in CANDIDATES]
    assert report['selected']['name']==choice.name
    assert all('SITE_0' not in '|'.join(x['held_sites']) for x in report['inner_folds'])
    pred,indices=fit_final_predict(c,tr,held,choice,17,2,adapter)
    assert len(pred)==len(held) and np.isfinite(pred).all()
    assert len(indices['structural_feature_indices']) == choice.k_anat


def test_paired_stratified_bootstrap_synthetic(tmp_path):
    root=create_fake_original(tmp_path/'old')
    a=load_one_model(root,'selected_equal',7,84)
    b=load_one_model(root,'v4_pca128',7,84)
    d=paired(a,b,draws=30)
    assert d['n_sites']==7 and d['n_subjects']==84
    assert len(d['participant_stratified_pair_bootstrap_CI_95'])==2
    with pytest.raises(ValueError):
        mutated=b.copy();mutated.loc[0,'subject_id']='wrong';paired(a,mutated,draws=5)


def test_export_never_includes_private(tmp_path):
    root=tmp_path/'results'
    for folder in ('quick','nested/private_predictions','nested/private_selection','nested/aggregates','paired'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    (root/'quick'/'PUBLIC.csv').write_text('x,y\n1,2\n')
    (root/'nested/private_predictions'/'people.csv').write_text('name,diagnosis\nsecret,1\n')
    (root/'nested/private_selection'/'selected.json').write_text('{"internal":42}')
    (root/'nested/aggregates'/'AUC.csv').write_text('auc\n0.7\n')
    archive=tmp_path/'safe.zip'
    subprocess.run([sys.executable,str(ROOT/'scripts/export_aggregate.py'),
      '--root',str(root),'--archive',str(archive)],check=True)
    with zipfile.ZipFile(archive) as z:
        files=z.namelist()
        assert z.testzip() is None
        assert any('PUBLIC.csv' in f for f in files)
        assert all('private' not in f.lower() for f in files)
        assert all('people.csv' not in f for f in files)
