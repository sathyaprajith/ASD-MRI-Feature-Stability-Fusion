import numpy as np
from pipeline.selection import FoldFeatureSelector
from pipeline.splits import outer_loso,inner_site_splits
from pipeline.tuning import tune_train_only
from pipeline.experts import SelectedExperts
from experiments.fusion_gated.model import AdaptiveGate
from experiments.fusion_attention.model import AttentionExperiment
from experiments.reference_v4.model import V4Reference


def test_outer_and_inner_do_not_mix_sites(synthetic):
    c=synthetic
    site,tr,held=next(iter(outer_loso(c)))
    assert not set(c.sites[tr]) & set(c.sites[held])
    for itr,iv,s in inner_site_splits(c.y,c.sites,tr,seed=17):
        assert not set(c.sites[itr]) & set(c.sites[iv])
        assert not set(c.sites[iv]) & set(c.sites[held])


def test_no_held_label_leakage(synthetic):
    c=synthetic
    _,tr,held=next(iter(outer_loso(c)))
    s1=FoldFeatureSelector(20,64,seed=17,n_boot=2).fit(c.anat[tr],c.fc[tr],c.sites[tr],c.y[tr],c.meta[tr])
    y2=c.y.copy();y2[held]=1-y2[held]
    s2=FoldFeatureSelector(20,64,seed=17,n_boot=2).fit(c.anat[tr],c.fc[tr],c.sites[tr],y2[tr],c.meta[tr])
    assert np.array_equal(s1.evidence.anat_indices,s2.evidence.anat_indices)
    assert np.array_equal(s1.evidence.fc_indices,s2.evidence.fc_indices)


def test_selected_experts_both_variants(synthetic):
    c=synthetic;_,tr,held=next(iter(outer_loso(c)))
    for mode in ('edges','roi'):
        model=SelectedExperts(20,64,mode,seed=17,nb=2).fit(c,tr)
        pa,pf=model.predict(c,held)
        assert pa.shape==pf.shape==(len(held),)
        assert np.all((pa>0)&(pa<1)) and np.all((pf>0)&(pf<1))
        assert len(model.selector.evidence.fc_indices)==64


def test_nested_candidate_choices(synthetic):
    c=synthetic;_,tr,_=next(iter(outer_loso(c)))
    opts=tune_train_only(c,tr,seed=17,n_boot=2,inner_splits=1)
    assert set(opts)=={'smri_stable','fmri_edges','fmri_roi','fusion'}
    assert all(int(opt['k_fc']) in (64,256,512) for opt in opts.values())


def test_oof_gate_and_original_v4(synthetic):
    c=synthetic;_,tr,held=next(iter(outer_loso(c)))
    gate=AdaptiveGate(20,64,seed=17,nb=2).fit(c,tr)
    pr,w=gate.predict(c,held)
    assert len(pr)==len(held) and np.all((w>=0)&(w<=1))
    for arm in ('v4_anatomy75','v4_pca128'):
        m=V4Reference(arm).fit(c,tr)
        p=m.predict(c,held)
        assert len(p)==len(held) and np.isfinite(p).all()


def test_attention_selection_and_full(synthetic):
    c=synthetic;_,tr,held=next(iter(outer_loso(c)))
    for full in (False,True):
        model=AttentionExperiment(full=full,ka=20,kf=64,seed=17,n_boot=2,
                                  inner_splits=1,max_epochs=8,device='cpu').fit(c,tr)
        pr,w=model.predict(c,held)
        assert np.all(np.isfinite(pr)) and pr.shape==(len(held),)
        assert np.allclose(w.sum(axis=1),1.,atol=1e-4)
        assert model.final_epochs>=6
