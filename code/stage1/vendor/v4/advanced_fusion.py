"""Fold-contained advanced multimodal fusion ablations.

All fitting (including the gated fusion out-of-fold head) occurs exclusively on
outer TRAIN participants. No held-site data is used for scaling or fitting.
"""
import numpy as np
from scipy.special import logit, expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from cv_utils import site_families
from controls import Reference

ADVANCED_ARMS=('bilinear_pca_fusion','oof_quality_gate')


class AdvancedFusion:
    def __init__(self, name):
        if name not in ADVANCED_ARMS: raise ValueError(name)
        self.name=name

    def fit(self,anat,fc,meta,site,y):
        anat=np.asarray(anat,float); fc=np.asarray(fc,float);meta=np.asarray(meta,float)
        site=np.asarray(site,str);y=np.asarray(y,int)
        if len(y)<50 or (y==0).sum()<20:raise ValueError('Insufficient training subjects')
        if self.name=='oof_quality_gate':
            # Cross-fitted base predictions are strictly out of fitting-fold.
            # Inner splits stratify by label; outer folds remain site-held-out.
            # Site-grouped inner OOF is preferable when viable but 3-site top4
            # training can have a TD-short inner fold; this uses fixed CV3.
            from feature_suite import FittedArm
            oofa=np.empty(len(y));ooff=np.empty(len(y))
            # Prefer site-family-held internal OOF for domain generalization.
            # Very small restricted training pools may not support enough TD
            # anchors per inner fit; log the fallback to subject-level OOF.
            try:
                fam=site_families(site)
                if len(set(fam))<3:raise ValueError('Fewer than three families')
                inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=17).split(anat,y,groups=fam))
                if any((y[tr]==0).sum()<20 or len(set(y[tr]))<2 or len(set(y[va]))<2 for tr,va in inner):
                    raise ValueError('Insufficient class coverage for site-grouped OOF')
                self.inner_gate_cv='grouped3'
            except ValueError:
                cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=17)
                inner=list(cv.split(anat,y));self.inner_gate_cv='stratified3_fallback'
            for tr,va in inner:
                model=FittedArm('logistic_equal').fit(anat[tr],fc[tr],meta[tr],site[tr],y[tr])
                pred=model.predict(anat[va],fc[va],meta[va],site[va])
                oofa[va]=pred['anatomy'];ooff[va]=pred['fc']
            self.meta_scaler=StandardScaler().fit(meta[:,[3,4,5]])
            x=self._gate_features(oofa,ooff,meta)
            # Gate is an OOF-trained regularized logistic STACKING head,
            # NOT a convex attention weight or a learned spatial ROI attention.
            self.head=LogisticRegression(C=.08,solver='liblinear',max_iter=2000,
                                         random_state=17).fit(x,y)
            self.base=FittedArm('logistic_equal').fit(anat,fc,meta,site,y)
            return self
        self.ar=Reference('bio_acq_site').fit(anat,meta,site,y)
        self.fr=Reference('bio_acq_site').fit(fc,meta,site,y)
        a0=self.ar.transform(anat,meta,site);f0=self.fr.transform(fc,meta,site)
        self.a_scale=StandardScaler().fit(a0);self.f_scale=StandardScaler().fit(f0)
        aa=self.a_scale.transform(a0); ff=self.f_scale.transform(f0)
        self.a_pca=PCA(n_components=min(12,len(y)-1,aa.shape[1]),svd_solver='randomized',random_state=17).fit(aa)
        self.f_pca=PCA(n_components=min(32,len(y)-1,ff.shape[1]),svd_solver='randomized',random_state=17).fit(ff)
        x=self._bilinear(aa,ff)
        self.head_scale=StandardScaler().fit(x)
        self.head=LogisticRegression(C=.04,solver='liblinear',max_iter=2000,random_state=17).fit(self.head_scale.transform(x),y)
        return self

    def _gate_features(self,pa,pf,meta):
        a=logit(np.clip(pa,.0001,.9999));f=logit(np.clip(pf,.0001,.9999))
        return np.column_stack((a,f,np.abs(a)-np.abs(f),a*f,
                                self.meta_scaler.transform(np.asarray(meta)[:,[3,4,5]])))

    def _bilinear(self,aa,ff):
        a=self.a_pca.transform(aa);f=self.f_pca.transform(ff)
        # Fixed orthogonal interactions, with regularization; NOT supervised feature-selection.
        interaction=np.einsum('ni,nj->nij',a,f).reshape(len(a),-1)
        return np.column_stack((a,f,interaction/np.sqrt(a.shape[1]*f.shape[1])))

    def predict(self,anat,fc,meta,site):
        if self.name=='oof_quality_gate':
            p=self.base.predict(anat,fc,meta,site)
            x=self._gate_features(p['anatomy'],p['fc'],meta)
            return {'equal':self.head.predict_proba(x)[:,1]}
        aa=self.a_scale.transform(self.ar.transform(anat,meta,site))
        ff=self.f_scale.transform(self.fr.transform(fc,meta,site))
        return {'equal':self.head.predict_proba(self.head_scale.transform(self._bilinear(aa,ff)))[:,1]}
