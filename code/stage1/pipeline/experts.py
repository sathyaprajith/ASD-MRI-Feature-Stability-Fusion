"""Fold-contained feature selected logistic experts and legacy-space full baseline."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pipeline.selection import FoldFeatureSelector
from pipeline.features import inverse_sqrt_site_weights


class SelectedExperts:
    def __init__(self,k_a=40,k_f=256,fc_mode='edges',seed=17,nb=8):
        self.k_a,self.k_f,self.fc_mode,self.seed,self.nb=k_a,k_f,fc_mode,seed,nb

    def fit(self,cohort,idx):
        a,f,y,sites=cohort.anat[idx],cohort.fc[idx],cohort.y[idx],cohort.sites[idx]
        self.selector=FoldFeatureSelector(self.k_a,self.k_f,self.fc_mode,self.seed,self.nb).fit(a,f,sites,y,cohort.meta[idx])
        aa,ff=self.selector.transform(a,f,cohort.meta[idx])
        weights=inverse_sqrt_site_weights(sites,y)
        self.am=LogisticRegression(C=.025,solver='liblinear',max_iter=2500,random_state=self.seed).fit(aa,y,sample_weight=weights)
        self.fm=LogisticRegression(C=.075,solver='liblinear',max_iter=2500,random_state=self.seed).fit(ff,y,sample_weight=weights)
        return self

    def predict(self,cohort,idx):
        aa,ff=self.selector.transform(cohort.anat[idx],cohort.fc[idx],cohort.meta[idx])
        return self.am.predict_proba(aa)[:,1],self.fm.predict_proba(ff)[:,1]


class FullInputPreprocessor:
    """Train-only TD-reference adjustment followed by 184 anatomy + FC PCA128."""
    def __init__(self,n_fc=128): self.n_fc=n_fc

    def fit(self,cohort,idx):
        from sklearn.impute import SimpleImputer
        from vendor.v4.controls import Reference
        y=cohort.y[idx]; sites=cohort.sites[idx]; meta=cohort.meta[idx]
        self.ar=Reference('bio_acq').fit(cohort.anat[idx],meta,sites,y)
        self.fr=Reference('bio_acq').fit(cohort.fc[idx],meta,sites,y)
        a=self.ar.transform(cohort.anat[idx],meta,sites)
        f=self.fr.transform(cohort.fc[idx],meta,sites)
        self.ai=SimpleImputer(strategy='median',keep_empty_features=True).fit(a)
        self.asc=StandardScaler().fit(self.ai.transform(a))
        self.fsc=StandardScaler().fit(f)
        ff=np.clip(self.fsc.transform(f),-6,6)
        self.pca=PCA(n_components=min(self.n_fc,len(idx)-1,cohort.fc.shape[1]),
                     svd_solver='randomized',random_state=17).fit(ff)
        self.psc=StandardScaler().fit(self.pca.transform(ff))
        return self

    def transform(self,cohort,idx):
        meta=cohort.meta[idx];sites=cohort.sites[idx]
        a0=self.ar.transform(cohort.anat[idx],meta,sites)
        f0=self.fr.transform(cohort.fc[idx],meta,sites)
        a=np.clip(self.asc.transform(self.ai.transform(a0)),-6,6).astype('float32')
        ff=np.clip(self.fsc.transform(f0),-6,6)
        f=self.psc.transform(self.pca.transform(ff)).astype('float32')
        return a,f
