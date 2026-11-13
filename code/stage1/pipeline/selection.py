"""Leakage-safe site-aware, supervised anatomical and FC feature ranking.

IMPORTANT: fit() may only receive participants in the ACTIVE training fold.
No selected feature set is reused between different outer folds.
Scores are predictive associations, NOT causal/clinical biomarkers.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from vendor.v4.controls import Reference

UPPER = np.triu_indices(116, 1)


def _site_effects(x, y, sites):
    """Site-conditional class differences with shrinkage on tiny families.

    Use training sites only. Effects are averaged over sites, not pooled across
    site confounding. No test-site labels or site means enter this function.
    """
    effects, weights = [], []
    for site in np.unique(sites):
        idx = np.flatnonzero(sites == site)
        a, b = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(a) < 2 or len(b) < 2: continue
        n_eff = len(a) * len(b) / (len(a) + len(b))
        # global fold-fitted scaling makes effects roughly comparable
        delta = np.mean(x[a],axis=0) - np.mean(x[b],axis=0)
        effects.append(np.clip(delta, -5, 5))
        weights.append(min(np.sqrt(n_eff), 4.))
    if len(effects) < 3:
        raise ValueError('Not enough training site families with both diagnoses')
    return np.asarray(effects), np.asarray(weights)


def _stable_site_score(x, y, sites, seed, n_resamples=10, top=256):
    effects, wt = _site_effects(x, y, sites)
    rng=np.random.default_rng(seed)
    wt=wt / wt.sum()
    full=np.average(effects, axis=0, weights=wt)
    # Penalize inconsistent signs, without requiring unanimity on small sites.
    agree=np.maximum((effects>0).mean(axis=0),(effects<0).mean(axis=0))
    score=np.abs(full) * np.maximum(agree,0.5)**1.5
    freq=np.zeros(x.shape[1],dtype=float)
    for _ in range(n_resamples):
        ix=rng.choice(len(effects),size=len(effects),replace=True)
        tmp=np.average(effects[ix],axis=0,weights=wt[ix])
        chosen=np.argsort(-np.abs(tmp),kind='stable')[:min(top,x.shape[1])]
        freq[chosen]+=1
    freq/=max(n_resamples,1)
    # combine continuous site-conditional effects + resampling stability
    norm=score/(np.max(score)+1e-12)
    return (0.82*norm + 0.18*freq),freq,effects


def _smri_l1_stability(x,y,sites,seed,n_boot=8):
    rng=np.random.default_rng(seed)
    families=np.unique(sites)
    freq=np.zeros(x.shape[1],dtype=float)
    for j in range(n_boot):
        picked=rng.choice(families,size=len(families),replace=True)
        idx=np.concatenate([np.flatnonzero(sites==s) for s in picked])
        if len(np.unique(y[idx]))!=2:continue
        model=LogisticRegression(penalty='l1',solver='liblinear',C=.08,
                                 class_weight='balanced',max_iter=1000,random_state=seed+j)
        model.fit(x[idx],y[idx])
        freq+=(np.abs(model.coef_[0])>1e-8).astype(float)
    return freq/n_boot


def _roi_candidate_order(edge_score, n_roi):
    strengths=np.zeros(116,dtype=float)
    for i in range(116):
        incidence=(UPPER[0]==i) | (UPPER[1]==i)
        s=edge_score[incidence]
        strengths[i]=np.mean(np.sort(s)[-min(len(s),10):])
    rois=np.argsort(-strengths,kind='stable')[:n_roi]
    candidate=np.isin(UPPER[0],rois) | np.isin(UPPER[1],rois)
    return np.flatnonzero(candidate),rois


@dataclass
class SelectionEvidence:
    anat_indices: np.ndarray
    fc_indices: np.ndarray
    anat_score: np.ndarray
    fc_score: np.ndarray
    anat_l1_frequency: np.ndarray
    fc_site_frequency: np.ndarray
    roi_indices: np.ndarray


class FoldFeatureSelector:
    def __init__(self,k_a=40,k_f=256,fc_mode='edges',seed=17,n_boot=8):
        if fc_mode not in ('edges','roi','full_pca'):raise ValueError(fc_mode)
        self.k_a,self.k_f,self.fc_mode,self.seed,self.n_boot = k_a,k_f,fc_mode,seed,n_boot

    def fit(self,anat,fc,sites,y,meta):
        anat=np.asarray(anat,dtype=float)
        fc=np.asarray(fc,dtype=float)
        sites=np.asarray(sites)
        y=np.asarray(y,dtype=int)
        meta=np.asarray(meta,dtype=float)
        if meta.shape!=(len(y),6):raise ValueError('Six known acquisition/demographic covariates required')
        # Learn confound reference only on TD training participants; the reference
        # transforms held-out data using pre-fitted coefficients (no held-site fit).
        self.a_reference=Reference('bio_acq').fit(anat,meta,sites,y)
        self.f_reference=Reference('bio_acq').fit(fc,meta,sites,y)
        anat=self.a_reference.transform(anat,meta,sites)
        fc=self.f_reference.transform(fc,meta,sites)
        if len(np.unique(sites))<3 or len(np.unique(y))!=2:raise ValueError('Bad training data')
        self.a_imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(anat)
        xa=self.a_imp.transform(anat)
        self.a_scaler=StandardScaler().fit(xa)
        a=np.clip(self.a_scaler.transform(xa),-6,6)
        self.f_scaler=StandardScaler().fit(fc)
        f=np.clip(self.f_scaler.transform(fc),-6,6)
        a_site,_,_= _stable_site_score(a,y,sites, self.seed, self.n_boot, top=40)
        a_l1 = _smri_l1_stability(a,y,sites,self.seed,self.n_boot)
        a_score=0.60*a_site/(a_site.max()+1e-12)+0.40*a_l1
        f_score, f_freq, _ = _stable_site_score(f,y,sites,self.seed+11,
                                                n_resamples=self.n_boot,top=256)
        ai=np.argsort(-a_score,kind='stable')[:min(self.k_a,a.shape[1])]
        rois=np.array([],dtype=int)
        if self.fc_mode=='roi':
            candidates,rois=_roi_candidate_order(f_score, min(30,116))
            fi=candidates[np.argsort(-f_score[candidates],kind='stable')[:min(self.k_f,len(candidates))]]
        else:
            fi=np.argsort(-f_score,kind='stable')[:min(self.k_f,f.shape[1])]
        self.evidence=SelectionEvidence(ai,fi,a_score,f_score,a_l1,f_freq,rois)
        return self

    def transform(self,anat,fc,meta):
        meta=np.asarray(meta,dtype=float)
        # No fit or access to labels happens on the target / held-out partition.
        anat=self.a_reference.transform(np.asarray(anat),meta,np.array(['UNSEEN']*len(meta)))
        fc=self.f_reference.transform(np.asarray(fc),meta,np.array(['UNSEEN']*len(meta)))
        a=np.clip(self.a_scaler.transform(self.a_imp.transform(anat)),-6,6)
        f=np.clip(self.f_scaler.transform(fc),-6,6)
        return (np.asarray(a[:,self.evidence.anat_indices],np.float32),
                np.asarray(f[:,self.evidence.fc_indices],np.float32))
