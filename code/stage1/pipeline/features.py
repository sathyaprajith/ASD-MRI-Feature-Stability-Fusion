"""Fit all feature transformations only on the current training fold."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

NROI = 116
UP = np.triu_indices(NROI,1)


def roi_matrices(fc):
    """Fixed, label-free Fisher-z connectivity -> bounded Pearson graph tensors."""
    fc = np.asarray(fc, dtype=np.float32)
    if fc.ndim != 2 or fc.shape[1] != 6670:
        raise ValueError('Expected 6670 AAL116 upper-triangle edges')
    corr = np.tanh(np.clip(fc, -6, 6))
    arr = np.zeros((len(fc), NROI, NROI), dtype=np.float32)
    arr[:,UP[0],UP[1]] = corr
    arr[:,UP[1],UP[0]] = corr
    return arr


def signed_edges(matrix, k=12):
    """Per-person, top-k strongest positive/negative edges; row normalized.

    Each adjacency is built from the test participant's own unlabelled fMRI
    scan, not from other held-site subjects or held-site labels.
    """
    a = np.asarray(matrix, dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (116,116): raise ValueError('Expected [N,116,116]')
    # Union top-k graph for symmetry; no target labels involved.
    weights = np.abs(a)
    idx = np.argpartition(weights, -k, axis=2)[:,:,-k:]
    mask = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=2)
    mask |= mask.transpose(0,2,1)
    w = np.where(mask, a, 0.0)
    pos = np.clip(w,0,None)
    neg = np.clip(-w,0,None)
    pos /= np.maximum(pos.sum(axis=2,keepdims=True),1e-5)
    neg /= np.maximum(neg.sum(axis=2,keepdims=True),1e-5)
    return pos, neg


class FoldPreprocessor:
    def __init__(self, n_pca=96):
        self.n_pca = n_pca

    def fit(self, anat, fc):
        self.imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(anat)
        a = self.imp.transform(anat)
        self.asc = StandardScaler().fit(a)
        self.fsc = StandardScaler().fit(fc)
        f = np.clip(self.fsc.transform(fc),-8,8)
        self.pca = PCA(n_components=min(self.n_pca,len(fc)-1,fc.shape[1]),
                       svd_solver='randomized', n_oversamples=12, iterated_power=3, random_state=101).fit(f)
        self.pcsc = StandardScaler().fit(self.pca.transform(f))
        return self

    def transform(self, anat, fc):
        a = np.clip(self.asc.transform(self.imp.transform(anat)), -6, 6).astype(np.float32)
        f = self.pcsc.transform(self.pca.transform(np.clip(self.fsc.transform(fc),-8,8)))
        f=np.clip(f,-6,6).astype(np.float32)
        if f.shape[1]<self.n_pca:
            f=np.pad(f,((0,0),(0,self.n_pca-f.shape[1])),mode='constant')
        return a,f


def inverse_sqrt_site_weights(sites, y):
    """Training-only group-balanced weights with mild diagnostic rebalancing."""
    sites = np.asarray(sites)
    y = np.asarray(y)
    uniq,inv,counts = np.unique(sites,return_inverse=True,return_counts=True)
    sw=1/np.sqrt(counts[inv].astype(np.float64))
    n0,n1=max((y==0).sum(),1),max((y==1).sum(),1)
    cw=np.where(y==1,np.sqrt((len(y)/2)/n1),np.sqrt((len(y)/2)/n0))
    w=np.clip(sw*cw,1e-6,None)
    w=w/w.mean()
    return np.clip(w,0.2,4).astype(np.float32)
