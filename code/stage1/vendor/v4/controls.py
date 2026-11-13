"""Training-fitted TD reference residualization, not ComBat or image harmonization."""
from collections import Counter
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression

class Reference:
    def __init__(self,mode):self.mode=mode
    def design(self,meta,sites,fit=False):
        # age, recorded male flag, log1p(ICV), FD, log usable frames, tensor TR
        columns=[meta[:,0],meta[:,0]**2,meta[:,1],meta[:,2]]
        if self.mode in ('bio_acq','bio_acq_site'):columns.extend([meta[:,3],meta[:,4],meta[:,5]])
        values=np.column_stack(columns)
        if fit:self.numeric_scaler=StandardScaler().fit(values)
        result=self.numeric_scaler.transform(values)
        if self.mode=='bio_acq_site':
            if fit:self.site_counts=dict(Counter(sites));self.site_vocab=sorted(s for s,n in self.site_counts.items() if n>=5)
            onehot=np.column_stack([sites==s for s in self.site_vocab]).astype(float) if self.site_vocab else np.empty((len(sites),0))
            result=np.column_stack([result,onehot])
        return result
    def fit(self,x,meta,sites,y):
        anchors=np.flatnonzero(y==0) if self.mode!='raw' else np.arange(len(y))
        if self.mode!='raw' and len(anchors)<20:raise ValueError('Fewer than 20 fitting TD reference participants')
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x[anchors])
        self.target_scaler=StandardScaler().fit(self.imputer.transform(x[anchors]))
        if self.mode!='raw':
            design=self.design(meta[anchors],sites[anchors],fit=True)
            self.regression=Ridge(alpha=10.).fit(design,self.target_scaler.transform(self.imputer.transform(x[anchors])))
            self.anchor_age_range=[float(meta[anchors,0].min()),float(meta[anchors,0].max())]
        self.n_anchors=len(anchors)
        return self
    def transform(self,x,meta,sites):
        result=self.target_scaler.transform(self.imputer.transform(x))
        if self.mode!='raw':result=result-self.regression.predict(self.design(meta,sites))
        return result

def fit_model(x,meta,sites,y,mode,c):
    prep=Reference(mode).fit(x,meta,sites,y)
    residual=prep.transform(x,meta,sites);scale=StandardScaler().fit(residual)
    model=LogisticRegression(C=c,solver='liblinear',max_iter=5000,random_state=17).fit(scale.transform(residual),y)
    return dict(prep=prep,scale=scale,model=model)

def predict(model,x,meta,sites):
    residual=model['prep'].transform(x,meta,sites)
    return model['model'].decision_function(model['scale'].transform(residual))

def metadata_design(meta,sites,fit=None):
    if fit is None:fit=dict(scale=StandardScaler().fit(meta),sites=sorted(set(sites)))
    site=np.column_stack([sites==s for s in fit['sites']]).astype(float)
    return np.column_stack([fit['scale'].transform(meta),site]),fit
