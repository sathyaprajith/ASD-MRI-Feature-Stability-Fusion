"""Train-site cross-fitted, low-parameter participant-adaptive fusion gate.

No base model sees a participant while generating that participant's OOF gate
training score. No outer site participates in training the gate.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pipeline.experts import SelectedExperts
from pipeline.features import inverse_sqrt_site_weights


def logit(x):
    x=np.clip(x,1.e-5,1-1.e-5)
    return np.log(x/(1-x))


class AdaptiveGate:
    def __init__(self,ka,kf,seed=17,nb=8):
        self.ka,self.kf,self.seed,self.nb=ka,kf,seed,nb

    def _inputs(self,pa,pf,meta,fit=False):
        if fit:self.q_scaler=StandardScaler().fit(meta[:,3:5])
        q=np.clip(self.q_scaler.transform(meta[:,3:5]),-4,4)
        la,lf=logit(pa),logit(pf)
        basis=np.column_stack([np.ones(len(pa)),np.abs(la),np.abs(lf),q])
        return basis,la,lf

    def fit(self,c,tr):
        y=c.y[tr]
        oofa=np.empty(len(tr));ooff=np.empty(len(tr))
        groups=c.sites[tr]
        cv=GroupKFold(n_splits=3)
        for i,(itr,iva) in enumerate(cv.split(tr,y,groups)):
            it,iv=tr[itr],tr[iva]
            if set(c.sites[it]) & set(c.sites[iv]): raise RuntimeError('OOF group leak')
            e=SelectedExperts(self.ka,self.kf,seed=self.seed+17*i,nb=self.nb).fit(c,it)
            oofa[iva],ooff[iva]=e.predict(c,iv)
        basis,la,lf=self._inputs(oofa,ooff,c.meta[tr],fit=True)
        wt=inverse_sqrt_site_weights(c.sites[tr],y)
        def loss(theta):
            gate=expit(np.clip(basis@theta[:basis.shape[1]],-14,14))
            z=gate*la+(1-gate)*lf+theta[-1]
            likelihood=np.average(np.logaddexp(0,z)-y*z,weights=wt)
            prior=np.array([1.10,0,0,0,0,0],float)
            return float(likelihood + .06*np.sum((theta-prior)**2))
        initial=np.array([1.1,0,0,0,0,0],dtype=float)
        opt=minimize(loss,initial,method='L-BFGS-B',options={'maxiter':200})
        if not opt.success and not np.isfinite(opt.fun):raise RuntimeError('Adaptive gate optimizer failed')
        self.theta=opt.x
        self.base=SelectedExperts(self.ka,self.kf,seed=self.seed,nb=self.nb).fit(c,tr)
        self.train_oof_auc=None
        return self

    def predict(self,c,idx):
        pa,pf=self.base.predict(c,idx)
        basis,la,lf=self._inputs(pa,pf,c.meta[idx])
        g=expit(np.clip(basis@self.theta[:-1],-14,14))
        p=expit(g*la+(1-g)*lf+self.theta[-1])
        return p,g
