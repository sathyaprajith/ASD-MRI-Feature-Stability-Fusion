"""Small two-modality query-attention classifier, with true train-only preprocessing.

Attention weights are not explanations/biomarkers. No test-site supervised fitting.
"""
from dataclasses import dataclass
import copy,random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from pipeline.experts import FullInputPreprocessor
from pipeline.selection import FoldFeatureSelector
from pipeline.features import inverse_sqrt_site_weights
from pipeline.splits import inner_site_splits
from pipeline.tuning import macro_score


class QueryModalityAttention(nn.Module):
    def __init__(self,a_dim,f_dim,d=32,drop=.25):
        super().__init__()
        def encoder(dim):
            return nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,64),nn.GELU(),
                                 nn.Dropout(drop),nn.Linear(64,d),nn.LayerNorm(d))
        self.anatomy=encoder(a_dim)
        self.functional=encoder(f_dim)
        self.query=nn.Parameter(torch.randn(1,1,d)*.05)
        self.attn=nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.alogit=nn.Linear(d,1)
        self.flogit=nn.Linear(d,1)
        self.correct=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,16),nn.GELU(),
                                   nn.Dropout(drop),nn.Linear(16,1))
        self.bias=nn.Parameter(torch.tensor(0.))

    def forward(self,xa,xf):
        a=self.anatomy(xa);f=self.functional(xf)
        tokens=torch.stack([a,f],dim=1)
        q=self.query.expand(len(xa),-1,-1)
        fused,w=self.attn(q,tokens,tokens,need_weights=True,average_attn_weights=True)
        w=w[:,0,:]
        scores=torch.cat([self.alogit(a),self.flogit(f)],dim=1)
        z=(w*scores).sum(dim=1)+.18*self.correct(fused[:,0,:]).squeeze(1)+self.bias
        return z,w


def _seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _data(c,train_idx,other_idx,full,ka,kf,seed,n_boot):
    if full:
        prep=FullInputPreprocessor(128).fit(c,train_idx)
    else:
        prep=FoldFeatureSelector(ka,kf,'edges',seed,n_boot).fit(c.anat[train_idx],c.fc[train_idx],
                                                             c.sites[train_idx],c.y[train_idx],c.meta[train_idx])
    if full:
        xtr=prep.transform(c,train_idx); xo=prep.transform(c,other_idx)
    else:
        xtr=prep.transform(c.anat[train_idx],c.fc[train_idx],c.meta[train_idx])
        xo=prep.transform(c.anat[other_idx],c.fc[other_idx],c.meta[other_idx])
    return prep,xtr,xo


def _optimize(xtr,ytr,site_tr,xv,yv,site_v,seed,device,max_epochs,
              early_stop=True,fixed_epochs=None):
    _seed_everything(seed)
    mod=QueryModalityAttention(xtr[0].shape[1],xtr[1].shape[1]).to(device)
    opt=torch.optim.AdamW(mod.parameters(),lr=6.e-4,weight_decay=.07)
    A=torch.from_numpy(np.asarray(xtr[0],np.float32)).to(device)
    F=torch.from_numpy(np.asarray(xtr[1],np.float32)).to(device)
    Y=torch.from_numpy(np.asarray(ytr,np.float32)).to(device)
    W=torch.from_numpy(inverse_sqrt_site_weights(site_tr,ytr)).to(device)
    if early_stop:
        VA=torch.from_numpy(np.asarray(xv[0],np.float32)).to(device)
        VF=torch.from_numpy(np.asarray(xv[1],np.float32)).to(device)
    batch=min(64,len(A))
    patience=10
    best=-np.inf; best_epoch=6; bad=0
    epochs=int(fixed_epochs or max_epochs)
    for epoch in range(1,epochs+1):
        mod.train()
        ix=torch.randperm(len(A),device=device)
        for b in ix.split(batch):
            z,_=mod(A[b],F[b])
            loss=nn.functional.binary_cross_entropy_with_logits(z,Y[b],reduction='none')
            loss=(loss*W[b]).sum()/W[b].sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite DL loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(),1.0)
            opt.step()
        if early_stop:
            mod.eval()
            with torch.no_grad():
                z,_=mod(VA,VF)
                p=torch.sigmoid(z).cpu().numpy()
            score=macro_score(yv,p,site_v)
            if score>best+1e-4:
                best=score;best_epoch=epoch;bad=0
            else:bad+=1
            if epoch>=8 and bad>=patience:break
    return mod,(int(best_epoch) if early_stop else int(epochs)),float(best)


class AttentionExperiment:
    def __init__(self,full=False,ka=40,kf=256,seed=17,n_boot=8,
                 inner_splits=2,max_epochs=65,device=None):
        self.full,self.ka,self.kf,self.seed,self.n_boot=full,ka,kf,seed,n_boot
        self.inner_splits,self.max_epochs=inner_splits,max_epochs
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    def fit(self,c,tr):
        epochs=[];scores=[];pairs=[]
        for i,(it,iv,vs) in enumerate(inner_site_splits(c.y,c.sites,tr,
                                      seed=self.seed,count=self.inner_splits)):
            _,xtr,xv=_data(c,it,iv,self.full,self.ka,self.kf,
                           self.seed+i,self.n_boot)
            _,ep,score=_optimize(xtr,c.y[it],c.sites[it],xv,c.y[iv],c.sites[iv],
                                 self.seed+i,self.device,self.max_epochs)
            epochs.append(ep);scores.append(score);pairs.append([str(x) for x in vs])
        self.final_epochs=int(np.clip(np.median(epochs),6,self.max_epochs))
        self.inner_scores=scores;self.inner_validation_sites=pairs
        # Fresh model trained on ALL outer-training participants for a locked epoch count.
        self.prep,xtr,_=_data(c,tr,tr,self.full,self.ka,self.kf,self.seed,self.n_boot)
        self.model,_,_=_optimize(xtr,c.y[tr],c.sites[tr],None,None,None,
                                  self.seed+777,self.device,self.max_epochs,
                                  early_stop=False,fixed_epochs=self.final_epochs)
        self.model.eval()
        return self

    def predict(self,c,idx):
        if self.full:a,f=self.prep.transform(c,idx)
        else:a,f=self.prep.transform(c.anat[idx],c.fc[idx],c.meta[idx])
        probs,weights=[],[]
        with torch.no_grad():
            for batch in range(0,len(a),64):
                ta=torch.as_tensor(a[batch:batch+64],dtype=torch.float32,device=self.device)
                tf=torch.as_tensor(f[batch:batch+64],dtype=torch.float32,device=self.device)
                z,w=self.model(ta,tf)
                probs.extend(torch.sigmoid(z).cpu().numpy().tolist())
                weights.extend(w.cpu().numpy().tolist())
        return np.asarray(probs), np.asarray(weights)
