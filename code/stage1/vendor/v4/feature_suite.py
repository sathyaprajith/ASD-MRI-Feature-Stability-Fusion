"""Fixed-recipe exploratory feature/model ablations for 720 ABIDE-I pairs.

All learned transforms are fit exclusively on the supplied fitting participants.
No held-out site labels or covariates used to calibrate a held-out scanner.
"""
import warnings
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from controls import Reference

from advanced_fusion import ADVANCED_ARMS, AdvancedFusion

N_ROIS=116
UPPER=np.triu_indices(N_ROIS,1)

# Each candidate is a fixed scientific hypothesis, NOT a hyperparameter optimized
# against the held-out sites. Test all candidates and report all, including failures.
ARMS=(
    'logistic_equal',             # archived C=0.01 site-adjusted experts
    'logistic_no_site',
    'logistic_site_balanced',
    'robust_scaling_equal',
    'anatomy_select50_equal',
    'pca64_equal',
    'pca128_equal',
    'univariate512_equal',
    'sparse512_equal',
    'roi_strength_equal',
    'rbf_svm_pca_equal',
    'rbf_svm_joint',
    'extra_trees_joint',
    'hist_gradient_joint',
    'mlp_joint',
    *ADVANCED_ARMS,
)


def graph_summary(z):
    """Per-participant, unsupervised ROI-level feature engineering.
    116 positive, 116 negative, 116 absolute and 116 top-12 strengths.
    The ROI ordering is the inherited AAL116 order. No site or diagnosis used.
    """
    z=np.asarray(z,dtype=float)
    if z.ndim!=2 or z.shape[1]!=6670:
        raise ValueError('Expected Fisher-z upper triangle with 6670 edges')
    n=len(z)
    result=np.empty((n,4*N_ROIS),dtype=np.float32)
    for j in range(n):
        a=np.zeros((N_ROIS,N_ROIS),dtype=np.float64)
        e=np.tanh(np.clip(z[j],-6,6))
        a[UPPER]=e; a[(UPPER[1],UPPER[0])]=e
        ab=np.abs(a)
        result[j,:116]=np.maximum(a,0).sum(axis=1)/115
        result[j,116:232]=np.maximum(-a,0).sum(axis=1)/115
        result[j,232:348]=ab.sum(axis=1)/115
        result[j,348:464]=np.sort(ab,axis=1)[:,-12:].mean(axis=1)
    return result


def site_weights(sites):
    sites=np.asarray(sites,dtype=str)
    _,inv,counts=np.unique(sites,return_inverse=True,return_counts=True)
    w=1/np.sqrt(counts[inv].astype(float))
    return w/w.mean()


class FittedArm:
    """Fold-contained imputation/TD norms, scaler, select-k/PCA, classifier."""
    def __init__(self,name):
        if name not in ARMS:raise ValueError(name)
        self.name=name

    def fit(self,anat,fc,meta,site,y):
        anat=np.asarray(anat,float);fc=np.asarray(fc,float)
        y=np.asarray(y,int);site=np.asarray(site,str)
        if len(np.unique(y))!=2 or (y==0).sum()<20:raise ValueError('Missing TD anchors')
        if self.name in ADVANCED_ARMS:
            self.advanced=AdvancedFusion(self.name).fit(anat,fc,meta,site,y)
            return self
        mode='bio_acq' if self.name=='logistic_no_site' else 'bio_acq_site'
        self.a_ref=Reference(mode).fit(anat,meta,site,y)
        scaler=RobustScaler if self.name=='robust_scaling_equal' else StandardScaler
        self.a_scale=scaler().fit(self.a_ref.transform(anat,meta,site))
        a=self.a_scale.transform(self.a_ref.transform(anat,meta,site))
        if self.name=='roi_strength_equal':
            self.f_ref=Reference(mode).fit(graph_summary(fc),meta,site,y)
            f0=self.f_ref.transform(graph_summary(fc),meta,site)
        else:
            self.f_ref=Reference(mode).fit(fc,meta,site,y)
            f0=self.f_ref.transform(fc,meta,site)
        self.f_scale=scaler().fit(f0)
        f=self.f_scale.transform(f0)
        if not np.isfinite(a).all() or not np.isfinite(f).all():raise ValueError('Nonfinite training features')
        self.mapper=None; self.postscale=None; self.a_mapper=None
        if self.name=='anatomy_select50_equal':
            self.a_mapper=SelectKBest(score_func=f_classif,k=50).fit(a,y)
            a=self.a_mapper.transform(a)
        if self.name.startswith('pca') or self.name in ('rbf_svm_pca_equal','rbf_svm_joint','extra_trees_joint','hist_gradient_joint','mlp_joint'):
            dim=128 if self.name=='pca128_equal' else 64
            dim=min(dim,len(y)-1,f.shape[1])
            self.mapper=PCA(n_components=dim,svd_solver='randomized',random_state=17).fit(f)
            f=self.mapper.transform(f)
            self.postscale=StandardScaler().fit(f)
            f=self.postscale.transform(f)
        elif self.name in ('univariate512_equal','sparse512_equal'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                self.mapper=SelectKBest(score_func=f_classif,k=min(512,f.shape[1])).fit(f,y)
            f=self.mapper.transform(f)
        weights=site_weights(site) if self.name=='logistic_site_balanced' else None
        if self.name in ('extra_trees_joint','hist_gradient_joint','mlp_joint','rbf_svm_joint'):
            joined=np.column_stack([a,f]);self.joined_dim=joined.shape[1]
            if self.name=='rbf_svm_joint':
                self.joint=SVC(C=1.,gamma='scale',kernel='rbf',probability=True,random_state=17).fit(joined,y)
            elif self.name=='extra_trees_joint':
                self.joint=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=7,max_features=.6,
                                                n_jobs=4,random_state=17).fit(joined,y)
            elif self.name=='hist_gradient_joint':
                self.joint=HistGradientBoostingClassifier(max_iter=125,max_leaf_nodes=9,min_samples_leaf=15,
                                                        learning_rate=.035,l2_regularization=5.,
                                                        early_stopping=False,random_state=17).fit(joined,y)
            else:
                self.joint=MLPClassifier(hidden_layer_sizes=(64,16),activation='relu',
                                         alpha=12.,max_iter=175,batch_size=32,random_state=17,
                                         early_stopping=False,learning_rate_init=.0005).fit(joined,y)
        else:
            self.a_clf=LogisticRegression(C=.01,solver='liblinear',max_iter=3500,random_state=17)
            self.a_clf.fit(a,y,sample_weight=weights)
            if self.name=='sparse512_equal':
                self.f_clf=LogisticRegression(C=.08,penalty='l1',solver='liblinear',max_iter=3000,random_state=17)
            elif self.name=='rbf_svm_pca_equal':
                # probability calibration is internal to the training pool only.
                self.f_clf=SVC(C=1.,gamma='scale',kernel='rbf',probability=True,random_state=17)
            else:
                c=.1 if self.name.startswith('pca') or self.name in ('univariate512_equal','roi_strength_equal') else .01
                self.f_clf=LogisticRegression(C=c,solver='liblinear',max_iter=3500,random_state=17)
            self.f_clf.fit(f,y,sample_weight=weights)
        return self

    def predict(self,anat,fc,meta,site):
        if self.name in ADVANCED_ARMS:
            return self.advanced.predict(anat,fc,meta,site)
        site=np.asarray(site,str)
        a=self.a_scale.transform(self.a_ref.transform(anat,meta,site))
        if self.name=='roi_strength_equal':
            f0=self.f_ref.transform(graph_summary(fc),meta,site)
        else:f0=self.f_ref.transform(fc,meta,site)
        f=self.f_scale.transform(f0)
        if self.a_mapper is not None:a=self.a_mapper.transform(a)
        if self.mapper is not None:f=self.mapper.transform(f)
        if self.postscale is not None:f=self.postscale.transform(f)
        if self.name in ('extra_trees_joint','hist_gradient_joint','mlp_joint','rbf_svm_joint'):
            p=self.joint.predict_proba(np.column_stack([a,f]))[:,1]
            return {'equal':p}
        pa=self.a_clf.predict_proba(a)[:,1]
        pf=self.f_clf.predict_proba(f)[:,1]
        result={'equal':.5*(pa+pf),'anatomy':pa,'fc':pf}
        if self.name=='logistic_equal':
            result['fc75']=.25*pa+.75*pf
            result['anatomy75']=.75*pa+.25*pf
        return result
