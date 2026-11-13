"""Exactly copied V4 FittedArm implementation, only wrapped for chosen recipe outputs."""
from pathlib import Path
import sys
SOURCE = Path(__file__).resolve().parents[2] / 'vendor' / 'v4'
if str(SOURCE) not in sys.path: sys.path.insert(0,str(SOURCE))
from feature_suite import FittedArm

class V4Reference:
    def __init__(self,name):
        assert name in ('v4_anatomy75','v4_pca128')
        self.name=name
        self.arm=FittedArm('logistic_equal' if name=='v4_anatomy75' else 'pca128_equal')
    def fit(self,cohort,tr):
        self.arm.fit(cohort.anat[tr],cohort.fc[tr],cohort.meta[tr],cohort.raw_sites[tr],cohort.y[tr]);return self
    def predict(self,cohort,idx):
        r=self.arm.predict(cohort.anat[idx],cohort.fc[idx],cohort.meta[idx],cohort.raw_sites[idx])
        return r['anatomy75' if self.name=='v4_anatomy75' else 'equal']
