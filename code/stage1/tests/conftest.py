import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from pipeline.data import Cohort

@pytest.fixture(scope='module')
def synthetic():
    rng=np.random.default_rng(92)
    nsite=8;per=12;n=nsite*per
    y=np.tile([0,1]*6,nsite)
    sites=np.repeat([f'ABIDE_I::S{i}' for i in range(nsite)],per)
    a=rng.normal(size=(n,184)).astype('float32')
    f=rng.normal(size=(n,6670)).astype('float32')
    a[:,3]+=.65*y
    f[:,120]+=.7*y
    meta=np.column_stack([rng.uniform(9,30,n),rng.binomial(1,.6,n),
        rng.normal(14,0.15,n),rng.uniform(0.05,.35,n),rng.uniform(4.5,5.6,n),
        rng.uniform(1.6,3.0,n)]).astype('float32')
    return Cohort(np.array([f'SYN::{i}' for i in range(n)]),sites,sites,y,a,f,meta,
                 [f'anatomy_feature_{i}' for i in range(184)])
