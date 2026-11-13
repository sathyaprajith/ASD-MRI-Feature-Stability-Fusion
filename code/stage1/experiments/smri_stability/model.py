"""Train-only stable FreeSurfer feature selection + regularized sMRI expert."""
from pipeline.experts import SelectedExperts

class StableStructuralModel(SelectedExperts):
    def __init__(self,k_a,k_f,seed=17,nb=8):
        super().__init__(k_a,k_f,'edges',seed,nb)
    def predict(self,c,idx):
        pa,_=super().predict(c,idx)
        return pa
