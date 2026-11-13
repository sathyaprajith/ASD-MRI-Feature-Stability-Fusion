"""ROI-prioritized connection selection from 116-node AAL atlas."""
from pipeline.experts import SelectedExperts

class StableFunctionalROI(SelectedExperts):
    def __init__(self,k_a,k_f,seed=17,nb=8):
        super().__init__(k_a,k_f,'roi',seed,nb)
    def predict(self,c,idx):
        _,pf=super().predict(c,idx)
        return pf
