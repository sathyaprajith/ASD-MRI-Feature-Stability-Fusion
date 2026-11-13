"""Prespecified fixed-weight late-fusion ablations after nested feature selection."""
from pipeline.experts import SelectedExperts

class FixedSelectedFusion(SelectedExperts):
    def __init__(self,k_a,k_f,mode,seed=17,nb=8):
        super().__init__(k_a,k_f,'edges',seed,nb)
        if mode not in ('equal','anatomy75'):raise ValueError(mode)
        self.fusion_mode=mode
    def predict(self,c,idx):
        pa,pf=super().predict(c,idx)
        return (.5*pa+.5*pf if self.fusion_mode=='equal' else .75*pa+.25*pf)
