"""Site-conditional edge-stability selection from AAL116 Fisher-z connectivity."""
from pipeline.experts import SelectedExperts

class StableFunctionalEdges(SelectedExperts):
    def __init__(self,k_a,k_f,seed=17,nb=8):
        super().__init__(k_a,k_f,'edges',seed,nb)
    def predict(self,c,idx):
        _,pf=super().predict(c,idx)
        return pf
