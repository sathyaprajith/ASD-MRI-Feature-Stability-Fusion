"""Preregistered candidate models. Outer site data NEVER selects candidates."""
EXPERIMENTS = (
    'v4_anatomy75', 'v4_pca128',
    'smri_stable', 'fmri_edges', 'fmri_roi',
    'selected_equal', 'selected_anatomy75',
    'selected_gate', 'selected_attention', 'full_attention',
)
DL = {'selected_attention', 'full_attention'}
CANDIDATES = ((20,64), (40,256), (80,512))
