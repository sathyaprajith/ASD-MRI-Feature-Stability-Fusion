# Figure generation

Run `python scripts/make_figures.py` from the repository root. It creates all
main Figures 1-5 as PDF/PNG plus supplementary Figures S1-S4 under
`figures/regenerated/`.

| Figure | Source | Implementation |
| --- | --- | --- |
| 1 Workflow | The documented architecture and evaluation design | New Matplotlib diagram in `make_figures.py`; no numerical performance data |
| 2 Model comparison | Additional file 3; neural arms averaged across their two seeds | New aggregate-based plotting code |
| 3 Site differences | Additional file 1; gate/PCA minus PCA-128 at each of 17 families | New aggregate-based plotting code |
| 4 Structural stability | Additional file 2; selected-equal anatomical rows | New aggregate-based histogram and seven deterministic recurring examples |
| 5 FC stability | Additional file 2; selected-equal FC rows | Original figure script with portable input discovery |
| S1-S4 | Additional files 1-3 | Original supplementary plot script with repository-relative paths |

The original plotting implementations for main Figures 1-4 were absent from the
handoff, so these are **new implementations using the same measured aggregate
inputs**, not falsely presented recovered originals. Appearance and the seven
illustrative anatomical examples may differ. `figures/archived/` preserves the
five original vector figures used in the reviewed manuscript.

Do not convert numbered AAL features to named anatomical connections until the
historical atlas lookup and ordering have been verified. No image-generation
model or synthetic experimental result is used for the scientific plots.
