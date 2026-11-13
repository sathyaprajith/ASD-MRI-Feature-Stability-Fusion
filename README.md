# Cross-Site Autism Classification with Structural and Functional MRI

Research software, aggregate results and reproducible figures for **Cross-Site Autism Classification from Structural and Functional MRI: A Feature-Stability-Guided Fusion Study**.

The study compares feature selection and lightweight multimodal fusion in **720 ABIDE-I participants (336 ASD, 384 TD) across 17 acquisition-site families**, using **184 structural predictors and 6,670 functional-connectivity features**.

This repository can verify reported aggregate summaries, regenerate all figures and run synthetic code tests without participant data. Full model training requires the original, authorized feature caches described in [Data access](docs/DATA_ACCESS.md). It is not a raw-MRI-to-results preprocessing pipeline or a clinically validated diagnostic tool.

## Reported results

| Model or procedure | Site-macro ROC-AUC | Site-macro balanced accuracy |
| --- | ---: | ---: |
| PCA-128 reference | 0.6770 | 0.6269 |
| Selected equal fusion | 0.6928 | 0.6120 |
| Adaptive gated fusion | 0.6920 | 0.6161 |
| Gate + PCA-128, 50:50 | 0.7023 | 0.6178 |
| Later nested search, seed 17 | 0.6685 | 0.6152 |
| Later nested search, seed 41 | 0.6707 | 0.6129 |

The 0.7023 mixture was selected retrospectively on a previously examined ABIDE-I resource. Its balanced accuracy is below the PCA-128 reference. The negative nested-search findings are included. These results do not establish independent superiority, external generalization or clinical utility.

## Start here

Tested release environment: Linux, Python **3.12.14**, CPU PyTorch. The exact direct dependency pins are in `requirements.txt` and the resolved dependency set is in `requirements-lock.txt`; this is a newly verified environment, not the original DGX environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/verify_results.py
python scripts/run_tests.py
python scripts/check_release.py
python scripts/make_figures.py
```

Run `check_release.py` **before** regenerating figures, or use a freshly extracted copy for the final check. Regenerated PDF metadata may change file hashes even when plotted values are unchanged. Installing the pinned CPU PyTorch wheel requires downloading approximately 180 MB plus dependencies. On Windows, use `.venv\Scripts\activate`; model-source hash conventions and full verification were tested on Linux.

The 18 tests cover seven original stage1 synthetic tests, six original stage2 synthetic tests and five release-tool tests. Synthetic tests check code behavior; they do not validate the reported ASD performance. See [Verification report](reports/VERIFICATION.md).

## Run models with authorized private inputs

```bash
# All ten first-stage model families (12 model/seed runs, 204 outer fits)
python scripts/run.py stage1 --anatomy data/anatomy --fmri data/fmri

# A single model
python scripts/run.py stage1 --anatomy data/anatomy --fmri data/fmri --models selected_equal

# The eight retrospective combinations, using complete seed-17 predictions
python scripts/run.py ensembles --original-out runs/stage1

# Nine-configuration train-site-only nested search; requires the original run contract
python scripts/run.py nested --anatomy data/anatomy --fmri data/fmri --original-out runs/stage1

# Conditional paired comparisons for nested results
python scripts/run.py paired --original-out runs/stage1 --nested-out runs/nested
```

The portable launcher requires explicit input paths and overrides the archived machine-specific defaults. It writes only under `runs/` or an external directory. The original model Python source is unchanged to preserve its SHA-256 contracts. Do not run the archived entrypoints directly with their legacy defaults.

Use a new `--out` directory after changing settings or code. Never bypass source/cohort checks or mix pilot and full runs. `--pilot` is a two-site smoke run, not the 17-site result. The default launcher uses CPU. GPU training requires a compatible PyTorch installation chosen for that system and is outside this release's CPU verification.

## Repository contents

| Path | Contents |
| --- | --- |
| `code/stage1/` | One byte-identical copy of the original reference, feature-selection, gating and attention implementation |
| `code/stage2/` | Original retrospective ensembles, nested optimization and paired-statistics code |
| `configs/experiments.json` | Twenty experiment/analysis entries replacing the duplicated source folders |
| `results/aggregate/` | Six original additional files, including site/model metrics, feature stability and the historical experiment inventory |
| `scripts/` | Portable launcher, tests, aggregate validation and plotting |
| `figures/archived/` | Five original vector figures from the reviewed study |
| `figures/regenerated/` | All five main figures and four supplementary figures generated from bundled aggregate inputs |
| `provenance/` | Original-source lineage, cohort fingerprints and release checksums |
| `docs/` | Methods, data contract, reproduction boundaries and permissions |
| `reports/` | Release verification and exact environment record |

Figures 1-4 use new plotting implementations and may differ visually from the archived PDFs. Figure 1 describes architecture; Figures 2-5 and S1-S4 use archived measured aggregate values. None of these figures uses generated or synthetic experimental results. See [Figures](docs/FIGURES.md).

## What is deliberately excluded

No raw scans, participant-level phenotype/feature tables, cached participant predictions, model weights, private run outputs, credentials, duplicate experiment ZIPs, internal reviews, cover letters or unpublished manuscript drafts are bundled. The separate normative reconstruction study is not included. The historical classifier inventory contains results from different cohorts/protocols and is not independent replication of the primary 720-participant analysis.

Historical fMRI preprocessing provenance, exact atlas release/label ordering, the FD convention and 248 tensor-versus-manifest TR discrepancies remain unresolved. See [Methods and limits](docs/METHODS_AND_LIMITS.md).

## Citation and permissions

GitHub can read the supplied `CITATION.cff`. A repository URL, paper DOI and release DOI have not been invented; add the actual identifiers once available. Contact from the supplied manuscript: **sireesha@neuroparadigm.in**.

The source archive had no explicit license. [LICENSE](LICENSE) retains rights rather than assigning an unverified open-source license. Check [third-party notices](docs/THIRD_PARTY_NOTICES.md) before granting broader reuse permissions. Source visibility alone does not make this an open-source release.

## Upload this package to GitHub

1. Extract the ZIP and open the `ASD_MRI_GitHub` directory.
2. Upload the files and subfolders **inside** that directory to your repository, including `.gitignore` and `.github/`.
3. Do not upload the distribution ZIP, your local `data/` or `runs/` contents, or earlier handoff archives.
4. After the repository exists, add its real URL to `CITATION.cff` and create a versioned release if desired. Editing any tracked file makes the supplied checksum manifest stale; refresh it deliberately for the next release.

No repository has been created or published by preparing this package.
