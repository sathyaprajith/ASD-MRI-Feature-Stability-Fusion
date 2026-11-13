# Data access and archived input contract

ABIDE-I source data: https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html

Structural derivatives were obtained from the Preprocessed Connectomes Project:
https://preprocessed-connectomes-project.org/abide/

The original data publications and access conditions apply. Public source data
availability does not mean that the project's private derived caches are bundled,
or that downloading ABIDE recreates the exact historical cohort automatically.
The corresponding author can discuss the availability of permitted derived inputs.

## Anatomy root supplied to --anatomy

The archived loader expects:

- `results/results.json`, with status `complete`.
- `results/contract.json`, containing the original `cohort_sha256` and `anatomy_sha256`.
- `data/cohort.json`, the original participant roster with `subject_id`, `historical_partition`, and `group` (and source metadata where available).
- `data/anatomy.csv`, indexed by `subject_id` with the **185 original structural columns**, including `EstimatedTotalIntraCranialVol`. The loader transforms volume values, removes ICV from the predictor matrix, and checks alignment. Do not supply an already-transformed 184-column table here.
- `data/raw/phenotypic.csv`, including `SUB_ID`, `FILE_ID`, `AGE_AT_SCAN`, and `SEX`.

## Functional root supplied to --fmri

- `results/results.json`, with status `complete`.
- `results/contract.json`, containing the eligible-roster fingerprint.
- `prepared/eligible_cohort.json`, including the archived `subject_id`, `partition`, `asd_true`, `cache_sha256`, `source_sha256`, `site_id`, `mean_fd`, `usable_frames`, and `tensor_tr` fields.
- `prepared/participants/<SHA256(subject_id)>.npz`. Each original cache contains `fc` (6,670 finite Fisher-z connections), `subject_id`, `asd_true`, and `source_sha256`. The hashed basename is an identity convention, not encryption.
- Optional `prepared/eligibility.csv` and `prepared/summary.json` are copied into the private audit directory by the historical loader.

These files are **not** included or generated as placeholders. The original
loader verifies roster/content fingerprints, participant matching, labels,
historical partitions, dimensions and site families before training.

## Prior predictions for ensembles and nested checks

`--original-out` points to the completed first-stage output directory:

- `RUN_CONTRACT.json` with its actual `code_sha256` and `cohort_sha256`.
- `private_predictions/<model>__s17__<SITE>.csv` for `selected_equal`, `selected_gate`, and `v4_pca128`.
- Each prediction table has `subject_id`, `site`, `label`, `prob`, `model`, and `seed`.
- The complete comparison expects 720 unique participants, 17 site families and both diagnoses at every site.

The nested procedure verifies the source digest of the shared stage1 code against
the original run contract. Its code digest remains
`1405fb37fa064628a5b02ca11f85420bef38f8bc3cc08f3163822b7ac6eea99c`.
Never edit a contract to conceal an input/source mismatch.

## Repository data policy

Keep these inputs outside Git or under the ignored local `data/` directory.
Training outputs include participant-level material under `runs/` and are also
ignored. The public inputs shipped here are only the reviewed aggregate tables
and figures. A Git ignore rule is not a retroactive deletion mechanism: do not
add private files to an upload or commit in the first place.
