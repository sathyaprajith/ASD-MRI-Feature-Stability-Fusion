# Methods and interpretation limits

The primary descriptive comparison uses 720 paired ABIDE-I participants, 336 ASD
and 384 TD, from 20 acquisition identifiers grouped into 17 site families. Each
outer fold trains on 16 families and scores the remaining family.

The archived structural parser produced 185 measurements. ICV is used as an
adjustment covariate and excluded as a predictor, leaving 184 structural inputs.
The functional representation comprises 6,670 Fisher-z-transformed correlations
between regional time series for 116 indexed AAL regions. The released classifier
loader consumes the original numerical cache; it does not reconstruct the full
historical image-level preprocessing from raw MRI.

Within active training sets, the models fit covariate adjustment, imputation,
scaling, feature ranking, model fitting and inner site-aware choices. The
biological/acquisition reference is fitted on training TD participants and then
applied without refitting to the held-out data. Adjustment is not equivalent to
scanner harmonization or complete removal of confounding.

The ten stage1 families comprise two historical references, selected structural,
edge and ROI experts, two fixed selected-feature fusions, an adaptive gate, and
two compact attention arms. Classical comparisons use recorded seed 17; the two
attention families use seeds 17 and 41. This gives 12 model/seed runs and 204
outer-fold evaluations. The twenty recipe records are ten model arms, eight
retrospective prediction combinations, one nested-search procedure and one paired
analysis; they are not twenty independent studies.

The eight combinations include three component-only references. Their declared
primary combination was selected-equal plus gate, not the highest observed
gate-plus-PCA mixture. A later nine-configuration nested procedure used seeds 17
and 41 and did not exceed the reference AUC.

Site-macro ROC-AUC is the unweighted mean of 17 site-specific AUCs. Balanced
accuracy is also averaged by site and uses threshold 0.5. Participant-count-
weighted mean site AUC is a different estimand from pooled ROC-AUC. The 0.7023
blend's lower balanced accuracy does not establish an improvement in clinical
classification at that threshold.

Feature recurrence uses overlapping outer-training sets. In selected equal
fusion, 137 structural variables and 1,186 FC edges were selected at least once;
24 structural variables and 112 edges occurred in all 17 folds. At least 75% of
17 folds means at least 13 folds (49 structural variables and 272 edges).
Recurrence is not independent replication, an effect-size estimate or a causal
biomarker result.

## What has not been established

- ABIDE-I was repeatedly examined during development; these are exploratory comparisons.
- The 0.7023 mixture was retrospectively selected, and uncertainty summaries are conditional on existing predictions, without selection/multiplicity adjustment or pipeline refitting.
- No independent ABIDE-II replication, clinical threshold validation or calibration evidence is supplied.
- Original BOLD preprocessing, the exact atlas release and ROI-label order, retained-frame identities and image-level QC are not fully linked to the stored tensors.
- The historical FD convention is unverified; no verified FD exclusion threshold of 0.5 was applied. Tensor and manifest TR differ for 248 included participants.
- The old clean-environment completion summary does not certify a fresh real-data training rerun. No such claim is made in this release.
- Additional file 6's 201 historical classifier records use different cohorts and protocols; they are not 201 repeat validations of this model.
