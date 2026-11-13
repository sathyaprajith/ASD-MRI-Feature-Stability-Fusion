# Release preparation

This package is intended to be extracted and uploaded as a source repository.
The supplied 11 November 2026 handoff remains a separate private backup.

Removed from this public package:

- Internal audits, review notes, author-confirmation checklists, draft cover letters and submission instructions.
- Older manuscripts and editable manuscript ZIPs; the repository provides software, figures and aggregates rather than redistributing an unpublished draft with unresolved declarations.
- Duplicate experiment source trees and nested backup/export ZIPs. The ten stage1 sources were identical and are represented once, with all experiment recipes retained.
- LaTeX build logs, .aux/.out files and redundant PNG/SVG copies of archived figures.
- Original machine-specific README instructions; public instructions use the portable launcher. Historical numerical Python is untouched for source-digest compatibility.
- Private participant data, predictions, checkpoints, raw scans and credentials. None is deliberately included.

Added:

- One repository README, twenty recipe records, explicit-path launchers, aggregate checks, meaningful synthetic tests and Git ignore rules.
- An exact tested CPU dependency set, environment file and verification report.
- Regeneration scripts for main Figures 1-4, plus portable access to existing Figure 5 and supplementary scripts.
- Source lineage and release checksums, a citation file and rights-retained license notice.
- GitHub Actions workflow for synthetic tests and aggregate verification.

No GitHub repository URL, DOI, independent evaluation result, ethics decision,
open-source permission or successful real-data training rerun has been invented.
