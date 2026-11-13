# Release verification

Prepared 12 November 2026.

| Check | Result |
| --- | --- |
| Original source archive inventory | 761 checksums matched; zero missing or mismatched |
| Deduplication | Ten identical stage1 Python trees replaced with one shared copy |
| Stage1 source digest | Original `1405fb37fa064628a5b02ca11f85420bef38f8bc3cc08f3163822b7ac6eea99c` preserved |
| Original stage1 synthetic suite | 7 passed, including all ten model families and restart/export checks |
| Original stage2 synthetic suite | 6 passed, including ensemble alignment, nested isolation and export checks |
| Portable release tools | 5 tests passed |
| Aggregate arithmetic | All 22 model/seed summaries matched the 17-site tables and 720-participant totals |
| Feature recurrence | 137 structural/1,186 FC ever selected; 24/112 selected in all 17 folds |
| Figure generation | Main Figures 1-5 and supplementary S1-S4 generated from included inputs |
| Supplementary PDF scan | No author-confirmation notes, machine paths or participant-style numeric IDs found in the two included PDFs |
| Real-data training | Not run: private caches and participant predictions are not included |
| GitHub workflow | Included; not run on GitHub during package preparation |

The synthetic tests passed in the exact direct dependency versions recorded in
`ENVIRONMENT.json` and `requirements.txt`. `requirements-lock.txt` additionally
records resolved runtime dependencies. Existing scikit-learn and environment
plugin deprecation warnings were observed; there were no test failures.

These checks establish code behavior, source integrity and aggregate arithmetic
within their scope. They do not certify original BOLD preprocessing, public
release permissions for all source, clinical validity or independent validation.
The rights-retained LICENSE reflects the absence of supplied upstream license
terms; an open-source permission grant has not been assumed.
