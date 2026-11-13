# Reproduction guide

## Without participant data

1. Install the pinned CPU environment in `requirements.txt` or use `environment.yml`.
2. Run `python scripts/verify_results.py` to check all 22 model/seed aggregate summaries against site tables, cohort totals, weighted/site-macro metrics, winner/reference site differences and selected-feature counts.
3. Run `python scripts/run_tests.py` for 18 synthetic and packaging tests.
4. On a freshly extracted tree, run `python scripts/check_release.py` to check the release manifest and prohibited file signatures.
5. Run `python scripts/make_figures.py` to regenerate the figures. Plot PDF metadata can change between runs; verify scientific inputs and values rather than interpreting every regenerated PDF hash difference as changed results.

## With authorized cached inputs

Follow `docs/DATA_ACCESS.md`, then use `scripts/run.py`. Its paths are explicit;
outputs are isolated. Core numerical Python files in `code/stage1/` and
`code/stage2/` are copied byte-for-byte from the supplied archive. The launchers
supply every relevant data/output path so legacy defaults in archived source are
not used. Keeping that source intact also preserves its exact historical source
digest; new packaging code lives outside the checksum-locked directory.

The input cache and historical BOLD provenance limits prevent a claim that this
is a self-contained raw-scan-to-results reproduction. The CPU release environment
was tested on synthetic inputs only and may not reproduce DGX floating-point
outputs bit-for-bit. Record the actual environment and outputs for each new run.

`provenance/source_lineage.json` records archive members and their original
hashes. `provenance/stage1_digest.json` distinguishes the feature-cohort and roster
fingerprints. The source archive's original checksum inventory was checked:
761 entries matched, with no missing or mismatched members.

`provenance/RELEASE_MANIFEST.json` describes this public package, not the private
handoff. Generated run outputs, local data and caches are excluded. If you modify
the public tree for a new release, regenerate its manifest with
`python scripts/update_manifest.py` only after reviewing those changes.
