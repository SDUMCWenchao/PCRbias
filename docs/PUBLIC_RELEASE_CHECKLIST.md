# Public release checklist

Before making the GitHub repository broadly visible or linking it in the thesis:

- [ ] Confirm the repository URL: `https://github.com/SDUMCWenchao/PCR_bias`.
- [ ] Confirm license: AGPL-3.0.
- [ ] Ensure no real FASTQ/BAM/CRAM/SAM data files are committed.
- [ ] Ensure no private access tokens, credentials, absolute home directories, or unpublished personal data are committed.
- [ ] Keep `configs/config.yaml` untracked.
- [ ] Run `bash tools/smoke_test.sh`.
- [ ] Run `python tools/audit_hardcoded_paths.py --legacy-mode report` and review hits.
- [ ] Confirm all hard-coded paths are either provenance-only legacy defaults or replaced with parameters.
- [ ] Decide which Chapter 4 variants are canonical.
- [ ] Add a minimal synthetic example dataset if reproducibility demonstration is required.
