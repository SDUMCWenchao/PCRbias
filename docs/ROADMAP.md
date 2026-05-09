# Refactoring roadmap

## P0 — safe public release

- Keep raw data, BAM/SAM/CRAM, model binaries, logs, and private configuration out of Git.
- Preserve legacy scripts under `legacy/` for thesis provenance.
- Keep AGPL-3.0 license, citation metadata, and README at repository root.
- Use `tools/smoke_test.sh` and GitHub Actions for syntax checks.

## P1 — reproducibility hardening

Completed for the public draft:

- Replaced private absolute server-path defaults with public placeholder paths that users can override with command-line arguments and/or `configs/config.yaml`.
- Added Chapter 4 status labels in `docs/CHAPTER4_SCRIPT_STATUS.tsv` so canonical/support, historical-variant, external-validation, utility, and deprecated-legacy scripts are distinguishable.
- Added a tiny synthetic fixture under `examples/synthetic/` for parser/path-wiring checks without exposing real sequencing data.
- Pinned the Python dependency baseline in `requirements.txt` and mirrored practical pins in `environment.yml`.

Remaining hardening work:

- Promote shared sequence, k-mer, statistics, and model-training functions into a package.
- Add tests that execute one or two complete synthetic fixture command paths after canonical command interfaces are stable.
- Replace placeholder paths in downstream deployments with site-local config files that remain uncommitted.

## P2 — package refactoring

- Move reusable code into a package, for example `src/pcr_bias/`.
- Add command-line entry points for feature extraction, k-mer statistics, paired PCR/non-PCR analysis, and model training.
- Add unit tests for sequence feature functions, metadata validation, and table schema checks.
- Add a Snakemake or Nextflow wrapper after canonical command interfaces are stable.
