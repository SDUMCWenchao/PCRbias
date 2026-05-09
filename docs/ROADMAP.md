# Refactoring roadmap

## P0 — safe public release

- Keep raw data, BAM/SAM/CRAM, model binaries, logs, and private configuration out of Git.
- Preserve legacy scripts under `legacy/` for thesis provenance.
- Keep AGPL-3.0 license, citation metadata, and README at repository root.
- Use `tools/smoke_test.sh` and GitHub Actions for syntax checks.

## P1 — reproducibility hardening

- Replace hard-coded `/datapool/...` defaults with `--project-dir` and/or `configs/config.yaml`.
- Mark Chapter 4 scripts as `canonical`, `historical`, or `deprecated`.
- Add a tiny synthetic dataset that validates 1–2 complete command paths without exposing real data.
- Pin dependency versions from the final thesis server environment.

## P2 — package refactoring

- Move reusable code into a package, for example `src/pcr_bias/`.
- Add command-line entry points for feature extraction, k-mer statistics, paired PCR/non-PCR analysis, and model training.
- Add unit tests for sequence feature functions, metadata validation, and table schema checks.
- Add a Snakemake or Nextflow wrapper after canonical command interfaces are stable.
