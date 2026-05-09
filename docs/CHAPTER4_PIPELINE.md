# Chapter 4 workflow notes

The Chapter 4 directory preserves the thesis analysis history while documenting the recommended public execution order. Script-level status labels are listed in `docs/CHAPTER4_SCRIPT_STATUS.tsv`.

## Current logical stages

1. Manifest and preprocessing: `00_manifest_builder.py`, `00_validate_manifest.py`, `01_preprocess_sequences_se.py`, `01b_build_global_counts_and_filter.py`
2. Sequence features: `02a_calc_features_chunk.py`, `02a_submit_features.sh`
3. K-mer vocabulary and regional features: `02b*`, `02c*`, `02d*`, `02e*`
4. Pair construction and DataWeaver tables: `03a_prepare_pairs_and_counts_db.py`, `03b_dataweaver_chunk.py`
5. Feature statistics and ranking: `04a_*`–`04k_*`
6. Bias plots and export: `05a_*`, `05b_*`, `08*`
7. Machine learning: `06a_*`–`06u_*`
8. Explainability: `07*`
9. External validation: `ext*`

## Canonical and variant status

- `canonical_or_support` scripts in `docs/CHAPTER4_SCRIPT_STATUS.tsv` are the recommended public path for the numbered Chapter 4 workflow.
- `historical_variant` scripts (`_v2`, `_v3`, `topbias`, high-k/deep/model-specific branches) are retained for thesis provenance and sensitivity comparisons.
- `external_validation` scripts are intentionally separate and should be run only after the main model-input artifacts exist.
- `utility` scripts are submission/export/environment helpers rather than independent analysis stages.

## Public-run requirements

- Replace placeholder paths such as `/path/to/PCR_bias_chapter4` with local paths through command-line arguments or `configs/config.yaml`.
- Keep large intermediate/model files excluded by `.gitignore`.
- Use `examples/synthetic/` only for parser/path-wiring checks; use local real data for scientific analyses.
