# Chapter 4 workflow notes

The Chapter 4 scripts are more exploratory and include multiple historical branches. Before public release, designate canonical scripts for each stage.

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

## Canonicalization needed

- Keep the newest verified version as canonical.
- Move older scripts to `legacy/deprecated/` or document them explicitly.
- Avoid two scripts with overlapping purpose unless the difference is scientifically meaningful.

## High-risk items before GitHub release

- Hard-coded `/datapool/.../2511_PCR_Bias` paths.
- Slurm scripts with absolute log paths.
- Large intermediate/model files should remain excluded by `.gitignore`.
- The scripts need a small public demo dataset for reproducibility testing.
