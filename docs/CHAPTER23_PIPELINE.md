# Chapter 2–3 workflow notes

Recommended public entry point: `legacy/chapter2_3/27_run_chapter3_pipeline_resume.sh`.

## Expected high-level workflow

1. Create directory structure: `00_setup_dirs.sh`
2. Prepare sample metadata: `01_make_sample_metadata_template.py`
3. QC and primer filtering: `02_single_end_qc_trim_filter.sh`
4. Build long abundance table: `03_build_master_long_abundance.py`
5. Summarize QC: `04_compute_basic_qc_report.py`
6. Build expected design templates: `05_make_expected_design_templates.py`
7. Threshold sensitivity and haplotype complexity: `06_*`, `07_*`
8. Sequence annotation and BLAST-assisted harmonization: `08_*`–`16_*`
9. Renormalized threshold tables and bias statistics: `16_renormalize_*`, `17_recompute_*`
10. Chapter 3 feature extraction and statistics: `18_*`–`25_*`

## Recommended command

```bash
cd legacy/chapter2_3
bash 27_run_chapter3_pipeline_resume.sh /path/to/chapter2_3_analysis 48 /path/to/RNAfold
```

## Notes

- `27_run_chapter3_pipeline_resume.sh` is preferred over `26_run_chapter3_pipeline_parallel.sh` because it has checkpoint files and per-step logs.
- `21_extract_rnafold_features_parallel_v2.py` should replace the older RNAfold extractor when possible because it supports resume mode and a custom RNAfold binary path.
