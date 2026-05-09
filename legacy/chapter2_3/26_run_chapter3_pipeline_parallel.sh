#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${1:-/path/to/chapter2_3_analysis}"
THREADS="${2:-${SLURM_CPUS_PER_TASK:-48}}"

REGION_DIR="${BASE_DIR}/04_features/chapter3/regions"
GLOBAL_DIR="${BASE_DIR}/04_features/chapter3/global_regional"
MISMATCH_DIR="${BASE_DIR}/04_features/chapter3/mismatch"
RNAFOLD_DIR="${BASE_DIR}/04_features/chapter3/rnafold"
MOTIF_DIR="${BASE_DIR}/04_features/chapter3/motif"
TABLE_DIR="${BASE_DIR}/05_stats/chapter3/tables"
STATS_DIR="${BASE_DIR}/05_stats/chapter3/stats"
CASE_DIR="${BASE_DIR}/05_stats/chapter3/cases"

mkdir -p "$REGION_DIR" "$GLOBAL_DIR" "$MISMATCH_DIR" "$RNAFOLD_DIR" "$MOTIF_DIR" "$TABLE_DIR" "$STATS_DIR" "$CASE_DIR"

echo "[INFO] BASE_DIR=$BASE_DIR"
echo "[INFO] THREADS=$THREADS"

python3 18_build_sequence_context_table.py   --renorm-threshold-dir "${BASE_DIR}/03_tables/annotated_tables/threshold_annotated_renorm"   --sequence-catalog "${BASE_DIR}/03_tables/abundance/sequence_catalog.tsv"   --outdir "$REGION_DIR"

python3 19_extract_global_regional_features_parallel.py   --sequence-regions "${REGION_DIR}/sequence_regions.tsv"   --outdir "$GLOBAL_DIR"   --threads "$THREADS"

python3 20_extract_primer_mismatch_features_parallel.py   --sequence-regions "${REGION_DIR}/sequence_regions.tsv"   --metadata "${BASE_DIR}/00_meta/sample_metadata.tsv"   --outdir "$MISMATCH_DIR"   --threads "$THREADS"

python3 21_extract_rnafold_features_parallel.py   --sequence-regions "${REGION_DIR}/sequence_regions.tsv"   --outdir "$RNAFOLD_DIR"   --threads "$THREADS"   --chunk-size 400

python3 22_extract_motif_candidates_parallel.py   --sequence-regions "${REGION_DIR}/sequence_regions.tsv"   --outdir "$MOTIF_DIR"   --k-values 4 5   --top-per-region 20   --threads "$THREADS"

python3 23_build_chapter3_analysis_tables.py   --renorm-threshold-dir "${BASE_DIR}/03_tables/annotated_tables/threshold_annotated_renorm"   --bias-dir "${BASE_DIR}/05_stats/chapter2/bias_multi_threshold_renorm"   --feature-files     "${GLOBAL_DIR}/global_regional_features.tsv"     "${MISMATCH_DIR}/primer_mismatch_features.tsv"     "${RNAFOLD_DIR}/rnafold_features.tsv"     "${MOTIF_DIR}/motif_candidate_features.tsv"   --outdir "$TABLE_DIR"

python3 24_run_chapter3_univariate_stats.py   --species-table "${TABLE_DIR}/species_level_feature_bias_table.tsv"   --haplotype-table "${TABLE_DIR}/haplotype_level_feature_bias_table.tsv"   --outdir "$STATS_DIR"

python3 25_select_case_candidates.py   --species-table "${TABLE_DIR}/species_level_feature_bias_table.tsv"   --outdir "$CASE_DIR"

echo "[DONE] Parallel Chapter 3 core pipeline completed."
