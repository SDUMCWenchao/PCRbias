#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/path/to/chapter2_3_analysis}"
THREADS="${2:-${SLURM_CPUS_PER_TASK:-48}}"
RNAFOLD_BIN="${3:-}"

REGION_DIR="${BASE_DIR}/04_features/chapter3/regions"
GLOBAL_DIR="${BASE_DIR}/04_features/chapter3/global_regional"
MISMATCH_DIR="${BASE_DIR}/04_features/chapter3/mismatch"
RNAFOLD_DIR="${BASE_DIR}/04_features/chapter3/rnafold"
MOTIF_DIR="${BASE_DIR}/04_features/chapter3/motif"
TABLE_DIR="${BASE_DIR}/05_stats/chapter3/tables"
STATS_DIR="${BASE_DIR}/05_stats/chapter3/stats"
CASE_DIR="${BASE_DIR}/05_stats/chapter3/cases"
STATE_DIR="${BASE_DIR}/05_stats/chapter3/pipeline_state"
LOG_DIR="${BASE_DIR}/09_logs/chapter3_resume"

mkdir -p "$REGION_DIR" "$GLOBAL_DIR" "$MISMATCH_DIR" "$RNAFOLD_DIR" "$MOTIF_DIR" "$TABLE_DIR" "$STATS_DIR" "$CASE_DIR" "$STATE_DIR" "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*"
}

is_done() {
  [[ -f "${STATE_DIR}/$1.done" ]]
}

mark_done() {
  date '+%F %T' > "${STATE_DIR}/$1.done"
}

run_step() {
  local step_id="$1"
  local expected_file="$2"
  shift 2

  if is_done "$step_id" && [[ -s "$expected_file" ]]; then
    log "SKIP ${step_id} (already done)"
    return 0
  fi

  log "RUN  ${step_id}"
  "$@" > "${LOG_DIR}/${step_id}.out" 2> "${LOG_DIR}/${step_id}.err"
  if [[ ! -s "$expected_file" ]]; then
    log "FAIL ${step_id}: expected output missing -> ${expected_file}"
    return 1
  fi
  mark_done "$step_id"
  log "DONE ${step_id}"
}

if [[ -z "$RNAFOLD_BIN" ]]; then
  if command -v RNAfold >/dev/null 2>&1; then
    RNAFOLD_BIN="$(command -v RNAfold)"
  else
    log "WARN RNAfold not found in PATH. You should pass absolute path as 3rd arg."
    RNAFOLD_BIN=""
  fi
fi
log "RNAFOLD_BIN=${RNAFOLD_BIN:-NOT_SET}"
log "THREADS=${THREADS}"

run_step "18_regions"   "${REGION_DIR}/sequence_regions.tsv"   python3 18_build_sequence_context_table.py     --renorm-threshold-dir "${BASE_DIR}/03_tables/annotated_tables/threshold_annotated_renorm"     --sequence-catalog "${BASE_DIR}/03_tables/abundance/sequence_catalog.tsv"     --outdir "$REGION_DIR"

run_step "19_global_regional"   "${GLOBAL_DIR}/global_regional_features.tsv"   python3 19_extract_global_regional_features_parallel.py     --sequence-regions "${REGION_DIR}/sequence_regions.tsv"     --outdir "$GLOBAL_DIR"     --threads "$THREADS"

run_step "20_mismatch"   "${MISMATCH_DIR}/primer_mismatch_features.tsv"   python3 20_extract_primer_mismatch_features_parallel.py     --sequence-regions "${REGION_DIR}/sequence_regions.tsv"     --metadata "${BASE_DIR}/00_meta/sample_metadata.tsv"     --outdir "$MISMATCH_DIR"     --threads "$THREADS"

if is_done "21_rnafold" && [[ -s "${RNAFOLD_DIR}/rnafold_features.tsv" ]]; then
  log "SKIP 21_rnafold (already done)"
else
  log "RUN  21_rnafold"
  CMD=(python3 21_extract_rnafold_features_parallel_v2.py
    --sequence-regions "${REGION_DIR}/sequence_regions.tsv"
    --outdir "$RNAFOLD_DIR"
    --threads "$THREADS"
    --chunk-size 400
    --resume)
  if [[ -n "$RNAFOLD_BIN" ]]; then
    CMD+=(--rnafold-bin "$RNAFOLD_BIN")
  fi
  "${CMD[@]}" > "${LOG_DIR}/21_rnafold.out" 2> "${LOG_DIR}/21_rnafold.err"
  [[ -s "${RNAFOLD_DIR}/rnafold_features.tsv" ]] || { log "FAIL 21_rnafold"; exit 1; }
  mark_done "21_rnafold"
  log "DONE 21_rnafold"
fi

run_step "22_motif"   "${MOTIF_DIR}/motif_candidate_features.tsv"   python3 22_extract_motif_candidates_parallel.py     --sequence-regions "${REGION_DIR}/sequence_regions.tsv"     --outdir "$MOTIF_DIR"     --k-values 4 5     --top-per-region 20     --threads "$THREADS"

run_step "23_tables"   "${TABLE_DIR}/species_level_feature_bias_table.tsv"   python3 23_build_chapter3_analysis_tables.py     --renorm-threshold-dir "${BASE_DIR}/03_tables/annotated_tables/threshold_annotated_renorm"     --bias-dir "${BASE_DIR}/05_stats/chapter2/bias_multi_threshold_renorm"     --feature-files       "${GLOBAL_DIR}/global_regional_features.tsv"       "${MISMATCH_DIR}/primer_mismatch_features.tsv"       "${RNAFOLD_DIR}/rnafold_features.tsv"       "${MOTIF_DIR}/motif_candidate_features.tsv"     --outdir "$TABLE_DIR"

run_step "24_stats"   "${STATS_DIR}/chapter3_top_signal_summary.tsv"   python3 24_run_chapter3_univariate_stats.py     --species-table "${TABLE_DIR}/species_level_feature_bias_table.tsv"     --haplotype-table "${TABLE_DIR}/haplotype_level_feature_bias_table.tsv"     --outdir "$STATS_DIR"

run_step "25_cases"   "${CASE_DIR}/case_candidates_marker_discordant.tsv"   python3 25_select_case_candidates.py     --species-table "${TABLE_DIR}/species_level_feature_bias_table.tsv"     --outdir "$CASE_DIR"

log "ALL DONE"
