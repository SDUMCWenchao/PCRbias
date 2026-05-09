#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"

NO_CORR="${PROJECT}/analysis_results/04_Stats/pair_feature_corr.tsv.gz"
NO_SHIFT="${PROJECT}/analysis_results/04_Stats/pair_feature_shift_top200.tsv.gz"

K_CORR="${PROJECT}/analysis_results/04_Stats_Kmer/pair_kmer_corr.tsv.gz"
K_SHIFT="${PROJECT}/analysis_results/04_Stats_Kmer/pair_kmer_shift.tsv.gz"

OUT_DIR="${PROJECT}/analysis_results/04_Stats_All"
mkdir -p "${OUT_DIR}"

OUT_CORR="${OUT_DIR}/pair_all_corr.tsv.gz"
OUT_SHIFT="${OUT_DIR}/pair_all_shift.tsv.gz"

# safety checks
for f in "${NO_CORR}" "${NO_SHIFT}" "${K_CORR}" "${K_SHIFT}"; do
  if [[ ! -s "${f}" ]]; then
    echo "[ERROR] missing input: ${f}"
    exit 2
  fi
done

echo "[INFO] merging corr..."
python "${PROJECT}/scripts/04a_merge_pair_stats.py" \
  --first "${NO_CORR}" \
  --second "${K_CORR}" \
  --out "${OUT_CORR}" \
  --prefer second

echo "[INFO] merging shift..."
python "${PROJECT}/scripts/04a_merge_pair_stats.py" \
  --first "${NO_SHIFT}" \
  --second "${K_SHIFT}" \
  --out "${OUT_SHIFT}" \
  --prefer second

echo "[INFO] joint ranking -> ${OUT_DIR}/joint"
python "${PROJECT}/scripts/04h_joint_rank_stratify.py" \
  --project_dir "${PROJECT}" \
  --pair_corr  "${OUT_CORR}" \
  --pair_shift "${OUT_SHIFT}" \
  --out_dir    "analysis_results/04_Stats_All/joint"

echo "[DONE] all-features report:"
echo "  ${OUT_DIR}/joint/report.md"
