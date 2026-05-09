#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/path/to/PCR_bias_chapter4"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
TASKS_TSV="${PROJECT_DIR}/analysis_results/02_Features/tasks.tsv"

FEAT_DIR="${PROJECT_DIR}/analysis_results/02_Features"
CHUNK_DIR="${FEAT_DIR}/chunks"
VOCAB_TSV="${FEAT_DIR}/kmervocab/kmer_union.tsv"

OUT_DIR="${FEAT_DIR}/kmer_region_chunks"
LOG_DIR="${FEAT_DIR}/logs_kmer_regions"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -s "${VOCAB_TSV}" ]]; then
  echo "[ERROR] missing vocab: ${VOCAB_TSV}"
  exit 1
fi

NJOBS=$(tail -n +2 "${TASKS_TSV}" | wc -l | awk '{print $1}')
echo "[INFO] chunks: ${NJOBS}"

MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then MAX_ARRAY_SIZE=1000; fi

CPUS=1
MEM="4G"
CONC=200

HEAD_WIN=30
TAIL_WIN=30
MID_BINS=3
XMODE="count"
MIN_K=6
MAX_K=8

submit_batch () {
  local START_TASK_ID=$1
  local BATCH_N=$2
  local END_INDEX=$((BATCH_N-1))
  local SLURM_SCRIPT="${FEAT_DIR}/run_02e_regions_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=kreg
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-${END_INDEX}%${CONC}
#SBATCH --output=${LOG_DIR}/kreg_%A_%a.out
#SBATCH --error=${LOG_DIR}/kreg_%A_%a.err
set -euo pipefail

OFFSET=${START_TASK_ID}
GLOBAL_ID=\$((OFFSET + SLURM_ARRAY_TASK_ID))
LINE=\$(awk -v n=\${GLOBAL_ID} 'NR==n+1{print; exit}' "${TASKS_TSV}")
CHUNK=\$(echo "\${LINE}" | cut -f2)

CHUNK_FASTA=""
for ext in fasta fa fna fasta.gz fa.gz fna.gz; do
  p="${CHUNK_DIR}/\${CHUNK}.\${ext}"
  if [[ -s "\${p}" ]]; then CHUNK_FASTA="\${p}"; break; fi
done
if [[ -z "\${CHUNK_FASTA}" ]]; then
  echo "[ERROR] chunk fasta not found for \${CHUNK} under ${CHUNK_DIR}"
  exit 2
fi

OUT="${OUT_DIR}/\${CHUNK}.kmer_regions.tsv.gz"
if [[ -s "\${OUT}" ]]; then
  echo "[SKIP] \${CHUNK} exists"
  exit 0
fi

python "${SCRIPTS_DIR}/02e_build_kmer_regions_chunk.py" \
  --chunk_fasta "\${CHUNK_FASTA}" \
  --kmer_union_tsv "${VOCAB_TSV}" \
  --out_tsv_gz "\${OUT}" \
  --min_k ${MIN_K} --max_k ${MAX_K} \
  --head_win ${HEAD_WIN} --tail_win ${TAIL_WIN} --mid_bins ${MID_BINS} \
  --x_mode ${XMODE} \
  --clip_primers
EOF

  echo "[INFO] submit batch start=${START_TASK_ID} n=${BATCH_N}"
  sbatch "${SLURM_SCRIPT}"
}

START=1
while [[ ${START} -le ${NJOBS} ]]; do
  REM=$((NJOBS - START + 1))
  BATCH_N=${MAX_ARRAY_SIZE}
  if [[ ${BATCH_N} -gt ${REM} ]]; then BATCH_N=${REM}; fi
  submit_batch "${START}" "${BATCH_N}"
  START=$((START + BATCH_N))
done

echo "[DONE] submitted kmer regions build"
