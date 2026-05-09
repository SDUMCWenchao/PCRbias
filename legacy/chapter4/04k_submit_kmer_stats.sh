#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/path/to/PCR_bias_chapter4"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
TASKS_TSV="${PROJECT_DIR}/analysis_results/02_Features/tasks.tsv"

OUT_DIR="${PROJECT_DIR}/analysis_results/04_Stats_Kmer"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

NJOBS=$(tail -n +2 "${TASKS_TSV}" | wc -l | awk '{print $1}')
echo "[INFO] chunks: ${NJOBS}"

MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then MAX_ARRAY_SIZE=1000; fi

CPUS=1
MEM="4G"
CONC=200

submit_batch () {
  local START_TASK_ID=$1
  local BATCH_N=$2
  local END_INDEX=$((BATCH_N-1))
  local SLURM_SCRIPT="${OUT_DIR}/run_04k_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=kstat
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-${END_INDEX}%${CONC}
#SBATCH --output=${LOG_DIR}/kstat_%A_%a.out
#SBATCH --error=${LOG_DIR}/kstat_%A_%a.err
set -euo pipefail

OFFSET=${START_TASK_ID}
GLOBAL_ID=\$((OFFSET + SLURM_ARRAY_TASK_ID))
LINE=\$(awk -v n=\${GLOBAL_ID} 'NR==n+1{print; exit}' "${TASKS_TSV}")
CHUNK=\$(echo "\${LINE}" | cut -f2)

python "${SCRIPTS_DIR}/04k_map_kmer_stats_chunk.py" --project_dir "${PROJECT_DIR}" --chunk_name "\${CHUNK}" --x_mode count
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

echo "[DONE] submitted kmer stats map"
