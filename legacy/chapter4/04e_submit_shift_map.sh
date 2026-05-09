#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
TASKS_TSV="${PROJECT_DIR}/analysis_results/02_Features/tasks.tsv"
OUT_DIR="${PROJECT_DIR}/analysis_results/04_Stats"
LOG_DIR="${OUT_DIR}/logs"
FEATURE_LIST="${OUT_DIR}/top_features_200.txt"

mkdir -p "${LOG_DIR}"

if [[ ! -s "${FEATURE_LIST}" ]]; then
  echo "[ERROR] missing ${FEATURE_LIST} (run 04d_pick_top_features.py first)"
  exit 1
fi

NJOBS=$(tail -n +2 "${TASKS_TSV}" | wc -l | awk '{print $1}')
echo "[INFO] chunks: ${NJOBS}"

MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then MAX_ARRAY_SIZE=1000; fi
MAX_INDEX=$((MAX_ARRAY_SIZE-1))
echo "[INFO] MaxArraySize=${MAX_ARRAY_SIZE} => array index 0..${MAX_INDEX}"

CPUS=1
MEM="2G"
CONC=200

submit_batch () {
  local START_TASK_ID=$1
  local BATCH_N=$2
  local END_INDEX=$((BATCH_N-1))
  local SLURM_SCRIPT="${OUT_DIR}/run_04e_shift_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=shift4E
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-${END_INDEX}%${CONC}
#SBATCH --output=${LOG_DIR}/shift4E_%A_%a.out
#SBATCH --error=${LOG_DIR}/shift4E_%A_%a.err

set -euo pipefail
OFFSET=${START_TASK_ID}
GLOBAL_ID=\$((OFFSET + SLURM_ARRAY_TASK_ID))

LINE=\$(awk -v n=\${GLOBAL_ID} 'NR==n+1{print; exit}' "${TASKS_TSV}")
CHUNK_NAME=\$(echo "\${LINE}" | cut -f2)

python "${SCRIPTS_DIR}/04e_shift_map_chunk.py" \
  --project_dir "${PROJECT_DIR}" \
  --chunk_name "\${CHUNK_NAME}" \
  --feature_list "${FEATURE_LIST}" \
  --min_count 1
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

echo "[DONE] Submitted Step 4E (shift map)."
