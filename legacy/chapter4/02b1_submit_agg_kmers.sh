#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
FEATURE_DIR="${PROJECT_DIR}/analysis_results/02_Features"

IN_DIR="${FEATURE_DIR}/kmer_sparse_chunks"
OUT_DIR="${FEATURE_DIR}/kmer_agg_chunks"
LOG_DIR="${FEATURE_DIR}/logs"
TASKS_TSV="${FEATURE_DIR}/agg_tasks.tsv"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# build task list
rm -f "${TASKS_TSV}" || true
echo -e "task_id\tchunk_name\tin_kmer\tout_agg" > "${TASKS_TSV}"

i=0
for fp in "${IN_DIR}"/*.kmer.tsv.gz; do
  [[ -e "$fp" ]] || { echo "[ERROR] no inputs in ${IN_DIR}"; exit 1; }
  bn=$(basename "$fp")
  chunk="${bn%%.*}"   # chunk_0001
  out="${OUT_DIR}/${chunk}.kmer_agg.tsv.gz"
  i=$((i+1))
  echo -e "${i}\t${chunk}\t${fp}\t${out}" >> "${TASKS_TSV}"
done

NJOBS=$i
echo "[INFO] agg tasks: ${NJOBS}"

# detect MaxArraySize; default 1000
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
  local SLURM_SCRIPT="${FEATURE_DIR}/run_02b1_agg_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=kmerAgg
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-${END_INDEX}%${CONC}
#SBATCH --output=${LOG_DIR}/kmerAgg_%A_%a.out
#SBATCH --error=${LOG_DIR}/kmerAgg_%A_%a.err

set -euo pipefail
OFFSET=${START_TASK_ID}
GLOBAL_ID=\$((OFFSET + SLURM_ARRAY_TASK_ID))

LINE=\$(awk -v n=\${GLOBAL_ID} 'NR==n+1{print; exit}' "${TASKS_TSV}")
CHUNK=\$(echo "\${LINE}" | cut -f2)
IN_KMER=\$(echo "\${LINE}" | cut -f3)
OUT_AGG=\$(echo "\${LINE}" | cut -f4)

# skip if exists
if [[ -s "\${OUT_AGG}" ]]; then
  echo "[SKIP] \${CHUNK} exists: \${OUT_AGG}"
  exit 0
fi

python "${SCRIPTS_DIR}/02b1_agg_kmers_chunk.py" --in_kmer "\${IN_KMER}" --out_agg "\${OUT_AGG}"
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

echo "[DONE] submitted kmer aggregation arrays"
