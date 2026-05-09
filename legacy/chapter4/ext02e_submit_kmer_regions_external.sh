#!/usr/bin/env bash
set -euo pipefail

# ---------- args ----------
PROJECT_DIR="/path/to/PCR_bias_chapter4/external_test"
HEAD_WIN=30
TAIL_WIN=30
MID_BINS=1
XMODE="count"
MIN_K=4
MAX_K=8
CPUS=1
MEM="6G"
CONC=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_dir) PROJECT_DIR="$2"; shift 2;;
    --head_win) HEAD_WIN="$2"; shift 2;;
    --tail_win) TAIL_WIN="$2"; shift 2;;
    --mid_bins) MID_BINS="$2"; shift 2;;
    --x_mode) XMODE="$2"; shift 2;;
    --min_k) MIN_K="$2"; shift 2;;
    --max_k) MAX_K="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --conc) CONC="$2"; shift 2;;
    *) echo "[ERROR] unknown arg: $1"; exit 2;;
  esac
done

# scripts dir = 当前脚本所在目录（即主项目 scripts）
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS_TSV="${PROJECT_DIR}/analysis_results/02_Features/tasks.tsv"
FEAT_DIR="${PROJECT_DIR}/analysis_results/02_Features"
CHUNK_DIR="${FEAT_DIR}/chunks"
VOCAB_TSV="${FEAT_DIR}/kmervocab/kmer_union.tsv"
OUT_DIR="${FEAT_DIR}/kmer_region_chunks"
LOG_DIR="${FEAT_DIR}/logs_kmer_regions"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -s "${TASKS_TSV}" ]]; then
  echo "[ERROR] missing tasks.tsv: ${TASKS_TSV}"
  echo "        -> 你需要先在 external_test 里跑完 02a_submit_features（生成 chunks + tasks.tsv）"
  exit 1
fi
if [[ ! -s "${VOCAB_TSV}" ]]; then
  echo "[ERROR] missing vocab: ${VOCAB_TSV}"
  echo "        -> 你需要先在 external_test 里跑完 02d_make_kmer_union_vocab（生成 kmervocab/kmer_union.tsv）"
  exit 1
fi

NJOBS=$(tail -n +2 "${TASKS_TSV}" | wc -l | awk '{print $1}')
echo "[INFO] chunks: ${NJOBS}"

MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then MAX_ARRAY_SIZE=1000; fi
echo "[INFO] MaxArraySize=${MAX_ARRAY_SIZE}"

submit_batch () {
  local START_TASK_ID=$1
  local BATCH_N=$2
  local END_INDEX=$((BATCH_N-1))
  local SLURM_SCRIPT="${FEAT_DIR}/run_02e_regions_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ext_kreg
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

echo "[DONE] submitted external kmer regions build -> ${OUT_DIR}"
