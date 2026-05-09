#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/path/to/PCR_bias_chapter4"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
FEATURE_DIR="${PROJECT_DIR}/analysis_results/02_Features"

# ✅使用过滤后的 fasta（count>=2）
SEQ_FASTA="${PROJECT_DIR}/analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.countge2.fasta"

# Slurm resources
CPUS_PER_TASK=1
MEM_PER_TASK="2G"
ARRAY_CONCURRENCY=200

# chunking
MAX_JOBS=1800
MIN_SEQS_PER_CHUNK=500

mkdir -p "${FEATURE_DIR}/chunks" "${FEATURE_DIR}/logs"

echo "[INFO] Counting sequences in ${SEQ_FASTA} ..."
NSEQ=$(grep -c "^>" "${SEQ_FASTA}" || true)
if [[ "${NSEQ}" -le 0 ]]; then
  echo "[ERROR] No sequences found in ${SEQ_FASTA}"
  echo "        Did you run: python 01b_build_global_counts_and_filter.py --min_count 2 ?"
  exit 1
fi
echo "[INFO] Total sequences (count>=2): ${NSEQ}"

TARGET_JOBS=$(( NSEQ < MAX_JOBS ? NSEQ : MAX_JOBS ))
SEQS_PER_CHUNK=$(( (NSEQ + TARGET_JOBS - 1) / TARGET_JOBS ))
if [[ "${SEQS_PER_CHUNK}" -lt "${MIN_SEQS_PER_CHUNK}" ]]; then
  SEQS_PER_CHUNK="${MIN_SEQS_PER_CHUNK}"
fi
echo "[INFO] TARGET_JOBS=${TARGET_JOBS}  SEQS_PER_CHUNK=${SEQS_PER_CHUNK}"

# regenerate chunks + tasks
rm -f "${FEATURE_DIR}/chunks/chunk_"*.fasta "${FEATURE_DIR}/tasks.tsv" || true

echo "[INFO] Splitting fasta into chunks ..."
export SEQ_FASTA
export OUT_DIR="${FEATURE_DIR}/chunks"
export SEQS_PER_CHUNK

python - << 'PY'
import os
from pathlib import Path

seq_fa = Path(os.environ["SEQ_FASTA"])
out_dir = Path(os.environ["OUT_DIR"])
seqs_per_chunk = int(os.environ["SEQS_PER_CHUNK"])

out_dir.mkdir(parents=True, exist_ok=True)
tasks = []
chunk_idx = 0
cur = []
n_in_chunk = 0

def flush(idx, buf):
    if not buf:
        return None
    name = f"chunk_{idx:04d}"
    fp = out_dir / f"{name}.fasta"
    fp.write_text("".join(buf))
    return name, fp

with seq_fa.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith(">"):
            if n_in_chunk >= seqs_per_chunk:
                name_fp = flush(chunk_idx, cur)
                if name_fp:
                    name, fp = name_fp
                    tasks.append((name, str(fp)))
                chunk_idx += 1
                cur = []
                n_in_chunk = 0
            n_in_chunk += 1
        cur.append(line)

name_fp = flush(chunk_idx, cur)
if name_fp:
    name, fp = name_fp
    tasks.append((name, str(fp)))

tasks_tsv = out_dir.parent / "tasks.tsv"
with tasks_tsv.open("w", encoding="utf-8") as fo:
    fo.write("task_id\tchunk_name\tchunk_fasta\n")
    for i, (name, fp) in enumerate(tasks, start=1):
        fo.write(f"{i}\t{name}\t{fp}\n")

print(f"[DONE] chunks={len(tasks)} -> {out_dir}")
print(f"[DONE] tasks file -> {tasks_tsv}")
PY

TASKS_TSV="${FEATURE_DIR}/tasks.tsv"
NJOBS=$(tail -n +2 "${TASKS_TSV}" | wc -l | awk '{print $1}')
echo "[INFO] Total chunk jobs: ${NJOBS}"

# ✅关键：MaxArraySize 限制的是 array index 最大值，必须用 0-based index，并用 OFFSET 映射
MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then
  MAX_ARRAY_SIZE=1000
fi
MAX_INDEX=$(( MAX_ARRAY_SIZE - 1 ))
if [[ "${MAX_INDEX}" -lt 0 ]]; then
  echo "[ERROR] Invalid MaxArraySize=${MAX_ARRAY_SIZE}"
  exit 1
fi
echo "[INFO] MaxArraySize=${MAX_ARRAY_SIZE} => allowed array index: 0..${MAX_INDEX}"

submit_batch () {
  local START_TASK_ID=$1   # global task_id in tasks.tsv (1-based)
  local BATCH_N=$2         # number of tasks in this batch
  local END_INDEX=$(( BATCH_N - 1 ))  # array index 0..END_INDEX

  local SLURM_SCRIPT="${FEATURE_DIR}/run_02a_features_${START_TASK_ID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=feat2A
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --array=0-${END_INDEX}%${ARRAY_CONCURRENCY}
#SBATCH --output=${FEATURE_DIR}/logs/feat2A_%A_%a.out
#SBATCH --error=${FEATURE_DIR}/logs/feat2A_%A_%a.err

set -euo pipefail

OFFSET=${START_TASK_ID}
GLOBAL_ID=\$(( OFFSET + SLURM_ARRAY_TASK_ID ))

LINE=\$(awk -v n=\${GLOBAL_ID} 'NR==n+1{print; exit}' "${TASKS_TSV}")
CHUNK_NAME=\$(echo "\${LINE}" | cut -f2)
CHUNK_FASTA=\$(echo "\${LINE}" | cut -f3)

echo "[INFO] array_id=\${SLURM_ARRAY_TASK_ID} global_id=\${GLOBAL_ID} chunk=\${CHUNK_NAME}"
python "${SCRIPTS_DIR}/02a_calc_features_chunk.py" "\${CHUNK_FASTA}" "\${CHUNK_NAME}" --project_dir "${PROJECT_DIR}"
EOF

  echo "[INFO] Submitting batch: start_task_id=${START_TASK_ID}, n=${BATCH_N}, array=0-${END_INDEX}%${ARRAY_CONCURRENCY}"
  sbatch "${SLURM_SCRIPT}"
}

# batch submit
START=1
while [[ ${START} -le ${NJOBS} ]]; do
  REMAIN=$(( NJOBS - START + 1 ))
  BATCH_N=${MAX_ARRAY_SIZE}
  if [[ ${BATCH_N} -gt ${REMAIN} ]]; then BATCH_N=${REMAIN}; fi
  # safety: ensure END_INDEX <= MAX_INDEX always true
  submit_batch "${START}" "${BATCH_N}"
  START=$(( START + BATCH_N ))
done

echo "[DONE] Submitted Step 2A with safe 0-based arrays + offsets."
