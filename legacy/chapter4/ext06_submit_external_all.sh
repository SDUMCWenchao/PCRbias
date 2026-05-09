#!/usr/bin/env bash
set -euo pipefail

PROJ="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
EXT="${PROJ}/external_test"

IN="${EXT}/analysis_results/05_ModelInputs_external_topbias_resplit_v1"
OUT="${EXT}/analysis_results/06_Models_external_topbias_v2_resplit_v1"   # 不覆盖你旧的 06_Models_external_topbias

TAGS="top1p,top0p5p,top0p1p"
VARS="no_kmer,kmer_only,all,k4,k5,k6,k7,k8"

TASKDIR="${EXT}/analysis_results/_tasks_ext06"
LOGDIR="${EXT}/logs_slurm"
mkdir -p "${TASKDIR}" "${LOGDIR}" "${OUT}"

FORCE="${FORCE:-0}"   # FORCE=1 会覆盖 out_dir

# 资源：你自己改这几行就行
CPUS_RF="${CPUS_RF:-8}"
MEM_RF="${MEM_RF:-24G}"
MAXJ_RF="${MAXJ_RF:-8}"      # 8*8=64 核

CPUS_XGB="${CPUS_XGB:-8}"
MEM_XGB="${MEM_XGB:-24G}"
MAXJ_XGB="${MAXJ_XGB:-8}"

CPUS_CNN="${CPUS_CNN:-8}"
MEM_CNN="${MEM_CNN:-32G}"
MAXJ_CNN="${MAXJ_CNN:-8}"    # 4*16=64 核

echo "[INFO] OUT=${OUT}"
echo "[INFO] FORCE=${FORCE}"

# RF
RF_TASKS="${TASKDIR}/tasks_rf.tsv"
python "${PROJ}/scripts/ext06_make_tasks_external.py" \
  --inputs_root "${IN}" --out_root "${OUT}" --tags "${TAGS}" --variants "${VARS}" \
  --model rf --out_tsv "${RF_TASKS}"
N_RF="$(wc -l < "${RF_TASKS}")"
echo "[INFO] RF tasks=${N_RF}"
sbatch --array=1-"${N_RF}"%"${MAXJ_RF}" --cpus-per-task="${CPUS_RF}" --mem="${MEM_RF}" \
  --export=ALL,TASKS_TSV="${RF_TASKS}",FORCE="${FORCE}" \
  "${PROJ}/scripts/ext06_rf_array.slurm"

# XGB
XGB_TASKS="${TASKDIR}/tasks_xgb.tsv"
python "${PROJ}/scripts/ext06_make_tasks_external.py" \
  --inputs_root "${IN}" --out_root "${OUT}" --tags "${TAGS}" --variants "${VARS}" \
  --model xgb --out_tsv "${XGB_TASKS}"
N_XGB="$(wc -l < "${XGB_TASKS}")"
echo "[INFO] XGB tasks=${N_XGB}"
sbatch --array=1-"${N_XGB}"%"${MAXJ_XGB}" --cpus-per-task="${CPUS_XGB}" --mem="${MEM_XGB}" \
  --export=ALL,TASKS_TSV="${XGB_TASKS}",FORCE="${FORCE}" \
  "${PROJ}/scripts/ext06_xgb_array.slurm"

# seqCNN
CNN_TASKS="${TASKDIR}/tasks_seqcnn.tsv"
python "${PROJ}/scripts/ext06_make_tasks_external.py" \
  --inputs_root "${IN}" --out_root "${OUT}" --tags "${TAGS}" --variants "${VARS}" \
  --model seqcnn --out_tsv "${CNN_TASKS}"
N_CNN="$(wc -l < "${CNN_TASKS}")"
echo "[INFO] seqCNN tasks=${N_CNN}"
sbatch --array=1-"${N_CNN}"%"${MAXJ_CNN}" --cpus-per-task="${CPUS_CNN}" --mem="${MEM_CNN}" \
  --export=ALL,TASKS_TSV="${CNN_TASKS}",FORCE="${FORCE}" \
  "${PROJ}/scripts/ext06_seqcnn_array.slurm"

echo "[DONE] submitted all 3 arrays. logs: ${LOGDIR}"
