#!/usr/bin/env bash
#SBATCH -J ext_xgb
#SBATCH -t 12:00:00
#SBATCH -o /datapool/zhangw/duwenchao/var/2511_PCR_Bias/external_test/logs_slurm/%x_%A_%a.out
#SBATCH -e /datapool/zhangw/duwenchao/var/2511_PCR_Bias/external_test/logs_slurm/%x_%A_%a.err

set -euo pipefail

source ~/.bashrc

TASKS_TSV="${TASKS_TSV:?need TASKS_TSV}"
FORCE="${FORCE:-0}"

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKS_TSV}")"
if [[ -z "${line}" ]]; then
  echo "[BAD] no line for task_id=${SLURM_ARRAY_TASK_ID} in ${TASKS_TSV}"
  exit 2
fi

dataset_dir="$(echo "${line}" | awk -F'\t' '{print $1}')"
out_dir="$(echo "${line}" | awk -F'\t' '{print $2}')"

python /datapool/zhangw/duwenchao/var/2511_PCR_Bias/scripts/ext06b_train_xgb_external_one.py \
  --dataset_dir "${dataset_dir}" \
  --out_dir "${out_dir}" \
  --n_jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --n_estimators "${XGB_N_ESTIMATORS:-4000}" \
  --learning_rate "${XGB_LR:-0.03}" \
  --max_depth "${XGB_MAX_DEPTH:-8}" \
  --subsample "${XGB_SUBSAMPLE:-0.8}" \
  --colsample_bytree "${XGB_COLSAMPLE:-0.8}" \
  --early_stopping_rounds "${XGB_ESR:-200}" \
  $( [[ "${FORCE}" == "1" ]] && echo "--force" )
