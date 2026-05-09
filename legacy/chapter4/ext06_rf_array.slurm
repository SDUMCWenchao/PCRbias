#!/usr/bin/env bash
#SBATCH -J ext_rf
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

python /datapool/zhangw/duwenchao/var/2511_PCR_Bias/scripts/ext06a_train_rf_external_one.py \
  --dataset_dir "${dataset_dir}" \
  --out_dir "${out_dir}" \
  --n_estimators "${N_ESTIMATORS:-1200}" \
  --n_jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --min_samples_leaf "${MIN_SAMPLES_LEAF:-2}" \
  --bootstrap \
  --max_samples "${MAX_SAMPLES:-0.5}" \
  --max_features "${MAX_FEATURES:-0.3}" \
  $( [[ "${FORCE}" == "1" ]] && echo "--force" )
