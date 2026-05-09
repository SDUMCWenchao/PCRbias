#!/usr/bin/env bash
#SBATCH -J ext_seqcnn
#SBATCH -t 24:00:00
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

python /datapool/zhangw/duwenchao/var/2511_PCR_Bias/scripts/ext06c_train_seqcnn_external_one.py \
  --dataset_dir "${dataset_dir}" \
  --out_dir "${out_dir}" \
  --epochs "${EPOCHS:-25}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --torch_threads "${SLURM_CPUS_PER_TASK:-1}" \
  $( [[ "${FORCE}" == "1" ]] && echo "--force" )
