#!/usr/bin/env bash
set -euo pipefail

MAIN="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
EXT="${MAIN}/external_test"
SCRIPTS="${MAIN}/scripts"

INPUTS_ROOT="${EXT}/analysis_results/05_ModelInputs_external_topbias"
MODELS_ROOT="${EXT}/analysis_results/06_Models_external_topbias"
LOG_ROOT="${MODELS_ROOT}/_logs"

# ---- resource policy (满足你要求：总并发不超过 64 核 / 500G) ----
CPUS_PER_TASK=4
MAX_CONC=$(( 64 / CPUS_PER_TASK ))   # 16
MEM_PER_TASK="24G"                  # 16*24=384G < 500G

# 训练哪些模型
MODELS="rf,xgb"
# 训练哪些组
TAGS="top1p,top0p5p,top0p1p"
VARIANTS="no_kmer,kmer_only,all"

RF_N=1200
RF_MIN_LEAF=2
RF_BOOTSTRAP=1
RF_MAX_SAMPLES=0.5
RF_MAX_FEATURES=0.3

XGB_TREE_METHOD="hist"
XGB_ETA=0.05
XGB_MAX_DEPTH=8
XGB_MIN_CHILD_WEIGHT=5
XGB_SUBSAMPLE=0.8
XGB_COLSAMPLE=0.6
XGB_LAMBDA=1
XGB_ALPHA=0
XGB_NUM_ROUND=5000
XGB_EARLY=150

FORCE=0  # 1=不管 DONE.ok 全部重跑

mkdir -p "${MODELS_ROOT}" "${LOG_ROOT}"

TASKS="${MODELS_ROOT}/_tasks_train.tsv"
rm -f "${TASKS}"
echo -e "task_id\ttag\tcompare_id\tvariant\tmodel\tdataset_dir\tout_dir" > "${TASKS}"

tid=0
IFS=',' read -r -a TAG_ARR <<< "${TAGS}"
IFS=',' read -r -a VAR_ARR <<< "${VARIANTS}"
IFS=',' read -r -a MOD_ARR <<< "${MODELS}"

for tag in "${TAG_ARR[@]}"; do
  for cid_dir in "${INPUTS_ROOT}/${tag}"/*; do
    [[ -d "${cid_dir}" ]] || continue
    cid=$(basename "${cid_dir}")
    for var in "${VAR_ARR[@]}"; do
      ddir="${cid_dir}/${var}"
      [[ -d "${ddir}" ]] || continue
      for m in "${MOD_ARR[@]}"; do
        odir="${MODELS_ROOT}/${tag}/${cid}/${m}/${var}"
        tid=$((tid+1))
        echo -e "${tid}\t${tag}\t${cid}\t${var}\t${m}\t${ddir}\t${odir}" >> "${TASKS}"
      done
    done
  done
done

NTASKS=$tid
echo "[INFO] tasks = ${NTASKS}"
echo "[INFO] tasks file = ${TASKS}"
echo "[INFO] cpus/task=${CPUS_PER_TASK} conc=${MAX_CONC} mem/task=${MEM_PER_TASK}"

MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2; exit}' || true)
if [[ -z "${MAX_ARRAY_SIZE}" ]]; then MAX_ARRAY_SIZE=1000; fi
echo "[INFO] MaxArraySize=${MAX_ARRAY_SIZE}"

submit_batch () {
  local START_TID=$1
  local BATCH_N=$2
  local END_INDEX=$((BATCH_N-1))
  local SLURM_SCRIPT="${MODELS_ROOT}/run_ext05_train_${START_TID}_n${BATCH_N}.slurm"

  cat > "${SLURM_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=extTrain
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --array=0-${END_INDEX}%${MAX_CONC}
#SBATCH --output=${LOG_ROOT}/extTrain_%A_%a.out
#SBATCH --error=${LOG_ROOT}/extTrain_%A_%a.err
set -euo pipefail
source ~/.bashrc

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

OFFSET=${START_TID}
TID=\$((OFFSET + SLURM_ARRAY_TASK_ID))

LINE=\$(awk -v n=\${TID} 'NR==n+1{print; exit}' "${TASKS}")
TAG=\$(echo "\${LINE}" | cut -f2)
CID=\$(echo "\${LINE}" | cut -f3)
VAR=\$(echo "\${LINE}" | cut -f4)
MODEL=\$(echo "\${LINE}" | cut -f5)
DDIR=\$(echo "\${LINE}" | cut -f6)
ODIR=\$(echo "\${LINE}" | cut -f7)

DONE="\${ODIR}/DONE.ok"
if [[ -f "\${DONE}" && "${FORCE}" == "0" ]]; then
  echo "[SKIP] DONE.ok exists: \${ODIR}"
  exit 0
fi

mkdir -p "\${ODIR}"

EXTRA="--n_jobs \${SLURM_CPUS_PER_TASK} --seed 1"
if [[ "\${MODEL}" == "rf" ]]; then
  RF_ARGS="--rf_n_estimators ${RF_N} --rf_min_samples_leaf ${RF_MIN_LEAF} --rf_max_samples ${RF_MAX_SAMPLES} --rf_max_features ${RF_MAX_FEATURES}"
  if [[ "${RF_BOOTSTRAP}" == "1" ]]; then RF_ARGS="\${RF_ARGS} --rf_bootstrap"; fi
  python "${SCRIPTS}/ext05_train_model_external.py" --dataset_dir "\${DDIR}" --out_dir "\${ODIR}" --model rf \${EXTRA} \${RF_ARGS}
elif [[ "\${MODEL}" == "xgb" ]]; then
  python "${SCRIPTS}/ext05_train_model_external.py" --dataset_dir "\${DDIR}" --out_dir "\${ODIR}" --model xgb \${EXTRA} \
    --xgb_tree_method ${XGB_TREE_METHOD} --xgb_eta ${XGB_ETA} --xgb_max_depth ${XGB_MAX_DEPTH} --xgb_min_child_weight ${XGB_MIN_CHILD_WEIGHT} \
    --xgb_subsample ${XGB_SUBSAMPLE} --xgb_colsample ${XGB_COLSAMPLE} --xgb_lambda ${XGB_LAMBDA} --xgb_alpha ${XGB_ALPHA} \
    --xgb_num_boost_round ${XGB_NUM_ROUND} --xgb_early_stopping ${XGB_EARLY}
else
  echo "[ERROR] unknown model=\${MODEL}"
  exit 2
fi
EOF

  echo "[INFO] submit batch start_tid=${START_TID} n=${BATCH_N} array=0-${END_INDEX}%${MAX_CONC}"
  sbatch "${SLURM_SCRIPT}"
}

START=1
while [[ ${START} -le ${NTASKS} ]]; do
  REM=$((NTASKS - START + 1))
  BATCH_N=${MAX_ARRAY_SIZE}
  if [[ ${BATCH_N} -gt ${REM} ]]; then BATCH_N=${REM}; fi
  submit_batch "${START}" "${BATCH_N}"
  START=$((START + BATCH_N))
done

echo "[DONE] submitted external full training. Models root: ${MODELS_ROOT}"
