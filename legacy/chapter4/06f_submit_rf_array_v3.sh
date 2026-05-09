#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
TASKS="$PROJECT/analysis_results/06_Models_v3/rf_tasks.tsv"
SLURM="$PROJECT/analysis_results/06_Models_v3/run_rf_array.slurm"

# 你可以调参：建议先 only_core4 跑一轮，确认趋势后再全量
python "$PROJECT/scripts/06e_make_rf_tasks_v3.py" --only_core4

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then
  echo "[ERROR] no tasks in $TASKS"
  exit 1
fi

# 资源策略（稳妥）：
# - 每个任务 32 线程；array 并发 2 -> 总线程 ~64
CPUS=32
CONC=2
MEM="220G"   # RF 2000 trees + 大样本，内存留足
N_EST=2000
MSL=5
MAXF=0.25
BOOT=1
MAXS=0.6

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=rf_v3
#SBATCH --output=$PROJECT/analysis_results/06_Models_v3/logs/rf_%A_%a.out
#SBATCH --error=$PROJECT/analysis_results/06_Models_v3/logs/rf_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

TASKS="$TASKS"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "\$TASKS")
IFS=\$'\\t' read -r SP LC V DATASET OUTDIR <<< "\$LINE"

mkdir -p "\$OUTDIR"

echo "[INFO] task_id=\$SLURM_ARRAY_TASK_ID  \$SP/\$LC/\$V"
echo "[INFO] dataset=\$DATASET"
echo "[INFO] outdir=\$OUTDIR"

python $PROJECT/scripts/06d_train_rf_weighted_v3.py \
  --dataset_dir "\$DATASET" \
  --out_dir     "\$OUTDIR" \
  --model rf \
  --n_estimators ${N_EST} --n_jobs ${CPUS} \
  --min_samples_leaf ${MSL} \
  --max_features ${MAXF} \
  $( [ "${BOOT}" -eq 1 ] && echo "--bootstrap --max_samples ${MAXS}" )

EOF

mkdir -p "$PROJECT/analysis_results/06_Models_v3/logs"
echo "[INFO] submit array size=$N  cpus=$CPUS  conc=$CONC  mem=$MEM"
sbatch "$SLURM"
echo "[DONE] submitted: $SLURM"
