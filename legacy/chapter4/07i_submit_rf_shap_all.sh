#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
TASKS="$PROJECT/analysis_results/06_Models_v3_topbias/rf_shap_tasks_all.tsv"
LOGDIR="$PROJECT/analysis_results/06_Models_v3_topbias/_rf_shap_logs"
mkdir -p "$LOGDIR"

# 你机器允许 200 并发；这里默认给 120，你要 200 就自己改
CPUS=4
CONC=120
MEM="80G"

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then echo "[ERROR] empty $TASKS"; exit 1; fi

SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/run_rf_shap_all.slurm"
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=rf_shap_all
#SBATCH --output=$LOGDIR/rfshap_%A_%a.out
#SBATCH --error=$LOGDIR/rfshap_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc


LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r TAG SP LC V SPLIT DATASET MODELDIR <<< "\$LINE"

# 不想重复算就保留跳过；想强制重算就把下面两行注释掉
if [ -s "\$MODELDIR/shap_tables/shap_global_\${SPLIT}.tsv" ]; then
  echo "[SKIP] exists: \$MODELDIR/shap_tables/shap_global_\${SPLIT}.tsv"
  exit 0
fi

python $PROJECT/scripts/07f_shap_rf_tables.py \
  --dataset_dir "\$DATASET" \
  --model_dir   "\$MODELDIR" \
  --split "\$SPLIT" \
  --explain_n 5000 \
  --top_features 500
EOF

sbatch "$SLURM"
