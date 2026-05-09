#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
TASKS="$PROJECT/analysis_results/06_Models_v3_topbias/xgb_shap_tasks_top0p5p_top0p1p.tsv"
LOGDIR="$PROJECT/analysis_results/06_Models_v3_topbias/_xgb_shap_logs"
mkdir -p "$LOGDIR"

CPUS=4
CONC=120
MEM="80G"

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then echo "[ERROR] empty $TASKS"; exit 1; fi

SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/run_xgb_shap_top0p5p_top0p1p.slurm"
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=xgb_shap
#SBATCH --output=$LOGDIR/xgbshap_%A_%a.out
#SBATCH --error=$LOGDIR/xgbshap_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r TAG SP LC V SPLIT DATASET MODELDIR <<< "\$LINE"

# 已经算过就跳过；想强制重算就注释掉
if [ -s "\$MODELDIR/shap_tables_v2/shap_global_full_\${SPLIT}.tsv.gz" ]; then
  echo "[SKIP] exists: \$MODELDIR/shap_tables_v2/shap_global_full_\${SPLIT}.tsv.gz"
  exit 0
fi

python $PROJECT/scripts/07c_shap_xgb_tables_v2.py \
  --dataset_dir "\$DATASET" \
  --model_dir   "\$MODELDIR" \
  --split "\$SPLIT" \
  --explain_n 5000 \
  --top_features 500
EOF

sbatch "$SLURM"
