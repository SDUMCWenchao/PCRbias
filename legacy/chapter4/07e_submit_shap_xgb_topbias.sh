#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
TASKS="$PROJECT/analysis_results/06_Models_v3_topbias/shap_xgb_tasks.tsv"

# 资源：SHAP 比训练更吃内存些（取决于 explain_n）
CPUS=4
CONC=8
MEM="80G"

# SHAP 参数（只出表，不画图）
SPLIT="test"
BG=4000
EX=20000
TOPF=500
LOCALTOP=0

mkdir -p "$PROJECT/analysis_results/06_Models_v3_topbias/_shap_logs"

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then
  echo "[ERROR] empty tasks: $TASKS"
  exit 1
fi

SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/run_shap_xgb.slurm"
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=shap_xgb
#SBATCH --output=$PROJECT/analysis_results/06_Models_v3_topbias/_shap_logs/shap_%A_%a.out
#SBATCH --error=$PROJECT/analysis_results/06_Models_v3_topbias/_shap_logs/shap_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r TAG SP LC V SPLIT DATASET MODELDIR <<< "\$LINE"

# 如果已经算过就跳过
if [ -s "\$MODELDIR/shap_tables/shap_global_${SPLIT}.tsv" ]; then
  echo "[SKIP] exists: \$MODELDIR/shap_tables/shap_global_${SPLIT}.tsv"
  exit 0
fi

python $PROJECT/scripts/07c_shap_xgb_tables.py \
  --dataset_dir "\$DATASET" \
  --model_dir   "\$MODELDIR" \
  --split ${SPLIT} \
  --background_n ${BG} \
  --explain_n ${EX} \
  --top_features ${TOPF} \
  --local_top ${LOCALTOP}

EOF

echo "[INFO] submit SHAP tasks=$N"
sbatch "$SLURM"
