#!/usr/bin/env bash
set -euo pipefail

PROJECT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
TASKS="$PROJECT/analysis_results/06_Models_v3_topbias/seqcnn_attr_tasks_all.tsv"
LOGDIR="$PROJECT/analysis_results/06_Models_v3_topbias/_seqcnn_attr_logs"
mkdir -p "$LOGDIR"

CPUS=4
CONC=120
MEM="80G"

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then echo "[ERROR] empty $TASKS"; exit 1; fi

SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/run_seqcnn_attr_all.slurm"
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=seqcnn_attr_all
#SBATCH --output=$LOGDIR/attr_%A_%a.out
#SBATCH --error=$LOGDIR/attr_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}
##SBATCH --gres=gpu:1

set -euo pipefail
source ~/.bashrc


LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r TAG SP LC V SPLIT DATASET MODELDIR <<< "\$LINE"

# 不想重复算就保留跳过；想强制重算就把下面两行注释掉
if [ -s "\$MODELDIR/attr_tables_v2/attr_region_\${SPLIT}.tsv" ]; then
  echo "[SKIP] exists: \$MODELDIR/attr_tables_v2/attr_region_\${SPLIT}.tsv"
  exit 0
fi

python $PROJECT/scripts/07g_attr_seqcnn_ig_tables.py \
  --dataset_dir "\$DATASET" \
  --model_dir   "\$MODELDIR" \
  --split "\$SPLIT" \
  --explain_n 2000 \
  --batch_size 128 \
  --head_win 30 --tail_win 30 --mid_bins 3
EOF

sbatch "$SLURM"
