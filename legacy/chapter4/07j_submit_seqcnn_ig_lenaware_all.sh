#!/usr/bin/env bash
set -euo pipefail

PROJECT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
TASKS="$PROJECT/analysis_results/06_Models_v3_topbias/seqcnn_attr_tasks_all.tsv"
LOGDIR="$PROJECT/analysis_results/06_Models_v3_topbias/_seqcnn_attr_lenaware_logs"
mkdir -p "$LOGDIR"

CPUS=4
CONC=200
MEM="80G"

N=$(wc -l < "$TASKS")
if [ "$N" -le 0 ]; then echo "[ERROR] empty $TASKS"; exit 1; fi

SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/run_seqcnn_attr_lenaware_all.slurm"
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=seqcnn_ig_lenaware
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

OUTFILE="\$MODELDIR/attr_tables_lenaware/attr_region_\${SPLIT}.tsv"
if [ "\${FORCE:-0}" != "1" ] && [ -s "\$OUTFILE" ]; then
  echo "[SKIP] exists: \$OUTFILE"
  exit 0
fi

python $PROJECT/scripts/07g_attr_seqcnn_ig_lenaware.py \
  --dataset_dir "\$DATASET" \
  --model_dir   "\$MODELDIR" \
  --split "\$SPLIT" \
  --explain_n 2000 \
  --batch_size 128 \
  --head_win 30 --tail_win 30 --mid_bins 3 \
  --ig_steps 32
EOF

sbatch "$SLURM"
