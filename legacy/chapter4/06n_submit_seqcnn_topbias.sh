#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
INROOT="$PROJECT/analysis_results/05_ModelInputs_v3_topbias"
OUTROOT="$PROJECT/analysis_results/06_Models_v3_topbias"

# 你要跑哪些 variant：按需改
VARS=("no_kmer" "no_kmer_noprimer" "kmer_only_all" "all_noprimer")

CPUS=8
CONC=8
MEM="80G"

for tag in top1p top0p5p top0p1p; do
  TASKS="$OUTROOT/$tag/seqcnn_tasks.tsv"
  mkdir -p "$OUTROOT/$tag/logs"
  : > "$TASKS"

  for sp in 10mix donkey pig cattle; do
    for lc in 12S 16S; do
      for v in "${VARS[@]}"; do
        ds="$INROOT/$tag/$sp/$lc/$v"
        if [ -d "$ds" ]; then
          out="$OUTROOT/$tag/$sp/$lc/seqcnn/$v"
          echo -e "$sp\t$lc\t$v\t$ds\t$out" >> "$TASKS"
        fi
      done
    done
  done

  N=$(wc -l < "$TASKS")
  if [ "$N" -le 0 ]; then
    echo "[WARN] no tasks for $tag"
    continue
  fi

  SLURM="$OUTROOT/$tag/run_seqcnn_${tag}.slurm"
  cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=seqcnn_${tag}
#SBATCH --output=$OUTROOT/${tag}/logs/seqcnn_%A_%a.out
#SBATCH --error=$OUTROOT/${tag}/logs/seqcnn_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}
##SBATCH --gres=gpu:1

set -euo pipefail
source ~/.bashrc

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r SP LC V DATASET OUTDIR <<< "\$LINE"
mkdir -p "\$OUTDIR"

# --- NEW SKIP LOGIC ---
# 只有当 metrics.json 和 model.pt 都存在时才跳过
if [ -s "\$OUTDIR/metrics.json" ] && [ -s "\$OUTDIR/model.pt" ]; then
  echo "[SKIP] exists: \$OUTDIR/metrics.json + model.pt"
  exit 0
fi

python $PROJECT/scripts/06m_train_seqcnn1d.py \
  --dataset_dir "\$DATASET" \
  --out_dir     "\$OUTDIR" \
  --use_pyfaidx \
  --max_len 500 \
  --y_clip 6 \
  --epochs 60 --patience 10 \
  --batch_size 512 \
  --lr 2e-3 --weight_decay 1e-4 \
  --min_pair_n 10
EOF

  echo "[INFO] submit $tag seqcnn: tasks=$N"
  sbatch "$SLURM"
done
