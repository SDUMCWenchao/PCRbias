#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
INPUTS_ROOT="$PROJECT/analysis_results/05_ModelInputs_v3_topbias"
OUT_ROOT="$PROJECT/analysis_results/06_Models_v3_topbias"

# CNN 资源（topbias 很小）
CPUS=8
CONC=8
MEM="40G"

# core4 variants
VARS=("no_kmer" "no_kmer_noprimer" "kmer_only_all" "all_noprimer")

for tag in top1p top0p5p top0p1p; do
  TASKS="$OUT_ROOT/$tag/cnn_tasks.tsv"
  mkdir -p "$OUT_ROOT/$tag/logs"

  # 生成 tasks：每行 = species locus variant dataset_dir out_dir
  : > "$TASKS"
  for sp in 10mix donkey pig cattle; do
    for lc in 12S 16S; do
      for v in "${VARS[@]}"; do
        ds="$INPUTS_ROOT/$tag/$sp/$lc/$v"
        if [ -d "$ds" ]; then
          out="$OUT_ROOT/$tag/$sp/$lc/cnn1d/$v"
          echo -e "$sp\t$lc\t$v\t$ds\t$out" >> "$TASKS"
        fi
      done
    done
  done

  N=$(wc -l < "$TASKS")
  if [ "$N" -le 0 ]; then
    echo "[WARN] no cnn tasks for $tag"
    continue
  fi

  SLURM="$OUT_ROOT/$tag/run_cnn_${tag}.slurm"
  cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=cnn_${tag}
#SBATCH --output=$OUT_ROOT/${tag}/logs/cnn_%A_%a.out
#SBATCH --error=$OUT_ROOT/${tag}/logs/cnn_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

TASKS="$TASKS"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "\$TASKS")
IFS=\$'\\t' read -r SP LC V DATASET OUTDIR <<< "\$LINE"

mkdir -p "\$OUTDIR"
if [ -s "\$OUTDIR/metrics.json" ]; then
  echo "[SKIP] exists: \$OUTDIR/metrics.json"
  exit 0
fi

python $PROJECT/scripts/06m_train_cnn1d.py \
  --dataset_dir "\$DATASET" \
  --out_dir     "\$OUTDIR" \
  --device auto \
  --epochs 200 --patience 20 \
  --batch_size 4096 --lr 1e-3 --weight_decay 1e-4

EOF

  echo "[INFO] submit cnn $tag: tasks=$N cpus=$CPUS conc=$CONC mem=$MEM"
  sbatch "$SLURM"
done
