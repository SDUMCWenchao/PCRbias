#!/usr/bin/env bash
set -euo pipefail

PROJECT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INROOT="$PROJECT/analysis_results/05_ModelInputs_v3_topbias"
OUTROOT="$PROJECT/analysis_results/06_Models_v3_topbias"

# 你要的全套 variants（存在就跑，不存在自动跳过）
VARS=("no_kmer" "no_kmer_noprimer" "all_noprimer" "kmer_only_all" "real_kmer_only_all")
for k in 1 2 3 4 5 6 7 8; do VARS+=("kmer_only_k${k}"); done

CPUS=8
CONC=6
MEM="100G"

# XGB 超参（你可后面再调）
MAX_DEPTH=7
ETA=0.05
SUBS=0.8
COLS=0.6
MINCH=2.0
L2=1.0
L1=0.0
NROUND=5000
ES=200
YCLIP=6
MINPAIR=10

for tag in top1p top0p5p top0p1p; do
  TASKS="$OUTROOT/$tag/xgb_tasks.tsv"
  mkdir -p "$OUTROOT/$tag/logs"
  : > "$TASKS"

  for sp in 10mix donkey pig cattle; do
    for lc in 12S 16S; do
      for v in "${VARS[@]}"; do
        ds="$INROOT/$tag/$sp/$lc/$v"
        if [ -d "$ds" ]; then
          out="$OUTROOT/$tag/$sp/$lc/xgb/$v"
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

  SLURM="$OUTROOT/$tag/run_xgb_${tag}.slurm"
  cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=xgb_${tag}
#SBATCH --output=$OUTROOT/${tag}/logs/xgb_%A_%a.out
#SBATCH --error=$OUTROOT/${tag}/logs/xgb_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS")
IFS=\$'\\t' read -r SP LC V DATASET OUTDIR <<< "\$LINE"

mkdir -p "\$OUTDIR"
if [ -s "\$OUTDIR/metrics.json" ]; then
  echo "[SKIP] exists: \$OUTDIR/metrics.json"
  exit 0
fi

python $PROJECT/scripts/06t_train_xgb.py \
  --dataset_dir "\$DATASET" \
  --out_dir     "\$OUTDIR" \
  --nthread ${CPUS} \
  --max_depth ${MAX_DEPTH} --eta ${ETA} \
  --subsample ${SUBS} --colsample_bytree ${COLS} \
  --min_child_weight ${MINCH} \
  --lambda_l2 ${L2} --alpha_l1 ${L1} \
  --num_boost_round ${NROUND} --early_stopping_rounds ${ES} \
  --y_clip ${YCLIP} --min_pair_n ${MINPAIR}

EOF

  echo "[INFO] submit $tag XGB: tasks=$N cpus=$CPUS conc=$CONC mem=$MEM"
  sbatch "$SLURM"
done
