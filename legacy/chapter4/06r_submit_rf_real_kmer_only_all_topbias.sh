#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"
INROOT="$PROJECT/analysis_results/05_ModelInputs_v3_topbias"
OUTROOT="$PROJECT/analysis_results/06_Models_v3_topbias"

# 资源（real_all 特征多一些，给足）
CPUS=16
CONC=4
MEM="160G"

N_EST=2000
MSL=5
MAXF=0.25
BOOT=1
MAXS=0.6

for tag in top1p top0p5p top0p1p; do
  TASKS="$OUTROOT/$tag/rf_tasks_real_all.tsv"
  mkdir -p "$OUTROOT/$tag/logs"
  : > "$TASKS"

  for sp in 10mix donkey pig cattle; do
    for lc in 12S 16S; do
      ds="$INROOT/$tag/$sp/$lc/real_kmer_only_all"
      if [ -d "$ds" ]; then
        out="$OUTROOT/$tag/$sp/$lc/rf/real_kmer_only_all"
        echo -e "$sp\t$lc\treal_kmer_only_all\t$ds\t$out" >> "$TASKS"
      fi
    done
  done

  N=$(wc -l < "$TASKS")
  if [ "$N" -le 0 ]; then
    echo "[WARN] no real_kmer_only_all tasks for $tag"
    continue
  fi

  # top0.1% 更小，可加并发降内存
  if [ "$tag" = "top0p1p" ]; then
    CPUS=8
    CONC=8
    MEM="80G"
  else
    CPUS=16
    CONC=4
    MEM="160G"
  fi

  SLURM="$OUTROOT/$tag/run_rf_real_all_${tag}.slurm"
  cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=rf_realall_${tag}
#SBATCH --output=$OUTROOT/${tag}/logs/rf_realall_%A_%a.out
#SBATCH --error=$OUTROOT/${tag}/logs/rf_realall_%A_%a.err
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

python $PROJECT/scripts/06d_train_rf_weighted_v3.py \
  --dataset_dir "\$DATASET" \
  --out_dir     "\$OUTDIR" \
  --model rf \
  --n_estimators ${N_EST} --n_jobs ${CPUS} \
  --min_samples_leaf ${MSL} \
  --max_features ${MAXF} \
  $( [ "${BOOT}" -eq 1 ] && echo "--bootstrap --max_samples ${MAXS}" )

EOF

  echo "[INFO] submit $tag real_kmer_only_all: tasks=$N cpus=$CPUS conc=$CONC mem=$MEM"
  sbatch "$SLURM"
done
