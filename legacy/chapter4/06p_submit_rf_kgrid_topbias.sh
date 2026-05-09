#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/PCR_bias_chapter4"

# 资源（kgrid 很小，建议多并发）
CPUS=16
CONC=4
MEM="120G"

N_EST=2000
MSL=5
MAXF=0.25
BOOT=1
MAXS=0.6

for tag in top1p top0p5p top0p1p; do
  echo "[RUN] $tag"

  python "$PROJECT/scripts/06e_make_rf_tasks_v3.py" \
    --inputs_root "analysis_results/05_ModelInputs_v3_topbias/$tag" \
    --models_root "analysis_results/06_Models_v3_topbias/$tag" \
    --out_tasks   "analysis_results/06_Models_v3_topbias/$tag/rf_tasks_all.tsv"

  TASKS_ALL="$PROJECT/analysis_results/06_Models_v3_topbias/$tag/rf_tasks_all.tsv"
  TASKS_K="$PROJECT/analysis_results/06_Models_v3_topbias/$tag/rf_tasks_kgrid.tsv"

  # 只保留 kmer_only_k1..k8
  grep -P $'\tkmer_only_k[1-8]\t' "$TASKS_ALL" > "$TASKS_K" || true

  N=$(wc -l < "$TASKS_K")
  if [ "$N" -le 0 ]; then
    echo "[WARN] no kgrid tasks for $tag (maybe datasets missing?)"
    continue
  fi

  mkdir -p "$PROJECT/analysis_results/06_Models_v3_topbias/$tag/logs"
  SLURM="$PROJECT/analysis_results/06_Models_v3_topbias/$tag/run_rf_kgrid_${tag}.slurm"

  # top0.1% 更小，可加并发
  if [ "$tag" = "top0p1p" ]; then
    CPUS=8
    CONC=8
    MEM="60G"
  else
    CPUS=16
    CONC=4
    MEM="120G"
  fi

  cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=rf_kgrid_${tag}
#SBATCH --output=$PROJECT/analysis_results/06_Models_v3_topbias/${tag}/logs/rf_kgrid_%A_%a.out
#SBATCH --error=$PROJECT/analysis_results/06_Models_v3_topbias/${tag}/logs/rf_kgrid_%A_%a.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=0-$(($N-1))%${CONC}

set -euo pipefail
source ~/.bashrc

TASKS="$TASKS_K"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID+1))p" "\$TASKS")
IFS=\$'\\t' read -r SP LC V DATASET OUTDIR <<< "\$LINE"

mkdir -p "\$OUTDIR"
if [ -s "\$OUTDIR/metrics.json" ]; then
  echo "[SKIP] exists: \$OUTDIR/metrics.json"
  exit 0
fi

echo "[INFO] \$SP/\$LC/\$V"
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

  echo "[INFO] submit $tag kgrid: tasks=$N cpus=$CPUS conc=$CONC mem=$MEM"
  sbatch "$SLURM"
done
