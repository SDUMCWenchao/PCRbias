#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/PCR_bias_chapter4"

PROJECT_DIR="/path/to/PCR_bias_chapter4/external_test"
INPUTS_ROOT="$PROJECT_DIR/analysis_results/05_ModelInputs_external_topbias_resplit_v1"
MODELS_ROOT="$PROJECT_DIR/analysis_results/06_Models_external_topbias_v2_resplit_v1"
WEAVER_SUBDIR="analysis_results/03_DataWeaver_Blayer_with_GCfix_GCall"

OUTDIR="$PROJECT_DIR/analysis_results/_missing_attr_resplit_v1"
mkdir -p "$OUTDIR"

# scan to build missing tasks
python "$ROOT/scripts/ext07a_scan_missing_attr_external_resplit_v1.py" \
  --project_dir "$PROJECT_DIR" \
  --inputs_root "$INPUTS_ROOT" \
  --models_root "$MODELS_ROOT" \
  --out_dir "$OUTDIR" \
  --split test

TASKS_RF="$OUTDIR/tasks_missing_rf.tsv"
TASKS_XGB="$OUTDIR/tasks_missing_xgb.tsv"
TASKS_IG="$OUTDIR/tasks_missing_ig.tsv"

rf_n=$(( $(wc -l < "$TASKS_RF") - 1 ))
xgb_n=$(( $(wc -l < "$TASKS_XGB") - 1 ))
ig_n=$(( $(wc -l < "$TASKS_IG") - 1 ))

echo "[INFO] missing: rf=$rf_n xgb=$xgb_n ig=$ig_n"
if [ $rf_n -le 0 ] && [ $xgb_n -le 0 ] && [ $ig_n -le 0 ]; then
  echo "[DONE] nothing to run."
  exit 0
fi

# resources
CORES=8
MEM="24G"
MAX_ARRAY_INDEX=1000
MAX_CORES_TOTAL=64
MAX_CONCURRENCY=$((MAX_CORES_TOTAL / CORES))  # 8

# attribution params
SPLIT="test"
EXPLAIN_N_RF=5000
EXPLAIN_N_XGB=5000
EXPLAIN_N_IG=2000
TOP_FEATURES=500
SEQBANK_DIR="$PROJECT_DIR/analysis_results/_seqbank"
HEAD_WIN=30
TAIL_WIN=30
MID_BINS=1
IG_STEPS=32
IG_BS=64

# ensure seqbank exists for IG
if [ $ig_n -gt 0 ] && [ ! -s "$SEQBANK_DIR/meta.json" ]; then
  echo "[INFO] building seqbank for IG ..."
  python "$ROOT/scripts/ext07_build_seqbank_external.py" \
    --fasta "$PROJECT_DIR/analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta" \
    --out_dir "$SEQBANK_DIR"
fi

submit_chunked () {
  local slurm="$1"
  local tasks="$2"
  local total="$3"
  local dep="${4:-}"
  local jobids=()
  local offset=0

  while [ $offset -lt $total ]; do
    local left=$((total - offset))
    local chunk=$left
    if [ $chunk -gt $MAX_ARRAY_INDEX ]; then chunk=$MAX_ARRAY_INDEX; fi

    if [ -n "$dep" ]; then
      out=$(sbatch --dependency="$dep" --export=ALL,TASKS_TSV="$tasks",OFFSET="$offset" \
        --array=1-$chunk%$MAX_CONCURRENCY "$slurm")
    else
      out=$(sbatch --export=ALL,TASKS_TSV="$tasks",OFFSET="$offset" \
        --array=1-$chunk%$MAX_CONCURRENCY "$slurm")
    fi

    echo "$out" >&2
    jobids+=( "$(echo "$out" | awk '{print $4}')" )
    offset=$((offset + chunk))
  done

  if [ ${#jobids[@]} -eq 0 ]; then
    echo ""
  else
    echo "afterany:$(IFS=:; echo "${jobids[*]}")"
  fi
}

# slurm templates
SLURM_RF="$OUTDIR/run_missing_rf.slurm"
cat > "$SLURM_RF" <<SLURM
#!/bin/bash
#SBATCH -J ext_miss_rf
#SBATCH -c $CORES
#SBATCH --mem=$MEM
#SBATCH -t 1-00:00:00
#SBATCH -o $OUTDIR/%x_%A_%a.out
#SBATCH -e $OUTDIR/%x_%A_%a.err
set -euo pipefail
: "\${TASKS_TSV:?}"
: "\${OFFSET:?}"
tid=\$((SLURM_ARRAY_TASK_ID + OFFSET))
line=\$(awk -v n=\$tid 'NR==n+1{print; exit}' "\$TASKS_TSV")
IFS=\$'\t' read -r tag compare variant dataset_dir model_dir <<<"\$line"
python "$ROOT/scripts/ext07_run_attr_external_one.py" \
  --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
  --model rf --split "$SPLIT" --explain_n $EXPLAIN_N_RF --seed 1 --top_features $TOP_FEATURES
SLURM

SLURM_XGB="$OUTDIR/run_missing_xgb.slurm"
cat > "$SLURM_XGB" <<SLURM
#!/bin/bash
#SBATCH -J ext_miss_xgb
#SBATCH -c $CORES
#SBATCH --mem=$MEM
#SBATCH -t 1-00:00:00
#SBATCH -o $OUTDIR/%x_%A_%a.out
#SBATCH -e $OUTDIR/%x_%A_%a.err
set -euo pipefail
: "\${TASKS_TSV:?}"
: "\${OFFSET:?}"
tid=\$((SLURM_ARRAY_TASK_ID + OFFSET))
line=\$(awk -v n=\$tid 'NR==n+1{print; exit}' "\$TASKS_TSV")
IFS=\$'\t' read -r tag compare variant dataset_dir model_dir <<<"\$line"
python "$ROOT/scripts/ext07_run_attr_external_one.py" \
  --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
  --model xgb --split "$SPLIT" --explain_n $EXPLAIN_N_XGB --seed 1 --top_features $TOP_FEATURES
SLURM

SLURM_IG="$OUTDIR/run_missing_ig.slurm"
cat > "$SLURM_IG" <<SLURM
#!/bin/bash
#SBATCH -J ext_miss_ig
#SBATCH -c $CORES
#SBATCH --mem=$MEM
#SBATCH -t 2-00:00:00
#SBATCH -o $OUTDIR/%x_%A_%a.out
#SBATCH -e $OUTDIR/%x_%A_%a.err
set -euo pipefail
: "\${TASKS_TSV:?}"
: "\${OFFSET:?}"
tid=\$((SLURM_ARRAY_TASK_ID + OFFSET))
line=\$(awk -v n=\$tid 'NR==n+1{print; exit}' "\$TASKS_TSV")
IFS=\$'\t' read -r tag compare variant dataset_dir model_dir <<<"\$line"
python "$ROOT/scripts/ext07_run_attr_external_one.py" \
  --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
  --model seqcnn --split "$SPLIT" --explain_n $EXPLAIN_N_IG --seed 1 \
  --seqbank_dir "$SEQBANK_DIR" \
  --head_win $HEAD_WIN --tail_win $TAIL_WIN --mid_bins $MID_BINS --ig_steps $IG_STEPS --batch_size $IG_BS \
  --project_dir "$PROJECT_DIR" --weaver_subdir "$WEAVER_SUBDIR"
SLURM

# submit in order with dependencies
dep=""
if [ $rf_n -gt 0 ]; then
  echo "[INFO] submit missing RF..."
  dep=$(submit_chunked "$SLURM_RF" "$TASKS_RF" "$rf_n" "")
  echo "[INFO] dep after RF: $dep"
fi

if [ $xgb_n -gt 0 ]; then
  echo "[INFO] submit missing XGB..."
  dep=$(submit_chunked "$SLURM_XGB" "$TASKS_XGB" "$xgb_n" "$dep")
  echo "[INFO] dep after XGB: $dep"
fi

if [ $ig_n -gt 0 ]; then
  echo "[INFO] submit missing IG..."
  submit_chunked "$SLURM_IG" "$TASKS_IG" "$ig_n" "$dep" >/dev/null
fi

echo "[DONE] submitted missing jobs."

