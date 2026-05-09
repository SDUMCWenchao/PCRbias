#!/usr/bin/env bash
set -euo pipefail

# ====== user configurable ======
PROJECT_DIR="/datapool/zhangw/duwenchao/var/2511_PCR_Bias/external_test"
INPUTS_ROOT="$PROJECT_DIR/analysis_results/05_ModelInputs_external_topbias_resplit_v1"
MODELS_ROOT="$PROJECT_DIR/analysis_results/06_Models_external_topbias_v2_resplit_v1"

# resources
CORES_PER_TASK=8
MEM_PER_TASK="24G"
MAX_CORES_TOTAL=64
MAX_MEM_TOTAL_GB=300
MAX_CONCURRENCY=$((MAX_CORES_TOTAL / CORES_PER_TASK))   # 8
MAX_ARRAY_INDEX=1000

# attribution settings
SPLIT="test"
EXPLAIN_N_RF=5000
EXPLAIN_N_XGB=5000
EXPLAIN_N_IG=2000
TOP_FEATURES=500
HEAD_WIN=30
TAIL_WIN=30
MID_BINS=1
IG_STEPS=32

# seqcnn re-train (only for missing model.ts)
SEQ_EPOCHS=25
SEQ_BS=256
# ==============================

ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
TASKDIR="$PROJECT_DIR/analysis_results/_tasks_attr_resplit_v1"
mkdir -p "$TASKDIR"

TASKS_TSV="$TASKDIR/tasks_attr.tsv"
TASKS_SEQ_TSV="$TASKDIR/tasks_seqcnn_train.tsv"

# build tasks
python - <<PY
import os
from pathlib import Path

project_dir = Path(os.environ["PROJECT_DIR"])
inputs_root = Path(os.environ["INPUTS_ROOT"])
models_root = Path(os.environ["MODELS_ROOT"])

tasks = []
tasks_seq = []

for tag_dir in sorted(models_root.glob("top*")):
    if not tag_dir.is_dir(): 
        continue
    tag = tag_dir.name
    for compare_dir in sorted(tag_dir.iterdir()):
        if not compare_dir.is_dir():
            continue
        compare = compare_dir.name
        for model_dir in sorted(compare_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name  # rf/xgb/seqcnn
            for variant_dir in sorted(model_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                variant = variant_dir.name
                dataset_dir = inputs_root / tag / compare / variant
                if not dataset_dir.exists():
                    # hard fail in bash later; but mark
                    pass
                tasks.append((tag, compare, model, variant, str(dataset_dir), str(variant_dir)))

                if model == "seqcnn":
                    ts = variant_dir / "model.ts"
                    if not ts.exists():
                        tasks_seq.append((tag, compare, variant, str(dataset_dir), str(variant_dir)))

Path(os.environ["TASKS_TSV"]).write_text("tag\tcompare_id\tmodel\tvariant\tdataset_dir\tmodel_dir\n" +
                                        "\n".join("\t".join(x) for x in tasks) + "\n")
Path(os.environ["TASKS_SEQ_TSV"]).write_text("tag\tcompare_id\tvariant\tdataset_dir\tmodel_dir\n" +
                                            "\n".join("\t".join(x) for x in tasks_seq) + "\n")

print("[DONE] attr tasks =", len(tasks), "->", os.environ["TASKS_TSV"])
print("[DONE] seqcnn-train tasks (missing model.ts) =", len(tasks_seq), "->", os.environ["TASKS_SEQ_TSV"])
PY

# helper to submit in chunks (array index <=1000)
submit_chunks () {
  local slurm_file="$1"
  local tasks_tsv="$2"
  local total="$3"
  local dep="$4"   # may be empty

  local offset=0
  while [ $offset -lt $total ]; do
    local left=$((total - offset))
    local chunk=$left
    if [ $chunk -gt $MAX_ARRAY_INDEX ]; then chunk=$MAX_ARRAY_INDEX; fi

    if [ -n "$dep" ]; then
      sbatch --dependency="$dep" \
        --export=ALL,TASKS_TSV="$tasks_tsv",OFFSET="$offset" \
        --array=1-$chunk%$MAX_CONCURRENCY "$slurm_file"
    else
      sbatch \
        --export=ALL,TASKS_TSV="$tasks_tsv",OFFSET="$offset" \
        --array=1-$chunk%$MAX_CONCURRENCY "$slurm_file"
    fi
    offset=$((offset + chunk))
  done
}

# slurm: seqcnn train (save model.ts)
SLURM_SEQ="$TASKDIR/run_seqcnn_train.slurm"
cat > "$SLURM_SEQ" <<'SLURM'
#!/bin/bash
#SBATCH -J ext_seqcnn_ts
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

set -euo pipefail
ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"

: "${TASKS_TSV:?need TASKS_TSV}"
: "${OFFSET:?need OFFSET}"

tid=$((SLURM_ARRAY_TASK_ID + OFFSET))
line=$(awk -v n=$tid 'NR==n+1{print; exit}' "$TASKS_TSV")
IFS=$'\t' read -r tag compare variant dataset_dir model_dir <<<"$line"

# 强校验 inputs，避免“inputs问题”
if [ ! -d "$dataset_dir" ]; then
  echo "[BAD] dataset_dir missing: $dataset_dir" >&2
  exit 2
fi
mkdir -p "$model_dir"

python "$ROOT/scripts/ext06c_train_seqcnn_external_one_save_ts.py" \
  --dataset_dir "$dataset_dir" \
  --out_dir "$model_dir" \
  --epochs 25 --batch_size 256 \
  --save_ts
SLURM

# slurm: attribution (rf/xgb/seqcnn)
SLURM_ATTR="$TASKDIR/run_attr.slurm"
cat > "$SLURM_ATTR" <<SLURM
#!/bin/bash
#SBATCH -J ext_attr
#SBATCH -p normal
#SBATCH -c $CORES_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH -t 1-00:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

set -euo pipefail
ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"

: "\${TASKS_TSV:?need TASKS_TSV}"
: "\${OFFSET:?need OFFSET}"

tid=\$((SLURM_ARRAY_TASK_ID + OFFSET))
line=\$(awk -v n=\$tid 'NR==n+1{print; exit}' "\$TASKS_TSV")
IFS=\$'\t' read -r tag compare model variant dataset_dir model_dir <<<"\$line"

# 强校验 inputs，避免“inputs问题”
if [ ! -d "\$dataset_dir" ]; then
  echo "[BAD] dataset_dir missing: \$dataset_dir" >&2
  exit 2
fi
if [ ! -d "\$model_dir" ]; then
  echo "[BAD] model_dir missing: \$model_dir" >&2
  exit 2
fi

# per-model explain_n
EXPLAIN_N=$EXPLAIN_N_RF
if [ "\$model" = "xgb" ]; then EXPLAIN_N=$EXPLAIN_N_XGB; fi
if [ "\$model" = "seqcnn" ]; then EXPLAIN_N=$EXPLAIN_N_IG; fi

python "\$ROOT/scripts/ext07_run_attr_external_one.py" \
  --dataset_dir "\$dataset_dir" \
  --model_dir   "\$model_dir" \
  --model "\$model" \
  --split "$SPLIT" \
  --explain_n "\$EXPLAIN_N" \
  --top_features $TOP_FEATURES \
  --head_win $HEAD_WIN --tail_win $TAIL_WIN --mid_bins $MID_BINS \
  --ig_steps $IG_STEPS
SLURM

# submit seqcnn train first if needed
SEQ_N=$(( $(wc -l < "$TASKS_SEQ_TSV") - 1 ))
ATTR_N=$(( $(wc -l < "$TASKS_TSV") - 1 ))

echo "[INFO] seqcnn_missing_model_ts_tasks = $SEQ_N"
echo "[INFO] attr_tasks_total = $ATTR_N"
echo "[INFO] MAX_ARRAY_INDEX = $MAX_ARRAY_INDEX  concurrency=$MAX_CONCURRENCY  cores/task=$CORES_PER_TASK mem/task=$MEM_PER_TASK"

DEP=""
if [ $SEQ_N -gt 0 ]; then
  echo "[INFO] submitting seqcnn (save model.ts) first..."
  # submit in chunks (<=1000)
  # capture jobid(s) is messy with multiple arrays; simplest: no dependency chaining here.
  # we instead recommend: wait until seqcnn tasks finish, then run attr.
  # BUT: 你要一键到底，我用 afterany 依赖最后一个提交的 jobid（足够稳，因为都是补缺失的）。
  jid_last=""
  offset=0
  while [ $offset -lt $SEQ_N ]; do
    left=$((SEQ_N - offset))
    chunk=$left; if [ $chunk -gt $MAX_ARRAY_INDEX ]; then chunk=$MAX_ARRAY_INDEX; fi
    out=$(sbatch --export=ALL,TASKS_TSV="$TASKS_SEQ_TSV",OFFSET="$offset" --array=1-$chunk%$((MAX_CORES_TOTAL/8)) "$SLURM_SEQ")
    echo "$out"
    jid_last=$(echo "$out" | awk '{print $4}')
    offset=$((offset + chunk))
  done
  DEP="afterok:$jid_last"
  echo "[INFO] will submit attr with dependency: $DEP"
fi

echo "[INFO] submitting attribution jobs..."
submit_chunks "$SLURM_ATTR" "$TASKS_TSV" "$ATTR_N" "$DEP"
echo "[DONE] submitted."
