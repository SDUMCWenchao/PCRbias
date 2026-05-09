#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/PCR_bias_chapter4"
INPUTS_ROOT="$PROJECT_ROOT/external_test/analysis_results/05_ModelInputs_external_topbias_resplit_v1_resplit_v1_resplit_v1"
MODELS_ROOT="$PROJECT_ROOT/external_test/analysis_results/06_Models_external_topbias_v2_resplit_v1_resplit_v1_resplit_v1"

TASKS_DIR="$PROJECT_ROOT/external_test/analysis_results/_tasks_ext06_resplit"
TASKS_TSV="$TASKS_DIR/tasks.tsv"
SLURM_FILE="$TASKS_DIR/run_ext06_train.slurm"

mkdir -p "$TASKS_DIR"
rm -f "$TASKS_TSV"

# 生成任务：每个 dataset_dir -> 3(model) 行
python - <<'PY'
import os, glob
from pathlib import Path

inputs_root = Path(os.environ["INPUTS_ROOT"])
models_root = Path(os.environ["MODELS_ROOT"])
tasks_tsv   = Path(os.environ["TASKS_TSV"])
tasks_tsv.parent.mkdir(parents=True, exist_ok=True)

# dataset_dir: .../<tag>/<compare>/<variant>/y_train.npy
ys = sorted(inputs_root.glob("top*/**/y_train.npy"))
out = []
for y in ys:
    dataset_dir = y.parent
    rel = dataset_dir.relative_to(inputs_root)  # tag/compare/variant
    tag, compare, variant = rel.parts[0], rel.parts[1], rel.parts[2]
    for model in ["rf","xgb","seqcnn"]:
        out_dir = models_root / tag / compare / model / variant
        out.append((tag, compare, variant, model, str(dataset_dir), str(out_dir)))

with open(tasks_tsv, "w") as f:
    f.write("tag\tcompare_id\tvariant\tmodel\tdataset_dir\tout_dir\n")
    for r in out:
        f.write("\t".join(r) + "\n")

print("[DONE] tasks =", len(out), "->", tasks_tsv)
PY

# 生成 slurm 文件（每任务 16 核，最多并发 4 -> 64 核）
cat > "$SLURM_FILE" <<'SLURM'
#!/bin/bash
#SBATCH -J ext06_train
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -o external_test/analysis_results/_tasks_ext06_resplit/%x_%A_%a.out
#SBATCH -e external_test/analysis_results/_tasks_ext06_resplit/%x_%A_%a.err
#SBATCH --array=1-1%4

set -euo pipefail

PROJECT_ROOT="/path/to/PCR_bias_chapter4"
TASKS_TSV="$PROJECT_ROOT/external_test/analysis_results/_tasks_ext06_resplit/tasks.tsv"

line=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1{print; exit}' "$TASKS_TSV")
IFS=$'\t' read -r tag compare variant model dataset_dir out_dir <<<"$line"

mkdir -p "$out_dir"

# 你可以按需调参（这里给相对稳的默认）
if [[ "$model" == "rf" ]]; then
  python "$PROJECT_ROOT/scripts/ext06a_train_rf_external_one.py" \
    --dataset_dir "$dataset_dir" \
    --out_dir "$out_dir" \
    --n_estimators 1200 --n_jobs 16 \
    --min_samples_leaf 2 \
    --bootstrap --max_samples 0.7 \
    --max_features 0.3
elif [[ "$model" == "xgb" ]]; then
  python "$PROJECT_ROOT/scripts/ext06b_train_xgb_external_one.py" \
    --dataset_dir "$dataset_dir" \
    --out_dir "$out_dir" \
    --nthread 16 \
    --n_estimators 4000 \
    --learning_rate 0.03 \
    --max_depth 6 \
    --subsample 0.8 \
    --colsample_bytree 0.8
elif [[ "$model" == "seqcnn" ]]; then
  python "$PROJECT_ROOT/scripts/ext06c_train_seqcnn_external_one.py" \
    --dataset_dir "$dataset_dir" \
    --out_dir "$out_dir" \
    --epochs 25 --batch_size 256 \
    --num_workers 8
else
  echo "[BAD] unknown model=$model" >&2
  exit 2
fi
SLURM

# 设 array 范围
N=$(($(wc -l < "$TASKS_TSV")-1))
sed -i "s/#SBATCH --array=1-1%4/#SBATCH --array=1-${N}%4/" "$SLURM_FILE"

echo "[INFO] INPUTS_ROOT=$INPUTS_ROOT"
echo "[INFO] MODELS_ROOT=$MODELS_ROOT"
echo "[INFO] tasks=$N"
echo "[INFO] submitting $SLURM_FILE"
sbatch "$SLURM_FILE"
