#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"

# === inputs/models roots (resplit v1) ===
INPUTS_ROOT="$PROJECT_ROOT/external_test/analysis_results/05_ModelInputs_external_topbias_resplit_v1"
MODELS_ROOT="$PROJECT_ROOT/external_test/analysis_results/06_Models_external_topbias_v2_resplit_v1"

# === tasks ===
TASKS_DIR="$PROJECT_ROOT/external_test/analysis_results/_tasks_ext07_attr"
TASKS_TSV="$TASKS_DIR/tasks.tsv"
SLURM_FILE="$TASKS_DIR/run_ext07_attr.slurm"

# === run params ===
SPLIT="test"
EXPLAIN_N=0      # 0 = explain all rows in split
TOPK=50
IG_STEPS=32
IG_BS=128

# === cluster constraints ===
MAX_SUBMIT=500    # 每次 sbatch array 任务数 <= 500（且 array index 不会超过 500）
CONCURRENCY=8     # 同时跑 8 个任务（每任务 8 核 → 总 64 核）
CPU_PER_TASK=8
MEM_PER_TASK="60G" # 60G * 8 = 480G <= 500G

mkdir -p "$TASKS_DIR"
rm -f "$TASKS_TSV"

# export 给 python heredoc
export INPUTS_ROOT MODELS_ROOT TASKS_TSV

python - <<'PY'
import os
from pathlib import Path

inputs_root = Path(os.environ["INPUTS_ROOT"])
models_root = Path(os.environ["MODELS_ROOT"])
tasks_tsv   = Path(os.environ["TASKS_TSV"])
tasks_tsv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for y in sorted(inputs_root.glob("top*/**/y_train.npy")):
    dataset_dir = y.parent
    rel = dataset_dir.relative_to(inputs_root)  # tag/compare/variant
    if len(rel.parts) < 3:
        continue
    tag, compare, variant = rel.parts[0], rel.parts[1], rel.parts[2]

    rf_dir   = models_root / tag / compare / "rf" / variant
    xgb_dir  = models_root / tag / compare / "xgb" / variant
    cnn_dir  = models_root / tag / compare / "seqcnn" / variant

    if rf_dir.exists():
        rows.append(("rf", str(dataset_dir), str(rf_dir)))
    if xgb_dir.exists():
        rows.append(("xgb", str(dataset_dir), str(xgb_dir)))
    if cnn_dir.exists():
        rows.append(("seqcnn", str(dataset_dir), str(cnn_dir)))

with open(tasks_tsv, "w") as f:
    f.write("job\tdataset_dir\tmodel_dir\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

print(f"[DONE] tasks={len(rows)} -> {tasks_tsv}")
PY

# slurm runner：array 永远 1..B，真正任务行号用 OFFSET 映射
cat > "$SLURM_FILE" <<SLURM
#!/bin/bash
#SBATCH -J ext07_attr
#SBATCH -c ${CPU_PER_TASK}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH -t 1-00:00:00
#SBATCH -o external_test/analysis_results/_tasks_ext07_attr/%x_%A_%a.out
#SBATCH -e external_test/analysis_results/_tasks_ext07_attr/%x_%A_%a.err
#SBATCH --array=1-1%${CONCURRENCY}

set -euo pipefail
PROJECT_ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
TASKS_TSV="\$PROJECT_ROOT/external_test/analysis_results/_tasks_ext07_attr/tasks.tsv"

SPLIT="${SPLIT}"
EXPLAIN_N="${EXPLAIN_N}"
TOPK="${TOPK}"
IG_STEPS="${IG_STEPS}"
IG_BS="${IG_BS}"

OFFSET="\${OFFSET:-0}"   # 由 sbatch --export 传入
TASK_ID=\$(( OFFSET + SLURM_ARRAY_TASK_ID ))  # 对应 tasks.tsv 的行号（不含表头）

line=\$(awk -v n=\$TASK_ID 'NR==n+1{print; exit}' "\$TASKS_TSV")
if [[ -z "\$line" ]]; then
  echo "[SKIP] no task line for TASK_ID=\$TASK_ID (OFFSET=\$OFFSET, AID=\$SLURM_ARRAY_TASK_ID)"
  exit 0
fi

IFS=\$'\\t' read -r job dataset_dir model_dir <<<"\$line"

if [[ "\$job" == "rf" ]]; then
  python "\$PROJECT_ROOT/scripts/ext07a_shap_rf_external_one.py" \
    --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
    --split "\$SPLIT" --explain_n "\$EXPLAIN_N" --topk "\$TOPK"
elif [[ "\$job" == "xgb" ]]; then
  python "\$PROJECT_ROOT/scripts/ext07b_shap_xgb_external_one.py" \
    --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
    --split "\$SPLIT" --explain_n "\$EXPLAIN_N" --topk "\$TOPK"
elif [[ "\$job" == "seqcnn" ]]; then
  python "\$PROJECT_ROOT/scripts/ext07c_ig_seqcnn_external_one.py" \
    --dataset_dir "\$dataset_dir" --model_dir "\$model_dir" \
    --split "\$SPLIT" --explain_n "\$EXPLAIN_N" --topk "\$TOPK" \
    --steps "\$IG_STEPS" --batch_size "\$IG_BS" --device cpu
else
  echo "[BAD] unknown job=\$job" >&2
  exit 2
fi
SLURM

N=$(( $(wc -l < "$TASKS_TSV") - 1 ))
if [[ $N -le 0 ]]; then
  echo "[BAD] no tasks generated. Check INPUTS_ROOT/MODELS_ROOT paths." >&2
  exit 2
fi

echo "[INFO] tasks_total=$N  batch_size=$MAX_SUBMIT  concurrency=%$CONCURRENCY  per_task=${CPU_PER_TASK}c ${MEM_PER_TASK}"
offset=0
while [[ $offset -lt $N ]]; do
  remain=$(( N - offset ))
  B=$MAX_SUBMIT
  if [[ $remain -lt $B ]]; then B=$remain; fi

  echo "[SUBMIT] sbatch --array=1-${B}%${CONCURRENCY}  OFFSET=$offset  $SLURM_FILE"
  sbatch --export=ALL,OFFSET=${offset} --array="1-${B}%${CONCURRENCY}" "$SLURM_FILE"

  offset=$(( offset + B ))
done

echo "[DONE] submitted all batches."
