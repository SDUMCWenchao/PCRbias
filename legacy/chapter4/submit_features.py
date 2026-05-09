import os
import subprocess
import shutil

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INPUT_FASTA = os.path.join(BASE_DIR, "analysis/07_Seq_Extraction/ALL_UNIQUE_SEQUENCES.fasta")
WORK_DIR = os.path.join(BASE_DIR, "analysis/08_Features")
SPLIT_DIR = os.path.join(WORK_DIR, "splits")
LOG_DIR = os.path.join(WORK_DIR, "logs")
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts/08_feature_worker.py")

# 清理并创建目录
for d in [WORK_DIR, SPLIT_DIR, LOG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# === 1. 拆分 FASTA 文件 ===
print("Splitting FASTA file...")

# 统计总序列数
total_seqs = 0
with open(INPUT_FASTA) as f:
    for line in f:
        if line.startswith('>'): total_seqs += 1

print(f"Total sequences: {total_seqs}")

# 决定拆分数量 (目标: 100 个 chunks)
# 每个文件包含的行数 = (序列数 * 2行/seq) / 100
lines_per_chunk = int((total_seqs * 2) / 100) + 2

# 使用 Linux split 命令快速拆分
# -l: 按行数拆分, -d: 使用数字后缀, --additional-suffix: 后缀名
split_cmd = f"split -l {lines_per_chunk} -d --additional-suffix=.fasta {INPUT_FASTA} {SPLIT_DIR}/chunk_"
subprocess.run(split_cmd, shell=True, check=True)

# 获取所有生成的 chunk 文件
chunk_files = sorted([f for f in os.listdir(SPLIT_DIR) if f.endswith('.fasta')])
num_chunks = len(chunk_files)
print(f"Split into {num_chunks} files.")

# === 2. 生成 Task List ===
task_file = os.path.join(WORK_DIR, "feature_tasks.txt")
with open(task_file, 'w') as f:
    for chunk in chunk_files:
        in_path = os.path.join(SPLIT_DIR, chunk)
        out_prefix = os.path.join(WORK_DIR, chunk.replace(".fasta", ""))
        # 命令行: python worker.py input output_prefix
        cmd = f"python {SCRIPT_PATH} {in_path} {out_prefix}"
        f.write(cmd + "\n")

# === 3. 提交 Slurm Job Array ===
# 使用 %64 限制最大并发核心数
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=FeatExt
#SBATCH --output={LOG_DIR}/%A_%a.out
#SBATCH --error={LOG_DIR}/%A_%a.err
#SBATCH --array=1-{num_chunks}%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
echo "Processing Task ${{SLURM_ARRAY_TASK_ID}}: $CMD"
eval $CMD
"""

slurm_file = os.path.join(WORK_DIR, "run_features.slurm")
with open(slurm_file, 'w') as f:
    f.write(slurm_script)

print(f"Submitting {num_chunks} feature extraction tasks...")
subprocess.run(f"sbatch {slurm_file}", shell=True)
print(f"Submitted. Check logs in {LOG_DIR}")