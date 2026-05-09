import os
import pandas as pd
import subprocess

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
RAW_DIR = os.path.join(BASE_DIR, "raw_data_trimmed")
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts/07_extract_sequences.py")
META_FILE = os.path.join(BASE_DIR, "samples_meta.tsv")
OUT_DIR = os.path.join(BASE_DIR, "analysis/07_Seq_Extraction")
LOG_DIR = os.path.join(OUT_DIR, "logs")

for d in [OUT_DIR, LOG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# === 1. 生成任务列表 ===
df = pd.read_csv(META_FILE, sep='\t')
tasks = []

for _, row in df.iterrows():
    sample_id = row['file_id'] # e.g. H12
    # 尝试寻找 .fq 或 .fastq
    fq_path = os.path.join(RAW_DIR, f"{sample_id}.fq")
    if not os.path.exists(fq_path):
        fq_path = os.path.join(RAW_DIR, f"{sample_id}.fastq")
    
    if os.path.exists(fq_path):
        cmd = f"python {SCRIPT_PATH} {fq_path} {sample_id} {OUT_DIR}"
        tasks.append(cmd)
    else:
        print(f"Warning: Raw file not found for {sample_id}")

# === 2. 写入任务文件 ===
task_file = os.path.join(OUT_DIR, "extraction_tasks.txt")
with open(task_file, 'w') as f:
    for t in tasks:
        f.write(t + "\n")

# === 3. 提交 Job Array ===
# 提取步骤主要耗 I/O，CPU 占用不高，可以跑多一点
# 64核，我们提交 32 个并行任务
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=SeqExtract
#SBATCH --output={LOG_DIR}/%A_%a.out
#SBATCH --error={LOG_DIR}/%A_%a.err
#SBATCH --array=1-{len(tasks)}%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
echo "Running: $CMD"
eval $CMD
"""

slurm_file = os.path.join(OUT_DIR, "run_extract.slurm")
with open(slurm_file, 'w') as f:
    f.write(slurm_script)

print(f"Submitting {len(tasks)} extraction tasks...")
subprocess.run(f"sbatch {slurm_file}", shell=True)
print("Submitted.")