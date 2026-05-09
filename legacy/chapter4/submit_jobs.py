import os
import pandas as pd
import subprocess

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts/05_bias_worker.py")
META_FILE = os.path.join(BASE_DIR, "samples_meta.tsv")
MAPPING_DIR = os.path.join(BASE_DIR, "analysis/02_Mapping")
OUT_DIR = os.path.join(BASE_DIR, "analysis/06_Advanced_Seq_Analysis")
REF_DIR = os.path.join(BASE_DIR, "refs")
LOG_DIR = os.path.join(OUT_DIR, "logs")

# 确保目录存在
for d in [OUT_DIR, LOG_DIR, os.path.dirname(SCRIPT_PATH)]:
    if not os.path.exists(d):
        os.makedirs(d)

# === 1. 解析 Metadata 并配对 ===
df = pd.read_csv(META_FILE, sep='\t')

# 参考基因组映射
ref_map = {
    '10mix': 'combined_mito_ref.fasta',
    'donkey': 'donkey.fasta',
    'cattle': 'cattle.fasta',
    'pig': 'pig.fasta'
}

# 按 (species, locus) 分组寻找 PCR 对
tasks = []
groups = df.groupby(['species', 'locus'])

for (species, locus), group in groups:
    # 找到 PCR=yes 和 PCR=no 的样本
    pcr_yes = group[group['pcr'] == 'yes']
    pcr_no = group[group['pcr'] == 'no']
    
    # 检查是否成对
    if pcr_yes.empty or pcr_no.empty:
        print(f"Skipping {species} {locus}: Unpaired (Yes: {len(pcr_yes)}, No: {len(pcr_no)})")
        continue
        
    # 获取文件 ID (假设组内样本需要两两比较，或者取第一个代表)
    # 这里我们取组内的第一个样本进行对比 (One-vs-One)
    # 也可以扩展为 Many-vs-Many，这里简化为每组取一对代表性样本
    
    # 注意：Metadata 中有些组有多个 No-PCR (如 Donkey 12S 有 L2, L4)。
    # 策略：优先取 n_individuals=10 的样本进行对比，或者取第一个。
    # 这里取列表中的第一个作为代表。
    id_yes = pcr_yes.iloc[0]['file_id']
    id_no = pcr_no.iloc[0]['file_id']
    
    # 构建路径
    bam_yes = os.path.join(MAPPING_DIR, f"{id_yes}.sorted.bam")
    bam_no = os.path.join(MAPPING_DIR, f"{id_no}.sorted.bam")
    
    ref_file = os.path.join(REF_DIR, ref_map.get(species, 'unknown.fasta'))
    
    if not os.path.exists(bam_yes) or not os.path.exists(bam_no):
        print(f"Warning: BAM files missing for {species} {locus}")
        continue
        
    label = f"{species}_{locus}_{id_no}_vs_{id_yes}"
    
    # 生成单行命令
    cmd = f"python {SCRIPT_PATH} {ref_file} {bam_no} {bam_yes} {OUT_DIR} {label}"
    tasks.append(cmd)

print(f"Generated {len(tasks)} comparison tasks.")

# === 2. 生成 Task List 文件 ===
task_list_file = os.path.join(OUT_DIR, "task_list.txt")
with open(task_list_file, 'w') as f:
    for t in tasks:
        f.write(t + "\n")

# === 3. 生成并提交 Slurm 脚本 ===
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=SeqBias
#SBATCH --output={LOG_DIR}/%A_%a.out
#SBATCH --error={LOG_DIR}/%A_%a.err
#SBATCH --array=1-{len(tasks)}%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 获取当前任务的命令行
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_list_file})

echo "Running task ${{SLURM_ARRAY_TASK_ID}}: $CMD"
eval $CMD
"""

slurm_file = os.path.join(OUT_DIR, "run_analysis.slurm")
with open(slurm_file, 'w') as f:
    f.write(slurm_script)

print(f"Submitting Slurm Job Array...")
subprocess.run(f"sbatch {slurm_file}", shell=True)
print(f"Check logs at: {LOG_DIR}")