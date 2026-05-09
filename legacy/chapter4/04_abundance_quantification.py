import os
import subprocess
import pandas as pd

# === 配置路径 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/"
MAPPING_DIR = os.path.join(BASE_DIR, "analysis", "02_Mapping")
OUTPUT_FILE = os.path.join(BASE_DIR, "analysis", "04_10mix_Abundance_Counts.tsv")

# === 1. 目标样本 (10mix) ===
target_ids = ["H2", "H6", "H12", "H16"]
bam_files = [os.path.join(MAPPING_DIR, f"{fid}.sorted.bam") for fid in target_ids]

# === 2. 统计丰度 (samtools idxstats) ===
all_data = {}

print("Starting abundance quantification...")
for bam in bam_files:
    if not os.path.exists(bam):
        print(f"Warning: Missing {bam}")
        continue
        
    sample_id = os.path.basename(bam).split('.')[0]
    # 运行 samtools idxstats
    cmd = f"samtools idxstats {bam}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 解析结果
    counts = {}
    total_mapped = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split('\t')
        chrom = parts[0]
        reads = int(parts[2])
        if chrom != '*':
            counts[chrom] = reads
            total_mapped += reads
            
    # 计算相对丰度
    for chrom, reads in counts.items():
        if total_mapped > 0:
            all_data.setdefault(chrom, {})[sample_id] = reads / total_mapped
        else:
             all_data.setdefault(chrom, {})[sample_id] = 0

# === 3. 输出结果 ===
df = pd.DataFrame(all_data).transpose() # 转置：行是物种，列是样本
df = df.fillna(0).round(4) # 填补空缺值并保留4位小数

# 添加理论值列 (假设是等比例混合，每种 0.1)
# 注意：如果实际不是等比例，请忽略此列
df['Expected'] = 0.1 

df.to_csv(OUTPUT_FILE, sep='\t')
print(f"Done! Results saved to: {OUTPUT_FILE}")
print("\nPreview of Relative Abundance:")
print(df)