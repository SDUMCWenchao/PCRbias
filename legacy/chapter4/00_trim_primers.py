import os
import pandas as pd
import subprocess

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
# 新的输出目录
OUT_DIR = os.path.join(BASE_DIR, "raw_data_trimmed")
META_FILE = os.path.join(BASE_DIR, "samples_meta.tsv")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# === 引物定义 (使用 IUPAC 通配符) ===
# F: 正向引物 (出现在 Read 5' 端)
# R_RC: 反向引物的反向互补 (出现在 Read 3' 端)

# 12S rRNA
# F: GGGATTAGATACCCCACTATGCYTA (Y=C/T)
# R: GAGGGTGACGGGCGGTGT -> RC: ACACCGCCCGTCACCCTC
PRIMERS_12S = {
    'g': 'GGGATTAGATACCCCACTATGCYTA', # -g: 锚定 5' 端 (必须以此开头)
    'a': 'ACACCGCCCGTCACCCTC'        # -a: 3' 端 adapter (切除 R 引物)
}

# 16S rRNA
# F: ACCAAAAACATCACCTCYAGCAT
# R: AATAGGATTGCGCTGTTATCCCTA -> RC: TAGGGATAACAGCGCAATCCTATT
PRIMERS_16S = {
    'g': 'ACCAAAAACATCACCTCYAGCAT',
    'a': 'TAGGGATAACAGCGCAATCCTATT'
}

# === 主流程 ===
df = pd.read_csv(META_FILE, sep='\t')

print(f"Start trimming primers for {len(df)} samples...")

for _, row in df.iterrows():
    sample_id = row['file_id']
    locus = row['locus']
    
    # 确定输入文件
    # 尝试 .fq 和 .fastq
    if os.path.exists(os.path.join(RAW_DIR, f"{sample_id}.fq")):
        in_file = os.path.join(RAW_DIR, f"{sample_id}.fq")
    elif os.path.exists(os.path.join(RAW_DIR, f"{sample_id}.fastq")):
        in_file = os.path.join(RAW_DIR, f"{sample_id}.fastq")
    else:
        print(f"Skipping {sample_id}: File not found.")
        continue
        
    out_file = os.path.join(OUT_DIR, f"{sample_id}.fq")
    
    # 选择引物
    if locus == '12S':
        p = PRIMERS_12S
    elif locus == '16S':
        p = PRIMERS_16S
    else:
        print(f"Skipping {sample_id}: Unknown locus {locus}")
        continue
    
    # 构造 cutadapt 命令
    # -g: 切除 5' 端引物 (Anchor)
    # -a: 切除 3' 端引物 (Adapter)
    # --discard-untrimmed: 丢弃没有找到引物的 reads (可选，如果你确定所有 reads 都是扩增子，建议加上，保证纯度)
    # -e 0.2: 允许 20% 的错配率 (处理测序错误和简并碱基)
    # --minimum-length 50: 丢弃切完后太短的序列
    
    cmd = [
        "cutadapt",
        "-g", p['g'],
        "-a", p['a'],
        "-e", "0.2",
        "--minimum-length", "50",
        "-o", out_file,
        in_file
    ]
    
    print(f"[{locus}] Trimming {sample_id}...")
    # print(" ".join(cmd)) # 调试用
    
    try:
        # 运行并捕获输出日志 (避免刷屏)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error trimming {sample_id}")

print("\n" + "="*50)
print(f"Trimming Complete. Clean data is in: {OUT_DIR}")
print("Please update your pipeline to use this directory.")
print("="*50)