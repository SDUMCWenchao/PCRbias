import os
import glob
import pandas as pd

BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/07_Seq_Extraction")
OUTPUT_FASTA = os.path.join(INPUT_DIR, "ALL_UNIQUE_SEQUENCES.fasta")
OUTPUT_META = os.path.join(INPUT_DIR, "ALL_UNIQUE_SEQUENCES_META.tsv")

print("Merging unique sequences from all samples...")

files = glob.glob(os.path.join(INPUT_DIR, "*_seq_stats.tsv"))
unique_pool = {} # Seq_ID -> Sequence

for f in files:
    print(f"Reading {os.path.basename(f)}...")
    # 只读取 Seq_ID 和 Sequence 两列，节省内存
    try:
        df = pd.read_csv(f, sep='\t', usecols=['Seq_ID', 'Sequence'])
        # 更新字典 (去重)
        for _, row in df.iterrows():
            unique_pool[row['Seq_ID']] = row['Sequence']
    except Exception as e:
        print(f"Skipping {f}: {e}")

print(f"Total global unique sequences: {len(unique_pool)}")

# 输出为 FASTA
print(f"Writing FASTA to {OUTPUT_FASTA}...")
with open(OUTPUT_FASTA, 'w') as f:
    for seq_id, seq in unique_pool.items():
        f.write(f">{seq_id}\n{seq}\n")

# 输出一个 ID 对照表
print(f"Writing Meta table to {OUTPUT_META}...")
pd.DataFrame(list(unique_pool.items()), columns=['Seq_ID', 'Sequence']).to_csv(OUTPUT_META, sep='\t', index=False)

print("Done. Ready for feature extraction.")