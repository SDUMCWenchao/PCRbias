import sys
import os
import gzip
import pandas as pd
import hashlib

# === 参数接收 ===
# python 07_extract_sequences.py <fq_file> <sample_id> <output_dir>
if len(sys.argv) != 4:
    print("Usage: python 07_extract_sequences.py <fq_file> <sample_id> <output_dir>")
    sys.exit(1)

FQ_FILE, SAMPLE_ID, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]

# === 函数：读取 FASTQ 并计数 ===
def count_sequences(fq_path):
    seq_counts = {}
    total_reads = 0
    
    # 自动识别 gzip
    open_func = gzip.open if fq_path.endswith('.gz') else open
    mode = 'rt' if fq_path.endswith('.gz') else 'r'
    
    try:
        with open_func(fq_path, mode) as f:
            while True:
                # FASTQ 4行一个单元
                header = f.readline()
                if not header: break
                seq = f.readline().strip().upper() # 第2行是序列
                f.readline() # +
                f.readline() # Quality
                
                if seq:
                    seq_counts[seq] = seq_counts.get(seq, 0) + 1
                    total_reads += 1
    except Exception as e:
        print(f"Error reading {fq_path}: {e}")
        sys.exit(1)
        
    return seq_counts, total_reads

# === 主流程 ===
print(f"Processing {SAMPLE_ID} from {FQ_FILE}...")

# 1. 计数
counts, total = count_sequences(FQ_FILE)
print(f"Total Reads: {total}, Unique Sequences: {len(counts)}")

# 2. 转换为 DataFrame
df = pd.DataFrame(list(counts.items()), columns=['Sequence', 'Count'])

# 3. 过滤 Count >= 2
df = df[df['Count'] >= 2].copy()
print(f"Unique Sequences (Count >= 2): {len(df)}")

if df.empty:
    print("Warning: No sequences found with count >= 2.")
    sys.exit(0)

# 4. 计算相对含量 (RPKM-like or Proportion)
df['Relative_Abundance'] = df['Count'] / total

# 5. 排序 (按 Count 降序)
df = df.sort_values('Count', ascending=False).reset_index(drop=True)

# 6. 生成唯一 Sequence ID (MD5 hash) 方便后续索引
# 这样长序列在特征表中就变成了一个短ID
df['Seq_ID'] = df['Sequence'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

# 7. 标记含量水平 (Top 1%, Top 0.5%)
# 这里定义为：在去重后的序列列表中，排名在前 1% 的序列
num_unique = len(df)
top1_cutoff_rank = int(num_unique * 0.01)
top05_cutoff_rank = int(num_unique * 0.005)

# 排名 (0-based)
df['Rank'] = df.index
df['Level_Tag'] = 'count_2' # 默认水平
df.loc[df['Rank'] < top1_cutoff_rank, 'Level_Tag'] = 'top1%'
df.loc[df['Rank'] < top05_cutoff_rank, 'Level_Tag'] = 'top0.5%'

# 为了方便筛选，也可以加布尔列
df['Is_Top1'] = df['Rank'] < top1_cutoff_rank
df['Is_Top0.5'] = df['Rank'] < top05_cutoff_rank

# 8. 保存结果
out_file = os.path.join(OUT_DIR, f"{SAMPLE_ID}_seq_stats.tsv")
# 调整列顺序，把 Seq_ID 放前面
cols = ['Seq_ID', 'Count', 'Relative_Abundance', 'Level_Tag', 'Is_Top1', 'Is_Top0.5', 'Sequence']
df[cols].to_csv(out_file, sep='\t', index=False)

print(f"Saved to {out_file}")