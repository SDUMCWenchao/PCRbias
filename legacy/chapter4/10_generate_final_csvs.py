import os
import glob
import pandas as pd
import gc

# === 配置路径 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
STATS_DIR = os.path.join(BASE_DIR, "analysis/07_Seq_Extraction") # 样本统计信息 (Count/Abundance)
FEATURE_FILE = os.path.join(BASE_DIR, "analysis/09_Feature_Summary/ALL_UNIQUE_FEATURES.tsv") # 特征库
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/09_Feature_Summary/Final_CSVs") # 输出目录

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. 加载全局特征库 (Master Feature Table) ===
print(f"Loading Global Feature Table from {FEATURE_FILE}...")
# 读取特征表
try:
    df_features = pd.read_csv(FEATURE_FILE, sep='\t')
    print(f"Loaded {len(df_features)} unique feature records.")
except Exception as e:
    print(f"Error loading features: {e}")
    exit(1)

# === 2. 遍历每个样本并合并 ===
sample_files = glob.glob(os.path.join(STATS_DIR, "*_seq_stats.tsv"))
print(f"Found {len(sample_files)} sample files to process.")

for i, s_file in enumerate(sample_files):
    sample_id = os.path.basename(s_file).replace("_seq_stats.tsv", "")
    print(f"[{i+1}/{len(sample_files)}] Processing sample: {sample_id}...")
    
    # 2.1 读取样本统计 (Seq_ID, Count, Relative_Abundance, Level_Tags)
    try:
        df_sample = pd.read_csv(s_file, sep='\t')
    except Exception as e:
        print(f"  Error reading sample file: {e}")
        continue

    # 2.2 合并特征 (Left Join: 以样本中的序列为准)
    # 样本表中有 Seq_ID, 特征表也有 Seq_ID
    df_merged = pd.merge(df_sample, df_features, on='Seq_ID', how='left')
    
    # 2.3 字段清洗 (保留核心字段)
    # 我们需要的列：
    # - 基础信息: Seq_ID, Sequence, Count, Relative_Abundance
    # - 物理特征: Length, Entropy, LZ_Complexity, Runs, MFE, Ensemble_Energy
    # - 高维特征: Kmer_Whole_JSON, Kmer_Head30_JSON, Kmer_Tail30_JSON, Enrichment_Summary_JSON
    
    keep_cols = [
        'Seq_ID', 'Sequence', 'Count', 'Relative_Abundance', 
        'Length', 'Entropy', 'LZ_Complexity', 'Runs', 'MFE', 'Ensemble_Energy',
        'Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON'
    ]
    
    # 确保列存在 (防止某些列丢失)
    final_cols = [c for c in keep_cols if c in df_merged.columns]
    
    # 2.4 分级输出
    
    #Level 1: Count >= 2 (Total)
    # 原始文件已经是 count >= 2 了，直接保存
    out_count2 = os.path.join(OUTPUT_DIR, f"{sample_id}_count2.csv")
    df_merged[final_cols].to_csv(out_count2, index=False)
    
    # Level 2: Top 1%
    # 利用 Is_Top1 列进行筛选
    if 'Is_Top1' in df_merged.columns:
        df_top1 = df_merged[df_merged['Is_Top1'] == True]
        out_top1 = os.path.join(OUTPUT_DIR, f"{sample_id}_top1pct.csv")
        df_top1[final_cols].to_csv(out_top1, index=False)
    
    # Level 3: Top 0.5%
    # 利用 Is_Top0.5 列进行筛选
    if 'Is_Top0.5' in df_merged.columns:
        df_top05 = df_merged[df_merged['Is_Top0.5'] == True]
        out_top05 = os.path.join(OUTPUT_DIR, f"{sample_id}_top0.5pct.csv")
        df_top05[final_cols].to_csv(out_top05, index=False)
        
    # 释放内存
    del df_merged
    gc.collect()

print("\n" + "="*50)
print(f"All Summary CSVs generated in: {OUTPUT_DIR}")
print("Files structure: {SampleID}_{Level}.csv")
print("="*50)