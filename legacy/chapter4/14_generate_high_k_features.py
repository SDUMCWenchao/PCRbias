import os
import glob
import pandas as pd
import numpy as np
import json
import gc

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Augmented")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_HighK")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 函数：展开 K-mer ===
def unpack_kmer_features(df, k_list):
    """从 JSON 中提取指定 K 值的 K-mer 并向量化"""
    print(f"  Unpacking K-mers: {k_list}...")
    
    # 解析 JSON
    def _parse(json_str):
        try:
            return json.loads(json_str)
        except:
            return {}
            
    # 将 JSON 列转换为字典列表
    dicts = df['Kmer_Whole_JSON'].apply(_parse)
    
    # 筛选需要的 K 值
    # 为了避免创建几十万列的稀疏矩阵，我们先构建 list of dicts
    filtered_dicts = []
    for d in dicts:
        new_d = {k: v for k, v in d.items() if len(k) in k_list}
        filtered_dicts.append(new_d)
    
    # 转换为 DataFrame (自动对齐列，缺失补0)
    # 使用 float32 节省内存
    kmer_df = pd.DataFrame(filtered_dicts).fillna(0).astype('float32')
    
    # 归一化 (Count -> Frequency)
    seq_lens = df['Length'].values[:, None] # 广播
    kmer_df = kmer_df / seq_lens
    
    # 重命名列
    kmer_df.columns = [f"Freq_{c}" for c in kmer_df.columns]
    
    return kmer_df

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Augmented.csv")))
print(f"Found {len(files)} files to process.")

for f in files:
    fname = os.path.basename(f)
    print(f"\nProcessing {fname}...")
    
    # 1. 读取现有数据
    df = pd.read_csv(f)
    
    # 2. 提取 K=4, 5, 6
    # 原文件中已经有了 K=1,2,3 的 Freq_ 列，我们需要保留它们
    # 并追加新的 Kmer 列
    
    new_kmers = unpack_kmer_features(df, k_list=[4, 5, 6])
    print(f"  Added {new_kmers.shape[1]} new high-order K-mer features.")
    
    # 3. 合并
    # axis=1 横向拼接
    df_high_k = pd.concat([df, new_kmers], axis=1)
    
    # 4. 保存
    # 使用 float32 压缩体积
    float_cols = df_high_k.select_dtypes(include=['float64']).columns
    df_high_k[float_cols] = df_high_k[float_cols].astype('float32')
    
    out_path = os.path.join(OUTPUT_DIR, fname.replace("_Augmented.csv", "_HighK.csv"))
    df_high_k.to_csv(out_path, index=False)
    
    print(f"  Saved to: {out_path}")
    
    # 内存清理
    del df, new_kmers, df_high_k
    gc.collect()

print("\nHigh-K Feature Generation Complete.")