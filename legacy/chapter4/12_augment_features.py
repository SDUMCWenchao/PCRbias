import os
import glob
import pandas as pd
import numpy as np
import json
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data") # 之前对齐好的数据
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Augmented")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. 定义新特征计算函数 ===

def calc_thermo_props(seq_str):
    """计算热力学特征 (Tm, GC Skew)"""
    try:
        # 1. Melting Temperature (Tm) - 使用标准 DNA/DNA 杂交参数
        # 假设标准的 PCR 盐浓度: Na+ 50mM, DNA 50nM
        my_seq = Seq(seq_str)
        tm_wallace = mt.Tm_Wallace(my_seq) # 简单法则
        tm_nn = mt.Tm_NN(my_seq, nn_table=mt.DNA_NN4) # 最近邻法 (更准)
        
        # 2. GC Skew: (G-C)/(G+C)
        # 衡量链的对称性，常用于复制原点预测，也会影响 PCR
        g = seq_str.count('G')
        c = seq_str.count('C')
        if g + c == 0:
            gc_skew = 0
        else:
            gc_skew = (g - c) / (g + c)
            
        return tm_wallace, tm_nn, gc_skew
    except:
        return 0, 0, 0

def detect_poly_n(seq_str):
    """检测连续的 Poly-N 区 (Poly-A/T/C/G)"""
    # 统计最长的连续相同碱基长度
    max_run = 0
    if not seq_str: return 0
    
    current_run = 1
    for i in range(1, len(seq_str)):
        if seq_str[i] == seq_str[i-1]:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    max_run = max(max_run, current_run)
    return max_run

# === 2. 批量处理 ===
csv_files = glob.glob(os.path.join(INPUT_DIR, "*_Aligned.csv"))
print(f"Found {len(csv_files)} files to augment.")

for f_path in csv_files:
    fname = os.path.basename(f_path)
    print(f"Augmenting: {fname}...")
    
    df = pd.read_csv(f_path)
    
    # --- A. 计算物理化学特征 ---
    print("  Calculating Thermodynamics & Poly-N...")
    
    # 使用 apply 批量计算
    # 返回: (Tm_Wallace, Tm_NN, GC_Skew)
    thermo_data = df['Sequence'].apply(lambda x: calc_thermo_props(x))
    
    # 拆分到列
    df['Tm_Wallace'] = [x[0] for x in thermo_data]
    df['Tm_NN'] = [x[1] for x in thermo_data]
    df['GC_Skew'] = [x[2] for x in thermo_data]
    
    # Poly-N
    df['Max_Homopolymer_Len'] = df['Sequence'].apply(detect_poly_n)
    
    # --- B. 展开 K-mer (Vectorization) ---
    # 我们只展开 K=1, K=2, K=3 (因为 K=4,5,6 会产生数千列，可能导致内存爆炸)
    # 对于 XGBoost，Top K-mers 已经包含在 JSON 里了，
    # 这里我们演示如何把 K=2 (16 cols) 和 K=3 (64 cols) 展开成独立特征
    
    print("  Vectorizing K-mers (K=2, K=3)...")
    
    # 解析 JSON
    def unpack_json(json_str, target_k_list):
        if pd.isna(json_str): return {}
        try:
            data = json.loads(json_str)
            # 筛选长度
            return {k: v for k, v in data.items() if len(k) in target_k_list}
        except:
            return {}

    # 应用转换
    # Kmer_Whole_JSON 包含 K=1..6
    kmer_dicts = df['Kmer_Whole_JSON'].apply(lambda x: unpack_json(x, [1, 2, 3]))
    
    # 转换为 DataFrame (这一步会自动 One-hot 展开)
    # fillna(0) 很关键，没出现的 kmer 频率为 0
    kmer_df = pd.DataFrame(list(kmer_dicts)).fillna(0)
    
    # 给列名加前缀，防止冲突
    kmer_df.columns = [f"Freq_{col}" for col in kmer_df.columns]
    
    # 计算频率 (归一化): Count / Sequence_Length
    # 这样长短序列才有可比性
    seq_lens = df['Length']
    for col in kmer_df.columns:
        kmer_df[col] = kmer_df[col] / seq_lens
        
    # --- C. 合并所有特征 ---
    # 将新的 K-mer 列拼接到主表右侧
    df_augmented = pd.concat([df, kmer_df], axis=1)
    
    # 移除原始的 JSON 列 (如果不需要了，或者保留以防万一)
    # 这里我们保留 JSON，因为它是原始数据
    
    # 保存
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".csv", "_Augmented.csv"))
    df_augmented.to_csv(out_path, index=False)
    
    print(f"  Saved {len(df_augmented.columns)} features to: {out_path}")

print("\nFeature Augmentation Complete!")