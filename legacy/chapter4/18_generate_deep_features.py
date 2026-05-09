import os
import glob
import pandas as pd
import numpy as np
import json
import re
import gc

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_HighK")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Deep")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. G-四链体检测函数 ===
def count_g4_motifs(seq):
    """
    检测典型 G-四链体特征: G{3,} N{1,7} G{3,} N{1,7} G{3,} N{1,7} G{3,}
    同时检测 C-四链体 (i-Motif) 潜力: C{3,} ...
    """
    if not isinstance(seq, str): return 0, 0
    seq = seq.upper()
    
    # 正则表达式 (Canonical G4)
    g4_pattern = r'([G]{3,}\w{1,7}){3,}[G]{3,}'
    c4_pattern = r'([C]{3,}\w{1,7}){3,}[C]{3,}' # i-Motif on complementary strand
    
    g4_matches = len(re.findall(g4_pattern, seq))
    c4_matches = len(re.findall(c4_pattern, seq))
    
    return g4_matches, c4_matches

# === 2. 局部 K-mer 展开函数 ===
def unpack_regional_kmers(df, col_name, prefix, target_k=[1, 2, 3]):
    """将 Head/Tail 的 JSON 展开为特征列"""
    print(f"  Unpacking {col_name} (K={target_k})...")
    
    def _parse(s):
        try:
            return json.loads(s)
        except:
            return {}
            
    dicts = df[col_name].apply(_parse)
    
    filtered_dicts = []
    for d in dicts:
        new_d = {k: v for k, v in d.items() if len(k) in target_k}
        filtered_dicts.append(new_d)
        
    # 转 DataFrame
    kmer_df = pd.DataFrame(filtered_dicts).fillna(0).astype('float32')
    
    # 归一化 (Count -> Frequency)
    # 假设 Head/Tail 区域长度约为 30
    kmer_df = kmer_df / 30.0
    
    # 重命名: Head_Freq_AAA
    kmer_df.columns = [f"{prefix}_Freq_{c}" for c in kmer_df.columns]
    
    return kmer_df

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_HighK.csv")))
print(f"Found {len(files)} files to process.")

for f in files:
    fname = os.path.basename(f)
    print(f"\nProcessing {fname}...")
    
    # 读取数据
    df = pd.read_csv(f)
    
    # --- A. 计算 G4 和 GC ---
    print("  Calculating G-Quadruplex & GC Stats...")
    
    g4_counts = []
    c4_counts = []
    gc_contents = []
    
    for seq in df['Sequence']:
        g, c = count_g4_motifs(seq)
        g4_counts.append(g)
        c4_counts.append(c)
        
        # GC Content
        l = len(seq)
        if l > 0:
            # === FIX: Rename variable 'gc' to 'gc_val' ===
            gc_val = (seq.count('G') + seq.count('C')) / l
        else:
            gc_val = 0
        gc_contents.append(gc_val)
        
    df['G4_Score'] = g4_counts
    df['iMotif_Score'] = c4_counts
    df['GC_Content'] = gc_contents
    df['GC_Squared'] = np.square(gc_contents) # 捕捉非线性
    
    # --- B. 展开 Head/Tail K-mers ---
    # Head (5' end)
    if 'Kmer_Head30_JSON' in df.columns:
        head_df = unpack_regional_kmers(df, 'Kmer_Head30_JSON', 'Head', target_k=[1, 2, 3])
        df = pd.concat([df, head_df], axis=1)
        
    # Tail (3' end)
    if 'Kmer_Tail30_JSON' in df.columns:
        tail_df = unpack_regional_kmers(df, 'Kmer_Tail30_JSON', 'Tail', target_k=[1, 2, 3])
        df = pd.concat([df, tail_df], axis=1)
        
    # --- C. 保存 ---
    # 清理不再需要的 JSON 列以减小体积
    drop_cols = ['Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    out_file = os.path.join(OUTPUT_DIR, fname.replace("_HighK.csv", "_Deep.csv"))
    df.to_csv(out_file, index=False)
    
    print(f"  Saved Deep features ({df.shape[1]} cols) to: {out_file}")
    
    del df
    if 'head_df' in locals(): del head_df
    if 'tail_df' in locals(): del tail_df
    
    # 现在 gc 指的是模块，可以正常回收了
    gc.collect()

print("\nDeep Feature Generation Complete.")