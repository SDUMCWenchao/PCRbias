import sys
import os
import zlib
import math
import json
import collections
import subprocess
import re
import pandas as pd
import numpy as np

# === 输入参数 ===
if len(sys.argv) != 3:
    print("Usage: python 08_feature_worker.py <input_fasta> <output_prefix>")
    sys.exit(1)

INPUT_FASTA = sys.argv[1]
OUT_PREFIX = sys.argv[2]

# === 1. ViennaRNA 批量计算函数 (CLI版) ===
def run_rnafold_batch(fasta_file):
    """
    调用命令行 RNAfold -p --noPS 处理整个 fasta 文件。
    解析输出获取 MFE 和 Ensemble Energy。
    返回字典: {seq_id: (mfe, ensemble_energy)}
    """
    print(f"Running RNAfold on {fasta_file}...")
    
    # 构造命令: 计算配分函数(-p)以获得系综能量, 不生成PS文件(--noPS)
    # 注意：确保 RNAfold 在系统 PATH 中 (既然您说环境已激活)
    cmd = f"RNAfold -p --noPS < {fasta_file}"
    
    try:
        # 运行命令并捕获标准输出
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold: {e.stderr}")
        return {}

    results = {}
    current_id = None
    mfe = 0.0
    ens = 0.0
    
    # 解析 RNAfold 输出
    # 典型输出格式:
    # >seq_id
    # SEQUENCE
    # ..((...)).. ( -3.50)  <-- MFE
    # ..((...)).. [ -3.60]  <-- Ensemble (如果有 -p)
    # frequency of mfe structure in ensemble 0.xxxx; ensemble diversity yyyy
    
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>'):
            # 提取 ID (去掉 >)
            current_id = line[1:].split()[0] # 只取第一个词作为ID
            i += 1
            if i >= len(lines): break
            
            # 下一行是序列 (跳过)
            # seq_line = lines[i] 
            i += 1
            if i >= len(lines): break
            
            # 下一行是 MFE 结构
            mfe_line = lines[i]
            # 提取括号里的数字 ( -3.50)
            mfe_match = re.search(r'\(\s*([-\d\.]+)\s*\)$', mfe_line)
            mfe = float(mfe_match.group(1)) if mfe_match else 0.0
            
            i += 1
            if i >= len(lines): break
            
            # 下一行通常是 Ensemble (因为用了 -p)
            # 格式: ...structure... [ -3.60]
            ens_line = lines[i]
            if '[' in ens_line and ']' in ens_line:
                 ens_match = re.search(r'\[\s*([-\d\.]+)\s*\]', ens_line)
                 ens = float(ens_match.group(1)) if ens_match else mfe # Fallback
                 i += 1 # Skip probability line
            else:
                 # 如果没有 -p 输出，Ensemble 就用 MFE 代替
                 ens = mfe
                 # 这一行不是 ensemble line，回退还是继续？
                 # 通常 RNAfold 输出比较固定，如果没有 -p 这一行就是下一条记录的 >ID 或者结束
                 if ens_line.startswith('>'):
                     i -= 1 # 回退，给下一次循环处理 >ID
            
            results[current_id] = (mfe, ens)
        else:
            i += 1
            
    return results

# === 2. 其他特征计算函数 ===

def calc_kmer_freq(seq, k_range):
    """统计 K-mer 频数 (稀疏存储)"""
    kmers = {}
    length = len(seq)
    for k in k_range:
        if length < k: continue
        for i in range(length - k + 1):
            kmer = seq[i : i+k]
            kmers[kmer] = kmers.get(kmer, 0) + 1
    return json.dumps(kmers)

def calc_complexity(seq):
    """计算复杂度和熵"""
    if not seq: return 0, 0, 0
    
    # 1. Shannon Entropy
    counts = collections.Counter(seq)
    total = len(seq)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    
    # 2. Lempel-Ziv Complexity Proxy
    lz_proxy = len(zlib.compress(seq.encode())) / total
    
    # 3. Runs
    runs = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs += 1
            
    return round(entropy, 4), round(lz_proxy, 4), runs

def find_enrichment(seq, motifs, windows, thresholds):
    """寻找碱基富集区"""
    summary = []
    details = []
    seq_len = len(seq)
    
    base_map = {b: i for i, b in enumerate("ATCG")}
    arr = np.zeros((len(seq) + 1, 4), dtype=int)
    
    for i, base in enumerate(seq):
        arr[i+1] = arr[i]
        if base in base_map:
            arr[i+1][base_map[base]] += 1
            
    def get_count(start, end, motif):
        vec = arr[end] - arr[start]
        count = 0
        for char in motif:
            count += vec[base_map[char]]
        return count

    for motif in motifs:
        for w in windows:
            if seq_len < w: continue
            for thresh in thresholds:
                min_count = w * thresh
                enriched_intervals = []
                current_start = -1
                current_end = -1
                
                for i in range(seq_len - w + 1):
                    cnt = get_count(i, i+w, motif)
                    if cnt >= min_count:
                        if current_start == -1:
                            current_start = i
                            current_end = i + w
                        elif i <= current_end:
                            current_end = max(current_end, i + w)
                        else:
                            enriched_intervals.append((current_start, current_end))
                            current_start = i
                            current_end = i + w
                if current_start != -1:
                    enriched_intervals.append((current_start, current_end))
                
                if enriched_intervals:
                    total_len = sum(end - start for start, end in enriched_intervals)
                    num_zones = len(enriched_intervals)
                    summary.append({
                        'Motif': motif, 'Window': w, 'Threshold': thresh,
                        'Num_Zones': num_zones, 'Total_Length': total_len
                    })
                    for start, end in enriched_intervals:
                        details.append({
                            'Motif': motif, 'Window': w, 'Threshold': thresh,
                            'Start': start + 1, 'End': end
                        })
    return summary, details

# === 3. 主处理流程 ===

# Step A: 先批量运行 RNAfold 获取结构数据
rna_struct_data = run_rnafold_batch(INPUT_FASTA)
print(f"Loaded structure data for {len(rna_struct_data)} sequences.")

features_data = []
enrich_viz_data = []

motifs = ['A', 'T', 'C', 'G', 'GC', 'AT', 'GT', 'AC', 'GA', 'CT']
windows = [10, 20, 30, 40, 50]
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

print(f"Processing sequence features for: {INPUT_FASTA}")

with open(INPUT_FASTA) as f:
    while True:
        line = f.readline()
        if not line: break
        if not line.startswith('>'): continue
        
        seq_id = line.strip()[1:] # 提取 ID
        seq = f.readline().strip().upper()
        if not seq: continue
        
        # --- A. 基础特征 ---
        entropy, lz, runs = calc_complexity(seq)
        
        # --- B. 结构特征 (查表) ---
        # 如果 RNAfold 失败或跳过，给默认值 0
        mfe, ens_energy = rna_struct_data.get(seq_id, (0.0, 0.0))
        
        # --- C. K-mer ---
        kmer_whole = calc_kmer_freq(seq, range(1, 7))
        head_seq = seq[:30]
        tail_seq = seq[-30:] if len(seq) > 30 else ""
        kmer_head = calc_kmer_freq(head_seq, range(1, 5))
        kmer_tail = calc_kmer_freq(tail_seq, range(1, 5))
        
        # --- D. 富集区 ---
        enrich_summary, enrich_details = find_enrichment(seq, motifs, windows, thresholds)
        
        # --- E. 整合 ---
        row = {
            'Seq_ID': seq_id,
            'Length': len(seq),
            'Entropy': entropy,
            'LZ_Complexity': lz,
            'Runs': runs,
            'MFE': mfe,
            'Ensemble_Energy': ens_energy,
            'Kmer_Whole_JSON': kmer_whole,
            'Kmer_Head30_JSON': kmer_head,
            'Kmer_Tail30_JSON': kmer_tail,
            'Enrichment_Summary_JSON': json.dumps(enrich_summary)
        }
        features_data.append(row)
        
        for d in enrich_details:
            d['Seq_ID'] = seq_id
            enrich_viz_data.append(d)

# === 4. 输出 ===
df_feat = pd.DataFrame(features_data)
df_viz = pd.DataFrame(enrich_viz_data)

df_feat.to_csv(f"{OUT_PREFIX}_features.tsv", sep='\t', index=False)

if not df_viz.empty:
    cols = ['Seq_ID', 'Motif', 'Window', 'Threshold', 'Start', 'End']
    df_viz[cols].to_csv(f"{OUT_PREFIX}_enrich_viz.tsv", sep='\t', index=False)
else:
    with open(f"{OUT_PREFIX}_enrich_viz.tsv", 'w') as f:
        f.write("Seq_ID\tMotif\tWindow\tThreshold\tStart\tEnd\n")

print(f"Done. Features saved to {OUT_PREFIX}_features.tsv")