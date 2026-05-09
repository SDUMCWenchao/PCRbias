import sys
import os
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict

# === 参数接收 ===
# 用法: python worker.py <Ref_Fasta> <NoPCR_BAM> <PCR_BAM> <Output_Dir> <Sample_Label>
if len(sys.argv) != 6:
    print("Usage: worker.py <Ref> <NoPCR> <PCR> <OutDir> <Label>")
    sys.exit(1)

REF_FASTA, BAM_NO, BAM_PCR, OUT_DIR, LABEL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

# 定义输出文件路径
FILE_KMER = os.path.join(OUT_DIR, f"{LABEL}_kmer_bias.tsv")
FILE_BASE = os.path.join(OUT_DIR, f"{LABEL}_base_freq.tsv")

# === 1. 断点续跑检查 ===
if os.path.exists(FILE_KMER) and os.path.exists(FILE_BASE):
    print(f"Output files exist for {LABEL}. Skipping (Checkpoint).")
    sys.exit(0)

print(f"Processing: {LABEL}")

# === 2. 工具函数 ===

def get_depth_and_freq(bam_file, ref_file):
    """
    运行 samtools mpileup 并解析:
    1. 每个位点的深度 (用于 K-mer 分析)
    2. 每个位点的碱基频率 (用于碱基分析)
    """
    # -d 0: 无深度限制; -Q 0: 不过滤低质量基(为了看错误)
    cmd = f"samtools mpileup -d 0 -Q 0 -f {ref_file} {bam_file}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    depths = {} # pos -> depth
    freqs = {}  # pos -> {A:0.1, T:0.9...}
    
    for line in process.stdout:
        parts = line.strip().split('\t')
        if len(parts) < 5: continue
        
        # mpileup columns: chrom, pos, ref, depth, bases, qual
        try:
            chrom, pos, ref_base, depth, base_str = parts[:5]
            pos = int(pos)
            depth = int(depth)
            
            if depth == 0: continue

            depths[pos] = depth
            
            # 简易解析 bases 字符串统计频率
            # 注意: 这不处理 Indel, 仅关注 SNV 频率
            base_str = base_str.upper()
            cnt = {'A':0, 'T':0, 'C':0, 'G':0}
            
            # 统计 Matches (. 和 ,)
            matches = base_str.count('.') + base_str.count(',')
            if ref_base.upper() in cnt:
                cnt[ref_base.upper()] += matches
                
            # 统计 Mismatches
            cnt['A'] += base_str.count('A')
            cnt['T'] += base_str.count('T')
            cnt['C'] += base_str.count('C')
            cnt['G'] += base_str.count('G')
            
            total_counted = sum(cnt.values())
            if total_counted > 0:
                freqs[pos] = {b: c/total_counted for b,c in cnt.items()}
                freqs[pos]['Ref'] = ref_base.upper()
                
        except ValueError:
            continue
            
    return depths, freqs

def read_reference(fasta_path):
    seq = ""
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip().upper()
    return seq

# === 3. 数据加载与预处理 ===
print("Reading Reference...")
ref_seq = read_reference(REF_FASTA)
ref_len = len(ref_seq)

print("Parsing No-PCR BAM...")
depth_no, freq_no = get_depth_and_freq(BAM_NO, REF_FASTA)

print("Parsing PCR BAM...")
depth_pcr, freq_pcr = get_depth_and_freq(BAM_PCR, REF_FASTA)

# 计算全局平均深度 (用于归一化)
# 防止除以零
mean_depth_no = np.mean(list(depth_no.values())) if depth_no else 1.0
mean_depth_pcr = np.mean(list(depth_pcr.values())) if depth_pcr else 1.0

print(f"Global Depth - NoPCR: {mean_depth_no:.2f}, PCR: {mean_depth_pcr:.2f}")

# === 4. K-mer 分析 (支持 3-mer, 4-mer 及扩展) ===
# K-mer Bias 定义: (Local_PCR_Depth / Global_PCR_Depth) / (Local_No_Depth / Global_No_Depth)
# > 1: PCR 偏好扩增; < 1: PCR 抑制扩增

kmer_stats = []
k_sizes = [3, 4] # 在这里扩展 [3, 4, 5, 6]

for k in k_sizes:
    print(f"Analyzing {k}-mers...")
    kmer_buffer = defaultdict(list)
    
    for i in range(ref_len - k + 1):
        pos_start = i + 1 # 1-based
        kmer_seq = ref_seq[i : i+k]
        
        # 获取该 k-mer 覆盖区域的平均深度
        d_no_list = [depth_no.get(p, 0) for p in range(pos_start, pos_start + k)]
        d_pcr_list = [depth_pcr.get(p, 0) for p in range(pos_start, pos_start + k)]
        
        local_mean_no = np.mean(d_no_list)
        local_mean_pcr = np.mean(d_pcr_list)
        
        # 仅分析 No-PCR 组有足够覆盖度的区域 (>5X)，避免噪声
        if local_mean_no > 5:
            norm_no = local_mean_no / mean_depth_no
            norm_pcr = local_mean_pcr / mean_depth_pcr
            
            ratio = norm_pcr / norm_no if norm_no > 0 else 0
            kmer_buffer[kmer_seq].append(ratio)
            
    # 汇总当前 k 长度的统计
    for km, ratios in kmer_buffer.items():
        kmer_stats.append({
            'K': k,
            'Kmer': km,
            'Count': len(ratios),
            'Mean_Ratio': np.mean(ratios),
            'Std_Ratio': np.std(ratios)
        })

# 保存 K-mer 结果
pd.DataFrame(kmer_stats).sort_values('Mean_Ratio').to_csv(FILE_KMER, sep='\t', index=False)

# === 5. 碱基频率差异分析 ===
# 寻找 No-PCR 和 PCR 组中，同一位点碱基频率差异最大的情况
print("Analyzing Base Frequencies...")
base_diff_stats = []

common_positions = set(freq_no.keys()) & set(freq_pcr.keys())

for pos in common_positions:
    f_no = freq_no[pos]
    f_pcr = freq_pcr[pos]
    ref = f_no['Ref']
    
    # 找到差异最大的碱基
    max_diff = 0
    change_desc = "-"
    
    for base in ['A', 'T', 'C', 'G']:
        # 频率: PCR - NoPCR
        diff = f_pcr.get(base, 0) - f_no.get(base, 0)
        if abs(diff) > abs(max_diff):
            max_diff = diff
            change_desc = f"Delta_{base}" # 正值代表 PCR 中该碱基增多
            
    # 仅记录显著差异 (>5% 频率变化) 以减小文件体积
    if abs(max_diff) > 0.05:
        base_diff_stats.append({
            'Position': pos,
            'Ref_Base': ref,
            'Depth_No': depth_no.get(pos, 0),
            'Depth_PCR': depth_pcr.get(pos, 0),
            'Max_Diff': round(max_diff, 4),
            'Change_Type': change_desc,
            'Freq_No_A': round(f_no.get('A', 0), 3),
            'Freq_No_T': round(f_no.get('T', 0), 3),
            'Freq_No_C': round(f_no.get('C', 0), 3),
            'Freq_No_G': round(f_no.get('G', 0), 3),
            'Freq_PCR_A': round(f_pcr.get('A', 0), 3),
            'Freq_PCR_T': round(f_pcr.get('T', 0), 3),
            'Freq_PCR_C': round(f_pcr.get('C', 0), 3),
            'Freq_PCR_G': round(f_pcr.get('G', 0), 3),
        })

# 保存碱基分析结果
pd.DataFrame(base_diff_stats).sort_values('Max_Diff', key=abs, ascending=False).to_csv(FILE_BASE, sep='\t', index=False)

print("Done.")