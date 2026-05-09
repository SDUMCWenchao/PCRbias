import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/06_Advanced_Seq_Analysis")
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "analysis/06_Advanced_Seq_Analysis/plots")

if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

# === 1. 加载数据 ===
print("Loading results...")
kmer_files = glob.glob(os.path.join(INPUT_DIR, "*_kmer_bias.tsv"))
base_files = glob.glob(os.path.join(INPUT_DIR, "*_base_freq.tsv"))

if not kmer_files:
    print("No result files found! Please wait for Slurm jobs to finish.")
    exit()

# --- K-mer 数据整合 ---
kmer_dfs = []
for f in kmer_files:
    label = os.path.basename(f).replace("_kmer_bias.tsv", "")
    # label 格式: species_locus_noID_vs_yesID
    # 简化 label 用于绘图
    simple_label = "_".join(label.split("_")[:2]) # e.g., donkey_16S
    
    df = pd.read_csv(f, sep='\t')
    df['Sample_Group'] = simple_label
    kmer_dfs.append(df)

df_kmer_all = pd.concat(kmer_dfs, ignore_index=True)

# --- 碱基突变数据整合 ---
base_dfs = []
for f in base_files:
    label = os.path.basename(f).replace("_base_freq.tsv", "")
    simple_label = "_".join(label.split("_")[:2])
    
    try:
        df = pd.read_csv(f, sep='\t')
        if not df.empty:
            df['Sample_Group'] = simple_label
            base_dfs.append(df)
    except pd.errors.EmptyDataError:
        continue

if base_dfs:
    df_base_all = pd.concat(base_dfs, ignore_index=True)
else:
    df_base_all = pd.DataFrame()

# === 2. 绘图: K-mer Bias Heatmap ===
# 我们关注 PCR 效率最低 (Depleted) 和 最高 (Enriched) 的 K-mer
print("Plotting K-mer Bias...")

# 筛选 3-mer 或 4-mer (这里展示 3-mer)
target_k = 3
subset = df_kmer_all[df_kmer_all['K'] == target_k].copy()

if not subset.empty:
    # 透视表: 行=Kmer, 列=Sample, 值=Mean_Ratio
    pivot = subset.pivot_table(index='Kmer', columns='Sample_Group', values='Mean_Ratio')
    
    # 找出变异最大的 K-mer (方差最大的 Top 30) 以避免图太大
    pivot['variance'] = pivot.var(axis=1)
    top_variable_kmers = pivot.sort_values('variance', ascending=False).head(30).index
    plot_data = pivot.loc[top_variable_kmers].drop(columns=['variance'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(plot_data, center=1.0, cmap="vlag", annot=True, fmt=".2f",
                cbar_kws={'label': 'PCR Bias Ratio (PCR/NoPCR)\n<1: Depleted, >1: Enriched'})
    plt.title(f"Top 30 Variable {target_k}-mer Bias Across Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, f"Kmer_{target_k}_Bias_Heatmap.pdf"))
    plt.close()

# === 3. 绘图: 碱基突变谱 (Mutation Spectrum) ===
if not df_base_all.empty:
    print("Plotting Mutation Spectrum...")
    
    # 统计每个样本中，每种突变类型 (Change_Type) 的数量
    # Change_Type 格式如 "Delta_A" (代表 PCR 中 A 增多)
    mut_counts = df_base_all.groupby(['Sample_Group', 'Change_Type']).size().reset_index(name='Count')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=mut_counts, x='Sample_Group', y='Count', hue='Change_Type')
    plt.title("Frequency of PCR-Induced Base Composition Changes (>5% shift)")
    plt.ylabel("Number of Genomic Sites with Significant Shift")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "PCR_Mutation_Counts.pdf"))
    plt.close()
    
    # 深入分析: 具体的突变方向 (A->G 等)
    # 这需要检查 Ref_Base 和 Change_Type
    # 如果 Change_Type 是 "Delta_G" 且 Ref 是 "A"，那就是 A->G
    # 这里做一个简单的表格输出
    print("\n=== Top PCR-Induced Site Changes ===")
    print(df_base_all.sort_values('Max_Diff', key=abs, ascending=False).head(10)[['Sample_Group', 'Position', 'Ref_Base', 'Change_Type', 'Max_Diff']])

print(f"\nAll plots saved to: {OUTPUT_PLOT_DIR}")