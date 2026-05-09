import os
import pandas as pd
import numpy as np

# === 配置路径 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
META_FILE = os.path.join(BASE_DIR, "samples_meta.tsv")
INPUT_CSV_DIR = os.path.join(BASE_DIR, "analysis/09_Feature_Summary/Final_CSVs")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. 解析 Metadata 寻找配对 ===
print("Parsing metadata to find PCR vs No-PCR pairs...")
meta = pd.read_csv(META_FILE, sep='\t')

# 分组键: 物种, 位点, 个体数 (确保是同源样本对比)
# 例如: species=10mix, locus=12S, n=10 -> H12(yes) vs H2(no)
groups = meta.groupby(['species', 'locus', 'n_individuals'])

pairs = []
for (species, locus, n_ind), group in groups:
    pcr_yes = group[group['pcr'] == 'yes']
    pcr_no = group[group['pcr'] == 'no']
    
    if pcr_yes.empty or pcr_no.empty:
        continue
        
    # 取每组的第一个样本作为代表 (One-to-One Pairing)
    # 实际 ID
    id_yes = pcr_yes.iloc[0]['file_id']
    id_no = pcr_no.iloc[0]['file_id']
    
    pairs.append({
        'group_name': f"{species}_{locus}_{n_ind}ind",
        'id_yes': id_yes,
        'id_no': id_no
    })

print(f"Found {len(pairs)} valid pairs for alignment.")

# === 2. 执行对齐与 Y 计算 ===
levels = ['count2', 'top1pct', 'top0.5pct']

for pair in pairs:
    group_name = pair['group_name']
    id_yes = pair['id_yes']
    id_no = pair['id_no']
    
    print(f"\nProcessing Pair: {group_name} ({id_yes} vs {id_no})")
    
    for level in levels:
        # 构建文件名
        file_yes = os.path.join(INPUT_CSV_DIR, f"{id_yes}_{level}.csv")
        file_no = os.path.join(INPUT_CSV_DIR, f"{id_no}_{level}.csv")
        
        if not os.path.exists(file_yes) or not os.path.exists(file_no):
            print(f"  [Skip] Missing CSV files for level: {level}")
            continue
            
        # 读取数据
        # No-PCR 组只保留 Abundance 用于计算，特征用 PCR 组的即可 (因为 SeqID 相同特征相同)
        # 实际上为了保险，我们取交集
        try:
            df_yes = pd.read_csv(file_yes)
            df_no = pd.read_csv(file_no)
        except Exception as e:
            print(f"  Error reading csv: {e}")
            continue
            
        # 准备 Merge
        # df_no 只需: Seq_ID, Relative_Abundance (重命名为 Abundance_NoPCR)
        df_no_clean = df_no[['Seq_ID', 'Relative_Abundance']].rename(
            columns={'Relative_Abundance': 'Abundance_NoPCR'}
        )
        
        # df_yes 保留所有特征, Relative_Abundance 重命名为 Abundance_PCR
        df_yes_clean = df_yes.rename(
            columns={'Relative_Abundance': 'Abundance_PCR'}
        )
        
        # Inner Join: 只保留两边都测到的序列
        # 如果你想研究 Dropout (在 NoPCR 有但 PCR 丢了)，可以用 Left Join
        # 这里为了做回归预测 Bias 程度，Inner Join 最稳健
        df_aligned = pd.merge(df_yes_clean, df_no_clean, on='Seq_ID', how='inner')
        
        if df_aligned.empty:
            print(f"  [Warning] No common sequences found for level: {level}")
            continue
            
        # === 核心: 计算 Target Y (Bias Score) ===
        # Y = log2( Abundance_PCR / Abundance_NoPCR )
        # 加一个极小值 epsilon 防止除零 (虽然 inner join 理论上不为0)
        epsilon = 1e-9
        df_aligned['Bias_Score_Y'] = np.log2(
            (df_aligned['Abundance_PCR'] + epsilon) / 
            (df_aligned['Abundance_NoPCR'] + epsilon)
        )
        
        # 移动 Y 列到前面方便查看
        cols = list(df_aligned.columns)
        cols.insert(2, cols.pop(cols.index('Bias_Score_Y')))
        df_aligned = df_aligned[cols]
        
        # 保存结果
        out_file = os.path.join(OUTPUT_DIR, f"{group_name}_{level}_Aligned.csv")
        df_aligned.to_csv(out_file, index=False)
        
        print(f"  [Done] Level: {level}, Aligned Seqs: {len(df_aligned)}, Saved to: {os.path.basename(out_file)}")

print("\n" + "="*50)
print(f"Data Alignment Complete. Ready for Modeling.")
print(f"Output Directory: {OUTPUT_DIR}")
print("="*50)