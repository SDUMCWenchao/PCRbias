import os
import glob
import pandas as pd

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
WORK_DIR = os.path.join(BASE_DIR, "analysis/08_Features")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/09_Feature_Summary")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. 合并主特征表 (Features) ===
print("Merging Feature Tables...")
feat_files = sorted(glob.glob(os.path.join(WORK_DIR, "*_features.tsv")))

if not feat_files:
    print("Error: No feature files found!")
    exit(1)

# 使用迭代器方式读取，防止内存爆炸，但这里是 unique sequences，应该还好
# 为了安全，我们分块写入
out_feat = os.path.join(OUTPUT_DIR, "ALL_UNIQUE_FEATURES.tsv")
first = True

with open(out_feat, 'w') as f_out:
    for i, f in enumerate(feat_files):
        if i % 10 == 0: print(f"Processing {i}/{len(feat_files)}: {os.path.basename(f)}")
        with open(f, 'r') as f_in:
            header = f_in.readline()
            if first:
                f_out.write(header)
                first = False
            # 写入剩余行
            for line in f_in:
                f_out.write(line)

print(f"Features saved to: {out_feat}")

# === 2. 合并可视化表 (Enrichment Viz) ===
print("Merging Enrichment Viz Tables...")
viz_files = sorted(glob.glob(os.path.join(WORK_DIR, "*_enrich_viz.tsv")))
out_viz = os.path.join(OUTPUT_DIR, "ALL_ENRICHMENT_VIZ.tsv")
first = True

with open(out_viz, 'w') as f_out:
    for i, f in enumerate(viz_files):
        with open(f, 'r') as f_in:
            header = f_in.readline()
            if first:
                f_out.write(header)
                first = False
            for line in f_in:
                f_out.write(line)

print(f"Enrichment Viz data saved to: {out_viz}")