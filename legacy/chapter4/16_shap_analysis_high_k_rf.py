import os
import glob
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_HighK")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/13_SHAP_Analysis_HighK_RF") # 新输出目录

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 只分析模型表现较好的样本
TARGET_KEYWORDS = ["top0.5pct", "top1pct"] 

def run_shap_rf(file_path):
    pair_name = os.path.basename(file_path).replace("_HighK.csv", "")
    
    if not any(k in pair_name for k in TARGET_KEYWORDS):
        return

    print(f"\n[RF-Mode] Running SHAP for: {pair_name}...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    # 2. 数据清洗
    exclude_cols = [
        'Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y',
        'Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 转换为 float32 节省内存
    X = df[feature_cols].astype('float32')
    y = df['Bias_Score_Y'].astype('float32')
    
    if len(X) < 50:
        print("  Skipping: Not enough data.")
        return

    # 3. 训练 Random Forest 模型
    # 为了速度，限制树的数量和深度，这对于特征重要性排序已经足够了
    print("  Training Random Forest (sklearn)...")
    model = RandomForestRegressor(
        n_estimators=100,      # 100棵树足够看特征了
        max_depth=10,          # 限制深度防止内存爆炸
        n_jobs=16,             # 并行加速
        random_state=42
    )
    model.fit(X, y)
    
    # 4. 计算 SHAP
    print("  Calculating SHAP values (TreeExplainer)...")
    
    # 降采样：随机森林的 SHAP 计算比较慢，我们取 200 个样本及其 Top 特征
    # 这足以展示规律
    if len(X) > 200:
        X_shap = X.sample(200, random_state=42)
    else:
        X_shap = X
        
    try:
        # Scikit-learn 模型原生支持 TreeExplainer，且非常稳定
        explainer = shap.TreeExplainer(model)
        
        # check_additivity=False 可以避免因浮点数精度导致的微小误差报错
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
        
        # 5. 绘图
        # Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.title(f"SHAP Summary (RF): {pair_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Summary.png"), dpi=150)
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Bar.png"), dpi=150)
        plt.close()
        
        print(f"  Plots saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"  SHAP Failed: {e}")
        import traceback
        traceback.print_exc()

    # 释放内存
    del df, X, y, model, explainer, shap_values, X_shap
    gc.collect()

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_HighK.csv")))
print(f"Found {len(files)} High-K files.")

for f in files:
    try:
        run_shap_rf(f)
    except Exception as e:
        print(f"Error processing {f}: {e}")

print("\nSHAP Analysis Complete.")