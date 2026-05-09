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
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
# 修改输入为 Deep 数据
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Deep")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/16_SHAP_Analysis_Deep")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 只分析模型表现较好的样本 (Top 0.5% 或 Top 1%)
TARGET_KEYWORDS = ["top0.5pct", "top1pct"] 

def run_shap_deep(file_path):
    pair_name = os.path.basename(file_path).replace("_Deep.csv", "")
    
    if not any(k in pair_name for k in TARGET_KEYWORDS):
        return

    print(f"\n[Deep-SHAP] Running analysis for: {pair_name}...")
    
    try:
        df = pd.read_csv(file_path)
    except:
        return

    # 清洗
    exclude_cols = ['Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].astype('float32')
    y = df['Bias_Score_Y'].astype('float32')
    
    if len(X) < 50: return

    # 训练 RF
    print("  Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=16, random_state=42)
    model.fit(X, y)
    
    # SHAP
    print("  Calculating SHAP...")
    if len(X) > 200:
        X_shap = X.sample(200, random_state=42)
    else:
        X_shap = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)
    
    # 绘图 1: Summary (Beeswarm)
    plt.figure(figsize=(12, 10))
    # max_display=20 让我们能看到更多特征（如 Head/Tail 是否在榜单前列）
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.title(f"SHAP Summary (Deep Features): {pair_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Summary.png"), dpi=150)
    plt.close()
    
    # 绘图 2: Dependence Plot (Top 1 特征)
    # 看看最重要的特征（如 G4_Score 或 Head_Freq_GGG）与 Bias 的具体关系
    vals = np.abs(shap_values).mean(0)
    top_idx = np.argsort(vals)[-1]
    top_name = X.columns[top_idx]
    
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(top_name, shap_values, X_shap, show=False)
    plt.title(f"Dependence: {top_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Dep_{top_name}.png"), dpi=150)
    plt.close()
    
    print(f"  Saved plots to {OUTPUT_DIR}")
    
    del df, X, y, model, explainer, shap_values
    gc.collect()

# 主循环
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Deep.csv")))
for f in files:
    run_shap_deep(f)

print("\nDeep SHAP Analysis Complete.")