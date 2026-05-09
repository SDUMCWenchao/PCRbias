import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === 配置 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
# 注意：这里改为 HighK 目录
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_HighK")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/12_HighK_Model_Results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def train_and_evaluate(file_path):
    pair_name = os.path.basename(file_path).replace("_HighK.csv", "")
    print(f"\nTraining High-K model for: {pair_name}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None

    # === 数据清洗 ===
    exclude_cols = [
        'Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y',
        'Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 简单清洗 (HighK 生成时已经做过一部分了，这里兜底)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].astype('float32')
    y = df['Bias_Score_Y'].astype('float32')
    
    print(f"  Data shape: {X.shape}")
    
    if len(X) < 10: return None

    # === 训练 ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 启用 tree_method='hist' 加速大规模特征训练
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6, # 特征多了，降低采样比例防止过拟合
        n_jobs=8,             # 多核加速
        tree_method='hist',   # 针对大数据集的优化模式
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # === 评估 ===
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    pearson_corr = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else 0
    
    print(f"  [Result] R2: {r2:.4f}, Pearson: {pearson_corr:.4f}")
    
    # 保存重要性 (只存 Top 100，否则文件太大)
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(100)
    
    importance.to_csv(os.path.join(OUTPUT_DIR, f"{pair_name}_Top100_Importance.csv"), index=False)
    
    return {'Pair': pair_name, 'R2': r2, 'Top1': importance.iloc[0]['Feature']}

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_HighK.csv")))
results = []
for f in files:
    res = train_and_evaluate(f)
    if res: results.append(res)

if results:
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "HighK_Performance.csv"), index=False)
    print("High-K Modeling Complete.")