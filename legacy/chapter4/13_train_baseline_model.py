import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Augmented")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/11_Model_Results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 函数: 训练单个文件 ===
def train_and_evaluate(file_path):
    pair_name = os.path.basename(file_path).replace("_Augmented.csv", "")
    print(f"\nTraining model for: {pair_name}...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None

    # === 核弹级数据清洗 (保留之前的修复) ===
    exclude_cols = [
        'Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y',
        'Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("  Coercing data types...")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Bias_Score_Y'] = pd.to_numeric(df['Bias_Score_Y'], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)

    print("  Clipping extreme values...")
    df[feature_cols] = df[feature_cols].clip(-1e10, 1e10)
    
    try:
        X = df[feature_cols].astype('float32')
        y = df['Bias_Score_Y'].astype('float32')
    except Exception as e:
        print(f"  Type conversion failed: {e}")
        return None
        
    print(f"  Cleaned Data: {X.shape[1]} features, {X.shape[0]} samples")
    
    if len(X) < 10:
        print("  Skipping: Not enough valid data points.")
        return None

    # =================================

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 初始化并训练 XGBoost
    # === FIX: Move early_stopping_rounds here ===
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=42,
        early_stopping_rounds=50  # <--- 新版写法：移到这里
    )
    
    try:
        # === FIX: Remove early_stopping_rounds form fit() ===
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except Exception as e:
        print(f"  XGBoost Training Failed: {e}")
        return None
    
    # 5. 预测与评估
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if len(y_test) > 1 and np.std(y_pred) > 1e-9:
        pearson_corr, _ = pearsonr(y_test, y_pred)
    else:
        pearson_corr = 0.0
    
    print(f"  [Result] R2: {r2:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson_corr:.4f}")
    
    # 6. 保存特征重要性
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_file = os.path.join(OUTPUT_DIR, f"{pair_name}_Importance.csv")
    importance.to_csv(importance_file, index=False)
    
    # 7. 可视化
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='blue', edgecolor=None)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f"{pair_name}\nR2={r2:.3f}, Corr={pearson_corr:.3f}")
    plt.xlabel("Actual Bias (Log Fold Change)")
    plt.ylabel("Predicted Bias")
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_DIR, f"{pair_name}_Pred_vs_Actual.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    return {
        'Pair': pair_name,
        'R2': r2,
        'RMSE': rmse,
        'Pearson': pearson_corr,
        'Top1_Feature': importance.iloc[0]['Feature'],
        'Top2_Feature': importance.iloc[1]['Feature'],
        'Top3_Feature': importance.iloc[2]['Feature']
    }

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Augmented.csv")))
results = []

for f in files:
    try:
        res = train_and_evaluate(f)
        if res:
            results.append(res)
    except Exception as e:
        print(f"CRITICAL ERROR processing {f}: {e}")

# 保存总报表
if results:
    df_res = pd.DataFrame(results)
    summary_file = os.path.join(OUTPUT_DIR, "Model_Performance_Summary.csv")
    df_res.to_csv(summary_file, index=False)
    print("\n" + "="*50)
    print("Modeling Complete.")
    print(f"Summary saved to: {summary_file}")
    print("="*50)
    print(df_res[['Pair', 'R2', 'Top1_Feature']])
else:
    print("No models trained successfully.")