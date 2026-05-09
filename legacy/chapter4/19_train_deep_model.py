import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Deep")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/15_Deep_Model_Results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def train_rf(file_path):
    pair_name = os.path.basename(file_path).replace("_Deep.csv", "")
    print(f"\nTraining Deep Model (RF) for: {pair_name}...")
    
    try:
        df = pd.read_csv(file_path)
    except:
        return None

    # 清洗
    exclude_cols = ['Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].astype('float32')
    y = df['Bias_Score_Y'].astype('float32')
    
    if len(X) < 50: return None
    
    # 训练
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用 RandomForest，开启多核
    model = RandomForestRegressor(
        n_estimators=150, 
        max_depth=12,      # 稍微加深一点深度，捕捉复杂交互
        n_jobs=16, 
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else 0
    
    print(f"  [Result] R2: {r2:.4f}, Pearson: {pearson:.4f}")
    
    # 特征重要性
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 保存 Top 50
    imp_file = os.path.join(OUTPUT_DIR, f"{pair_name}_Importance.csv")
    importance.head(50).to_csv(imp_file, index=False)
    
    # 检查 Top 特征是否包含新特征 (Head/Tail/G4)
    top_feats = importance.head(10)['Feature'].tolist()
    new_feat_hits = [f for f in top_feats if "Head" in f or "Tail" in f or "G4" in f or "GC" in f]
    
    if new_feat_hits:
        print(f"  >>> Discovery! New features in Top 10: {new_feat_hits}")
    
    return {
        'Pair': pair_name,
        'R2': r2,
        'Pearson': pearson,
        'Top1': importance.iloc[0]['Feature'],
        'Top2': importance.iloc[1]['Feature'],
        'Top3': importance.iloc[2]['Feature'],
        'New_Feats_In_Top10': len(new_feat_hits)
    }

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Deep.csv")))
results = []
for f in files:
    res = train_rf(f)
    if res: results.append(res)

if results:
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "Deep_Model_Summary.csv"), index=False)
    print("\nDeep Modeling Complete.")