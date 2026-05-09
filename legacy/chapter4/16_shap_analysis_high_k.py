import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import gc

# === Configuration ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_HighK")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/13_SHAP_Analysis_HighK")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Only analyze samples with good model performance
TARGET_KEYWORDS = ["top0.5pct", "top1pct"] 

def run_shap_high_k(file_path):
    pair_name = os.path.basename(file_path).replace("_HighK.csv", "")
    
    # Filter
    if not any(k in pair_name for k in TARGET_KEYWORDS):
        return

    print(f"\nRunning SHAP for High-K data: {pair_name}...")
    
    # 1. Read Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    # 2. Data Cleaning
    exclude_cols = [
        'Seq_ID', 'Sequence', 'Count', 'Abundance_PCR', 'Abundance_NoPCR', 'Bias_Score_Y',
        'Kmer_Whole_JSON', 'Kmer_Head30_JSON', 'Kmer_Tail30_JSON', 'Enrichment_Summary_JSON'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Simple cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Convert to numpy to avoid index issues
    X = df[feature_cols].astype('float32')
    y = df['Bias_Score_Y'].astype('float32')
    
    if len(X) < 50:
        print("  Skipping: Not enough data.")
        return

    # 3. Train Model (Native API)
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    
    # Using exact method for best compatibility
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'exact', 
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_jobs': 8,
        'eval_metric': 'rmse'
    }
    
    print("  Training model...")
    model = xgb.train(params, dtrain, num_boost_round=300)
    
    # 4. Calculate SHAP (Black-box Approach)
    print("  Calculating SHAP values (using Generic Explainer)...")
    
    # Wrapper function: input dataframe/numpy -> output predictions
    def predict_wrapper(data):
        if isinstance(data, pd.DataFrame):
            dm = xgb.DMatrix(data, feature_names=feature_cols)
        else:
            dm = xgb.DMatrix(data, feature_names=feature_cols)
        return model.predict(dm)

    try:
        # Background data for baseline (100 samples)
        if len(X) > 100:
            background_data = X.sample(100, random_state=42)
        else:
            background_data = X

        # Explain data
        # IMPORTANT: Reduced to 100 samples for speed because PermutationExplainer is slow
        if len(X) > 100:
            X_shap = X.sample(100, random_state=42)
        else:
            X_shap = X

        # Use Generic Explainer
        explainer = shap.Explainer(predict_wrapper, background_data)
        
        # Calculate SHAP values (Removed check_additivity=False)
        shap_values_obj = explainer(X_shap)
        
        # Extract values
        shap_values = shap_values_obj.values
        
        # 5. Plotting
        # Summary Plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.title(f"SHAP Summary (High-K): {pair_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Summary.png"), dpi=150)
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_SHAP_Bar.png"), dpi=150)
        plt.close()
        
        print(f"  Plots saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"  SHAP Calculation Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Free memory
    del df, X, y, model, dtrain
    if 'explainer' in locals(): del explainer
    if 'shap_values_obj' in locals(): del shap_values_obj
    gc.collect()

# === Main Loop ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_HighK.csv")))
print(f"Found {len(files)} High-K files.")

for f in files:
    try:
        run_shap_high_k(f)
    except Exception as e:
        print(f"Error processing {f}: {e}")

print("\nSHAP Analysis Complete.")