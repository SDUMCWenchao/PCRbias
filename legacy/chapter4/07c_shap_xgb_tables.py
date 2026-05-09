#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

KMER_RE = re.compile(r"^k(\d+)_(head30|tail30|mid\d+)_")

def load_X(ds: Path, split: str):
    npz = ds / f"X_{split}.npz"
    npy = ds / f"X_{split}.npy"
    if npz.exists():
        return sparse.load_npz(npz).tocsr().astype(np.float32)
    if npy.exists():
        X = np.load(npy, allow_pickle=False)
        return sparse.csr_matrix(X.astype(np.float32, copy=False))
    raise FileNotFoundError(f"missing X_{split}.npz or X_{split}.npy in {ds}")

def read_feat_names(ds: Path):
    fn = ds / "feature_names.txt"
    if not fn.exists():
        return None
    return [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]

def parse_group(feature: str):
    m = KMER_RE.match(feature)
    if m:
        return ("kmer", f"k{m.group(1)}", m.group(2))
    return ("non_kmer", "NA", "NA")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True, help=".../xgb/<variant>")
    ap.add_argument("--split", default="test", choices=["train","val","test"])

    # 下面这些参数保留（为了和你旧脚本/任务兼容），但 xgb 原生 TreeSHAP 不需要 background
    ap.add_argument("--background_n", type=int, default=4000)
    ap.add_argument("--explain_n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--top_features", type=int, default=500, help="keep top-N by mean|SHAP|")
    ap.add_argument("--local_top", type=int, default=0, help="0 disables; >0 writes per-sample top contributions")
    args = ap.parse_args()

    import xgboost as xgb

    ds = Path(args.dataset_dir)
    md = Path(args.model_dir)
    out = md / "shap_tables"
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    X = load_X(ds, args.split)
    y = np.load(ds / f"y_{args.split}.npy").astype(np.float32, copy=False)
    meta = pd.read_csv(ds / f"meta_{args.split}.tsv.gz", sep="\t", compression="gzip")

    feat = read_feat_names(ds)
    if feat is None:
        feat = [f"f{i}" for i in range(X.shape[1])]

    # Load model
    booster = xgb.Booster()
    booster.load_model(str(md / "model.json"))

    rng = np.random.default_rng(args.seed)
    n = X.shape[0]
    e_idx = rng.choice(np.arange(n), size=min(args.explain_n, n), replace=False)
    X_explain = X[e_idx]

    # Build DMatrix with feature names (important for alignment)
    dmat = xgb.DMatrix(X_explain, feature_names=feat)

    # ---- Key fix: use XGBoost native TreeSHAP ----
    # pred_contribs=True returns (n, p+1) with last column = bias term
    contrib = booster.predict(dmat, pred_contribs=True, approx_contribs=False)
    contrib = np.asarray(contrib, dtype=np.float32)
    shap_vals = contrib[:, :-1]   # (m, p)
    bias = contrib[:, -1]         # (m,)

    # Normal prediction (for local output)
    y_pred = booster.predict(dmat)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_val = np.mean(shap_vals, axis=0)

    df = pd.DataFrame({
        "feature": feat,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_val,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if args.top_features and len(df) > args.top_features:
        df = df.iloc[:args.top_features].copy()

    df.to_csv(out / f"shap_global_{args.split}.tsv", sep="\t", index=False)

    # region aggregation
    rows = []
    for _, r in df.iterrows():
        tp, kk, reg = parse_group(r["feature"])
        rows.append((tp, kk, reg, float(r["mean_abs_shap"]), float(r["mean_shap"])))
    g = pd.DataFrame(rows, columns=["type","k","region","mean_abs_shap","mean_shap"])
    gsum = g.groupby(["type","k","region"], as_index=False)[["mean_abs_shap","mean_shap"]].sum()
    gsum = gsum.sort_values(["type","mean_abs_shap"], ascending=[True, False])
    gsum.to_csv(out / f"shap_region_{args.split}.tsv", sep="\t", index=False)

    # optional local top contributions
    if args.local_top and args.local_top > 0:
        top = int(args.local_top)
        sub_meta = meta.iloc[e_idx].reset_index(drop=True)
        local = []
        for i in range(shap_vals.shape[0]):
            sv = shap_vals[i]
            idx = np.argsort(np.abs(sv))[::-1][:top]
            feats = [feat[j] for j in idx]
            vals = [float(sv[j]) for j in idx]
            local.append({
                "Seq_ID": str(sub_meta.loc[i, "Seq_ID"]) if "Seq_ID" in sub_meta.columns else str(i),
                "pair_id": str(sub_meta.loc[i, "pair_id"]) if "pair_id" in sub_meta.columns else "",
                "y_true": float(y[e_idx[i]]),
                "y_pred": float(y_pred[i]),
                "bias": float(bias[i]),
                "top_features": json.dumps(feats, ensure_ascii=False),
                "top_shap": json.dumps(vals, ensure_ascii=False),
            })
        pd.DataFrame(local).to_csv(out / f"shap_local_top{top}_{args.split}.tsv.gz",
                                  sep="\t", index=False, compression="gzip")

    info = {
        "method": "xgboost_pred_contribs_treeSHAP",
        "dataset_dir": str(ds),
        "model_dir": str(md),
        "split": args.split,
        "explain_n": int(X_explain.shape[0]),
        "top_features": int(args.top_features),
        "seed": args.seed,
        "note": "background_n ignored (native TreeSHAP does not require background dataset)",
    }
    (out / f"shap_info_{args.split}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")

    print(f"[DONE] SHAP tables -> {out}")

if __name__ == "__main__":
    main()
