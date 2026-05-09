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

def read_feat_names(ds: Path, ncol: int):
    fn = ds / "feature_names.txt"
    if fn.exists():
        feat = [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(feat) == ncol:
            return feat
    return [f"f{i}" for i in range(ncol)]

def parse_group(feature: str):
    m = KMER_RE.match(feature)
    if m:
        return ("kmer", f"k{m.group(1)}", m.group(2))
    return ("non_kmer", "NA", "NA")

def find_model_file(model_dir: Path):
    # robust search
    cands = [
        "model.joblib", "model.pkl", "model.pickle",
        "rf.joblib", "rf.pkl", "rf.pickle",
        "estimator.joblib", "estimator.pkl", "estimator.pickle",
    ]
    for c in cands:
        p = model_dir / c
        if p.exists() and p.stat().st_size > 0:
            return p
    for p in sorted(model_dir.glob("*.joblib")) + sorted(model_dir.glob("*.pkl")) + sorted(model_dir.glob("*.pickle")):
        if p.stat().st_size > 0:
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True, help=".../rf/<variant>")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top_features", type=int, default=500)
    ap.add_argument("--local_top", type=int, default=0)
    args = ap.parse_args()

    import joblib
    import shap

    ds = Path(args.dataset_dir)
    md = Path(args.model_dir)
    out = md / "shap_tables"
    out.mkdir(parents=True, exist_ok=True)

    X = load_X(ds, args.split)
    y = np.load(ds / f"y_{args.split}.npy").astype(np.float32, copy=False)
    meta = pd.read_csv(ds / f"meta_{args.split}.tsv.gz", sep="\t", compression="gzip")

    feat = read_feat_names(ds, X.shape[1])

    mf = find_model_file(md)
    if mf is None:
        raise SystemExit(f"[ERROR] cannot find RF model file in {md} (need *.joblib/*.pkl)")

    model = joblib.load(mf)

    rng = np.random.default_rng(args.seed)
    n = X.shape[0]
    e_idx = rng.choice(np.arange(n), size=min(args.explain_n, n), replace=False)
    Xex = X[e_idx]
    Xex_dense = Xex.toarray() if sparse.issparse(Xex) else np.asarray(Xex)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    # handle both old/new shap APIs
    sv = explainer.shap_values(Xex_dense, check_additivity=False)
    sv = np.asarray(sv, dtype=np.float32)  # (m,p)

    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_val = np.mean(sv, axis=0)

    df = pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs, "mean_shap": mean_val})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    if args.top_features and len(df) > args.top_features:
        df = df.iloc[:args.top_features].copy()
    df.to_csv(out / f"shap_global_{args.split}.tsv", sep="\t", index=False)

    rows = []
    for _, r in df.iterrows():
        tp, kk, reg = parse_group(r["feature"])
        rows.append((tp, kk, reg, float(r["mean_abs_shap"]), float(r["mean_shap"])))
    g = pd.DataFrame(rows, columns=["type","k","region","mean_abs_shap","mean_shap"])
    gsum = g.groupby(["type","k","region"], as_index=False)[["mean_abs_shap","mean_shap"]].sum()
    gsum = gsum.sort_values(["type","mean_abs_shap"], ascending=[True, False])
    gsum.to_csv(out / f"shap_region_{args.split}.tsv", sep="\t", index=False)

    if args.local_top and args.local_top > 0:
        top = int(args.local_top)
        sub_meta = meta.iloc[e_idx].reset_index(drop=True)
        try:
            y_pred = model.predict(Xex_dense)
        except Exception:
            y_pred = np.full((Xex_dense.shape[0],), np.nan, dtype=float)

        local = []
        for i in range(sv.shape[0]):
            s = sv[i]
            idx = np.argsort(np.abs(s))[::-1][:top]
            local.append({
                "Seq_ID": str(sub_meta.loc[i, "Seq_ID"]) if "Seq_ID" in sub_meta.columns else str(i),
                "pair_id": str(sub_meta.loc[i, "pair_id"]) if "pair_id" in sub_meta.columns else "",
                "y_true": float(y[e_idx[i]]),
                "y_pred": float(y_pred[i]) if np.isfinite(y_pred[i]) else None,
                "top_features": json.dumps([feat[j] for j in idx], ensure_ascii=False),
                "top_shap": json.dumps([float(s[j]) for j in idx], ensure_ascii=False),
            })
        pd.DataFrame(local).to_csv(out / f"shap_local_top{top}_{args.split}.tsv.gz",
                                  sep="\t", index=False, compression="gzip")

    info = {
        "method": "TreeSHAP(RandomForest)",
        "model_file": str(mf),
        "dataset_dir": str(ds),
        "model_dir": str(md),
        "split": args.split,
        "explain_n": int(len(e_idx)),
        "top_features": int(args.top_features),
        "seed": args.seed,
    }
    (out / f"shap_info_{args.split}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")

    print(f"[DONE] RF SHAP -> {out}")

if __name__ == "__main__":
    main()
