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
    cands = ["model.joblib","model.pkl","rf.joblib","rf.pkl","estimator.joblib","estimator.pkl"]
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
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top_features", type=int, default=500)
    args = ap.parse_args()

    import joblib, shap

    ds = Path(args.dataset_dir)
    md = Path(args.model_dir)
    out = md / "shap_tables_v2"
    out.mkdir(parents=True, exist_ok=True)

    X = load_X(ds, args.split)
    feat = read_feat_names(ds, X.shape[1])

    mf = find_model_file(md)
    if mf is None:
        raise SystemExit(f"[ERROR] cannot find RF model file in {md}")
    model = joblib.load(mf)

    rng = np.random.default_rng(args.seed)
    n = X.shape[0]
    e_idx = rng.choice(np.arange(n), size=min(args.explain_n, n), replace=False)
    Xex = X[e_idx].toarray() if sparse.issparse(X) else np.asarray(X[e_idx])

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = np.asarray(explainer.shap_values(Xex, check_additivity=False), dtype=np.float32)  # (m,p)

    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_val = np.mean(sv, axis=0)

    full = pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs, "mean_shap": mean_val})
    full.to_csv(out / f"shap_global_full_{args.split}.tsv.gz", sep="\t", index=False, compression="gzip")

    top = full.sort_values("mean_abs_shap", ascending=False).head(args.top_features).reset_index(drop=True)
    top.to_csv(out / f"shap_global_top{args.top_features}_{args.split}.tsv", sep="\t", index=False)

    rows = []
    for _, r in full.iterrows():
        tp, kk, reg = parse_group(r["feature"])
        rows.append((tp, kk, reg, float(r["mean_abs_shap"]), float(r["mean_shap"])))
    g = pd.DataFrame(rows, columns=["type","k","region","mean_abs_shap","mean_shap"])
    gsum = g.groupby(["type","k","region"], as_index=False)[["mean_abs_shap","mean_shap"]].sum()
    gsum = gsum.sort_values(["type","mean_abs_shap"], ascending=[True, False])
    gsum.to_csv(out / f"shap_region_full_{args.split}.tsv", sep="\t", index=False)

    info = {"method":"TreeSHAP(RF)","model_file":str(mf),"explain_n":int(len(e_idx)),"seed":args.seed}
    (out / f"shap_info_{args.split}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] RF SHAP(v2) -> {out}")

if __name__ == "__main__":
    main()
