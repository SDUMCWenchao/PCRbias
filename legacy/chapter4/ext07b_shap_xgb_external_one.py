#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb

def load_feature_names(dataset_dir: Path, nfeat: int):
    cand = [
        "feature_names.tsv", "feature_names.txt", "feat_cols.tsv",
        "feature_cols.tsv", "columns.tsv", "X_cols.tsv"
    ]
    for fn in cand:
        p = dataset_dir / fn
        if p.exists():
            df = pd.read_csv(p, sep="\t", header=None)
            names = df.iloc[:, 0].astype(str).tolist()
            if len(names) == nfeat:
                return names
    return [f"f{i}" for i in range(nfeat)]

def load_split(dataset_dir: Path, split: str):
    Xp = dataset_dir / f"X_{split}.npz"
    yp = dataset_dir / f"y_{split}.npy"
    if not Xp.exists():
        raise FileNotFoundError(f"[BAD] missing {Xp}")
    if not yp.exists():
        raise FileNotFoundError(f"[BAD] missing {yp}")
    X = sparse.load_npz(Xp).tocsr()
    y = np.load(yp)
    if X.shape[0] != len(y):
        raise RuntimeError(f"[BAD] X/y mismatch: X={X.shape} y={y.shape}")
    return X, y

def find_xgb_model_file(model_dir: Path):
    cand = ["model.json", "model.bin", "xgb.json", "xgb.bin", "booster.json", "booster.bin"]
    for fn in cand:
        p = model_dir / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"[BAD] cannot find xgb model file under {model_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=0, help="0=all rows of split")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out_dir", default="", help="default: <model_dir>/shap_tables")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / "shap_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / f"xgb_shap_global_{args.split}.tsv.gz"
    if done_flag.exists() and (not args.force):
        print(f"[SKIP] exists: {done_flag}")
        return

    X, y = load_split(dataset_dir, args.split)
    n = X.shape[0]
    if n == 0:
        raise RuntimeError(f"[BAD] split {args.split} has 0 rows: {dataset_dir}")

    idx = np.arange(n)
    if args.explain_n and args.explain_n < n:
        idx = idx[:args.explain_n]

    Xs = X[idx]
    ys = y[idx]

    model_file = find_xgb_model_file(model_dir)
    booster = xgb.Booster()
    booster.load_model(str(model_file))

    nfeat = Xs.shape[1]
    feat_names = load_feature_names(dataset_dir, nfeat)

    dmat = xgb.DMatrix(Xs, feature_names=feat_names)
    pred = booster.predict(dmat)  # (n,)
    contrib = booster.predict(dmat, pred_contribs=True)  # (n, nfeat+1) last=bias

    base = contrib[:, -1].astype(np.float32)
    shap_values = contrib[:, :-1].astype(np.float32)

    abs_sv = np.abs(shap_values)
    df_g = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": abs_sv.mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "std_abs_shap": abs_sv.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    df_g.to_csv(done_flag, sep="\t", index=False, compression="gzip")

    topk = min(args.topk, nfeat)
    rows = []
    Xsd = Xs.toarray().astype(np.float32) if sparse.issparse(Xs) else np.asarray(Xs, dtype=np.float32)
    for i in range(len(idx)):
        sv = shap_values[i]
        top = np.argsort(np.abs(sv))[::-1][:topk]
        for r, j in enumerate(top, 1):
            rows.append({
                "row_i": int(idx[i]),
                "rank": int(r),
                "feature": feat_names[j],
                "x": float(Xsd[i, j]),
                "shap": float(sv[j]),
                "abs_shap": float(abs(sv[j])),
                "base_value": float(base[i]),
                "y_true": float(ys[i]),
                "y_pred": float(pred[i]),
            })
    df_l = pd.DataFrame(rows)
    out_l = out_dir / f"xgb_shap_local_top{topk}_{args.split}.tsv.gz"
    df_l.to_csv(out_l, sep="\t", index=False, compression="gzip")

    meta = {
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "model_file": str(model_file),
        "split": args.split,
        "n_explained": int(len(idx)),
        "n_features": int(nfeat),
        "global_table": str(done_flag),
        "local_table": str(out_l),
    }
    (out_dir / f"xgb_shap_meta_{args.split}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[DONE] XGB SHAP (pred_contribs) -> {out_dir}")

if __name__ == "__main__":
    main()
