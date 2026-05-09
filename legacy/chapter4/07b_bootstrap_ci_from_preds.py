#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ssr = np.sum((y - p)**2)
    sst = np.sum((y - y.mean())**2)
    return float(1.0 - ssr/(sst + 1e-12))

def rmse(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.sqrt(np.mean((y - p)**2)))

def mae(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs(y - p)))

def sign_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean((y>=0)==(p>=0)))

def spearman(y, p):
    rr = spearmanr(y, p)
    return float(rr.correlation) if np.isfinite(rr.correlation) else float("nan")

def pair_spearman(df, min_pair_n=30):
    if "pair_id" not in df.columns:
        return float("nan"), 0
    out = []
    for pid, g in df.groupby("pair_id"):
        if len(g) < min_pair_n:
            continue
        rr = spearmanr(g["y_true"].values, g["y_pred"].values)
        if np.isfinite(rr.correlation):
            out.append(float(rr.correlation))
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), int(len(out))

def bootstrap_ci(y, p, B=5000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {"r2":[], "rmse":[], "mae":[], "spearman":[], "sign_acc":[]}
    idx = np.arange(n)
    for _ in range(B):
        b = rng.choice(idx, size=n, replace=True)
        yb = y[b]; pb = p[b]
        stats["r2"].append(r2(yb, pb))
        stats["rmse"].append(rmse(yb, pb))
        stats["mae"].append(mae(yb, pb))
        stats["spearman"].append(spearman(yb, pb))
        stats["sign_acc"].append(sign_acc(yb, pb))
    out = {}
    for k, arr in stats.items():
        a = np.array(arr, dtype=float)
        out[k] = {
            "mean": float(np.nanmean(a)),
            "p2.5": float(np.nanpercentile(a, 2.5)),
            "p50": float(np.nanpercentile(a, 50)),
            "p97.5": float(np.nanpercentile(a, 97.5)),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help=".../rf/<variant> or .../seqcnn/<variant>")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min_pair_n", type=int, default=10)
    args = ap.parse_args()

    md = Path(args.model_dir)
    pred = md / f"pred_{args.split}.tsv.gz"
    if not pred.exists():
        raise SystemExit(f"[ERROR] missing {pred}")

    df = pd.read_csv(pred, sep="\t", compression="gzip")
    if not {"y_true","y_pred"}.issubset(df.columns):
        raise SystemExit("[ERROR] pred file must contain y_true and y_pred")

    y = df["y_true"].values.astype(float)
    p = df["y_pred"].values.astype(float)

    base = {
        "n": int(len(df)),
        "split": args.split,
        "metrics": {
            "r2": r2(y, p),
            "rmse": rmse(y, p),
            "mae": mae(y, p),
            "spearman": spearman(y, p),
            "sign_acc": sign_acc(y, p),
        }
    }
    ps, pn = pair_spearman(df, min_pair_n=args.min_pair_n)
    base["pair"] = {"pair_spearman_mean": ps, "pair_spearman_n": pn, "min_pair_n": args.min_pair_n}

    ci = bootstrap_ci(y, p, B=args.B, seed=args.seed)
    out = {"base": base, "bootstrap_CI": ci}

    out_json = md / f"ci_{args.split}_B{args.B}.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] {out_json}")

if __name__ == "__main__":
    main()
