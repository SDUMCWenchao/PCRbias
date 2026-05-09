#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    sse = np.sum((y-p)**2)
    sst = np.sum((y-y.mean())**2)
    return float(1.0 - sse/(sst+1e-12))

def rmse(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.sqrt(np.mean((y-p)**2)))

def mae(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs(y-p)))

def sign_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean((y>=0)==(p>=0)))

def spearman(y, p):
    y = np.asarray(y); p = np.asarray(p)
    # 常数直接返回 NaN（不调用 spearmanr，就不会有 warning）
    if np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rr = spearmanr(y, p)
        return float(rr.correlation) if np.isfinite(rr.correlation) else float("nan")
    except Exception:
        yr = pd.Series(y).rank().values
        pr = pd.Series(p).rank().values
        v = float(np.corrcoef(yr, pr)[0,1])
        return v if np.isfinite(v) else float("nan")

def metrics(y, p):
    return {
        "r2": r2(y,p),
        "rmse": rmse(y,p),
        "mae": mae(y,p),
        "spearman": spearman(y,p),
        "sign_acc": sign_acc(y,p),
    }

def bootstrap_cluster(df, B, seed):
    rng = np.random.default_rng(seed)
    y = df["y_true"].values
    p = df["y_pred"].values

    if "pair_id" in df.columns:
        gids = df["pair_id"].astype(str).values
        uniq = np.unique(gids)
        # map id -> row idx
        idx_map = {}
        for i, g in enumerate(gids):
            idx_map.setdefault(g, []).append(i)
        uniq = list(uniq)
        n_g = len(uniq)

        out = []
        for _ in range(B):
            samp = rng.choice(uniq, size=n_g, replace=True)
            rows = []
            for g in samp:
                rows.extend(idx_map[g])
            rows = np.asarray(rows, dtype=int)
            yy = y[rows]; pp = p[rows]
            out.append(metrics(yy, pp))
        return out, n_g
    else:
        n = len(df)
        out = []
        for _ in range(B):
            rows = rng.integers(0, n, size=n, endpoint=False)
            out.append(metrics(y[rows], p[rows]))
        return out, 0

def summarize(boot_list):
    # boot_list: list of dict
    keys = list(boot_list[0].keys())
    arr = {k: np.array([d[k] for d in boot_list], dtype=float) for k in keys}
    res = {}
    for k, v in arr.items():
        v = v[np.isfinite(v)]
        if len(v) == 0:
            res[k+"_ci_low"] = np.nan
            res[k+"_ci_high"] = np.nan
            res[k+"_boot_sd"] = np.nan
        else:
            res[k+"_ci_low"] = float(np.quantile(v, 0.025))
            res[k+"_ci_high"] = float(np.quantile(v, 0.975))
            res[k+"_boot_sd"] = float(np.std(v))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis_results/06_Models_v3_topbias")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.models_root)
    outp = Path(args.out_tsv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred in sorted(root.rglob("pred_test.tsv.gz")):
        rel = pred.relative_to(root).parts
        if len(rel) < 6:
            continue
        tag, sp, lc, model, variant = rel[0], rel[1], rel[2], rel[3], rel[4]

        df = pd.read_csv(pred, sep="\t", compression="gzip")
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            continue
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true","y_pred"])
        if len(df) < 10:
            continue

        point = metrics(df["y_true"].values, df["y_pred"].values)
        boot, n_pairs = bootstrap_cluster(df, args.B, args.seed)
        ci = summarize(boot)

        rec = {
            "tag": tag, "species": sp, "locus": lc, "model": model, "variant": variant,
            "n_test": int(len(df)),
            "n_pairs": int(n_pairs) if n_pairs else (int(df["pair_id"].nunique()) if "pair_id" in df.columns else 0),
            "pred_path": str(pred),
            **{f"test_{k}": v for k, v in point.items()},
            **{f"test_{k}": v for k, v in ci.items()},
        }
        # instability flags
        rec["flag_small_test"] = int(rec["n_test"] < 50)
        rec["flag_tiny_test"] = int(rec["n_test"] < 20)
        rec["flag_few_pairs"] = int(rec["n_pairs"] > 0 and rec["n_pairs"] < 5)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(outp, sep="\t", index=False)
    print(f"[DONE] wrote -> {outp}  rows={len(out_df)}")

if __name__ == "__main__":
    main()
