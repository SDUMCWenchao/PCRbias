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
    # avoid ConstantInputWarning
    if len(y) == 0 or np.all(y == y[0]) or np.all(p == p[0]):
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

def summarize(boot_list):
    keys = list(boot_list[0].keys())
    res = {}
    for k in keys:
        v = np.array([d[k] for d in boot_list], dtype=float)
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

def pick_cols(df):
    ytrue_cands = ["y_true","y","truth","target"]
    ypred_cands = ["y_pred","pred","yhat","prediction","y_hat"]
    yt = next((c for c in ytrue_cands if c in df.columns), None)
    yp = next((c for c in ypred_cands if c in df.columns), None)
    return yt, yp

def bootstrap_cluster(df, B, seed):
    rng = np.random.default_rng(seed)
    y = df["__y_true__"].values
    p = df["__y_pred__"].values

    if "pair_id" in df.columns:
        gids = df["pair_id"].astype(str).values
        uniq = np.unique(gids)
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
            out.append(metrics(y[rows], p[rows]))
        return out, n_g
    else:
        n = len(df)
        out = []
        for _ in range(B):
            rows = rng.integers(0, n, size=n, endpoint=False)
            out.append(metrics(y[rows], p[rows]))
        return out, 0

def one_file(pred_path: str, root: Path, B: int, seed: int, min_n: int):
    pred = Path(pred_path)
    rel = pred.relative_to(root).parts
    if len(rel) < 6:
        return None

    tag, sp, lc, model, variant = rel[0], rel[1], rel[2], rel[3], rel[4]

    df = pd.read_csv(pred, sep="\t", compression="gzip")
    yt, yp = pick_cols(df)
    if yt is None or yp is None:
        return None

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[yt, yp]).copy()
    if len(df) < min_n:
        return None

    df["__y_true__"] = df[yt].astype(float).values
    df["__y_pred__"] = df[yp].astype(float).values

    point = metrics(df["__y_true__"].values, df["__y_pred__"].values)
    boot, n_pairs = bootstrap_cluster(df, B, seed)
    ci = summarize(boot)

    rec = {
        "tag": tag, "species": sp, "locus": lc, "model": model, "variant": variant,
        "n_test": int(len(df)),
        "n_pairs": int(n_pairs) if n_pairs else (int(df["pair_id"].nunique()) if "pair_id" in df.columns else 0),
        "pred_path": str(pred),
        "ytrue_col": yt,
        "ypred_col": yp,
        **{f"test_{k}": v for k, v in point.items()},
        **{f"test_{k}": v for k, v in ci.items()},
        "flag_small_test": int(len(df) < 50),
        "flag_tiny_test": int(len(df) < 20),
        "flag_few_pairs": int((("pair_id" in df.columns) and (df["pair_id"].nunique() < 5))),
    }
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min_n", type=int, default=10)
    ap.add_argument("--n_jobs", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.models_root)
    outp = Path(args.out_tsv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    preds = sorted([str(p) for p in root.rglob("pred_test.tsv.gz")])
    if len(preds) == 0:
        raise SystemExit(f"[ERROR] no pred_test.tsv.gz under {root}")

    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
        delayed(one_file)(pp, root, int(args.B), int(args.seed), int(args.min_n)) for pp in preds
    )
    recs = [r for r in recs if r is not None]
    df = pd.DataFrame(recs).sort_values(["tag","species","locus","model","variant"]).reset_index(drop=True)
    df.to_csv(outp, sep="\t", index=False)
    print(f"[DONE] wrote -> {outp}  rows={len(df)}")

if __name__ == "__main__":
    main()
