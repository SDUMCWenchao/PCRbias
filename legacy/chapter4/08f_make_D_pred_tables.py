#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import gzip

def read_pred(path: Path):
    p = Path(path)
    if not p.exists():
        return None
    # tsv / tsv.gz / csv / csv.gz
    suf = "".join(p.suffixes)
    if suf.endswith(".tsv.gz") or suf.endswith(".csv.gz"):
        sep = "\t" if ".tsv" in suf else ","
        return pd.read_csv(p, sep=sep, compression="gzip")
    if suf.endswith(".tsv") or suf.endswith(".csv"):
        sep = "\t" if suf.endswith(".tsv") else ","
        return pd.read_csv(p, sep=sep)
    # fallback: try parquet
    if suf.endswith(".parquet"):
        return pd.read_parquet(p)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--master_tsv", required=True)
    ap.add_argument("--best_tsv", required=True, help="02_best_by_group.tsv (use this to limit to best models)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_n", type=int, default=5000, help="scatter subsample per group")
    ap.add_argument("--bins", type=int, default=20, help="bins for calibration/residual summary")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    proj = Path(args.project_dir).resolve()
    master = pd.read_csv(proj / args.master_tsv, sep="\t")
    best = pd.read_csv(proj / args.best_tsv, sep="\t")
    out = proj / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # only keep the exact best rows (tag/species/locus/model/variant match)
    key = ["tag","species","locus","model","variant"]
    keep = best[key].drop_duplicates()
    m = master.merge(keep, on=key, how="inner").copy()

    scat_rows = []
    cal_rows = []
    resid_rows = []
    quant_rows = []

    for _, r in m.iterrows():
        pred_path = proj / str(r["pred_path"])
        df = read_pred(pred_path)
        if df is None:
            print(f"[WARN] missing/unsupported pred file: {pred_path}")
            continue

        yt = str(r.get("ytrue_col","y_true"))
        yp = str(r.get("ypred_col","y_pred"))
        if yt not in df.columns or yp not in df.columns:
            print(f"[WARN] columns not found in {pred_path.name}: need {yt},{yp} but have {list(df.columns)[:10]}")
            continue

        y_true = df[yt].to_numpy(dtype=float)
        y_pred = df[yp].to_numpy(dtype=float)
        n = len(y_true)
        if n == 0:
            continue

        # scatter sample
        take = min(args.sample_n, n)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        sub = pd.DataFrame({
            "tag": r["tag"], "species": r["species"], "locus": r["locus"],
            "model": r["model"], "variant": r["variant"],
            "y_true": y_true[idx],
            "y_pred": y_pred[idx],
        })
        sub["resid"] = sub["y_pred"] - sub["y_true"]
        sub["abs_resid"] = np.abs(sub["resid"])
        scat_rows.append(sub)

        # calibration by y_true quantile bins
        q = pd.qcut(y_true, q=min(args.bins, max(2, len(np.unique(y_true)))), duplicates="drop")
        tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bin": q})
        g = tmp.groupby("bin", observed=True)
        cal = g.agg(n=("y_true","size"), y_true_mean=("y_true","mean"), y_pred_mean=("y_pred","mean"),
                    y_true_med=("y_true","median"), y_pred_med=("y_pred","median")).reset_index()
        cal["tag"] = r["tag"]; cal["species"]=r["species"]; cal["locus"]=r["locus"]
        cal["model"]=r["model"]; cal["variant"]=r["variant"]
        cal_rows.append(cal)

        # residual summary by same bins
        tmp["resid"] = tmp["y_pred"] - tmp["y_true"]
        rr = g.agg(n=("resid","size"),
                   resid_mean=("resid","mean"),
                   resid_med=("resid","median"),
                   resid_rmse=("resid", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                   resid_mae=("resid", lambda x: float(np.mean(np.abs(x))))).reset_index()
        rr["tag"]=r["tag"]; rr["species"]=r["species"]; rr["locus"]=r["locus"]
        rr["model"]=r["model"]; rr["variant"]=r["variant"]
        resid_rows.append(rr)

        # global residual quantiles
        qs = np.quantile(y_pred - y_true, [0.01,0.05,0.5,0.95,0.99])
        quant_rows.append({
            "tag": r["tag"], "species": r["species"], "locus": r["locus"],
            "model": r["model"], "variant": r["variant"],
            "n": n,
            "resid_q01": qs[0], "resid_q05": qs[1], "resid_q50": qs[2], "resid_q95": qs[3], "resid_q99": qs[4],
        })

    if scat_rows:
        pd.concat(scat_rows, ignore_index=True).to_csv(out / "D6_pred_scatter_sample.tsv.gz", sep="\t", index=False, compression="gzip")
    if cal_rows:
        pd.concat(cal_rows, ignore_index=True).to_csv(out / "D6_calibration_bins.tsv.gz", sep="\t", index=False, compression="gzip")
    if resid_rows:
        pd.concat(resid_rows, ignore_index=True).to_csv(out / "D6_residual_bins.tsv.gz", sep="\t", index=False, compression="gzip")
    if quant_rows:
        pd.DataFrame(quant_rows).to_csv(out / "D6_residual_quantiles.tsv", sep="\t", index=False)

    print("[DONE] D6 tables ->", out)

if __name__ == "__main__":
    main()
