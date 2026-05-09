#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def read_table(p: Path):
    suf = "".join(p.suffixes)
    if suf.endswith(".tsv.gz") or suf.endswith(".csv.gz"):
        sep = "\t" if ".tsv" in suf else ","
        return pd.read_csv(p, sep=sep, compression="gzip")
    if suf.endswith(".tsv") or suf.endswith(".csv"):
        sep = "\t" if suf.endswith(".tsv") else ","
        return pd.read_csv(p, sep=sep)
    if suf.endswith(".parquet"):
        return pd.read_parquet(p)
    return None

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def spearman_np(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0,1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--master_tsv", required=True)
    ap.add_argument("--best_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--perm_B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pair_min_n", type=int, default=200, help="min rows per pair for per-pair metric")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    proj = Path(args.project_dir).resolve()
    out = proj / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(proj / args.master_tsv, sep="\t")
    best = pd.read_csv(proj / args.best_tsv, sep="\t")

    # F1: ci_width + stability ranking
    m = master.copy()
    if "test_spearman_ci_low" in m.columns and "test_spearman_ci_high" in m.columns:
        m["ci_width"] = m["test_spearman_ci_high"] - m["test_spearman_ci_low"]
    m.to_csv(out / "F1_master_with_ciwidth.tsv.gz", sep="\t", index=False, compression="gzip")

    # F2: tag sensitivity (just reuse master)
    # (you plot: x=tag, y=test_spearman, facet species/locus, color variant/model)
    m.to_csv(out / "F2_tag_sensitivity_long.tsv.gz", sep="\t", index=False, compression="gzip")

    # F4: primer ablation contrasts (derive paired comparisons)
    # expects variants: no_kmer vs no_kmer_noprimer, all vs all_noprimer if present
    rows=[]
    keys = ["tag","species","locus","model"]
    for (tag,sp,lc,md), g in m.groupby(keys):
        def get(v):
            gg = g[g["variant"]==v]
            return None if gg.empty else gg.iloc[0]
        for a,b in [("no_kmer","no_kmer_noprimer"), ("all","all_noprimer")]:
            ra, rb = get(a), get(b)
            if ra is None or rb is None:
                continue
            rows.append({
                "tag":tag,"species":sp,"locus":lc,"model":md,
                "A":a,"B":b,
                "A_test_spearman":float(ra["test_spearman"]),
                "B_test_spearman":float(rb["test_spearman"]),
                "delta_spearman":float(rb["test_spearman"]-ra["test_spearman"]),
                "A_rmse":float(ra.get("test_rmse", np.nan)),
                "B_rmse":float(rb.get("test_rmse", np.nan)),
                "delta_rmse":float(rb.get("test_rmse", np.nan)-ra.get("test_rmse", np.nan)),
            })
    pd.DataFrame(rows).to_csv(out / "F4_primer_ablation_contrasts.tsv", sep="\t", index=False)

    # F3 & F5: per-pair robustness and permutation null on best models
    keep = best[["tag","species","locus","model","variant"]].drop_duplicates()
    mb = master.merge(keep, on=["tag","species","locus","model","variant"], how="inner").copy()

    pair_rows=[]
    perm_rows=[]
    for _, r in mb.iterrows():
        pred_path = proj / str(r.get("pred_path",""))
        df = read_table(pred_path)
        if df is None:
            continue
        yt = r.get("ytrue_col","y_true")
        yp = r.get("ypred_col","y_pred")
        if yt not in df.columns or yp not in df.columns:
            # try fallback
            yt2 = pick_col(df, ["y_true","y","label","target","log2fc"])
            yp2 = pick_col(df, ["y_pred","pred","prediction"])
            if yt2 is None or yp2 is None:
                continue
            yt, yp = yt2, yp2

        y = df[yt].to_numpy(dtype=float)
        p = df[yp].to_numpy(dtype=float)

        # per-pair if possible
        pair_col = pick_col(df, ["pair_id","pair","pairid","Pair_ID"])
        if pair_col is not None:
            for pid, g in df.groupby(pair_col):
                if len(g) < args.pair_min_n:
                    continue
                yy = g[yt].to_numpy(dtype=float)
                pp = g[yp].to_numpy(dtype=float)
                pair_rows.append({
                    "tag":r["tag"],"species":r["species"],"locus":r["locus"],
                    "model":r["model"],"variant":r["variant"],
                    "pair_id":pid,"n":len(g),
                    "pair_spearman":spearman_np(yy, pp),
                    "pair_rmse":float(np.sqrt(np.mean((pp-yy)**2))),
                    "pair_mae":float(np.mean(np.abs(pp-yy))),
                })

            # permutation within pair (preserve pair structure)
            for b in range(args.perm_B):
                pp_all=[]
                yy_all=[]
                for pid, g in df.groupby(pair_col):
                    yy = g[yt].to_numpy(dtype=float)
                    perm = rng.permutation(yy)
                    yy_all.append(perm)
                    pp_all.append(g[yp].to_numpy(dtype=float))
                yy_all = np.concatenate(yy_all)
                pp_all = np.concatenate(pp_all)
                perm_rows.append({
                    "tag":r["tag"],"species":r["species"],"locus":r["locus"],
                    "model":r["model"],"variant":r["variant"],
                    "perm_id":b,
                    "perm_spearman":spearman_np(yy_all, pp_all)
                })
        else:
            # fallback: global permutation
            for b in range(args.perm_B):
                yy = rng.permutation(y)
                perm_rows.append({
                    "tag":r["tag"],"species":r["species"],"locus":r["locus"],
                    "model":r["model"],"variant":r["variant"],
                    "perm_id":b,
                    "perm_spearman":spearman_np(yy, p)
                })

    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(out / "F3_pairwise_metrics.tsv.gz", sep="\t", index=False, compression="gzip")
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(out / "F5_permutation_null.tsv.gz", sep="\t", index=False, compression="gzip")

    print("[DONE] F tables ->", out)

if __name__ == "__main__":
    main()
