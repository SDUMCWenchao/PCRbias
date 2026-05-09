#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

def feature_category(name: str):
    n = str(name)
    if n.startswith("feat_pr_") or n.startswith("primer_"):
        return "primer"
    if re.match(r"^k[1-8]_(head30|tail30|mid\d+)_[ACGT]{1,8}$", n):
        return "kmer"
    low = n.lower()
    if "gc" in low:
        return "gc"
    if "entropy" in low or "dust" in low or "lz" in low or "lingcomp" in low:
        return "complexity"
    if n.startswith("feat_di_") or "cpg" in low or "upa" in low or n.startswith("feat_p"):
        return "composition"
    if "run" in low or "tandem" in low or "rep" in low or "pal_" in low:
        return "repeats_pal"
    if "mfe" in low or "hairpin" in low or "stem" in low or "loop" in low or "tm" in low:
        return "structure_tm"
    if "g4" in low or "zdna" in low:
        return "alt_struct"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="xgb,rf")
    ap.add_argument("--variants", default="no_kmer,no_kmer_noprimer,all_noprimer,real_kmer_only_all")
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    sg = pd.read_csv(plot_dir / "04_shap_global_full_long.tsv", sep="\t")
    sg = sg[sg["model"].isin(models) & sg["variant"].isin(variants)].copy()

    # ensure cols exist
    for c in ["mean_abs_shap","mean_shap"]:
        if c not in sg.columns:
            # fallback: pick numeric cols
            num = [x for x in sg.columns if pd.api.types.is_numeric_dtype(sg[x])]
            raise RuntimeError(f"missing {c} in shap table. numeric candidates={num[:10]}")

    sg["category"] = sg["feature"].map(feature_category)
    sg["sign"] = np.sign(sg["mean_shap"].to_numpy(dtype=float))

    # E1: top features per group
    sg2 = sg.sort_values(["tag","species","locus","model","variant","mean_abs_shap"],
                         ascending=[True,True,True,True,True,False]).copy()
    sg2["rank"] = sg2.groupby(["tag","species","locus","model","variant"]).cumcount() + 1
    e1 = sg2[sg2["rank"] <= args.topN].copy()
    e1.to_csv(out_dir / "E1_top_features_long.tsv.gz", sep="\t", index=False, compression="gzip")

    # E2: category budget
    e2 = (sg.groupby(["tag","species","locus","model","variant","category"], as_index=False)["mean_abs_shap"]
          .sum()
          .rename(columns={"mean_abs_shap":"sum_mean_abs_shap"}))
    tot = e2.groupby(["tag","species","locus","model","variant"])["sum_mean_abs_shap"].transform("sum")
    e2["category_frac"] = e2["sum_mean_abs_shap"] / tot.replace(0, np.nan)
    e2.to_csv(out_dir / "E2_category_budget.tsv.gz", sep="\t", index=False, compression="gzip")

    # E3: xgb vs rf concordance (only where both exist)
    wide = (sg.pivot_table(index=["tag","species","locus","variant","feature"],
                           columns="model", values="mean_abs_shap", aggfunc="first")
            .reset_index())
    if "xgb" in wide.columns and "rf" in wide.columns:
        # per group spearman on ranks
        rows=[]
        for keys, g in wide.groupby(["tag","species","locus","variant"]):
            a = g["xgb"].to_numpy(dtype=float)
            b = g["rf"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10:
                continue
            ra = pd.Series(a[ok]).rank().to_numpy()
            rb = pd.Series(b[ok]).rank().to_numpy()
            rho = np.corrcoef(ra, rb)[0,1]
            rows.append({"tag":keys[0],"species":keys[1],"locus":keys[2],"variant":keys[3],
                         "n_features":int(ok.sum()),"rank_corr":float(rho)})
        pd.DataFrame(rows).to_csv(out_dir / "E3_rank_concordance_xgb_rf.tsv", sep="\t", index=False)

    # E4 direction summary for top features
    e4 = e1.copy()
    e4["direction"] = np.where(e4["mean_shap"]>0, "positive", np.where(e4["mean_shap"]<0, "negative", "zero"))
    e4.to_csv(out_dir / "E4_top_features_direction.tsv.gz", sep="\t", index=False, compression="gzip")

    # IG tables if exist
    ig_region = plot_dir / "06_ig_region_long.tsv"
    ig_pos = plot_dir / "07_ig_pos_from_end_long.tsv"
    if ig_region.exists():
        df = pd.read_csv(ig_region, sep="\t")
        df.to_csv(out_dir / "E5_ig_region_long.tsv.gz", sep="\t", index=False, compression="gzip")
    if ig_pos.exists():
        df = pd.read_csv(ig_pos, sep="\t")
        df.to_csv(out_dir / "E5_ig_pos_from_end_long.tsv.gz", sep="\t", index=False, compression="gzip")

    print("[DONE] E tables ->", out_dir)

if __name__ == "__main__":
    main()
