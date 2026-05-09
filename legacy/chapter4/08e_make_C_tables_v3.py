#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

PRIMERS = {
    "12S_F": "GGGATTAGATACCCCACTATGCYTA",
    "12S_R": "GAGGGTGACGGGCGGTGT",
    "16S_F": "ACCAAAAACATCACCTCYAGCAT",
    "16S_R": "AATAGGATTGCGCTGTTATCCCTA",
}

IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "R": set("AG"), "Y": set("CT"), "S": set("GC"), "W": set("AT"),
    "K": set("GT"), "M": set("AC"),
    "B": set("CGT"), "D": set("AGT"), "H": set("ACT"), "V": set("ACG"),
    "N": set("ACGT"),
}

# strict kmer feature pattern in your global SHAP table:
# k7_tail30_AGGCCAT
# k8_mid2_TTGATCCA
KM_RE = re.compile(r"^k(?P<k>[1-8])_(?P<region>head30|tail30|mid\d+)_(?P<kmer>[ACGT]{1,8})$")

def revcomp(seq: str):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def iupac_contains(hay: str, needle: str):
    hay = hay.upper()
    needle = needle.upper()
    n = len(needle)
    if n == 0 or len(hay) < n:
        return False
    for i in range(len(hay) - n + 1):
        ok = True
        for j in range(n):
            b = hay[i + j]
            p = needle[j]
            if b not in "ACGT":
                ok = False
                break
            if b not in IUPAC.get(p, set("ACGT")):
                ok = False
                break
        if ok:
            return True
    return False

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="xgb,rf")
    ap.add_argument("--variants",
                    default="kmer_only_all,real_kmer_only_all,kmer_only_k1,kmer_only_k2,kmer_only_k3,kmer_only_k4,kmer_only_k5,kmer_only_k6,kmer_only_k7,kmer_only_k8,all_noprimer",
                    help="which variants to include")
    ap.add_argument("--topN", type=int, default=30)
    ap.add_argument("--primer_min_k", type=int, default=6, help="only annotate primer-like for kmers with len>=primer_min_k")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    variants_keep = set([x.strip() for x in args.variants.split(",") if x.strip()])

    master = pd.read_csv(plot_dir / "01_master_metrics_ci.tsv", sep="\t")
    sg = pd.read_csv(plot_dir / "04_shap_global_full_long.tsv", sep="\t")
    sr = pd.read_csv(plot_dir / "05_shap_region_full_long.tsv", sep="\t")

    # filter model/variant
    master = master[master["model"].isin(models)].copy()
    sg = sg[sg["model"].isin(models)].copy()
    sr = sr[sr["model"].isin(models)].copy()

    master = master[master["variant"].isin(variants_keep)].copy()
    sg = sg[sg["variant"].isin(variants_keep)].copy()
    sr = sr[sr["variant"].isin(variants_keep)].copy()

    # ---------- parse kmers from global shap feature ----------
    # sg columns expected: feature, mean_abs_shap, mean_shap, tag, species, locus, model, variant, ...
    m = sg["feature"].astype(str).str.match(KM_RE)
    sgk = sg[m].copy()
    if sgk.empty:
        raise RuntimeError("No kmer features matched regex ^k[1-8]_(head30|tail30|mid\\d+)_[ACGT]{1,8}$ in 04_shap_global_full_long.tsv")

    ex = sgk["feature"].astype(str).str.extract(KM_RE)
    sgk["k"] = ex["k"].astype(int)
    sgk["region"] = ex["region"].astype(str)
    sgk["kmer"] = ex["kmer"].astype(str)

    # ---------- C1: region budget from global (kmer only) ----------
    c1 = (sgk.groupby(["tag","species","locus","model","variant","k","region"], as_index=False)["mean_abs_shap"]
          .sum()
          .rename(columns={"mean_abs_shap":"sum_abs_shap"}))
    # add proportion within each group
    tot = c1.groupby(["tag","species","locus","model","variant","k"])["sum_abs_shap"].transform("sum")
    c1["region_frac"] = c1["sum_abs_shap"] / tot.replace(0, np.nan)

    # ---------- C1b: region budget from region table (type=kmer/non_kmer) ----------
    # sr columns: type,k,region,mean_abs_shap,...
    c1b = sr.copy()
    # keep only columns we need
    keep = [c for c in ["tag","species","locus","model","variant","k","region","type","mean_abs_shap","mean_shap","topbias_frac"] if c in c1b.columns]
    c1b = c1b[keep].copy()

    # ---------- C2: top kmers by region ----------
    sgk2 = sgk.sort_values(["tag","species","locus","model","variant","k","region","mean_abs_shap"],
                          ascending=[True,True,True,True,True,True,True,False]).copy()
    sgk2["rank_in_region"] = sgk2.groupby(["tag","species","locus","model","variant","k","region"]).cumcount() + 1
    c2 = sgk2[sgk2["rank_in_region"] <= args.topN].copy()

    # ---------- C3: same kmer across regions (within group+k) ----------
    # if a kmer appears in >=2 regions, keep it and show region-wise mean_abs_shap
    c3 = (sgk.groupby(["tag","species","locus","model","variant","k","kmer","region"], as_index=False)["mean_abs_shap"]
          .mean())
    reg_ct = c3.groupby(["tag","species","locus","model","variant","k","kmer"])["region"].nunique().reset_index(name="n_regions")
    c3 = c3.merge(reg_ct, on=["tag","species","locus","model","variant","k","kmer"], how="left")
    c3 = c3[c3["n_regions"] >= 2].copy()

    # ---------- C4: k-curve metrics ----------
    keep_cols = [
        "tag","species","locus","model","variant","k",
        "test_spearman","test_rmse","test_r2",
        "test_spearman_lo","test_spearman_hi",
        "test_rmse_lo","test_rmse_hi",
        "test_r2_lo","test_r2_hi",
        "n_test","n_pairs","flag_small_test","flag_few_pairs"
    ]
    cols = [c for c in keep_cols if c in master.columns]
    c4 = master[cols].copy()

    # ---------- C5: primer-like annotation among top kmers (global) ----------
    sgk3 = sgk.sort_values(["tag","species","locus","model","variant","k","mean_abs_shap"],
                           ascending=[True,True,True,True,True,True,False]).copy()
    sgk3["rank_global_k"] = sgk3.groupby(["tag","species","locus","model","variant","k"]).cumcount() + 1
    c5 = sgk3[sgk3["rank_global_k"] <= args.topN].copy()

    hits = []
    for _, r in c5.iterrows():
        kmer = r["kmer"]
        h = []
        if isinstance(kmer, str) and len(kmer) >= args.primer_min_k:
            for pname, pseq in PRIMERS.items():
                if iupac_contains(pseq, kmer) or iupac_contains(revcomp(pseq), kmer):
                    h.append(pname)
        hits.append(",".join(h) if h else "")
    c5["primer_hits"] = hits
    c5["is_primerlike"] = (c5["primer_hits"].astype(str).str.len() > 0).astype(int)

    c5b = (c5.groupby(["tag","species","locus","model","variant","k"], as_index=False)
           .agg(n_top=("kmer","size"),
                n_primerlike=("is_primerlike","sum")))
    c5b["primerlike_frac"] = c5b["n_primerlike"] / c5b["n_top"].replace(0, np.nan)

    # ---------- C6: logo basefreq from region top kmers ----------
    # build from C2 (region top kmers)
    logo_rows = []
    for keys, g in c2.groupby(["tag","species","locus","model","variant","k","region"]):
        kk = int(keys[5])
        kmers = g["kmer"].astype(str).tolist()
        kmers = [x for x in kmers if len(x) == kk]
        if not kmers:
            continue
        for pos in range(kk):
            col = [x[pos] for x in kmers]
            for base in "ACGT":
                logo_rows.append({
                    "tag": keys[0], "species": keys[1], "locus": keys[2], "model": keys[3], "variant": keys[4],
                    "k": kk, "region": keys[6],
                    "pos": pos+1, "base": base,
                    "freq": float(np.mean([b == base for b in col]))
                })
    c6 = pd.DataFrame(logo_rows)

    # ---------- write ----------
    def write(df: pd.DataFrame, name: str):
        p = out_dir / name
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"exists: {p} (add --overwrite)")
        df.to_csv(p, sep="\t", index=False)

    write(c1, "C1_region_budget_kmer.tsv")
    write(c1b, "C1b_region_budget_type.tsv")
    write(c2, "C2_top_kmers_by_region.tsv")
    write(c3, "C3_same_kmer_across_regions.tsv")
    write(c4, "C4_k_curve_metrics.tsv")
    write(c5, "C5_primerlike_kmers.tsv")
    write(c5b, "C5b_primerlike_summary.tsv")
    write(c6, "C6_logo_basefreq.tsv")

    print("[DONE] C tables v3 ->", out_dir)
    print("[INFO] rows:",
          "C1", len(c1),
          "C1b", len(c1b),
          "C2", len(c2),
          "C3", len(c3),
          "C4", len(c4),
          "C5", len(c5),
          "C5b", len(c5b),
          "C6", len(c6))

if __name__ == "__main__":
    main()
