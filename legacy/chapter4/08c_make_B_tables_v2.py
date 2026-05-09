#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def read_json(p: Path):
    return json.loads(p.read_text())

def load_feature_names(d: Path):
    for fn in ["feature_cols.json", "feature_names.json", "cols.json"]:
        p = d / fn
        if p.exists():
            obj = read_json(p)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "feature_cols" in obj:
                return obj["feature_cols"]
    for fn in ["feature_cols.txt", "feature_names.txt", "cols.txt"]:
        p = d / fn
        if p.exists():
            return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    raise FileNotFoundError(f"cannot find feature names in {d}")

def load_X(d: Path, split: str):
    p_npz = d / f"X_{split}.npz"
    p_npy = d / f"X_{split}.npy"
    if p_npz.exists():
        return sparse.load_npz(p_npz)
    if p_npy.exists():
        return np.load(p_npy, mmap_mode="r")
    raise FileNotFoundError(f"missing X for split={split} in {d}")

def load_y(d: Path, split: str):
    p = d / f"y_{split}.npy"
    if p.exists():
        return np.load(p, allow_pickle=False).astype(np.float64)
    p = d / f"y_{split}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if len(z.files) != 1:
            raise ValueError(f"npz has multiple arrays: {p} -> {z.files}")
        return z[z.files[0]].astype(np.float64)
    raise FileNotFoundError(f"missing y for split={split} in {d}")

def feature_group(name: str):
    if name.startswith("feat_pr_") or name.startswith("primer_"):
        return "primer"
    if name.startswith("kmer_") or name.startswith("km_") or "kmer" in name:
        return "kmer"
    return "nonprimer"

def feature_category(name: str):
    if name.startswith("feat_pr_"):
        return "primer"
    if name.startswith("feat_gc") or name == "feat_gc" or "gc_" in name:
        return "gc"
    if "entropy" in name or "dust" in name or "lz" in name or "lingcomp" in name:
        return "complexity"
    if name.startswith("feat_di_") or name.startswith("feat_at") or name.startswith("feat_p") or "CpG" in name or "UpA" in name:
        return "composition"
    if "run" in name or "tandem" in name or "trirep" in name or "direp" in name or "pal_" in name:
        return "repeats_pal"
    if "mfe" in name or "hairpin" in name or "stem" in name or "loop" in name or "tm" in name:
        return "structure_tm"
    if "g4" in name or "zdna" in name:
        return "alt_struct"
    return "other"

def ensure_dense_col(X, j):
    if sparse.issparse(X):
        return np.asarray(X[:, j].todense()).ravel()
    return np.asarray(X[:, j]).ravel()

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan)
    if ok.sum() == 0:
        return q
    q_ok = multipletests(pvals[ok], method="fdr_bh")[1]
    q[ok] = q_ok
    return q

def cliff_delta_from_u(U, n1, n2):
    return (2.0 * U) / (n1 * n2) - 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--out_dir", default="analysis_results/07_PlotTables_v3_topbias/B_tables_v2")
    ap.add_argument("--tags", default="top1p,top0p5p,top0p1p")
    ap.add_argument("--species", default="donkey,pig,cattle,10mix")
    ap.add_argument("--loci", default="12S,16S")
    ap.add_argument("--variants", default="no_kmer,no_kmer_noprimer,all_noprimer")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--topK_corr", type=int, default=40)
    ap.add_argument("--ybins", type=int, default=10)
    ap.add_argument("--min_shift_n", type=int, default=20, help="min samples per side; if smaller, output NA but keep row")
    ap.add_argument("--min_bin_n", type=int, default=50, help="min samples per bin for trend; if smaller, still keep row but mark small_bin=1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    proj = Path(args.project_dir).resolve()
    inputs_root = (proj / args.inputs_root).resolve()
    out_dir = (proj / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [x.strip() for x in args.tags.split(",") if x.strip()]
    species = [x.strip() for x in args.species.split(",") if x.strip()]
    loci = [x.strip() for x in args.loci.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    assoc_rows, shift_rows, corr_rows, trend_rows, top_rows = [], [], [], [], []

    for tag in tags:
        for sp in species:
            for lc in loci:
                for var in variants:
                    d = inputs_root / tag / sp / lc / var
                    if not d.exists():
                        continue

                    feat = load_feature_names(d)
                    X = load_X(d, args.split)
                    y = load_y(d, args.split)
                    n = len(y)
                    p = len(feat)

                    # ---------- B1: Spearman assoc ----------
                    rhos, pvals, means, sds, miss = [], [], [], [], []
                    for j, name in enumerate(feat):
                        x = ensure_dense_col(X, j).astype(np.float64)
                        means.append(float(np.mean(x)))
                        sds.append(float(np.std(x)))
                        miss.append(float(np.mean(~np.isfinite(x))))
                        rr, pp = spearmanr(x, y)
                        rhos.append(float(rr))
                        pvals.append(float(pp))
                    qvals = bh_fdr(pvals)

                    for j, name in enumerate(feat):
                        assoc_rows.append({
                            "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                            "feature": name,
                            "group": feature_group(name),
                            "category": feature_category(name),
                            "n": int(n),
                            "rho": float(rhos[j]),
                            "p": float(pvals[j]),
                            "q": float(qvals[j]),
                            "mean": float(means[j]),
                            "sd": float(sds[j]),
                            "missing_rate": float(miss[j]),
                        })

                    order = np.argsort(-np.abs(np.asarray(rhos)))
                    topN = order[:20]
                    for rk, j in enumerate(topN, start=1):
                        top_rows.append({
                            "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                            "rank": rk, "feature": feat[j],
                            "rho": float(rhos[j]), "q": float(qvals[j]),
                            "group": feature_group(feat[j]),
                            "category": feature_category(feat[j]),
                        })

                    # ---------- B2: shift (no skipping; NA + flags if insufficient) ----------
                    idx_pos = (y > 0)
                    idx_neg = (y < 0)
                    qlo = np.quantile(y, 0.1)
                    qhi = np.quantile(y, 0.9)
                    idx_lo = (y <= qlo)
                    idx_hi = (y >= qhi)

                    def do_shift(idxA, idxB, contrast):
                        n1 = int(idxA.sum())
                        n2 = int(idxB.sum())
                        tmp = []
                        p_list = []
                        for j, name in enumerate(feat):
                            x = ensure_dense_col(X, j).astype(np.float64)
                            a = x[idxA]
                            b = x[idxB]
                            med_a = float(np.median(a)) if len(a) else float("nan")
                            med_b = float(np.median(b)) if len(b) else float("nan")
                            delta_med = med_a - med_b if np.isfinite(med_a) and np.isfinite(med_b) else float("nan")

                            insufficient = int((n1 < args.min_shift_n) or (n2 < args.min_shift_n) or (n1 == 0) or (n2 == 0))
                            if insufficient:
                                cd = float("nan")
                                pval = float("nan")
                            else:
                                res = mannwhitneyu(a, b, alternative="two-sided")
                                U = float(res.statistic)
                                pval = float(res.pvalue)
                                cd = float(cliff_delta_from_u(U, len(a), len(b)))

                            tmp.append((j, med_a, med_b, delta_med, cd, pval, insufficient))
                            p_list.append(pval)

                        q_list = bh_fdr(p_list)
                        for k, (j, med_a, med_b, delta_med, cd, pval, insufficient) in enumerate(tmp):
                            shift_rows.append({
                                "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                                "contrast": contrast,
                                "nA": n1, "nB": n2,
                                "insufficient": int(insufficient),
                                "feature": feat[j],
                                "group": feature_group(feat[j]),
                                "category": feature_category(feat[j]),
                                "median_A": med_a,
                                "median_B": med_b,
                                "delta_median": float(delta_med),
                                "cliff_delta": float(cd),
                                "p": float(pval),
                                "q": float(q_list[k]),
                            })

                    do_shift(idx_pos, idx_neg, "pos_vs_neg")
                    do_shift(idx_hi, idx_lo, "top10_vs_bottom10")

                    # ---------- B3: corr heatmap (topK by |rho|) ----------
                    topK = int(args.topK_corr)
                    idx_top = order[:min(topK, p)]
                    Xtop = np.vstack([ensure_dense_col(X, j).astype(np.float64) for j in idx_top]).T
                    C = np.corrcoef(Xtop, rowvar=False)
                    names = [feat[j] for j in idx_top]
                    for i in range(len(names)):
                        for j in range(len(names)):
                            corr_rows.append({
                                "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                                "feature_i": names[i],
                                "feature_j": names[j],
                                "corr": float(C[i, j]),
                            })

                    # ---------- B4: trend by y-bins (robust; never drop whole tag) ----------
                    # Use qcut with duplicates drop; if only 1 bin possible, keep 1 bin.
                    s = pd.Series(y)
                    try:
                        bins = pd.qcut(s, q=args.ybins, duplicates="drop")
                        codes = bins.cat.codes.to_numpy()
                        intervals = bins.cat.categories
                        bin_count = len(intervals)
                    except Exception:
                        codes = np.zeros_like(y, dtype=int)
                        intervals = [pd.Interval(left=float(np.min(y)), right=float(np.max(y)), closed="right")]
                        bin_count = 1

                    # trend only for top20 features (same as B1_top20)
                    for j in topN:
                        x = ensure_dense_col(X, j).astype(np.float64)
                        for b in range(bin_count):
                            msk = (codes == b)
                            bn = int(msk.sum())
                            if bn == 0:
                                continue
                            small_bin = int(bn < args.min_bin_n)
                            iv = intervals[b] if b < len(intervals) else None
                            y_lo = float(iv.left) if iv is not None else float("nan")
                            y_hi = float(iv.right) if iv is not None else float("nan")
                            trend_rows.append({
                                "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                                "feature": feat[j],
                                "bin": int(b),
                                "bin_n": bn,
                                "small_bin": small_bin,
                                "y_lo": y_lo,
                                "y_hi": y_hi,
                                "y_median": float(np.median(y[msk])),
                                "y_mean": float(np.mean(y[msk])),
                                "x_median": float(np.median(x[msk])),
                                "x_mean": float(np.mean(x[msk])),
                            })

    df_assoc = pd.DataFrame(assoc_rows)
    df_shift = pd.DataFrame(shift_rows)
    df_corr  = pd.DataFrame(corr_rows)
    df_trend = pd.DataFrame(trend_rows)
    df_top   = pd.DataFrame(top_rows)

    def out_tsv(df, p: Path):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"exists: {p} (use --overwrite)")
        if str(p).endswith(".gz"):
            df.to_csv(p, sep="\t", index=False, compression="gzip")
        else:
            df.to_csv(p, sep="\t", index=False)

    out_tsv(df_assoc, out_dir / "B1_feature_y_assoc_long.tsv.gz")
    out_tsv(df_top,   out_dir / "B1_top20_by_abs_rho.tsv")
    out_tsv(df_shift, out_dir / "B2_feature_shift_long.tsv.gz")
    out_tsv(df_corr,  out_dir / "B3_topK_feature_corr_long.tsv.gz")
    out_tsv(df_trend, out_dir / "B4_feature_trend_by_ybins.tsv.gz")

    # Consistency tables (same as before)
    if not df_top.empty:
        keycols = ["species","locus","variant","feature"]
        ct = (df_top.groupby(["tag"] + keycols).size().reset_index(name="in_top20"))
        g = ct.groupby(keycols)["tag"].nunique().reset_index(name="n_tags_in_top20")
        g = g[g["n_tags_in_top20"] == len(set(df_top["tag"]))].copy()
        out_tsv(g, out_dir / "Table_B2_consistent_features_across_tags.tsv")

        summ = (df_top.groupby(["tag","species","locus","variant","group"])
                .size().reset_index(name="n_in_top20"))
        out_tsv(summ, out_dir / "Table_B3_top20_group_composition.tsv")

    print("[DONE] B tables v2 ->", out_dir)
    print("Check tags:")
    print(" - B4 should include top0p1p now, even if only 1-2 bins exist due to y ties.")

if __name__ == "__main__":
    main()
