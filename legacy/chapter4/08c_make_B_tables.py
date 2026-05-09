#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, math
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def read_json(p: Path):
    return json.loads(p.read_text())

def load_feature_names(d: Path):
    # prefer json
    for fn in ["feature_cols.json", "feature_names.json", "cols.json"]:
        p = d / fn
        if p.exists():
            obj = read_json(p)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "feature_cols" in obj:
                return obj["feature_cols"]
    # fallback txt
    for fn in ["feature_cols.txt", "feature_names.txt", "cols.txt"]:
        p = d / fn
        if p.exists():
            return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    raise FileNotFoundError(f"cannot find feature names in {d}")

def load_X(d: Path, split: str):
    # support X_{split}.npz (sparse) or .npy (dense)
    p_npz = d / f"X_{split}.npz"
    p_npy = d / f"X_{split}.npy"
    if p_npz.exists():
        X = sparse.load_npz(p_npz)
        return X
    if p_npy.exists():
        # mmap to save RAM
        X = np.load(p_npy, mmap_mode="r")
        return X
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
    # 你可以后期再细化；这个分类足够支撑B组图
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
    # return 1D dense vector
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
    # delta = (2U / (n1*n2)) - 1
    return (2.0 * U) / (n1 * n2) - 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--out_dir", default="analysis_results/07_PlotTables_v3_topbias/B_tables")
    ap.add_argument("--tags", default="top1p,top0p5p,top0p1p")
    ap.add_argument("--species", default="donkey,pig,cattle,10mix")
    ap.add_argument("--loci", default="12S,16S")
    ap.add_argument("--variants", default="no_kmer,no_kmer_noprimer,all_noprimer")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--topK_corr", type=int, default=40, help="topK features (by |rho|) used for corr heatmap")
    ap.add_argument("--ybins", type=int, default=10, help="number of y bins (quantile bins) for trend tables")
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

    assoc_rows = []
    shift_rows = []
    corr_rows = []
    trend_rows = []
    top_rows = []

    for tag in tags:
        for sp in species:
            for lc in loci:
                for var in variants:
                    d = inputs_root / tag / sp / lc / var
                    if not d.exists():
                        # 不报错：有些组不一定有该variant
                        continue

                    feat = load_feature_names(d)
                    X = load_X(d, args.split)
                    y = load_y(d, args.split)

                    n = len(y)
                    p = len(feat)
                    # 1) Spearman：一次性算 rho/p（矩阵列较少时很快）
                    #    注意：spearmanr 会返回 (p+1)*(p+1) 的相关矩阵，我们只取 y vs feature
                    #    为避免对稀疏矩阵 densify 过大，这里逐列取值（p~100-150，OK）
                    rhos = []
                    pvals = []
                    means = []
                    sds = []
                    miss = []
                    for j, name in enumerate(feat):
                        x = ensure_dense_col(X, j).astype(np.float64)
                        m = np.mean(x)
                        s = np.std(x)
                        means.append(m)
                        sds.append(s)
                        miss.append(float(np.mean(~np.isfinite(x))))
                        # spearman
                        rr, pp = spearmanr(x, y)
                        rhos.append(float(rr))
                        pvals.append(float(pp))

                    qvals = bh_fdr(pvals)

                    # assoc long
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

                    # top table（每组top20，方便做Table B-1 / 条形图）
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

                    # 2) shift：pos_vs_neg 与 top10_vs_bottom10
                    y_pos = y[y > 0]
                    y_neg = y[y < 0]
                    idx_pos = (y > 0)
                    idx_neg = (y < 0)

                    # 分位
                    qlo = np.quantile(y, 0.1)
                    qhi = np.quantile(y, 0.9)
                    idx_lo = (y <= qlo)
                    idx_hi = (y >= qhi)

                    def do_shift(idxA, idxB, contrast):
                        p_list = []
                        tmp = []
                        n1 = int(idxA.sum())
                        n2 = int(idxB.sum())
                        if n1 < 20 or n2 < 20:
                            return  # 太小就不做（避免无意义统计）
                        for j, name in enumerate(feat):
                            x = ensure_dense_col(X, j).astype(np.float64)
                            a = x[idxA]
                            b = x[idxB]
                            # robust summary
                            med_a = float(np.median(a))
                            med_b = float(np.median(b))
                            delta_med = med_a - med_b
                            # MWU + Cliff's delta
                            try:
                                res = mannwhitneyu(a, b, alternative="two-sided")
                                U = float(res.statistic)
                                pval = float(res.pvalue)
                                cd = float(cliff_delta_from_u(U, len(a), len(b)))
                            except Exception:
                                pval = float("nan")
                                cd = float("nan")
                            tmp.append((j, med_a, med_b, delta_med, cd, pval))
                            p_list.append(pval)

                        q_list = bh_fdr(p_list)
                        for k, (j, med_a, med_b, delta_med, cd, pval) in enumerate(tmp):
                            shift_rows.append({
                                "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                                "contrast": contrast,
                                "nA": n1, "nB": n2,
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

                    # 3) corr heatmap：取 topK features by |rho|
                    topK = int(args.topK_corr)
                    idx_top = order[:min(topK, p)]
                    # assemble dense matrix of topK features
                    Xtop = np.vstack([ensure_dense_col(X, j).astype(np.float64) for j in idx_top]).T  # n x K
                    # pearson corr
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

                    # 4) trend：按 y 分位分箱，看特征随 y 增强的趋势（用于画“关系曲线”）
                    # 这里默认做 top20 features（与 top_rows 一致）
                    edges = np.quantile(y, np.linspace(0, 1, args.ybins + 1))
                    # 防止重复边界
                    edges = np.unique(edges)
                    if len(edges) >= 3:
                        bin_id = np.digitize(y, edges[1:-1], right=True)  # 0..nbin-1
                        # 用 top20 feature
                        for j in topN:
                            x = ensure_dense_col(X, j).astype(np.float64)
                            for b in range(bin_id.min(), bin_id.max() + 1):
                                msk = (bin_id == b)
                                if msk.sum() < 50:
                                    continue
                                trend_rows.append({
                                    "tag": tag, "species": sp, "locus": lc, "variant": var, "split": args.split,
                                    "feature": feat[j],
                                    "bin": int(b),
                                    "bin_n": int(msk.sum()),
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

    out_dir.mkdir(parents=True, exist_ok=True)

    def out_gz(df, name):
        p = out_dir / name
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"exists: {p} (use --overwrite)")
        df.to_csv(p, sep="\t", index=False, compression="gzip" if str(p).endswith(".gz") else None)

    out_gz(df_assoc, "B1_feature_y_assoc_long.tsv.gz")
    out_gz(df_top,   "B1_top20_by_abs_rho.tsv")
    out_gz(df_shift, "B2_feature_shift_long.tsv.gz")
    out_gz(df_corr,  "B3_topK_feature_corr_long.tsv.gz")
    out_gz(df_trend, "B4_feature_trend_by_ybins.tsv.gz")

    # 额外：Table B-2（跨tag一致性：在同 species×locus×variant 内，三tag都进top20）
    # 这里用 split=train 的 top20 表做一致性统计
    if not df_top.empty:
        keycols = ["species","locus","variant","feature"]
        ct = (df_top.groupby(["tag"] + keycols).size().reset_index(name="in_top20"))
        # tag 必须三种都出现
        g = ct.groupby(keycols)["tag"].nunique().reset_index(name="n_tags_in_top20")
        g = g[g["n_tags_in_top20"] == len(set(df_top["tag"]))].copy()
        g.to_csv(out_dir / "Table_B2_consistent_features_across_tags.tsv", sep="\t", index=False)

    # 额外：Table B-3（primer vs nonprimer 占比）
    if not df_top.empty:
        summ = (df_top.groupby(["tag","species","locus","variant","group"])
                .size().reset_index(name="n_in_top20"))
        summ.to_csv(out_dir / "Table_B3_top20_group_composition.tsv", sep="\t", index=False)

    print("[DONE] B tables ->", out_dir)
    print(" - B1_feature_y_assoc_long.tsv.gz")
    print(" - B1_top20_by_abs_rho.tsv")
    print(" - B2_feature_shift_long.tsv.gz")
    print(" - B3_topK_feature_corr_long.tsv.gz")
    print(" - B4_feature_trend_by_ybins.tsv.gz")
    print(" - Table_B2_consistent_features_across_tags.tsv")
    print(" - Table_B3_top20_group_composition.tsv")

if __name__ == "__main__":
    main()
