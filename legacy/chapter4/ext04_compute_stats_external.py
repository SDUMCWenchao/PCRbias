#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, math
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None


def log(msg: str):
    print(msg, flush=True)


def open_tsv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def open_tsv_w(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "wt", encoding="utf-8")


def safe_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "none"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def constant_or_nan(a: np.ndarray):
    a = a[np.isfinite(a)]
    if a.size < 3:
        return True
    return np.nanmin(a) == np.nanmax(a)


def calc_corr(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 20 or constant_or_nan(x) or constant_or_nan(y) or spearmanr is None:
        return n, np.nan, np.nan, np.nan, np.nan
    try:
        sr = spearmanr(x, y)
        pr = pearsonr(x, y)
        # scipy can return objects; normalize
        s_r = float(sr.correlation) if hasattr(sr, "correlation") else float(sr[0])
        s_p = float(sr.pvalue) if hasattr(sr, "pvalue") else float(sr[1])
        p_r = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
        p_p = float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1])
        return n, s_r, s_p, p_r, p_p
    except Exception:
        return n, np.nan, np.nan, np.nan, np.nan


def cohen_d(a: np.ndarray, b: np.ndarray):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return np.nan
    va = np.nanvar(a, ddof=1)
    vb = np.nanvar(b, ddof=1)
    sp = ((a.size - 1) * va + (b.size - 1) * vb) / max(1, (a.size + b.size - 2))
    if sp <= 0 or not np.isfinite(sp):
        return np.nan
    return (np.nanmean(a) - np.nanmean(b)) / math.sqrt(sp)


def detect_training_chunks_dir(project_dir: Path, prefer_subdir: str | None):
    # 1) user-specified
    if prefer_subdir:
        cand = project_dir / prefer_subdir / "training_chunks"
        if (cand / "index.tsv").exists():
            return cand

    # 2) scan analysis_results/03_DataWeaver*
    ar = project_dir / "analysis_results"
    cands = []
    if ar.exists():
        for d in ar.glob("03_DataWeaver*"):
            tc = d / "training_chunks"
            if (tc / "index.tsv").exists():
                cands.append(tc)

    if not cands:
        raise FileNotFoundError(
            f"[BAD] cannot find training_chunks. Tried:\n"
            f"  - {project_dir}/{prefer_subdir}/training_chunks\n"
            f"  - {project_dir}/analysis_results/03_DataWeaver*/training_chunks"
        )

    # pick newest by mtime of index.tsv
    cands.sort(key=lambda p: (p / "index.tsv").stat().st_mtime, reverse=True)
    return cands[0]


def iter_rows(training_chunks_dir: Path, split_keep: str):
    idx = training_chunks_dir / "index.tsv"
    with idx.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            chunk = line.split("\t")[0].strip()
            fp = training_chunks_dir / chunk
            with open_tsv(fp) as fh:
                dr = csv.DictReader(fh, delimiter="\t")
                for row in dr:
                    if split_keep != "all":
                        if (row.get("split") or "") != split_keep:
                            continue
                    yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True, help="external_test root")
    ap.add_argument("--weaver_subdir", default=None, help="e.g. analysis_results/03_DataWeaver_Blayer_with_GCfix_GCall")
    ap.add_argument("--out_dir", default="analysis_results/04_Stats", help="under project_dir")

    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    ap.add_argument("--topN", type=int, default=200, help="for shift: topN vs bottomN by log2fc")
    ap.add_argument("--min_rows_pair", type=int, default=500, help="skip compare_id with too few rows")

    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    out_dir = project / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    training_chunks_dir = detect_training_chunks_dir(project, args.weaver_subdir)
    log(f"[INFO] using training_chunks_dir = {training_chunks_dir}")

    # 1) first row to get feature columns
    first = None
    for row in iter_rows(training_chunks_dir, args.split):
        first = row
        break
    if first is None:
        raise SystemExit("[ABORT] no rows found (split filter too strict?)")

    base_cols = {
        "dataset", "exp_no", "exp_yes", "pcr_cycle_no", "pcr_cycle_yes",
        "compare_id", "split",
        "pair_id", "yes_file_id", "no_file_id",
        "Seq_ID",
        "count_yes", "count_no", "total_yes", "total_no",
        "rel_yes", "rel_no", "log2fc"
    }
    feat_cols = [c for c in first.keys() if c not in base_cols]
    feat_cols = sorted(feat_cols)
    log(f"[INFO] detected feature columns = {len(feat_cols)}")

    # 2) collect per compare_id arrays (log2fc + features)
    # store: meta + y + features lists
    meta = {}  # compare_id -> dict
    ys = defaultdict(list)  # compare_id -> list[float]
    xs = defaultdict(lambda: defaultdict(list))  # compare_id -> feat -> list[float]

    n_rows = 0
    for row in iter_rows(training_chunks_dir, args.split):
        cid = row.get("compare_id") or ""
        if not cid:
            continue
        if cid not in meta:
            meta[cid] = {
                "dataset": row.get("dataset", ""),
                "exp_no": row.get("exp_no", ""),
                "exp_yes": row.get("exp_yes", ""),
                "pcr_cycle_no": row.get("pcr_cycle_no", ""),
                "pcr_cycle_yes": row.get("pcr_cycle_yes", ""),
            }
        y = safe_float(row.get("log2fc"))
        ys[cid].append(y)
        for c in feat_cols:
            xs[cid][c].append(safe_float(row.get(c)))
        n_rows += 1
        if n_rows % 200000 == 0:
            log(f"[INFO] loaded rows={n_rows}")

    log(f"[INFO] total rows loaded = {n_rows}")
    log(f"[INFO] compare_ids = {len(meta)}")

    # 3) write overview
    overview_path = out_dir / "pairs_overview.tsv"
    with overview_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["compare_id", "dataset", "exp_yes", "exp_no", "pcr_cycle_yes", "pcr_cycle_no", "rows"])
        for cid in sorted(meta.keys()):
            m = meta[cid]
            w.writerow([cid, m["dataset"], m["exp_yes"], m["exp_no"], m["pcr_cycle_yes"], m["pcr_cycle_no"], len(ys[cid])])
    log(f"[DONE] overview -> {overview_path}")

    # 4) corr + shift
    corr_path = out_dir / "pair_feature_corr.tsv.gz"
    shift_path = out_dir / f"pair_feature_shift_top{args.topN}.tsv.gz"

    with open_tsv_w(corr_path) as fc, open_tsv_w(shift_path) as fs:
        wc = csv.writer(fc, delimiter="\t")
        ws = csv.writer(fs, delimiter="\t")

        wc.writerow([
            "compare_id", "dataset", "exp_yes", "exp_no", "pcr_cycle_yes", "pcr_cycle_no",
            "feature", "n", "spearman_r", "spearman_p", "pearson_r", "pearson_p"
        ])
        ws.writerow([
            "compare_id", "dataset", "exp_yes", "exp_no", "pcr_cycle_yes", "pcr_cycle_no",
            "feature", "topN", "n_pos", "n_neg",
            "mean_pos", "mean_neg", "delta_mean",
            "median_pos", "median_neg", "delta_median",
            "cohen_d"
        ])

        kept = 0
        for cid in sorted(meta.keys()):
            y = np.array(ys[cid], dtype=np.float64)
            if y.size < args.min_rows_pair:
                continue
            m = meta[cid]
            order = np.argsort(y)  # ascending
            bottom = order[:args.topN]
            top = order[-args.topN:] if order.size >= args.topN else order

            kept += 1
            for fcol in feat_cols:
                x = np.array(xs[cid][fcol], dtype=np.float64)

                n, sr, sp, pr, pp = calc_corr(x, y)
                wc.writerow([cid, m["dataset"], m["exp_yes"], m["exp_no"], m["pcr_cycle_yes"], m["pcr_cycle_no"],
                             fcol, n, sr, sp, pr, pp])

                xb = x[bottom]
                xt = x[top]
                n_neg = int(np.isfinite(xb).sum())
                n_pos = int(np.isfinite(xt).sum())

                mean_pos = float(np.nanmean(xt)) if np.isfinite(xt).any() else np.nan
                mean_neg = float(np.nanmean(xb)) if np.isfinite(xb).any() else np.nan
                med_pos = float(np.nanmedian(xt)) if np.isfinite(xt).any() else np.nan
                med_neg = float(np.nanmedian(xb)) if np.isfinite(xb).any() else np.nan

                ws.writerow([
                    cid, m["dataset"], m["exp_yes"], m["exp_no"], m["pcr_cycle_yes"], m["pcr_cycle_no"],
                    fcol, args.topN, n_pos, n_neg,
                    mean_pos, mean_neg, (mean_pos - mean_neg) if (np.isfinite(mean_pos) and np.isfinite(mean_neg)) else np.nan,
                    med_pos, med_neg, (med_pos - med_neg) if (np.isfinite(med_pos) and np.isfinite(med_neg)) else np.nan,
                    cohen_d(xt, xb)
                ])

        log(f"[INFO] compare_ids kept (rows>={args.min_rows_pair}) = {kept}")

    log(f"[DONE] corr  -> {corr_path}")
    log(f"[DONE] shift -> {shift_path}")
    log("[DONE] Step04 external stats finished.")


if __name__ == "__main__":
    main()
