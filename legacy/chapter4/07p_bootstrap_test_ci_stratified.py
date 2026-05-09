#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified bootstrap CI on TEST predictions for all models under models_root.

Key features:
- Stratified resampling by y_true (log2fc) distribution (adaptive quantile bins).
- Robust to constant y/pred causing Spearman NaN (records nan_frac; CI computed ignoring NaN).
- Scans model directories and automatically finds pred_test.tsv(.gz) etc.
- Optionally merges CI back into an existing master metrics TSV (write to new file; no overwrite by default).

Expected prediction file format: TSV/TSV.GZ with columns containing y_true and y_pred
(autodetected from common names).
"""

import argparse
import csv
import gzip
import math
import os
from pathlib import Path
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# scipy is preferred for Spearman; fallback to numpy if unavailable
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None


PRED_FILE_CANDIDATES = [
    "pred_test.tsv.gz", "pred_test.tsv",
    "predictions_test.tsv.gz", "predictions_test.tsv",
    "test_pred.tsv.gz", "test_pred.tsv",
    "test_predictions.tsv.gz", "test_predictions.tsv",
]

Y_TRUE_KEYS = ["y_true", "y", "log2fc_true", "true", "target", "label"]
Y_PRED_KEYS = ["y_pred", "pred", "log2fc_pred", "prediction", "output"]

KEY_FIELDS = ["tag", "species", "locus", "model", "variant"]


def stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def read_tsv_any(path: Path) -> Tuple[List[str], List[List[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    return header, rows


def detect_cols(header: List[str]) -> Tuple[Optional[int], Optional[int]]:
    idx_true = None
    idx_pred = None
    lower = [h.strip() for h in header]
    low2i = {h.lower(): i for i, h in enumerate(lower)}
    for k in Y_TRUE_KEYS:
        if k.lower() in low2i:
            idx_true = low2i[k.lower()]
            break
    for k in Y_PRED_KEYS:
        if k.lower() in low2i:
            idx_pred = low2i[k.lower()]
            break
    return idx_true, idx_pred


def load_y_pred(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    header, rows = read_tsv_any(path)
    it, ip = detect_cols(header)
    if it is None or ip is None:
        raise ValueError(f"Cannot detect y_true/y_pred columns in {path.name}. header={header[:20]}")
    y = []
    p = []
    for r in rows:
        try:
            yt = float(r[it])
            yp = float(r[ip])
        except Exception:
            continue
        if math.isnan(yt) or math.isnan(yp):
            continue
        y.append(yt)
        p.append(yp)
    if len(y) == 0:
        raise ValueError(f"No valid rows parsed from {path}")
    return np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(p - y)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    # avoid sklearn dependency
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def sign_acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y >= 0) == (p >= 0)))


def safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    if spearmanr is None:
        # crude fallback: rank then pearson
        ry = np.argsort(np.argsort(y))
        rp = np.argsort(np.argsort(p))
        return float(np.corrcoef(ry, rp)[0, 1])
    return float(spearmanr(y, p).correlation)


def safe_pearson(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    if pearsonr is None:
        return float(np.corrcoef(y, p)[0, 1])
    return float(pearsonr(y, p)[0])


def choose_nbins(n: int, max_bins: int) -> int:
    # adaptive, stable for small n (top0.1p)
    # sqrt(n) but capped; minimum 2
    k = int(round(math.sqrt(n)))
    k = max(2, min(max_bins, k))
    # avoid too many bins when n is tiny
    if n < 20:
        k = min(k, 3)
    return k


def make_quantile_bins(y: np.ndarray, nbins: int) -> np.ndarray:
    # Returns bin edges length nbins+1, increasing.
    # Ensure strictly increasing by de-duplicating.
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(y, qs)
    # enforce monotonic
    edges2 = [edges[0]]
    for v in edges[1:]:
        if v <= edges2[-1]:
            edges2.append(edges2[-1] + 1e-12)
        else:
            edges2.append(v)
    return np.asarray(edges2, dtype=np.float64)


def stratified_indices(y: np.ndarray, rng: np.random.Generator, max_bins: int) -> Tuple[np.ndarray, int]:
    n = len(y)
    nbins = choose_nbins(n, max_bins)
    edges = make_quantile_bins(y, nbins)
    # bin id in [0, nbins-1]
    bin_id = np.clip(np.digitize(y, edges[1:-1], right=False), 0, nbins - 1)

    idx_all = []
    for b in range(nbins):
        idx = np.where(bin_id == b)[0]
        if len(idx) == 0:
            continue
        # sample with replacement within bin (preserve bin size)
        samp = rng.choice(idx, size=len(idx), replace=True)
        idx_all.append(samp)
    if len(idx_all) == 0:
        # fallback: plain bootstrap
        return rng.choice(np.arange(n), size=n, replace=True), nbins
    return np.concatenate(idx_all), nbins


@dataclass
class OneResult:
    tag: str
    species: str
    locus: str
    model: str
    variant: str
    n_test: int
    nbins: int
    seed: int
    B: int

    spearman: float
    pearson: float
    rmse: float
    mae: float
    r2: float
    sign_acc: float

    spearman_ci_low: float
    spearman_ci_high: float
    spearman_boot_mean: float
    spearman_boot_sd: float
    spearman_nan_frac: float

    rmse_ci_low: float
    rmse_ci_high: float
    rmse_boot_mean: float
    rmse_boot_sd: float

    def to_dict(self) -> Dict[str, str]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}
        # format
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                out[k] = "nan" if (math.isnan(v) or math.isinf(v)) else f"{v:.6g}"
            else:
                out[k] = str(v)
        return out


def percentile_ci(arr: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo = np.nanpercentile(arr, 100.0 * (alpha / 2.0))
    hi = np.nanpercentile(arr, 100.0 * (1.0 - alpha / 2.0))
    return float(lo), float(hi)


def compute_bootstrap(y: np.ndarray, p: np.ndarray, B: int, seed: int, max_bins: int, alpha: float) -> Tuple[dict, int]:
    rng = np.random.default_rng(seed)
    sp_list = []
    rmse_list = []
    nbins_used = None

    for _ in range(B):
        idx, nb = stratified_indices(y, rng, max_bins=max_bins)
        if nbins_used is None:
            nbins_used = nb
        yy = y[idx]
        pp = p[idx]
        sp = safe_spearman(yy, pp)
        sp_list.append(sp)
        rmse_list.append(rmse(yy, pp))

    sp_arr = np.asarray(sp_list, dtype=np.float64)
    rmse_arr = np.asarray(rmse_list, dtype=np.float64)

    sp_nan = np.mean(np.isnan(sp_arr)) if len(sp_arr) else float("nan")
    sp_valid = sp_arr[~np.isnan(sp_arr)]
    if len(sp_valid) >= max(10, int(0.1 * B)):
        sp_lo, sp_hi = percentile_ci(sp_valid, alpha)
        sp_mean = float(np.nanmean(sp_valid))
        sp_sd = float(np.nanstd(sp_valid, ddof=1)) if len(sp_valid) > 1 else 0.0
    else:
        sp_lo, sp_hi, sp_mean, sp_sd = float("nan"), float("nan"), float("nan"), float("nan")

    rm_lo, rm_hi = percentile_ci(rmse_arr, alpha)
    rm_mean = float(np.mean(rmse_arr))
    rm_sd = float(np.std(rmse_arr, ddof=1)) if len(rmse_arr) > 1 else 0.0

    return {
        "spearman_ci_low": sp_lo,
        "spearman_ci_high": sp_hi,
        "spearman_boot_mean": sp_mean,
        "spearman_boot_sd": sp_sd,
        "spearman_nan_frac": float(sp_nan),
        "rmse_ci_low": rm_lo,
        "rmse_ci_high": rm_hi,
        "rmse_boot_mean": rm_mean,
        "rmse_boot_sd": rm_sd,
    }, (nbins_used if nbins_used is not None else choose_nbins(len(y), max_bins))


def find_pred_file(model_variant_dir: Path) -> Optional[Path]:
    for fn in PRED_FILE_CANDIDATES:
        p = model_variant_dir / fn
        if p.exists():
            return p
    # sometimes in subdir like "eval" or "preds"
    for sub in ["eval", "preds", "pred", "outputs"]:
        for fn in PRED_FILE_CANDIDATES:
            p = model_variant_dir / sub / fn
            if p.exists():
                return p
    return None


def scan_models(models_root: Path) -> Dict[Tuple[str, str, str, str, str], Path]:
    """
    Return mapping key=(tag,species,locus,model,variant) -> pred_file_path
    by rglob scanning for candidate prediction files.
    """
    mapping = {}
    for fn in PRED_FILE_CANDIDATES:
        for p in models_root.rglob(fn):
            rel = p.relative_to(models_root)
            parts = rel.parts
            if len(parts) < 5:
                continue
            tag, species, locus, model, variant = parts[0], parts[1], parts[2], parts[3], parts[4]
            if model not in {"rf", "xgb", "seqcnn", "cnn1d"}:
                continue
            key = (tag, species, locus, model, variant)
            # prefer the shortest path (closest to variant dir)
            if key not in mapping or len(str(p)) < len(str(mapping[key])):
                mapping[key] = p
    return mapping


def write_tsv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def merge_master(master_tsv: Path, ci_tsv: Path, out_master: Path):
    # load master
    with open(master_tsv, "r", encoding="utf-8", newline="") as f:
        m = list(csv.DictReader(f, delimiter="\t"))
        m_fields = f"{open(master_tsv,'r',encoding='utf-8').readline()}".strip().split("\t")

    # load ci
    with open(ci_tsv, "r", encoding="utf-8", newline="") as f:
        c = list(csv.DictReader(f, delimiter="\t"))

    idx = {}
    for r in c:
        key = tuple(r[k] for k in KEY_FIELDS)
        idx[key] = r

    # columns to inject/replace
    inject_cols = [
        "spearman_ci_low", "spearman_ci_high", "spearman_boot_mean", "spearman_boot_sd", "spearman_nan_frac",
        "rmse_ci_low", "rmse_ci_high", "rmse_boot_mean", "rmse_boot_sd",
        "nbins", "B", "seed",
    ]

    # expand master fields if missing
    out_fields = list(m_fields)
    for col in inject_cols:
        if col not in out_fields:
            out_fields.append(col)

    out_rows = []
    hit = 0
    for r in m:
        key = tuple(r.get(k, "") for k in KEY_FIELDS)
        if key in idx:
            hit += 1
            for col in inject_cols:
                r[col] = idx[key].get(col, "")
        out_rows.append(r)

    out_master.parent.mkdir(parents=True, exist_ok=True)
    with open(out_master, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=out_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in out_fields})

    print(f"[DONE] merged master -> {out_master} (matched={hit}/{len(m)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True, help="e.g. analysis_results/06_Models_v3_topbias")
    ap.add_argument("--out_tsv", required=True, help="output CI table")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap replicates (default 1000)")
    ap.add_argument("--seed", type=int, default=1, help="global seed (default 1)")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (default 0.05 => 95% CI)")
    ap.add_argument("--max_bins", type=int, default=8, help="max strat bins (adaptive; default 8)")
    ap.add_argument("--merge_master", action="store_true", help="also merge CI into master metrics tsv")
    ap.add_argument("--master_tsv", default="", help="path to 01_master_metrics_ci.tsv")
    ap.add_argument("--out_master", default="", help="where to write merged master tsv")
    args = ap.parse_args()

    models_root = Path(args.models_root)
    out_tsv = Path(args.out_tsv)

    print(f"[INFO] scanning preds under: {models_root}")
    mapping = scan_models(models_root)
    print(f"[INFO] found prediction files for models: {len(mapping)}")

    rows_out = []
    n_ok = 0
    n_fail = 0

    # deterministic order
    for key in sorted(mapping.keys()):
        tag, species, locus, model, variant = key
        pred_path = mapping[key]
        try:
            y, p = load_y_pred(pred_path)
            n = len(y)
            # point estimates
            sp = safe_spearman(y, p)
            pe = safe_pearson(y, p)
            rm = rmse(y, p)
            ma = mae(y, p)
            r2v = r2(y, p)
            sa = sign_acc(y, p)

            # per-model seed (stable)
            s_model = args.seed + stable_int_hash(str(pred_path)) % 10_000_000
            boot, nbins_used = compute_bootstrap(y, p, B=args.B, seed=s_model, max_bins=args.max_bins, alpha=args.alpha)

            res = OneResult(
                tag=tag, species=species, locus=locus, model=model, variant=variant,
                n_test=n, nbins=nbins_used, seed=args.seed, B=args.B,
                spearman=sp, pearson=pe, rmse=rm, mae=ma, r2=r2v, sign_acc=sa,
                spearman_ci_low=boot["spearman_ci_low"],
                spearman_ci_high=boot["spearman_ci_high"],
                spearman_boot_mean=boot["spearman_boot_mean"],
                spearman_boot_sd=boot["spearman_boot_sd"],
                spearman_nan_frac=boot["spearman_nan_frac"],
                rmse_ci_low=boot["rmse_ci_low"],
                rmse_ci_high=boot["rmse_ci_high"],
                rmse_boot_mean=boot["rmse_boot_mean"],
                rmse_boot_sd=boot["rmse_boot_sd"],
            )
            rows_out.append(res.to_dict())
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] failed {key} pred={pred_path}: {e}")

    if not rows_out:
        raise RuntimeError("No CI computed. Check if prediction files exist under models_root.")

    write_tsv(out_tsv, rows_out)
    print(f"[DONE] wrote CI table: {out_tsv}  ok={n_ok}  fail={n_fail}")

    if args.merge_master:
        if not args.master_tsv or not args.out_master:
            raise ValueError("--merge_master requires --master_tsv and --out_master")
        merge_master(Path(args.master_tsv), out_tsv, Path(args.out_master))


if __name__ == "__main__":
    main()
