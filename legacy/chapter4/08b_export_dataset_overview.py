#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export dataset overview for Results writing (ModelInputs_v3_topbias).

Expected layout per dataset_dir (<tag>/<species>/<locus>/<variant>/):
  - y_train.npy, y_val.npy, y_test.npy    (required for y distribution)
  - meta_train.tsv(.gz), meta_val.tsv(.gz), meta_test.tsv(.gz) (optional, for unique pair/seq counts)
  - X_train.npz ... (not used)

Output per (tag,species,locus,variant,split):
  n_rows, n_pairs, n_seqs, y mean/std/frac_pos, and quantiles.

Important: meta_*.tsv.gz may NOT contain y/log2fc; y comes from y_*.npy.
"""

import argparse
import gzip
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SPLITS = ["train", "val", "test"]

PAIR_KEYS = ["pair_id", "pairid"]
SEQ_KEYS = ["Seq_ID", "seq_id", "seqid", "SeqID"]


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else open(path, "r", encoding="utf-8", newline="")


def find_meta_file(d: Path, split: str) -> Optional[Path]:
    for p in [d / f"meta_{split}.tsv.gz", d / f"meta_{split}.tsv"]:
        if p.exists():
            return p
    return None


def find_y_file(d: Path, split: str) -> Optional[Path]:
    # primary
    p = d / f"y_{split}.npy"
    if p.exists():
        return p
    # tolerate alternate names (just in case)
    for alt in [d / f"Y_{split}.npy", d / f"label_{split}.npy", d / f"target_{split}.npy"]:
        if alt.exists():
            return alt
    return None


def detect_meta_cols(header: List[str]) -> Tuple[Optional[int], Optional[int]]:
    low = [h.strip().lower() for h in header]
    low2i = {h: i for i, h in enumerate(low)}
    idx_pair = None
    idx_seq = None
    for k in PAIR_KEYS:
        if k.lower() in low2i:
            idx_pair = low2i[k.lower()]
            break
    for k in SEQ_KEYS:
        if k.lower() in low2i:
            idx_seq = low2i[k.lower()]
            break
    return idx_pair, idx_seq


def quantiles(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {k: float("nan") for k in ["min","q01","q05","q10","q25","q50","q75","q90","q95","q99","max"]}
    return {
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "nan"
        return f"{v:.6g}"
    return str(v)


def summarize_meta(meta_path: Optional[Path], unique_cap: int) -> Tuple[str, str]:
    """Return (n_pairs, n_seqs) as strings. If meta missing -> empty."""
    if meta_path is None or not meta_path.exists():
        return ("", "")
    with open_text(meta_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_pair, idx_seq = detect_meta_cols(header)
        pair_set = set()
        seq_set = set()
        pair_over = False
        seq_over = False
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if idx_pair is not None and not pair_over and len(parts) > idx_pair:
                pair_set.add(parts[idx_pair])
                if len(pair_set) >= unique_cap:
                    pair_over = True
            if idx_seq is not None and not seq_over and len(parts) > idx_seq:
                seq_set.add(parts[idx_seq])
                if len(seq_set) >= unique_cap:
                    seq_over = True
    n_pairs = (f">={unique_cap}" if pair_over else str(len(pair_set)) if idx_pair is not None else "")
    n_seqs  = (f">={unique_cap}" if seq_over  else str(len(seq_set))  if idx_seq  is not None else "")
    return (n_pairs, n_seqs)


def summarize_y(y_path: Path) -> Dict[str, str]:
    y = np.load(y_path).astype(np.float64)
    # drop NaN just in case
    y = y[~np.isnan(y)]
    out = {
        "n_rows": str(int(y.size)),
        "y_mean": fmt(float(np.mean(y)) if y.size else float("nan")),
        "y_std": fmt(float(np.std(y, ddof=1)) if y.size > 1 else float("nan")),
        "y_frac_pos": fmt(float(np.mean(y >= 0)) if y.size else float("nan")),
    }
    qs = quantiles(y)
    for k, v in qs.items():
        out[f"y_{k}"] = fmt(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--out_tsv", default="analysis_results/07_PlotTables_v3_topbias/dataset_overview.tsv")
    ap.add_argument("--unique_cap", type=int, default=200000)
    ap.add_argument("--include_missing", action="store_true",
                    help="also emit rows when y/meta missing (with blanks)")
    args = ap.parse_args()

    root = Path(args.inputs_root)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    dirs = sorted([p for p in root.glob("*/*/*/*") if p.is_dir()])

    fields = [
        "tag","species","locus","variant","split",
        "y_file","meta_file",
        "n_rows","n_pairs","n_seqs",
        "y_mean","y_std","y_frac_pos",
        "y_min","y_q01","y_q05","y_q10","y_q25","y_q50","y_q75","y_q90","y_q95","y_q99","y_max",
    ]

    # cache by (tag,species,locus,split) because all variants share identical y/meta distributions
    cache: Dict[Tuple[str,str,str,str], Dict[str,str]] = {}

    rows = []
    for d in dirs:
        rel = d.relative_to(root)
        if len(rel.parts) != 4:
            continue
        tag, species, locus, variant = rel.parts

        for split in SPLITS:
            key = (tag, species, locus, split)
            if key in cache:
                base = cache[key].copy()
                base["variant"] = variant
                rows.append({k: base.get(k, "") for k in fields})
                continue

            y_path = find_y_file(d, split)
            meta_path = find_meta_file(d, split)

            if y_path is None:
                if args.include_missing:
                    row = {
                        "tag":tag,"species":species,"locus":locus,"variant":variant,"split":split,
                        "y_file":"", "meta_file": str(meta_path) if meta_path else "",
                    }
                    rows.append({k: row.get(k, "") for k in fields})
                continue

            try:
                y_stats = summarize_y(y_path)
            except Exception as e:
                print(f"[WARN] failed y for {rel} split={split} file={y_path}: {e}")
                if args.include_missing:
                    row = {"tag":tag,"species":species,"locus":locus,"variant":variant,"split":split,
                           "y_file":str(y_path)+" [ERROR]", "meta_file": str(meta_path) if meta_path else ""}
                    rows.append({k: row.get(k, "") for k in fields})
                continue

            n_pairs, n_seqs = summarize_meta(meta_path, unique_cap=args.unique_cap)

            row = {
                "tag":tag, "species":species, "locus":locus, "variant":variant, "split":split,
                "y_file": str(y_path),
                "meta_file": str(meta_path) if meta_path else "",
                "n_pairs": n_pairs,
                "n_seqs": n_seqs,
            }
            row.update(y_stats)

            # store cache (without variant)
            cached = row.copy()
            cached["variant"] = "__cached__"
            cache[key] = cached

            rows.append({k: row.get(k, "") for k in fields})

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k,"")) for k in fields) + "\n")

    print(f"[DONE] wrote: {out_tsv}")
    print(f"[INFO] dataset_dirs={len(dirs)}  rows={len(rows)}")

if __name__ == "__main__":
    main()
